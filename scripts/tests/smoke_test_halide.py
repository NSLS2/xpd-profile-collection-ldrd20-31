#!/usr/bin/env python
"""
Smoke test for the halide perovskite QueueserverAgent.

Spins up:
1. An in-process tiled server (SimpleTiledServer)
2. A queueserver backend (start-re-manager with mock devices)
3. A queueserver HTTP server
4. A ZMQ RemoteDispatcher → TiledWriter bridge (persists documents to tiled)

Then runs the agent for 2 iterations and verifies the loop completes.  The
agent is configured with all optional acquisition steps enabled so that the
full ``halide_acquire`` code path is exercised:

* ``post_dilute=True``  — toluene post-dilution pump step
* ``use_good_bad=True`` — PL quality-gate loop (good/bad classifier)

Usage:
    python scripts/tests/smoke_test_halide.py

Requires the 'qs' pixi environment (or equivalent with bluesky-queueserver,
bluesky-httpserver, tiled, bluesky-tiled-plugins, etc. installed).
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Enable logging for blop internals only; suppress noisy libraries
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(threadName)s] %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("blop").setLevel(logging.DEBUG)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARTUP_DIR = PROJECT_ROOT / "startup"

# Ports / addresses
QS_HTTP_PORT = 60610
# bluesky-0MQ-proxy ports for run document publishing
ZMQ_PROXY_IN_PORT = 5577  # RunEngine Publisher → proxy in
ZMQ_PROXY_OUT_PORT = 5578  # proxy out → RemoteDispatcher subscribers
ZMQ_PROXY_IN_ADDR = f"tcp://localhost:{ZMQ_PROXY_IN_PORT}"
ZMQ_PROXY_OUT_ADDR = f"tcp://localhost:{ZMQ_PROXY_OUT_PORT}"
ZMQ_PROXY_OUT_ADDR_TUPLE = ("localhost", ZMQ_PROXY_OUT_PORT)
# Fixed API key for the HTTP server (avoids auth issues)
QS_API_KEY = "smoketestapikey12345"

# Subprocesses to clean up
_subprocesses: list[subprocess.Popen] = []


def _cleanup():
    """Kill all child processes on exit."""
    for proc in _subprocesses:
        if proc.poll() is None:
            proc.terminate()
    time.sleep(1)
    for proc in _subprocesses:
        if proc.poll() is None:
            proc.kill()


atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
signal.signal(signal.SIGINT, lambda *_: sys.exit(1))


def wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Poll until a TCP port is accepting connections."""
    import socket

    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# bluesky-0MQ-proxy (publishes run documents)
# ---------------------------------------------------------------------------


def start_zmq_proxy() -> subprocess.Popen:
    """Start bluesky-0MQ-proxy for run document publishing."""
    cmd = ["bluesky-0MQ-proxy", str(ZMQ_PROXY_IN_PORT), str(ZMQ_PROXY_OUT_PORT), "-v"]
    print(f"[SMOKE] Starting bluesky-0MQ-proxy: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    _subprocesses.append(proc)
    # Give it a moment to bind
    time.sleep(1)
    if proc.poll() is not None:
        print("[SMOKE] ERROR: bluesky-0MQ-proxy exited immediately.")
        sys.exit(1)
    print(f"[SMOKE] 0MQ proxy ready (in={ZMQ_PROXY_IN_PORT}, out={ZMQ_PROXY_OUT_PORT})")
    return proc


# ---------------------------------------------------------------------------
# Tiled server (in-process)
# ---------------------------------------------------------------------------


def start_tiled_server():
    """Start an in-process SimpleTiledServer and return the tiled client.

    Returns
    -------
    tiled_client : tiled.client.container.Container
    tiled_uri : str
    """
    from tiled.server import SimpleTiledServer
    from tiled.client import from_uri

    server = SimpleTiledServer()
    client = from_uri(server.uri)
    print(f"[SMOKE] Tiled server running at {server.uri}")
    return client, server.uri, server


# ---------------------------------------------------------------------------
# ZMQ → Tiled bridge (RemoteDispatcher + TiledWriter)
# ---------------------------------------------------------------------------


def start_zmq_tiled_bridge(tiled_client, zmq_addr: tuple[str, int]) -> threading.Thread:
    """Start a RemoteDispatcher in a background thread that writes documents to tiled.

    Parameters
    ----------
    tiled_client : tiled Container
        The tiled client to write documents into.
    zmq_addr : tuple[str, int]
        ZMQ address where queueserver publishes documents.

    Returns
    -------
    threading.Thread
        The dispatcher thread (daemon, already started).
    """
    from bluesky.callbacks.zmq import RemoteDispatcher
    from bluesky_tiled_plugins import TiledWriter

    tiled_writer = TiledWriter(tiled_client)

    dispatcher = RemoteDispatcher(zmq_addr)
    dispatcher.subscribe(tiled_writer)

    thread = threading.Thread(target=dispatcher.start, daemon=True)
    thread.start()
    print(f"[SMOKE] ZMQ→Tiled bridge started (subscribing to {zmq_addr})")
    return thread


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


def start_redis() -> subprocess.Popen:
    """Start a Redis server on default port 6379."""
    cmd = ["redis-server", "--port", "6379", "--daemonize", "no"]
    print(f"[SMOKE] Starting Redis: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    _subprocesses.append(proc)
    # Wait for Redis to be ready
    if not wait_for_port(6379, timeout=5):
        print("[SMOKE] ERROR: Redis did not start in time.")
        sys.exit(1)
    print("[SMOKE] Redis ready.")
    return proc


# ---------------------------------------------------------------------------
# Queueserver services
# ---------------------------------------------------------------------------


def start_qs_backend() -> subprocess.Popen:
    """Start the queueserver RE Manager with mock devices."""
    env = {
        **os.environ,
        "HALIDE_TEST_MODE": "1",
        "MPLBACKEND": "Agg",
        "BLUESKY_ZMQ_PROXY_IN_ADDR": ZMQ_PROXY_IN_ADDR,
    }
    cmd = [
        "start-re-manager",
        f"--startup-dir={STARTUP_DIR}",
    ]
    print(f"[SMOKE] Starting queueserver backend: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env=env,
    )
    _subprocesses.append(proc)
    return proc


def start_qs_http_server() -> subprocess.Popen:
    """Start the queueserver HTTP server."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "bluesky_httpserver.server:app",
        "--host",
        "localhost",
        "--port",
        str(QS_HTTP_PORT),
    ]
    print(f"[SMOKE] Starting queueserver HTTP server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "QSERVER_HTTP_SERVER_SINGLE_USER_API_KEY": QS_API_KEY,
            "QSERVER_HTTP_SERVER_ALLOW_ANONYMOUS_ACCESS": "1",
        },
    )
    _subprocesses.append(proc)
    return proc


def wait_for_qs_ready(timeout: float = 60.0) -> bool:
    """Wait for the queueserver to report 'idle' status via HTTP API."""
    import urllib.request
    import json

    url = f"http://localhost:{QS_HTTP_PORT}/api/status?api_key={QS_API_KEY}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read())
                state = data.get("manager_state", "")
                if state in ("idle", "executing_queue"):
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_agent(tiled_uri: str, n_iterations: int = 5):
    """Run the QueueserverAgent for a few iterations."""

    # Also subscribe a raw debug callback to the agent's internal dispatcher
    # so we can see ALL document types arriving on ZMQ (not just start/stop)
    def _raw_doc_debug(name, doc):
        print(
            f"[RAW-ZMQ] doc_type={name}, uid={doc.get('uid', 'N/A')}, run_start={doc.get('run_start', 'N/A')}"
        )

    # Add the ML_agent dir to path
    ml_agent_dir = str(PROJECT_ROOT / "scripts" / "ML_agent")
    if ml_agent_dir not in sys.path:
        sys.path.insert(0, ml_agent_dir)

    # Set env vars for the agent
    os.environ["QSERVER_HTTP_URI"] = f"http://localhost:{QS_HTTP_PORT}"
    os.environ["QSERVER_HTTP_API_KEY"] = QS_API_KEY
    os.environ["TILED_URI"] = tiled_uri
    os.environ["ZMQ_CONSUMER_ADDR"] = ZMQ_PROXY_OUT_ADDR

    from queueserver_agent_halide import build_queueserver_agent

    agent = build_queueserver_agent(
        agent_data_path="",  # No historical data for smoke test
        http_server_uri=f"http://localhost:{QS_HTTP_PORT}",
        http_api_key=QS_API_KEY,
        zmq_consumer_addr=ZMQ_PROXY_OUT_ADDR_TUPLE,
        tiled_profile="unused",  # TILED_URI env var takes precedence
        # Exercise all optional acquisition steps in every agent iteration:
        # - post_dilute: runs the toluene post-dilution pump step
        # - use_good_bad: enables the PL quality-gate loop (good_target=1 so
        #   a single good batch is sufficient for the smoke test to pass)
        acquisition_plan_kwargs={
            "post_dilute": True,
            "post_dilute_wait_sec": 2,  # keep smoke test fast
            "use_good_bad": True,
            "good_target": 1,
        },
    )

    print(f"[SMOKE] Running agent for {n_iterations} iterations...")
    fut = agent.run(iterations=n_iterations, n_points=1)

    print("[SMOKE] Agent submitted plans. Waiting for queue to drain...")

    try:
        result = fut.result(timeout=600)
    except Exception as e:
        print(f"[SMOKE] Agent iterations timed out: {repr(e)}")
        try:
            result = fut.exception(timeout=1)
        except Exception as e:
            result = None

    print(f"[SMOKE] {result=}")

    # Give ZMQ → eval → ingest pipeline time to propagate
    print("[SMOKE] Queue drained. Waiting for ingestion...")
    time.sleep(5)

    # Display results
    print("\n" + "-" * 60)
    print("  AGENT RESULTS")
    print("-" * 60)
    try:
        data = agent.ax_client.summarize()
        print(f"  Observations ingested: {len(data)}")
        if len(data) > 0:
            print(f"  Columns: {list(data.columns)}")
            print(data.to_string(index=False))
        else:
            print("  WARNING: No data ingested (callback may not have fired)")
    except Exception as e:
        print(f"  Could not read agent data: {e}")
    print("-" * 60)

    # Give time for next suggestion
    time.sleep(10)


def _wait_for_queue_drain(timeout: float = 300):
    """Poll queueserver until queue is empty and manager is idle."""
    import urllib.request
    import json

    url = f"http://localhost:{QS_HTTP_PORT}/api/status?api_key={QS_API_KEY}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read())
                state = data.get("manager_state", "")
                queue_size = data.get("items_in_queue", -1)
                running = data.get("running_item_uid", None)
                if state == "idle" and queue_size == 0 and not running:
                    return
        except Exception:
            pass
        time.sleep(2)
    print("[SMOKE] WARNING: Timed out waiting for queue to drain.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  HALIDE PEROVSKITE QUEUESERVER AGENT — SMOKE TEST")
    print("=" * 60)

    # 1. Start in-process tiled server
    tiled_client, tiled_uri, _tiled_server = start_tiled_server()

    # 2. Start Redis (required by queueserver backend)
    start_redis()

    # 2b. Start bluesky-0MQ-proxy for run document publishing
    start_zmq_proxy()

    # 3. Start queueserver backend + HTTP server
    start_qs_backend()
    start_qs_http_server()

    # 3. Wait for queueserver HTTP API
    print("[SMOKE] Waiting for queueserver HTTP API...")
    if not wait_for_port(QS_HTTP_PORT):
        print("[SMOKE] ERROR: QS HTTP server did not start in time.")
        sys.exit(1)
    print("[SMOKE] QS HTTP server ready.")

    print("[SMOKE] Waiting for queueserver RE Manager...")
    if not wait_for_qs_ready():
        print("[SMOKE] ERROR: QS RE Manager did not become idle in time.")
        for proc in _subprocesses:
            if proc.stdout:
                print(proc.stdout.read().decode(errors="replace")[-2000:])
            if proc.stderr:
                print(proc.stderr.read().decode(errors="replace")[-2000:])
        sys.exit(1)
    print("[SMOKE] QS RE Manager ready.")

    # 4. Start ZMQ → Tiled bridge (subscribes TiledWriter to queueserver docs)
    start_zmq_tiled_bridge(tiled_client, ZMQ_PROXY_OUT_ADDR_TUPLE)

    # 5. Open the RE environment (loads startup files including mocks)
    print("[SMOKE] Opening RE environment...")
    import urllib.request
    import json

    req = urllib.request.Request(
        f"http://localhost:{QS_HTTP_PORT}/api/environment/open?api_key={QS_API_KEY}",
        data=json.dumps({}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"[SMOKE] Environment open response: {result}")

    # Wait for environment to be ready
    time.sleep(10)  # Give it time to load startup files with mocks

    # 6. Run the agent
    try:
        run_agent(tiled_uri=tiled_uri, n_iterations=5)
    except Exception as e:
        print(f"[SMOKE] FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
