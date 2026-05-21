#!/usr/bin/env python
"""
Local Tiled server that ingests Bluesky documents from a ZMQ stream.

The analysis server for xray_uvvis_acquire publishes analyzed artifacts
(in the Bluesky event model) to a ZMQ PUB socket. This script spins up
a local Tiled server and bridges those documents into it via
RemoteDispatcher + TiledWriter, providing a consistent data access point.

Usage:
    python scripts/zmq/tiled_server.py <zmq_address> [--port 8000]

Example:
    python scripts/zmq/tiled_server.py tcp://localhost:5578 --port 8000
"""

from __future__ import annotations

import argparse
import threading

from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.server import SimpleTiledServer


def main():
    parser = argparse.ArgumentParser(
        description="Local Tiled server ingesting Bluesky documents from ZMQ."
    )
    parser.add_argument(
        "zmq_address",
        help="ZMQ PUB address to subscribe to (e.g. tcp://localhost:5578)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the Tiled HTTP server (default: 8000)",
    )
    args = parser.parse_args()

    # 1. Start local Tiled server
    server = SimpleTiledServer(port=args.port)
    client = from_uri(server.uri)
    print(f"Tiled server running at: {server.uri}")

    # 2. Wire RemoteDispatcher → TiledWriter
    writer = TiledWriter(client)
    dispatcher = RemoteDispatcher(args.zmq_address)
    dispatcher.subscribe(writer)

    # 3. Run dispatcher (blocks in background thread)
    thread = threading.Thread(target=dispatcher.start, daemon=True)
    thread.start()
    print(f"Listening for Bluesky documents on: {args.zmq_address}")
    print("Press Ctrl+C to stop.")

    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
