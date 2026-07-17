"""
QueueserverAgent for halide perovskite UV-Vis optimization.

This module replaces the old macro-based ZMQ dispatcher loop with Blop's
built-in QueueserverAgent, which handles the suggest → acquire → evaluate → ingest
cycle automatically via the queueserver.

Usage
-----
    python queueserver_agent_halide.py

Configuration is loaded once from an Excel sheet at startup (pump lists, PLQY
params, target wavelength, etc.), then the agent runs autonomously.
"""

from __future__ import annotations

import os
import sys
import functools

import numpy as np
import pandas as pd
from bluesky_queueserver_api.http import REManagerAPI

from blop import RangeDOF, Objective, OutcomeConstraint
from blop.ax.agent import QueueserverAgent, Agent
from blop.queueserver import (
    CORRELATION_UID_KEY,
    QueueserverClient,
    QueueserverOptimizationRunner,
)
from bluesky_queueserver_api import BPlan
from ax.api.protocols import IMetric

# Add utils to path for evaluation function dependencies
_utils_dir = os.path.join(os.path.dirname(__file__), "..", "utils")
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)

from evaluation_halide import HalideEvaluation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Queueserver connection
HTTP_SERVER_URI = os.environ.get("QSERVER_HTTP_URI", "https://xf28id2-xpd-qs1.nsls2.bnl.gov")
HTTP_API_KEY = os.environ.get("QSERVER_HTTP_SERVER_API_KEY", "")
ZMQ_CONSUMER_ADDR = os.environ.get(
    "ZMQ_CONSUMER_ADDR",
    "ipc:///var/lib/bluesky-zmq-proxy/xpd-ipc-in-ipc-out/out.sock",
)

# Optimization parameters
PEAK_TARGET = 660  # nm
PEAK_TOLERANCE = 5  # nm

# PLQY reference parameters: [flag, reference_type, excitation_wl, abs_ref, PL_int_ref, ri_ref, plqy_ref]
PLQY_PARAMS = [1, "quinine", 365, 0.06, 1.2e6, 1.33, 0.546]

# Historical data path
AGENT_DATA_PATH = ""

# Tiled URI for evaluation function
TILED_URI = os.environ.get("TILED_URI", "https://tiled.nsls2.bnl.gov")
TILED_PROFILE = os.environ.get("TILED_PROFILE", "xpd")

# Acquisition plan name (must be registered on the queueserver)
ACQUISITION_PLAN_NAME = "xray_uvvis_acquire"

AGENT_CHECKPOINT_PATH = "/nsls2/data/xpd-new/legacy/processed/LDRD_chl/checkpoints/agent_halide.json"

# ---------------------------------------------------------------------------
# DOFs
# ---------------------------------------------------------------------------


def build_dofs(use_OAm: bool = False) -> list[RangeDOF]:
    """Build DOFs for the halide perovskite optimization."""
    if use_OAm:
        return [
            RangeDOF(name="infusion_rate_CsPb", bounds=(5, 200), parameter_type="float"),
            RangeDOF(name="infusion_rate_Br", bounds=(5, 250), parameter_type="float"),
            RangeDOF(name="infusion_rate_Cl", bounds=(5, 200), parameter_type="float"),
            RangeDOF(name="infusion_rate_OAm", bounds=(0, 70), parameter_type="float"),
        ]
    else:
        return [
            RangeDOF(name="infusion_rate_CsPb", bounds=(10, 200), parameter_type="float"),
            RangeDOF(name="infusion_rate_Br", bounds=(5, 200), parameter_type="float"),
            RangeDOF(name="infusion_rate_I2", bounds=(0, 200), parameter_type="float"),
        ]


# ---------------------------------------------------------------------------
# Objectives & Constraints
# ---------------------------------------------------------------------------


def build_objectives() -> list[Objective]:
    """Build objectives for the halide perovskite optimization."""
    return [
        Objective(name="log_FWHM", minimize=True),
        Objective(name="log_PLQY", minimize=False),
        Objective(name="peak_distance", minimize=True),
    ]


def build_outcome_constraints(
    peak_target: float = PEAK_TARGET,
    peak_tolerance: float = PEAK_TOLERANCE,
) -> list[OutcomeConstraint]:
    """Build outcome constraints on peak emission wavelength."""
    peak_up = peak_target + peak_tolerance
    peak_down = peak_target - peak_tolerance
    peak_metric = IMetric(name="Peak")
    return [
        OutcomeConstraint(f"p >= {peak_down}", p=peak_metric),
        OutcomeConstraint(f"p <= {peak_up}", p=peak_metric),
    ]


# ---------------------------------------------------------------------------
# Historical data loading
# ---------------------------------------------------------------------------


def load_historical_data(
    agent_data_path: str,
    dof_names: list[str],
    r2_min: float = 0.70,
) -> list[dict]:
    """Load historical data from CSV and format for agent.ingest().

    Parameters
    ----------
    agent_data_path : str
        Path to the CSV file with historical observations.
    dof_names : list[str]
        Names of the DOF columns in the CSV.
    r2_min : float
        Minimum R-squared to include a data point.

    Returns
    -------
    list[dict]
        Each dict contains DOF values and objective values, ready for ingest().
    """
    # names = [
    #     "infusion_rate_CsPb",
    #     "infusion_rate_Br",
    #     "infusion_rate_I2",
    #     "infusion_rate_Cl",
    #     "Peak",
    #     "FWHM",
    #     "PLQY",
    #     "time",
    #     "uid",
    #     "r_2",
    # ]
    
    names = [
        'infusion_rate_CsPb',
        'infusion_rate_Br', 
        'infusion_rate_Cl', 
        'infusion_rate_OAm', 
        'N/A', 
        'Peak', 
        'FWHM', 
        'PLQY'
    ] 

    # df = pd.read_csv(agent_data_path, sep=" ", names=names, skiprows=1, index_col=False)
    df = pd.read_csv(agent_data_path, sep=",", names=names, skiprows=1, index_col=False)

    points = []
    for _, row in df.iterrows():
        # if row.get("r_2", 0) < r2_min:
        #     continue

        point = {}
        for name in dof_names:
            if name in row.index:
                point[name] = row[name]

        point["Peak"] = row["Peak"]
        point["log_FWHM"] = np.log(row["FWHM"]) if row["FWHM"] > 0 else np.log(1000)
        point["log_PLQY"] = np.log(row["PLQY"]) if row["PLQY"] > 0 else np.log(1e-10)
        points.append(point)

    return points


# ---------------------------------------------------------------------------
# Tiled client
# ---------------------------------------------------------------------------


def _get_tiled_client(tiled_profile: str):
    """Get a tiled client, supporting both profile names and URIs.

    If TILED_URI env var is set, connects directly to that URI.
    Otherwise falls back to from_profile().
    """
    tiled_uri = os.environ.get("TILED_URI", "")
    if tiled_uri:
        from tiled.client import from_uri

        return from_uri(tiled_uri)
    else:
        from tiled.client import from_profile

        return from_profile(tiled_profile)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def build_agent(
    queueserver: bool = True,
    peak_target: float = PEAK_TARGET,
    peak_tolerance: float = PEAK_TOLERANCE,
    use_OAm: bool = False,
    agent_data_path: str = AGENT_DATA_PATH,
    plqy_params: list | None = None,
    http_server_uri: str = HTTP_SERVER_URI,
    http_api_key: str = HTTP_API_KEY,
    zmq_consumer_addr: str = ZMQ_CONSUMER_ADDR,
    tiled_profile: str = TILED_PROFILE,
    acquisition_plan_kwargs: dict | None = None,
    checkpoint_path: str | None = AGENT_CHECKPOINT_PATH,
) -> Agent | QueueserverAgent:
    """Build and return a QueueserverAgent ready to run.

    Parameters
    ----------
    peak_target : float
        Target peak emission wavelength (nm).
    peak_tolerance : float
        Acceptable deviation from target (nm).
    use_OAm : bool
        Whether to include OAm DOF (alternative precursor set).
    agent_data_path : str
        Path to historical data CSV for seeding the agent.
    plqy_params : list
        PLQY reference parameters for the evaluation function.
    http_server_uri : str
        Queueserver HTTP URI.
    http_api_key : str
        Queueserver API key.
    zmq_consumer_addr : str
        ZMQ address to consume Bluesky documents from.
    tiled_profile : str
        Tiled profile name for data access.
    acquisition_plan_kwargs : dict | None
        Extra keyword arguments forwarded to every ``xray_uvvis_acquire`` call
        submitted by the agent (e.g. ``{"post_dilute": True,
        "use_good_bad": True}``).  When ``None`` (default) the plan runs with
        its own module-level defaults.

    Returns
    -------
    Agent | QueueserverAgent
        Configured agent, seeded with historical data.
    """
    if plqy_params is None:
        plqy_params = PLQY_PARAMS

    # Build components
    dofs = build_dofs(use_OAm=use_OAm)
    objectives = build_objectives()
    outcome_constraints = build_outcome_constraints(peak_target, peak_tolerance)

    # Evaluation function
    tiled_client = _get_tiled_client(tiled_profile)
    evaluation_function = HalideEvaluation(
        tiled_client=tiled_client,
        plqy_params=plqy_params,
        peak_target=peak_target,
    )

    if queueserver:
        # Queueserver connection
        RM = REManagerAPI(http_server_uri=http_server_uri)
        if http_api_key:
            RM.set_authorization_key(api_key=http_api_key)

        # Build agent
        agent = QueueserverAgent(
            re_manager_api=RM,
            zmq_consumer_addr=zmq_consumer_addr,
            sensors=["qepro"],
            dofs=dofs,
            objectives=objectives,
            evaluation_function=evaluation_function,
            acquisition_plan=ACQUISITION_PLAN_NAME,
            outcome_constraints=outcome_constraints,
            acquisition_plan_kwargs=acquisition_plan_kwargs,
            checkpoint_path=checkpoint_path,
        )
    else:
        halide_acquisition = functools.partial(
            globals().get(ACQUISITION_PLAN_NAME),
            use_good_bad=True,
            do_xray=False,
        )

        agent = Agent(
            [],
            dofs,
            objectives,
            evaluation_function=evaluation_function,
            acquisition_plan=halide_acquisition,
            outcome_constraints=outcome_constraints,
            checkpoint_path=checkpoint_path,
        )

    # Seed with historical data
    if agent_data_path and os.path.exists(agent_data_path):
        dof_names = [d.parameter_name for d in dofs]
        historical = load_historical_data(agent_data_path, dof_names)
        if historical:
            agent.ingest(historical)
            print(f"Ingested {len(historical)} historical observations.")

    print(f"Agent built. Target peak: {peak_target} ± {peak_tolerance} nm")
    return agent
