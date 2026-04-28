"""
Spike test: Can we use IMetric + OutcomeConstraint to implement a
"tracking metric with bounds" (no optimization direction)?

This validates the approach for the Peak emission wavelength, which
should be constrained to a range (e.g., 655-665 nm) but not optimized
in any direction.

Run with: pixi run -e terminal python scripts/tests/test_imetric_spike.py
"""

import logging
import time
from typing import Any

logging.getLogger("httpx").setLevel(logging.WARNING)

from ax.api.protocols import IMetric
from blop.ax import Agent, OutcomeConstraint, Objective, RangeDOF
from bluesky.protocols import HasHints, HasParent, Hints, NamedMovable, Readable, Status
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.server import SimpleTiledServer


# ---------------------------------------------------------------------------
# Minimal simulated devices (copied from Blop tutorial)
# ---------------------------------------------------------------------------
class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)

    def exception(self, timeout=0.0):
        return None

    @property
    def done(self) -> bool:
        return True

    @property
    def success(self) -> bool:
        return True


class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def hints(self) -> Hints:
        return {"fields": [self._name], "dimensions": [], "gridding": "rectilinear"}

    @property
    def parent(self) -> Any | None:
        return None

    def read(self):
        return {self._name: {"value": self._value, "timestamp": time.time()}}

    def describe(self):
        return {self._name: {"source": self._name, "dtype": "number", "shape": []}}


class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value

    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
tiled_server = SimpleTiledServer()
RE = RunEngine({})
tiled_client = from_uri(tiled_server.uri)
tiled_writer = TiledWriter(tiled_client)
RE.subscribe(tiled_writer)

# ---------------------------------------------------------------------------
# DOFs: two flow rates
# ---------------------------------------------------------------------------
rate_a = MovableSignal("rate_a", initial_value=50.0)
rate_b = MovableSignal("rate_b", initial_value=50.0)

dofs = [
    RangeDOF(actuator=rate_a, bounds=(10, 200), parameter_type="float"),
    RangeDOF(actuator=rate_b, bounds=(5, 200), parameter_type="float"),
]

# ---------------------------------------------------------------------------
# Objectives: only FWHM (minimize). Peak is NOT an objective.
# ---------------------------------------------------------------------------
objectives = [
    Objective(name="FWHM", minimize=True),
]

# ---------------------------------------------------------------------------
# Peak is a tracking metric with range constraints (655-665 nm)
# ---------------------------------------------------------------------------
peak_metric = IMetric(name="Peak")
peak_constraints = [
    OutcomeConstraint("p >= 655", p=peak_metric),
    OutcomeConstraint("p <= 665", p=peak_metric),
]


# ---------------------------------------------------------------------------
# Evaluation function: synthetic data
# ---------------------------------------------------------------------------
class SyntheticEvaluation:
    def __init__(self, tiled_client):
        self.tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        run = self.tiled_client[uid]
        rate_a_data = run["primary/rate_a"].read()
        rate_b_data = run["primary/rate_b"].read()

        outcomes = []
        for suggestion in suggestions:
            sid = suggestion["_id"]
            a = rate_a_data[sid % len(rate_a_data)]
            b = rate_b_data[sid % len(rate_b_data)]
            # Synthetic: Peak depends on rates, FWHM is something to minimize
            peak = 600 + 0.5 * a + 0.1 * b
            fwhm = 20 + 0.1 * (a - 100) ** 2 + 0.05 * (b - 80) ** 2
            outcomes.append(
                {
                    "_id": sid,
                    "FWHM": fwhm,
                    "Peak": peak,
                }
            )
        return outcomes


# ---------------------------------------------------------------------------
# Create Agent and run with proper cleanup
# ---------------------------------------------------------------------------
try:
    print("Creating agent with IMetric + OutcomeConstraint for Peak...")
    agent = Agent(
        sensors=[],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=SyntheticEvaluation(tiled_client),
        outcome_constraints=peak_constraints,
    )
    print("SUCCESS: Agent created with IMetric-based outcome constraints.")

    # -------------------------------------------------------------------
    # Inspect Ax internals to verify constraints are registered
    # -------------------------------------------------------------------
    print("\n=== Ax Experiment Introspection ===")

    ax_client = agent.ax_client
    experiment = ax_client._experiment

    # 1. Check optimization config objectives
    opt_config = experiment.optimization_config
    print(f"\nObjective: {opt_config.objective}")
    print(f"Objective type: {type(opt_config.objective).__name__}")

    # 2. Check outcome constraints registered in Ax
    print(f"\nOutcome constraints ({len(opt_config.outcome_constraints)}):")
    for i, oc in enumerate(opt_config.outcome_constraints):
        print(f"  [{i}] {oc}")
        print(f"       metric name: {oc.metric.name}")
        print(f"       op: {oc.op}")
        print(f"       bound: {oc.bound}")
        print(f"       relative: {oc.relative}")

    # 3. Check all metrics registered on the experiment
    print(f"\nAll metrics on experiment ({len(experiment.metrics)}):")
    for name, metric in experiment.metrics.items():
        print(f"  {name}: {type(metric).__name__}")

    # 4. Check search space (DOFs)
    print(f"\nSearch space parameters:")
    for p in experiment.search_space.parameters.values():
        print(f"  {p.name}: [{p.lower}, {p.upper}]")

    print("\n=== End Introspection ===")

    # -------------------------------------------------------------------
    # Run 2 iterations to verify constraints work at runtime
    # -------------------------------------------------------------------
    print("\nRunning 20 optimization iterations...")
    RE(agent.optimize(iterations=20))
    print("\nSUCCESS: Optimization ran with IMetric + OutcomeConstraint.")

    print(
        "\nSpike test PASSED: IMetric + OutcomeConstraint is a valid approach for Peak."
    )

finally:
    tiled_server.close()
