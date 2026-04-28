"""
Unit tests for the halide agent builder (prepare_agent_halide_20260404.py).

These tests validate the SEMANTIC CONTRACT of build_agent() — the invariants
that must hold before and after refactoring from the old blop API to the new
Ax-based blop v1.0.0b1 API. They mock the blop module so no actual blop
installation is required.

Semantic invariants tested:
    1. DOF names and search domains (default and use_OAm modes)
    2. Objective names, optimization directions, and transforms
    3. Peak target range calculation from peak_target ± peak_tolerance
    4. Data quality filtering by r_2 >= 0.70
    5. Correct mapping of CSV data to DOF/objective/metadata dicts
    6. Model training invoked after all data ingestion
    7. build_agent() returns the agent object
    8. Expected CSV column schema
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "test_fixtures"
SAMPLE_CSV = FIXTURES_DIR / "agent_data_sample.csv"
OAM_CSV = FIXTURES_DIR / "agent_data_oam.csv"

# We need the ML_agent directory importable
ML_AGENT_DIR = str(Path(__file__).resolve().parent.parent / "ML_agent")
if ML_AGENT_DIR not in sys.path:
    sys.path.insert(0, ML_AGENT_DIR)


# ---------------------------------------------------------------------------
# Helpers to build a mock blop module and capture all calls
# ---------------------------------------------------------------------------
class _DOFRecord:
    """Captures old-API DOF constructor args for inspection."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # The old API stores the name in the 'name' kwarg
        self.name = kwargs.get("name")
        self.search_domain = kwargs.get("search_domain")
        self.description = kwargs.get("description")
        self.units = kwargs.get("units")


class _ObjectiveRecord:
    """Captures old-API Objective constructor args for inspection."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name = kwargs.get("name")
        self.target = kwargs.get("target")
        self.weight = kwargs.get("weight")
        self.transform = kwargs.get("transform")
        self.max_noise = kwargs.get("max_noise")
        self.description = kwargs.get("description")


def _make_mock_blop():
    """Return (mock_blop_module, mock_agent_instance) with realistic structure."""
    mock_blop = MagicMock()
    mock_agent = MagicMock()

    # Track DOFs/Objectives created
    created_dofs = []
    created_objectives = []

    def dof_factory(**kwargs):
        rec = _DOFRecord(**kwargs)
        created_dofs.append(rec)
        return rec

    def obj_factory(**kwargs):
        rec = _ObjectiveRecord(**kwargs)
        created_objectives.append(rec)
        return rec

    mock_blop.DOF = dof_factory
    mock_blop.Objective = obj_factory

    # Agent constructor returns the mock agent
    mock_blop.Agent = MagicMock(return_value=mock_agent)

    # agent.dofs.names / agent.objectives.names — filled dynamically
    # We'll set these as a side_effect of Agent() so they reflect the DOFs passed in
    def agent_init(**kwargs):
        dofs = kwargs.get("dofs", [])
        objectives = kwargs.get("objectives", [])
        mock_agent.dofs = MagicMock()
        mock_agent.dofs.names = [d.name for d in dofs]
        mock_agent.objectives = MagicMock()
        mock_agent.objectives.names = [o.name for o in objectives]
        return mock_agent

    mock_blop.Agent = MagicMock(side_effect=agent_init)

    return mock_blop, mock_agent, created_dofs, created_objectives


@pytest.fixture()
def blop_env():
    """Patch 'blop' in sys.modules and return test handles."""
    mock_blop, mock_agent, created_dofs, created_objectives = _make_mock_blop()

    with patch.dict(sys.modules, {"blop": mock_blop}):
        # Force re-import so the module picks up our mock
        if "prepare_agent_halide_20260404" in sys.modules:
            del sys.modules["prepare_agent_halide_20260404"]
        import prepare_agent_halide_20260404 as mod

        yield {
            "mod": mod,
            "mock_blop": mock_blop,
            "mock_agent": mock_agent,
            "created_dofs": created_dofs,
            "created_objectives": created_objectives,
        }


# ===================================================================
# Test 1: DOF configuration — default (no OAm)
# ===================================================================
class TestDOFConfigurationDefault:
    def test_dof_names(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        dofs = blop_env["created_dofs"]
        names = [d.name for d in dofs]
        assert names == ["infusion_rate_CsPb", "infusion_rate_Br", "infusion_rate_I2"]

    def test_dof_count(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        assert len(blop_env["created_dofs"]) == 3

    def test_dof_search_domains(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        dofs = blop_env["created_dofs"]
        expected = {
            "infusion_rate_CsPb": (10, 200),
            "infusion_rate_Br": (5, 200),
            "infusion_rate_I2": (0, 200),
        }
        for d in dofs:
            assert d.search_domain == expected[d.name], (
                f"Search domain mismatch for {d.name}"
            )


# ===================================================================
# Test 2: DOF configuration — use_OAm=True
# ===================================================================
class TestDOFConfigurationOAm:
    """Tests for the use_OAm=True DOF path.

    NOTE: The current source has a bug where the OAm DOF name
    'infusion_rate_OAm' is not present in the hardcoded CSV column names,
    causing a KeyError during data ingestion. These tests are marked xfail
    to document this known issue. The refactor should fix this bug.
    """

    @pytest.mark.xfail(
        reason="Bug: 'infusion_rate_OAm' not in CSV column names list",
        raises=(KeyError, NameError),
        strict=True,
    )
    def test_dof_names_oam(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(OAM_CSV), use_OAm=True
        )
        dofs = blop_env["created_dofs"]
        names = [d.name for d in dofs]
        assert names == [
            "infusion_rate_CsPb",
            "infusion_rate_Cl",
            "infusion_rate_OAm",
        ]

    @pytest.mark.xfail(
        reason="Bug: 'infusion_rate_OAm' not in CSV column names list",
        raises=(KeyError, NameError),
        strict=True,
    )
    def test_dof_count_oam(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(OAM_CSV), use_OAm=True
        )
        assert len(blop_env["created_dofs"]) == 3

    @pytest.mark.xfail(
        reason="Bug: 'infusion_rate_OAm' not in CSV column names list",
        raises=(KeyError, NameError),
        strict=True,
    )
    def test_dof_search_domains_oam(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(OAM_CSV), use_OAm=True
        )
        dofs = blop_env["created_dofs"]
        expected = {
            "infusion_rate_CsPb": (20, 80),
            "infusion_rate_Cl": (10, 190),
            "infusion_rate_OAm": (0, 70),
        }
        for d in dofs:
            assert d.search_domain == expected[d.name], (
                f"Search domain mismatch for {d.name}"
            )


# ===================================================================
# Test 3: Objective configuration
# ===================================================================
class TestObjectiveConfiguration:
    def test_objective_names(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        objs = blop_env["created_objectives"]
        names = [o.name for o in objs]
        assert names == ["Peak", "FWHM", "PLQY"]

    def test_objective_count(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        assert len(blop_env["created_objectives"]) == 3

    def test_fwhm_is_minimized(self, blop_env):
        """FWHM should target 'min' — the optimizer should minimize it."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        fwhm = [o for o in blop_env["created_objectives"] if o.name == "FWHM"][0]
        assert fwhm.target == "min"

    def test_plqy_is_maximized(self, blop_env):
        """PLQY should target 'max' — the optimizer should maximize it."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        plqy = [o for o in blop_env["created_objectives"] if o.name == "PLQY"][0]
        assert plqy.target == "max"

    def test_fwhm_log_transform(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        fwhm = [o for o in blop_env["created_objectives"] if o.name == "FWHM"][0]
        assert fwhm.transform == "log"

    def test_plqy_log_transform(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        plqy = [o for o in blop_env["created_objectives"] if o.name == "PLQY"][0]
        assert plqy.transform == "log"

    def test_objective_weights(self, blop_env):
        """Validate relative weighting: Peak=100, FWHM=50, PLQY=100."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        objs = {o.name: o for o in blop_env["created_objectives"]}
        assert objs["Peak"].weight == 100.0
        assert objs["FWHM"].weight == 50.0
        assert objs["PLQY"].weight == 100.0


# ===================================================================
# Test 4: Peak target range calculation
# ===================================================================
class TestPeakTargetRange:
    @pytest.mark.parametrize(
        "peak_target, peak_tolerance, expected_range",
        [
            (660, 5, (655, 665)),
            (525, 10, (515, 535)),
            (500, 0, (500, 500)),
            (700, 20, (680, 720)),
        ],
    )
    def test_peak_target_range(
        self, blop_env, peak_target, peak_tolerance, expected_range
    ):
        blop_env["mod"].build_agent(
            peak_target=peak_target,
            peak_tolerance=peak_tolerance,
            agent_data_path=str(SAMPLE_CSV),
            use_OAm=False,
        )
        peak_obj = [o for o in blop_env["created_objectives"] if o.name == "Peak"][0]
        assert peak_obj.target == expected_range, (
            f"Expected Peak target {expected_range}, got {peak_obj.target}"
        )


# ===================================================================
# Test 5: Data filtering by r_2 >= 0.70
# ===================================================================
class TestDataFilteringByR2:
    def test_rows_with_low_r2_are_skipped(self, blop_env):
        """
        Sample CSV has 6 rows with r_2 values: 0.85, 0.92, 0.55, 0.78, 0.65, 0.71
        Rows with r_2 < 0.70 should be skipped: uid_003 (0.55), uid_005 (0.65)
        So agent.tell() should be called exactly 4 times.
        """
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        assert agent.tell.call_count == 4

    @pytest.mark.xfail(
        reason="Bug: 'infusion_rate_OAm' not in CSV column names list",
        raises=(KeyError, NameError),
        strict=True,
    )
    def test_all_rows_pass_when_r2_high(self, blop_env):
        """OAm CSV has 2 rows both with r_2 >= 0.70."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(OAM_CSV), use_OAm=True
        )
        agent = blop_env["mock_agent"]
        assert agent.tell.call_count == 2


# ===================================================================
# Test 6: Data ingestion maps CSV columns correctly
# ===================================================================
class TestDataIngestionMapping:
    def test_x_dict_uses_dof_names(self, blop_env):
        """The x dict keys must match DOF names exactly."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        expected_keys = {"infusion_rate_CsPb", "infusion_rate_Br", "infusion_rate_I2"}
        for c in agent.tell.call_args_list:
            x = c.kwargs.get("x") or c[1].get("x")
            assert set(x.keys()) == expected_keys

    def test_y_dict_uses_objective_names(self, blop_env):
        """The y dict keys must match objective names exactly."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        expected_keys = {"Peak", "FWHM", "PLQY"}
        for c in agent.tell.call_args_list:
            y = c.kwargs.get("y") or c[1].get("y")
            assert set(y.keys()) == expected_keys

    def test_metadata_contains_expected_keys(self, blop_env):
        """Metadata should include time, uid, r_2."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        expected_keys = {"time", "uid", "r_2"}
        for c in agent.tell.call_args_list:
            metadata = c.kwargs.get("metadata") or c[1].get("metadata")
            assert set(metadata.keys()) == expected_keys

    def test_first_passing_row_values(self, blop_env):
        """First row (uid_001, r_2=0.85) should pass filter with correct values."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        first_call = agent.tell.call_args_list[0]
        x = first_call.kwargs.get("x") or first_call[1].get("x")
        y = first_call.kwargs.get("y") or first_call[1].get("y")

        assert x["infusion_rate_CsPb"] == [50.0]
        assert x["infusion_rate_Br"] == [100.0]
        assert x["infusion_rate_I2"] == [80.0]
        assert y["Peak"] == [520.5]
        assert y["FWHM"] == [25.3]
        assert y["PLQY"] == [0.45]

    def test_tell_called_with_train_false(self, blop_env):
        """During batch loading, train=False and update_models=False."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        for c in agent.tell.call_args_list:
            assert c.kwargs.get("train") is False
            assert c.kwargs.get("update_models") is False


# ===================================================================
# Test 7: Model training called after data ingestion
# ===================================================================
class TestModelTrainingOrder:
    def test_construct_models_called(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        agent._construct_all_models.assert_called_once()

    def test_train_models_called(self, blop_env):
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]
        agent._train_all_models.assert_called_once()

    def test_training_after_all_tells(self, blop_env):
        """Model training must happen AFTER all tell() calls, not interleaved."""
        blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        agent = blop_env["mock_agent"]

        # Build ordered call log from mock
        call_order = []
        for name, args, kwargs in agent.method_calls:
            call_order.append(name)

        tell_indices = [i for i, n in enumerate(call_order) if n == "tell"]
        construct_idx = next(
            i for i, n in enumerate(call_order) if n == "_construct_all_models"
        )
        train_idx = next(
            i for i, n in enumerate(call_order) if n == "_train_all_models"
        )

        # All tells must come before construct, which comes before train
        assert all(t < construct_idx for t in tell_indices), (
            "All tell() calls must precede _construct_all_models()"
        )
        assert construct_idx < train_idx, (
            "_construct_all_models() must precede _train_all_models()"
        )


# ===================================================================
# Test 8: build_agent returns the agent object
# ===================================================================
class TestBuildAgentReturn:
    def test_returns_agent(self, blop_env):
        result = blop_env["mod"].build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        assert result is blop_env["mock_agent"]


# ===================================================================
# Test 9: Expected CSV column schema
# ===================================================================
class TestCSVColumnSchema:
    def test_expected_columns(self, blop_env):
        """
        The CSV is expected to have these columns (in order):
        infusion_rate_CsPb, infusion_rate_Br, infusion_rate_I2, infusion_rate_Cl,
        Peak, FWHM, PLQY, time, uid, r_2

        This is the full schema regardless of which DOFs are active.
        """
        expected = [
            "infusion_rate_CsPb",
            "infusion_rate_Br",
            "infusion_rate_I2",
            "infusion_rate_Cl",
            "Peak",
            "FWHM",
            "PLQY",
            "time",
            "uid",
            "r_2",
        ]
        # Read the module source to verify the 'names' list matches
        mod = blop_env["mod"]
        # Call build_agent and verify the CSV was parseable with these columns
        result = mod.build_agent(
            peak_target=660, agent_data_path=str(SAMPLE_CSV), use_OAm=False
        )
        # If the CSV was parsed successfully (tell was called), the schema is correct
        agent = blop_env["mock_agent"]
        assert agent.tell.call_count > 0, "CSV parsing failed — no data ingested"

        # Also verify the hardcoded names list in source matches expectations
        import inspect

        source = inspect.getsource(mod.build_agent)
        for col in expected:
            assert f'"{col}"' in source, (
                f"Column '{col}' not found in build_agent source"
            )
