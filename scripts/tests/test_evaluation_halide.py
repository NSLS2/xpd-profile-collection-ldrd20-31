"""
Regression tests for HalideEvaluation (evaluation_halide.py).

Uses synthetic QEPro data with known analytical properties to verify that the
evaluation function produces the same results as the old macro 10-13 pipeline.

Test categories:
    1. PL processing: Gaussian fit recovers known peak center and FWHM
    2. Absorbance baseline: correction removes known linear baseline
    3. PLQY: formula reproduces analytical result
    4. Full __call__: end-to-end with mocked Tiled client
    5. Edge cases: no PL peak → default values
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy import integrate

# ---------------------------------------------------------------------------
# Path setup — mirror the codebase convention
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_UTILS_DIR = _SCRIPTS_DIR / "utils"
_ML_AGENT_DIR = _SCRIPTS_DIR / "ML_agent"
_FIXTURES_DIR = Path(__file__).parent / "test_fixtures"

for p in [str(_UTILS_DIR), str(_ML_AGENT_DIR), str(_FIXTURES_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import _data_analysis as da
from synthetic_qepro import (
    make_absorbance_qepro,
    make_fluorescence_qepro,
    make_metadata,
    make_plqy_params,
    make_wavelength,
)
from evaluation_halide import HalideEvaluation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wavelength():
    return make_wavelength()


@pytest.fixture
def fl_data(wavelength):
    """Fluorescence QEPro data with a Gaussian peak at 520 nm, sigma=15."""
    return make_fluorescence_qepro(wavelength, peak_center=520.0, sigma=15.0, amplitude=500.0)


@pytest.fixture
def abs_data(wavelength):
    """Absorbance QEPro data with known abs_at_365=0.5."""
    return make_absorbance_qepro(wavelength, abs_at_365=0.5)


@pytest.fixture
def metadata():
    return make_metadata()


@pytest.fixture
def plqy_params():
    return make_plqy_params()


@pytest.fixture
def evaluator(plqy_params):
    """HalideEvaluation with a dummy tiled_client (overridden per-test)."""
    return HalideEvaluation(
        tiled_client=None,
        plqy_params=plqy_params,
        key_height=200,
        distance=100,
        height=50,
    )


# ===========================================================================
# 1. PL processing tests
# ===========================================================================

class TestProcessPL:
    """Verify Gaussian fit recovers known peak parameters."""

    def test_peak_center_recovery(self, evaluator, fl_data, metadata):
        qepro, expected = fl_data
        peak_emission, fwhm, PL_integral, r_2, has_peak = evaluator._process_pl(qepro, metadata)

        assert has_peak is True
        # Peak center should be within 2 nm of the true value
        assert abs(peak_emission - expected["peak_center"]) < 2.0, (
            f"Peak center {peak_emission:.2f} not close to {expected['peak_center']}"
        )

    def test_fwhm_recovery(self, evaluator, fl_data, metadata):
        qepro, expected = fl_data
        peak_emission, fwhm, PL_integral, r_2, has_peak = evaluator._process_pl(qepro, metadata)

        assert has_peak is True
        # FWHM should be within 3 nm of the true value
        assert abs(fwhm - expected["fwhm"]) < 3.0, (
            f"FWHM {fwhm:.2f} not close to {expected['fwhm']:.2f}"
        )

    def test_r_squared_high(self, evaluator, fl_data, metadata):
        qepro, expected = fl_data
        _, _, _, r_2, has_peak = evaluator._process_pl(qepro, metadata)

        assert has_peak is True
        assert r_2 > 0.95, f"R^2 = {r_2:.4f} is too low for clean Gaussian data"

    def test_pl_integral_positive(self, evaluator, fl_data, metadata):
        qepro, expected = fl_data
        _, _, PL_integral, _, has_peak = evaluator._process_pl(qepro, metadata)

        assert has_peak is True
        assert PL_integral > 0


class TestProcessPLNoPeak:
    """Edge case: fluorescence with no detectable peak."""

    def test_no_peak_returns_defaults(self, evaluator, wavelength, metadata):
        """Flat/low signal should return default values."""
        n = len(wavelength)
        n_spectra = 5
        qepro = {
            "QEPro_x_axis": np.tile(wavelength, (n_spectra + 1, 1)),
            "QEPro_output": np.ones((n_spectra + 1, n)) * 10,  # flat, below key_height
            "QEPro_spectrum_type": np.zeros(n_spectra + 1, dtype=int),
            "QEPro_integration_time": np.array([100000] * (n_spectra + 1)),
            "QEPro_num_spectra": np.array([1] * (n_spectra + 1)),
            "QEPro_buff_capacity": np.array([1] * (n_spectra + 1)),
        }

        peak_emission, fwhm, PL_integral, r_2, has_peak = evaluator._process_pl(qepro, metadata)

        assert has_peak is False
        assert peak_emission == 0.0
        assert fwhm == 1000.0
        assert PL_integral == 0.0


# ===========================================================================
# 2. Absorbance baseline correction tests
# ===========================================================================

class TestProcessAbsorbance:
    """Verify baseline correction removes known linear baseline."""

    def test_baseline_removal(self, evaluator, abs_data):
        qepro, expected = abs_data
        wavelength, abs_offset = evaluator._process_absorbance(qepro)

        # After baseline removal, values in the 750-950 nm range should be near zero
        # (the exponential decay is negligible there)
        idx750, _ = da.find_nearest(wavelength, 750)
        idx950, _ = da.find_nearest(wavelength, 950)
        tail_mean = np.mean(np.abs(abs_offset[idx750:idx950]))
        assert tail_mean < 0.05, f"Tail mean {tail_mean:.4f} should be near zero after baseline removal"

    def test_abs_at_365(self, evaluator, abs_data):
        qepro, expected = abs_data
        wavelength, abs_offset = evaluator._process_absorbance(qepro)

        idx_365, _ = da.find_nearest(wavelength, 365)
        measured_abs = abs_offset[idx_365]

        # Should be close to the expected value (within 10% tolerance due to
        # noise and percentile filtering)
        assert abs(measured_abs - expected["abs_at_365"]) / expected["abs_at_365"] < 0.15, (
            f"Abs@365 {measured_abs:.4f} not close to expected {expected['abs_at_365']:.4f}"
        )


# ===========================================================================
# 3. PLQY computation tests
# ===========================================================================

class TestComputePLQY:
    """Verify PLQY formula against direct analytical calculation."""

    def test_plqy_quinine_formula(self, evaluator, wavelength):
        """PLQY computed by evaluator should match direct da.plqy_quinine call."""
        abs_offset = np.zeros_like(wavelength)
        idx_365, _ = da.find_nearest(wavelength, 365)
        abs_offset[idx_365] = 0.5  # known absorbance

        PL_integral = 100000.0

        plqy = evaluator._compute_plqy(abs_offset, wavelength, PL_integral)

        # Compute expected directly
        params = evaluator.plqy_params
        expected = da.plqy_quinine(
            0.5, PL_integral, 1.506,
            params[3], params[4], params[5], params[6],
        )

        assert abs(plqy - expected) < 1e-10, f"PLQY {plqy} != expected {expected}"

    def test_plqy_fluorescein_formula(self, wavelength):
        """Test fluorescein PLQY path."""
        params = make_plqy_params(method="fluorescein")
        ev = HalideEvaluation(tiled_client=None, plqy_params=params)

        abs_offset = np.zeros_like(wavelength)
        idx_365, _ = da.find_nearest(wavelength, 365)
        abs_offset[idx_365] = 0.5

        PL_integral = 100000.0
        plqy = ev._compute_plqy(abs_offset, wavelength, PL_integral)

        expected = da.plqy_fluorescein(
            0.5, PL_integral, 1.506,
            params[3], params[4], params[5], params[6],
        )

        assert abs(plqy - expected) < 1e-10


# ===========================================================================
# 4. Full __call__ end-to-end test (mocked Tiled)
# ===========================================================================

class TestFullCall:
    """End-to-end test with mocked _read_streams."""

    def test_end_to_end_with_peak(self, evaluator, fl_data, abs_data, metadata):
        qepro_fl, fl_expected = fl_data
        qepro_abs, abs_expected = abs_data

        # Patch _read_streams to return our synthetic data
        evaluator._read_streams = MagicMock(
            return_value=(qepro_fl, qepro_abs, metadata)
        )

        suggestions = [{"_id": 0}]
        results = evaluator("test-uid", suggestions)

        assert len(results) == 1
        r = results[0]
        assert "_id" in r
        assert r["_id"] == 0
        assert "Peak" in r
        assert "FWHM" in r
        assert "PLQY" in r

        # Peak should be near 520 nm
        assert abs(r["Peak"] - 520.0) < 2.0
        # FWHM should be near 2.355 * 15 ≈ 35.3 nm
        assert abs(r["FWHM"] - 35.325) < 3.0
        # PLQY should be a positive number
        assert r["PLQY"] > 0

    def test_end_to_end_no_peak(self, evaluator, abs_data, metadata, wavelength):
        """Flat fluorescence signal → default values."""
        n = len(wavelength)
        n_spectra = 5
        qepro_fl = {
            "QEPro_x_axis": np.tile(wavelength, (n_spectra + 1, 1)),
            "QEPro_output": np.ones((n_spectra + 1, n)) * 10,
            "QEPro_spectrum_type": np.zeros(n_spectra + 1, dtype=int),
            "QEPro_integration_time": np.array([100000] * (n_spectra + 1)),
            "QEPro_num_spectra": np.array([1] * (n_spectra + 1)),
            "QEPro_buff_capacity": np.array([1] * (n_spectra + 1)),
        }
        qepro_abs, _ = abs_data

        evaluator._read_streams = MagicMock(
            return_value=(qepro_fl, qepro_abs, metadata)
        )

        results = evaluator("test-uid", [{"_id": 42}])
        r = results[0]
        assert r["Peak"] == 0.0
        assert r["FWHM"] == 1000.0
        assert r["PLQY"] == 0.0
        assert r["_id"] == 42

    def test_multiple_suggestions(self, evaluator, fl_data, abs_data, metadata):
        """Multiple suggestions should all get the same result."""
        qepro_fl, _ = fl_data
        qepro_abs, _ = abs_data

        evaluator._read_streams = MagicMock(
            return_value=(qepro_fl, qepro_abs, metadata)
        )

        suggestions = [{"_id": 0}, {"_id": 1}, {"_id": 2}]
        results = evaluator("test-uid", suggestions)

        assert len(results) == 3
        # All should have the same Peak/FWHM/PLQY but different _id
        for i, r in enumerate(results):
            assert r["_id"] == i
            assert r["Peak"] == results[0]["Peak"]
            assert r["FWHM"] == results[0]["FWHM"]
            assert r["PLQY"] == results[0]["PLQY"]


# ===========================================================================
# 5. Consistency with old pipeline functions
# ===========================================================================

class TestConsistencyWithOldPipeline:
    """Verify that individual steps match direct calls to _data_analysis functions."""

    def test_pl_fitting_matches_direct_call(self, fl_data, metadata):
        """_process_pl should produce the same peak/fwhm as calling
        _identify_multi_in_kafka + _fitting_in_kafka directly."""
        qepro, _ = fl_data
        evaluator = HalideEvaluation(
            tiled_client=None,
            plqy_params=make_plqy_params(),
            key_height=200,
            distance=100,
            height=50,
        )

        # Direct pipeline call (old macros 10 + 12)
        x0, y0, data_id, peak, prop = da._identify_multi_in_kafka(
            qepro, metadata, key_height=200, distance=100, height=50,
            dummy_test=False, percent_range=[40, 100],
        )
        assert isinstance(peak, np.ndarray) and len(peak) > 0

        x, y, shifted_peak, f_fit, popt = da._fitting_in_kafka(
            x0, y0, data_id, peak, prop, is_one_peak=True, dummy_test=False,
        )

        # Extract expected values the same way macro_12 does
        constant = 2.355 if "gauss" in f_fit.__name__ else 1
        intensity_list, peak_list, fwhm_list = [], [], []
        for i in range(int(len(popt) / 3)):
            intensity_list.append(popt[i * 3])
            peak_list.append(popt[i * 3 + 1])
            fwhm_list.append(popt[i * 3 + 2] * constant)
        eid = np.argmax(np.asarray(intensity_list))
        expected_peak = peak_list[eid]
        expected_fwhm = fwhm_list[eid]

        # Now via evaluator
        peak_emission, fwhm, _, _, _ = evaluator._process_pl(qepro, metadata)

        assert peak_emission == pytest.approx(expected_peak, abs=1e-6)
        assert fwhm == pytest.approx(expected_fwhm, abs=1e-6)

    def test_absorbance_matches_direct_call(self, abs_data):
        """_process_absorbance should match direct macro_11 logic."""
        qepro, _ = abs_data
        evaluator = HalideEvaluation(
            tiled_client=None,
            plqy_params=make_plqy_params(),
        )

        # Direct pipeline (macro 11)
        abs_per = da.percentile_abs(
            qepro["QEPro_x_axis"], qepro["QEPro_output"], percent_range=[10, 70],
        )
        abs_array = abs_per.mean(axis=0)
        wavelength = qepro["QEPro_x_axis"][0]
        popt01, _ = da.fit_line_2D(wavelength, abs_array, da.line_2D, x_range=[205, 240])
        popt02, _ = da.fit_line_2D(wavelength, abs_array, da.line_2D, x_range=[750, 950])
        if abs(popt01[0]) >= abs(popt02[0]):
            popt = popt02
        else:
            popt = popt01
        expected_offset = abs_array - da.line_2D(wavelength, *popt)

        # Via evaluator
        wl, abs_offset = evaluator._process_absorbance(qepro)

        np.testing.assert_allclose(abs_offset, expected_offset, atol=1e-10)
        np.testing.assert_allclose(wl, wavelength, atol=1e-10)
