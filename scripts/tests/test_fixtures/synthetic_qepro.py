"""Generate synthetic QEPro data for regression-testing HalideEvaluation.

The generated data has known analytical properties so tests can verify
correctness of peak fitting, baseline correction, and PLQY calculation.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate


def make_wavelength(start: float = 200.0, stop: float = 1000.0, n_points: int = 1800):
    """Wavelength axis mimicking the Ocean Optics QEPro."""
    return np.linspace(start, stop, n_points)


def make_fluorescence_qepro(
    wavelength: np.ndarray,
    peak_center: float = 520.0,
    sigma: float = 15.0,
    amplitude: float = 500.0,
    n_spectra: int = 5,
    noise_std: float = 5.0,
    seed: int = 42,
):
    """Build a ``qepro_dic`` for a fluorescence stream.

    Returns
    -------
    qepro_dic : dict
        Keys match ``_data_keys()`` output: ``QEPro_x_axis``, ``QEPro_output``,
        ``QEPro_spectrum_type``, etc.
    expected : dict
        Analytically-derived expected values:
        ``peak_center``, ``fwhm``, ``PL_integral``.
    """
    rng = np.random.default_rng(seed)
    n = len(wavelength)

    # Pure Gaussian signal
    signal = amplitude * np.exp(-((wavelength - peak_center) ** 2) / (2 * sigma**2))

    # Row 0 is a "reference" spectrum (spectrum_type row); rows 1..n_spectra are measurements
    x_axis = np.tile(wavelength, (n_spectra + 1, 1))
    output = np.zeros((n_spectra + 1, n))
    output[0] = signal  # row 0 — reference/dark (not used in PL pipeline)
    for i in range(1, n_spectra + 1):
        output[i] = signal + rng.normal(0, noise_std, n)

    spectrum_type = np.zeros(n_spectra + 1, dtype=int)
    spectrum_type[0] = 2  # "fluorescence" type flag

    qepro_dic = {
        "QEPro_x_axis": x_axis,
        "QEPro_output": output,
        "QEPro_spectrum_type": spectrum_type,
        "QEPro_integration_time": np.array([100000] * (n_spectra + 1)),
        "QEPro_num_spectra": np.array([1] * (n_spectra + 1)),
        "QEPro_buff_capacity": np.array([1] * (n_spectra + 1)),
    }

    # Expected analytical values
    fwhm = sigma * 2.355
    # Integral of Gaussian: A * sigma * sqrt(2*pi)
    # But the pipeline integrates the *windowed* spectrum via simpson, so provide
    # the analytical full integral as a reference.
    analytical_integral = amplitude * sigma * np.sqrt(2 * np.pi)

    expected = {
        "peak_center": peak_center,
        "sigma": sigma,
        "fwhm": fwhm,
        "amplitude": amplitude,
        "analytical_integral": analytical_integral,
    }

    return qepro_dic, expected


def make_absorbance_qepro(
    wavelength: np.ndarray,
    abs_at_365: float = 0.5,
    baseline_slope: float = 1e-4,
    baseline_intercept: float = 0.02,
    n_spectra: int = 5,
    noise_std: float = 0.005,
    seed: int = 123,
):
    """Build a ``qepro_dic`` for an absorbance stream.

    The absorbance spectrum is modelled as an exponential decay (UV-absorbing
    material) plus a linear baseline.

    Returns
    -------
    qepro_dic : dict
    expected : dict
        ``abs_at_365`` — the known absorbance at 365 nm *after* baseline
        correction (the baseline is constructed so it can be exactly removed
        by the two-range fitting procedure).
    """
    rng = np.random.default_rng(seed)
    n = len(wavelength)

    # Construct a "true" absorbance that has value abs_at_365 at 365 nm
    # Use an exponential decay: A * exp(-k * (wl - 200))
    # Solve: A * exp(-k * 165) = abs_at_365
    k = 0.01
    A = abs_at_365 / np.exp(-k * (365 - 200))
    true_abs = A * np.exp(-k * (wavelength - 200))

    # Add linear baseline
    baseline = baseline_slope * wavelength + baseline_intercept
    raw_abs = true_abs + baseline

    # Build multi-row array (row 0 = reference wavelength, rows 1+ = spectra)
    x_axis = np.tile(wavelength, (n_spectra + 1, 1))
    output = np.zeros((n_spectra + 1, n))
    output[0] = raw_abs
    for i in range(1, n_spectra + 1):
        output[i] = raw_abs + rng.normal(0, noise_std, n)

    spectrum_type = np.zeros(n_spectra + 1, dtype=int)
    spectrum_type[0] = 1  # absorbance type

    qepro_dic = {
        "QEPro_x_axis": x_axis,
        "QEPro_output": output,
        "QEPro_spectrum_type": spectrum_type,
        "QEPro_integration_time": np.array([100000] * (n_spectra + 1)),
        "QEPro_num_spectra": np.array([1] * (n_spectra + 1)),
        "QEPro_buff_capacity": np.array([1] * (n_spectra + 1)),
    }

    # The expected abs_at_365 after baseline removal.
    # The pipeline fits a line in [750, 950] or [205, 240] and picks the flatter one.
    # Our baseline is very shallow (slope=1e-4), so both ranges should fit it well.
    # After removing it, abs_at_365 ≈ true_abs at 365 nm.
    idx_365 = int(np.abs(wavelength - 365).argmin())
    expected_abs_365 = true_abs[idx_365]

    expected = {
        "abs_at_365": expected_abs_365,
        "baseline_slope": baseline_slope,
        "baseline_intercept": baseline_intercept,
    }

    return qepro_dic, expected


def make_metadata(uid: str = "abc12345-dead-beef-cafe-000000000001"):
    """Minimal metadata_dic compatible with ``_identify_multi_in_kafka``."""
    import time as _time

    return {
        "uid": uid,
        "time": _time.time(),
        "pumps": ["pump_CsPb", "pump_Br", "pump_I2", "pump_Cl"],
        "precursors": ["CsPb", "TOABr", "ZnI2", "ZnCl2"],
        "infuse_rate": [50.0, 100.0, 80.0, 60.0],
        "infuse_rate_unit": ["ul/min", "ul/min", "ul/min", "ul/min"],
        "pump_status": ["Infusing", "Infusing", "Infusing", "Infusing"],
        "mixer": ["mixer1"],
        "sample_type": ["perovskite"],
        "note": ["synthetic test"],
        "stream_name": "fluorescence",
    }


def make_plqy_params(
    method: str = "quinine",
    excitation_wl: float = 365.0,
    abs_ref: float = 0.376390,
    PL_int_ref: float = 468573.0,
    ri_ref: float = 1.337,
    plqy_ref: float = 0.546,
):
    """PLQY reference parameters in the format expected by HalideEvaluation."""
    return [1, method, excitation_wl, abs_ref, PL_int_ref, ri_ref, plqy_ref]
