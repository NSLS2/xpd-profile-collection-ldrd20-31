"""Evaluation function for halide perovskite agent (new Blop v1.0.0b1 API).

Composes the data pipeline from macros 10-13 in _LDRD_Kafka.py into a single
callable class compatible with the Blop ``Agent`` evaluation_function interface::

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]

Each returned dict contains ``Peak``, ``FWHM``, ``PLQY``, and ``_id`` keys.
"""

from __future__ import annotations

import sys
import os
import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Add utils to path so we can import _data_analysis / _data_export
# This mirrors the pattern used throughout the codebase.
_utils_dir = os.path.join(os.path.dirname(__file__), "..", "utils")
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)

import _data_analysis as da
import _data_export as de


class HalideEvaluation:
    """Evaluation function that reads QEPro data from Tiled and computes
    optical properties (Peak, FWHM, PLQY) for the halide perovskite agent.

    Parameters
    ----------
    tiled_client
        A Tiled ``Container`` (or compatible mapping) keyed by Bluesky run UID.
    plqy_params : list
        PLQY reference parameters, matching ``self.inputs.PLQY`` from the old
        dispatcher:
        ``[flag, 'quinine'|'fluorescein', excitation_wl, abs_ref, PL_int_ref,
          ri_ref, plqy_ref]``
    key_height : float
        Minimum peak height to consider a spectrum "good" (passed to
        ``good_bad_data``).  Default 200.
    distance : int
        Minimum distance between peaks in ``find_peaks``.  Default 100.
    height : float
        Minimum height for ``find_peaks``.  Default 50.
    percent_range_pl : list[float]
        Percentile range for PL filtering.  Default ``[40, 100]``.
    percent_range_abs : list[float]
        Percentile range for absorbance filtering.  Default ``[10, 70]``.
    """

    def __init__(
        self,
        tiled_client,
        plqy_params: list,
        key_height: float = 200,
        distance: int = 100,
        height: float = 50,
        percent_range_pl: list[float] | None = None,
        percent_range_abs: list[float] | None = None,
    ):
        self.tiled_client = tiled_client
        self.plqy_params = plqy_params
        self.key_height = key_height
        self.distance = distance
        self.height = height
        self.percent_range_pl = (
            percent_range_pl if percent_range_pl is not None else [40, 100]
        )
        self.percent_range_abs = (
            percent_range_abs if percent_range_abs is not None else [10, 70]
        )

    # ------------------------------------------------------------------
    # Internal helpers (exposed for testability)
    # ------------------------------------------------------------------

    def _read_streams(self, uid: str):
        """Read fluorescence and absorbance QEPro data from Tiled.

        Returns
        -------
        qepro_fl : dict
            QEPro dictionary for the fluorescence stream.
        qepro_abs : dict
            QEPro dictionary for the absorbance stream.
        metadata : dict
            Metadata dictionary (from fluorescence stream's start doc).
        """
        qepro_fl, metadata = de.read_qepro_by_stream(
            uid, stream_name="fluorescence", data_agent="tiled"
        )
        qepro_abs, _ = de.read_qepro_by_stream(
            uid, stream_name="absorbance", data_agent="tiled"
        )
        return qepro_fl, qepro_abs, metadata

    def _process_pl(self, qepro_dic: dict, metadata_dic: dict):
        """Run PL percentile filtering, peak finding, and Gaussian fitting.

        Corresponds to macros 10 + 12.

        Returns
        -------
        peak_emission : float
            Center of the highest Gaussian peak (nm), or 0 if no peak found.
        fwhm : float
            Full-width at half-maximum (nm), or 1000 if no peak found.
        PL_integral : float
            Simpson integral of the PL spectrum, or 0 if no peak found.
        r_2 : float
            R-squared of the Gaussian fit, or 0 if no peak found.
        has_peak : bool
            Whether a valid peak was found.
        """
        # Macro 10: percentile filtering + peak identification
        try:
            x0, y0, data_id, peak, prop = da._identify_multi_in_kafka(
                qepro_dic,
                metadata_dic,
                key_height=self.key_height,
                distance=self.distance,
                height=self.height,
                dummy_test=False,
                percent_range=self.percent_range_pl,
            )
        except (ValueError, IndexError):
            # Upstream good_bad_data raises ValueError when no peaks are found
            # at all (empty spectrum / signal below height threshold).
            return 0.0, 1000.0, 0.0, 0.0, False

        has_peak = isinstance(peak, np.ndarray) and len(peak) > 0

        if not has_peak:
            return 0.0, 1000.0, 0.0, 0.0, False

        # Macro 12: Gaussian fitting
        x, y, shifted_peak, f_fit, popt = da._fitting_in_kafka(
            x0, y0, data_id, peak, prop, is_one_peak=True, dummy_test=False
        )

        # Extract peak_emission and fwhm from popt
        # popt layout for _1gauss: [A, x0, sigma]  (groups of 3 for multi-gauss)
        if "gauss" in f_fit.__name__:
            constant = 2.355
        else:
            constant = 1

        intensity_list = []
        peak_list = []
        fwhm_list = []
        for i in range(int(len(popt) / 3)):
            intensity_list.append(popt[i * 3 + 0])
            peak_list.append(popt[i * 3 + 1])
            fwhm_list.append(popt[i * 3 + 2] * constant)

        peak_emission_id = np.argmax(np.asarray(intensity_list))
        peak_emission = peak_list[peak_emission_id]
        fwhm = fwhm_list[peak_emission_id]

        # R-squared
        fitted_y = f_fit(x, *popt)
        r2_idx1, _ = da.find_nearest(x, popt[1] - 3 * popt[2])
        r2_idx2, _ = da.find_nearest(x, popt[1] + 3 * popt[2])
        r_2 = da.r_square(
            x[r2_idx1:r2_idx2],
            y[r2_idx1:r2_idx2],
            fitted_y[r2_idx1:r2_idx2],
            y_low_limit=0,
        )

        # PL integral (macro 13 step 1)
        PL_integral = integrate.simpson(y)

        return peak_emission, fwhm, PL_integral, r_2, True

    def _process_absorbance(self, qepro_dic: dict):
        """Run absorbance percentile filtering and baseline correction.

        Corresponds to macro 11.

        Returns
        -------
        wavelength : np.ndarray
            Wavelength array (nm).
        abs_offset : np.ndarray
            Baseline-corrected absorbance array.
        """
        abs_per = da.percentile_abs(
            qepro_dic["QEPro_x_axis"],
            qepro_dic["QEPro_output"],
            percent_range=self.percent_range_abs,
        )
        abs_array = abs_per.mean(axis=0)
        wavelength = qepro_dic["QEPro_x_axis"][0]

        # Two-range baseline fit; pick the one with flatter slope
        popt01, _ = da.fit_line_2D(
            wavelength, abs_array, da.line_2D, x_range=[205, 240]
        )
        popt02, _ = da.fit_line_2D(
            wavelength, abs_array, da.line_2D, x_range=[750, 950]
        )
        if abs(popt01[0]) >= abs(popt02[0]):
            popt = popt02
        else:
            popt = popt01

        abs_offset = abs_array - da.line_2D(wavelength, *popt)
        return wavelength, abs_offset

    def _compute_plqy(
        self, abs_offset: np.ndarray, wavelength: np.ndarray, PL_integral: float
    ) -> float:
        """Compute PLQY from baseline-corrected absorbance and PL integral.

        Corresponds to the PLQY portion of macro 13.

        Returns
        -------
        float
            Photoluminescence quantum yield.
        """
        excitation_wl = self.plqy_params[2]
        idx, _ = da.find_nearest(wavelength, excitation_wl)
        absorbance_s = abs_offset[idx]

        # Reference params: [abs_ref, PL_int_ref, ri_ref, plqy_ref]
        ref_params = self.plqy_params[3:]
        refractive_index_solvent = 1.506  # toluene

        if self.plqy_params[1] == "fluorescein":
            plqy = da.plqy_fluorescein(
                absorbance_s, PL_integral, refractive_index_solvent, *ref_params
            )
        else:
            plqy = da.plqy_quinine(
                absorbance_s, PL_integral, refractive_index_solvent, *ref_params
            )
        return plqy

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        """Evaluate a Bluesky run and return objective outcomes.

        Parameters
        ----------
        uid : str
            Bluesky run UID.
        suggestions : list[dict]
            Optimizer suggestions; each must contain an ``_id`` key.

        Returns
        -------
        list[dict]
            One outcome dict per suggestion, with keys ``Peak``, ``FWHM``,
            ``PLQY``, and ``_id``.
        """
        qepro_fl, qepro_abs, metadata = self._read_streams(uid)

        # PL processing (macros 10 + 12 + part of 13)
        peak_emission, fwhm, PL_integral, r_2, has_peak = self._process_pl(
            qepro_fl, metadata
        )

        # Absorbance processing (macro 11)
        wavelength, abs_offset = self._process_absorbance(qepro_abs)

        # PLQY (macro 13)
        if has_peak:
            plqy = self._compute_plqy(abs_offset, wavelength, PL_integral)
        else:
            plqy = 0.0

        # Return one result per suggestion (typically len(suggestions) == 1
        # since each uid corresponds to one experimental run).
        results = []
        for s in suggestions:
            results.append(
                {
                    "Peak": peak_emission,
                    "FWHM": fwhm,
                    "PLQY": plqy,
                    "log_FWHM": np.log(fwhm),
                    "log_PLQY": np.log(plqy),
                    "_id": s["_id"],
                }
            )
        return results
