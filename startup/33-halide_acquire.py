"""
Acquisition plan for the halide perovskite QueueserverAgent.

This plan follows the Blop AcquisitionPlan protocol signature::

    def __call__(suggestions, actuators, sensors, md=None) -> uid

It performs the full synthesis + measurement sequence for one optimization step:

1. Stop pumps (defensive), set pump infusion rates, start pumps, wait for flow
   equilibrium (hardware read via ``wait_equilibrium2``). Optionally configure
   toluene dilution.
2. Collect absorbance spectra.
3. Collect fluorescence spectra. When ``USE_GOOD_BAD`` is enabled, additional
   PL batches are taken (in the same Bluesky run) until either ``GOOD_TARGET``
   good batches or ``MAX_BAD`` bad batches have been classified.
4. Stop all pumps that were started (guaranteed even on exception, via
   ``bpp.finalize_wrapper``).
5. Return the run UID.

Design:

* :func:`steady_state_flow` is a wrapper plan that owns pump setup and
  teardown. Pumps stop on success and on exception.
* :func:`measure_absorbance` / :func:`measure_pl` are pure measurement
  plan stubs.
* :class:`PLQualityMonitor` is a ``CallbackBase`` subscribed locally via
  ``bpp.subs_decorator``. It runs synchronously in the RunEngine thread
  between event docs, so the plan can read ``monitor.good_count`` /
  ``monitor.bad_count`` immediately after each batch
* The decision policy (continue / stop) is inline in
  :func:`_pl_with_quality_gate`.

This file is loaded into the queueserver environment via startup. All devices
(``qepro``, ``LED``, ``UV_shutter``, pump objects) and helper plans
(``stop_group``, ``set_group_infuse2``, ``start_group_infuse``,
``wait_equilibrium2``, ``sleep_sec_q``) are available as globals from earlier
startup files.
"""

import numpy as np
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.callbacks import CallbackBase
from ophyd import Signal


# ---------------------------------------------------------------------------
# Static configuration (physical setup — update per beamtime)
# ---------------------------------------------------------------------------

# Syringe sizes (mL) for each pump, in order matching DOF order
SYRINGE_LIST = [50, 50, 50]

# Syringe materials
SYRINGE_MATER_LIST = ["steel", "steel", "steel"]

# Target volumes (format: "value unit")
TARGET_VOL_LIST = ["30 ml", "30 ml", "30 ml"]

# Whether to auto-set target for each pump
SET_TARGET_LIST = [True, True, True]

# Rate unit
RATE_UNIT = "ul/min"

# Mixer tubing: single segment of 30 cm (matches historical xlsx config).
# Format matches what wait_equilibrium2 expects: "value unit".
MIXER_LENGTHS_CM = [30.0]

# Residence time multiplier (wait this many multiples of the residence time)
RESIDENT_T_RATIO = 1.0

# Number of absorbance and fluorescence spectra per measurement
NUM_ABS = 10
NUM_FLU = 10

# Precursor names (for metadata only)
PRECURSOR_LIST = ["CsPbOA", "TOABr", "ZnI2"]

# Post-dilution with toluene
POST_DILUTE = False
POST_DILUTE_RATIO = 1.0  # toluene rate = sum(active_rates) * ratio
POST_DILUTE_WAIT_SEC = 30  # wait time after starting toluene pump (seconds)

# Default mapping: DOF name -> pump device name in queueserver namespace
DOF_TO_PUMP = {
    "infusion_rate_CsPb": "dds2_p1",
    "infusion_rate_Br": "dds2_p2",
    "infusion_rate_I2": "dds3_p1",
    "infusion_rate_Cl": "dds1_p1",
    "infusion_rate_OAm": "dds1_p2",
}

# Toluene dilution pump device name
DILUTE_PUMP_NAME = "dds1_p2"

# ---------------------------------------------------------------------------
# Good/bad fluorescence reacquisition
# ---------------------------------------------------------------------------
# When enabled, after each batch of NUM_FLU PL shots the most recent qepro
# spectrum is classified by `_classify_pl`. Additional batches are taken (in
# the same Bluesky run, into the same 'fluorescence' stream) until either
# GOOD_TARGET good batches or MAX_BAD bad batches are accumulated. One small
# bookkeeping event per batch is emitted into a single auxiliary stream
# 'fluorescence_quality' for traceability.
USE_GOOD_BAD = False
GOOD_TARGET = 3  # success once this many good batches collected
MAX_BAD = 3  # give up after this many bad batches (log + proceed)

# Classifier thresholds — names + defaults mirror legacy
# scripts/utils/_data_analysis.good_bad_data exactly. Production callers
# (macro_10_good_bad) leave c2_c3=False so only c1 is evaluated; c2/c3 are
# kept available as an opt-in.
DEFAULT_THRESHOLDS = {
    "key_height": 2000,  # c1 threshold
    "height": 30,  # scipy.find_peaks height param
    "distance": 30,  # scipy.find_peaks distance param
    "c2_c3": False,  # evaluate c2/c3? legacy default False
    "threshold": [560, 100000, 200000],  # [split_wl_nm, integral_low, integral_high]
    "int_boundary": [340, 400, 800],  # [LED_lo, LED_hi == PL_lo, PL_hi] (nm)
}

# ---------------------------------------------------------------------------
# Module-level cached Signals for the 'fluorescence_quality' stream.
# Created once at import time so we don't churn descriptor UIDs across runs.
# ---------------------------------------------------------------------------
_Q_BATCH_INDEX = Signal(name="batch_index", value=0)
_Q_VERDICT = Signal(name="verdict", value="bad")
_Q_PEAK_WL = Signal(name="peak_wavelength_nm", value=float("nan"))
_Q_N_GOOD = Signal(name="n_good_total", value=0)
_Q_N_BAD = Signal(name="n_bad_total", value=0)
_Q_N_EVENTS = Signal(name="n_events_in_batch", value=0)
_Q_SIGS = [_Q_BATCH_INDEX, _Q_VERDICT, _Q_PEAK_WL, _Q_N_GOOD, _Q_N_BAD, _Q_N_EVENTS]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _find_nearest_idx(arr, value):
    """Return the index in `arr` whose value is closest to `value`."""
    arr = np.asarray(arr)
    return int(np.abs(arr - value).argmin())


def _classify_pl(x, y, thresholds=None):
    """Classify a PL spectrum as good/bad.

    Faithful port of ``scripts/utils/_data_analysis.good_bad_data`` (the
    legacy production classifier). Behavior identical to a call to the
    legacy with ``c2_c3=False`` (the only configuration ever used by
    ``macro_10_good_bad``), or with ``c2_c3=True`` when explicitly enabled
    via ``thresholds['c2_c3']``.

    Rejection criteria (returns ``(False, top_wl)``):

    - **c1** the highest peak (after suppressing peaks at wavelengths
      < 400 nm by zeroing their heights) has intensity below ``key_height``.
    - **c2** (only if ``c2_c3`` is True) the highest peak is below
      ``threshold[0]`` nm and ``(PL_integral - LED_integral) < threshold[1]``.
    - **c3** (only if ``c2_c3`` is True) the highest peak is **strictly above**
      ``threshold[0]`` nm and ``(PL_integral - LED_integral) < threshold[2]``.
      (An exact equality at ``threshold[0]`` falls through both c2 and c3,
      matching legacy.)

    Integrals (when ``c2_c3`` is True) use ``scipy.integrate.simpson`` over
    ``y[w1:w2]`` for the LED band and ``y[w2:w3]`` for PL, where
    ``w1, w2, w3`` are the indices nearest ``int_boundary[0..2]`` — matching
    the legacy implementation (no x-spacing passed, no double-counting).

    Parameters
    ----------
    x, y : array_like
        Wavelength (nm) and intensity arrays from the QEPro.
    thresholds : dict | None
        Threshold dict; falls back to :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    (is_good, peak_wavelength_nm) : tuple[bool, float]
        ``peak_wavelength_nm`` is ``NaN`` when no qualifying peak is found.
    """
    from scipy.signal import find_peaks
    from scipy import integrate

    t = thresholds or DEFAULT_THRESHOLDS
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    key_height = t["key_height"]
    height = t["height"]
    distance = t["distance"]
    c2_c3 = t.get("c2_c3", False)
    thr = t["threshold"]
    ib = t["int_boundary"]

    # Step 1: find all peaks in the full spectrum.
    peaks, _ = find_peaks(y, height=height, distance=distance)
    if peaks.size == 0:
        return False, float("nan")

    # Step 2 (optional, c2_c3): integrate the LED and PL bands by index slice.
    peak_diff = None
    if c2_c3 and len(ib) >= 3:
        w1 = _find_nearest_idx(x, ib[0])
        w2 = _find_nearest_idx(x, ib[1])
        w3 = _find_nearest_idx(x, ib[2])
        LED_integration = float(integrate.simpson(y[w1:w2]))
        PL_integration = float(integrate.simpson(y[w2:w3]))
        peak_diff = PL_integration - LED_integration

    # Step 3: suppress LED peaks by zeroing height for any peak with x < 400 nm,
    # then argmax over the resulting heights (legacy mechanism).
    peak_heights_2 = [0.0 if x[i] < 400 else float(y[i]) for i in peaks]
    max_idx = int(np.argmax(peak_heights_2))
    top_int = peak_heights_2[max_idx]
    top_wl = float(x[peaks[max_idx]])

    # c1
    if top_int < key_height:
        return False, top_wl

    # c2 / c3 — only when explicitly enabled. Inequalities are strict on
    # both sides to match legacy: an exact equality at threshold[0]
    # (e.g. top_wl == 560.0) falls through both checks.
    if c2_c3 and peak_diff is not None:
        if top_wl < thr[0] and peak_diff < thr[1]:
            return False, top_wl
        if top_wl > thr[0] and peak_diff < thr[2]:
            return False, top_wl

    return True, top_wl


# ---------------------------------------------------------------------------
# Quality monitor (local, blocking callback)
# ---------------------------------------------------------------------------
class PLQualityMonitor(CallbackBase):
    """Subscribed via ``bpp.subs_decorator`` so it runs synchronously on the
    RunEngine thread between event docs.

    The plan reads ``.good_count`` / ``.bad_count`` after each batch and calls
    :meth:`finalize_batch` to classify the most recent spectrum and update
    the counters.

    Parameters
    ----------
    qepro : ophyd device
        The QEPro device whose ``x_axis`` / ``output`` keys carry the spectrum
        in the event document. Used to derive event-data field names so this
        module does not hardcode string keys.
    stream_name : str
        Name of the event stream to monitor (default: ``"fluorescence"``).
    thresholds : dict | None
        Classifier thresholds; defaults to :data:`DEFAULT_THRESHOLDS`.
    """

    def __init__(self, qepro, stream_name="fluorescence", thresholds=None):
        super().__init__()
        self.stream_name = stream_name
        self.x_field = qepro.x_axis.name
        self.y_field = qepro.output.name
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

        self._target_descriptors = set()
        self._latest_spectrum = None
        self._batch_event_count = 0

        self.good_count = 0
        self.bad_count = 0
        self.batch_index = 0
        self.batch_results = []

    def descriptor(self, doc):
        if doc.get("name") == self.stream_name:
            self._target_descriptors.add(doc["uid"])

    def event(self, doc):
        if doc["descriptor"] not in self._target_descriptors:
            return

        data = doc["data"]
        if self.x_field not in data or self.y_field not in data:
            return

        self._latest_spectrum = (
            np.asarray(data[self.x_field]),
            np.asarray(data[self.y_field]),
        )
        self._batch_event_count += 1

    def finalize_batch(self):
        """Classify the most recent spectrum in the just-finished batch."""
        if self._latest_spectrum is None:
            return None

        x, y = self._latest_spectrum
        is_good, peak_wl = _classify_pl(x, y, self.thresholds)
        if is_good:
            self.good_count += 1
        else:
            self.bad_count += 1

        result = {
            "batch_index": self.batch_index,
            "verdict": "good" if is_good else "bad",
            "peak_wavelength_nm": float(peak_wl),
            "n_good_total": self.good_count,
            "n_bad_total": self.bad_count,
            "n_events_in_batch": self._batch_event_count,
        }
        self.batch_results.append(result)
        self.batch_index += 1
        self._batch_event_count = 0
        self._latest_spectrum = None
        return result


# ---------------------------------------------------------------------------
# Measurement sub-plans
# ---------------------------------------------------------------------------


def measure_absorbance(qepro, n_shots, *, stream="absorbance", settle_sec=2):
    """Configure optics for absorbance and trigger ``n_shots`` reads.

    Preserves the state-guard from the previous ``_acquire_uvvis``: if the
    optics are already in the target state, skip the ``mv`` + settle.
    """
    if not (
        LED.get() == "Low"
        and UV_shutter.get() == "High"
        and qepro.correction.get() == "Reference"
        and qepro.spectrum_type.get() == "Absorbtion"
    ):
        yield from bps.mv(
            qepro.correction,
            "Reference",
            qepro.spectrum_type,
            "Absorbtion",
        )
        yield from bps.mv(LED, "Low", UV_shutter, "High")
        yield from bps.sleep(settle_sec)

    for _ in range(n_shots):
        yield from bps.trigger_and_read([qepro], name=stream)


def measure_pl(qepro, n_shots, *, stream="fluorescence", settle_sec=2):
    """Configure optics for PL and trigger ``n_shots`` reads.

    Preserves the state-guard from the previous ``_acquire_uvvis``: if the
    optics are already in the target state, skip the ``mv`` + settle. This
    is what makes the retry loop in :func:`_pl_with_quality_gate` cheap —
    subsequent batches do not pay another settle.
    """
    if not (
        LED.get() == "High"
        and UV_shutter.get() == "Low"
        and qepro.correction.get() == "Dark"
        and qepro.spectrum_type.get() == "Corrected Sample"
    ):
        yield from bps.mv(
            qepro.correction,
            "Dark",
            qepro.spectrum_type,
            "Corrected Sample",
        )
        yield from bps.mv(LED, "High", UV_shutter, "Low")
        yield from bps.sleep(settle_sec)

    for _ in range(n_shots):
        yield from bps.trigger_and_read([qepro], name=stream)


def _emit_quality_event(result):
    """Emit one event in the ``fluorescence_quality`` stream from a result dict."""
    if result is None:
        return
    yield from bps.mv(
        _Q_BATCH_INDEX,
        int(result["batch_index"]),
        _Q_VERDICT,
        result["verdict"],
        _Q_PEAK_WL,
        float(result["peak_wavelength_nm"]),
        _Q_N_GOOD,
        int(result["n_good_total"]),
        _Q_N_BAD,
        int(result["n_bad_total"]),
        _Q_N_EVENTS,
        int(result["n_events_in_batch"]),
    )
    yield from bps.create(name="fluorescence_quality")
    for s in _Q_SIGS:
        yield from bps.read(s)
    yield from bps.save()


def _pl_with_quality_gate(qepro, monitor, num_flu, good_target, max_bad):
    """Run PL batches until good/bad termination.

    Always runs at least one batch. When ``monitor`` is ``None``
    (quality gating disabled), returns after that single batch. Otherwise
    keeps running batches until ``good_count >= GOOD_TARGET`` or
    ``bad_count >= MAX_BAD``.
    """
    # First batch always runs.
    yield from measure_pl(qepro, num_flu)

    if monitor is None:
        return

    yield from _emit_quality_event(monitor.finalize_batch())

    while monitor.good_count < good_target and monitor.bad_count < max_bad:
        yield from measure_pl(qepro, num_flu)
        yield from _emit_quality_event(monitor.finalize_batch())

    if monitor.good_count >= good_target:
        print(f"*** {monitor.good_count} good PL batches, proceeding ***")
    else:
        print(f"*** {monitor.bad_count} bad PL batches, proceeding anyway ***")


# ---------------------------------------------------------------------------
# Steady-state flow context (setup + guaranteed teardown)
# ---------------------------------------------------------------------------


def steady_state_flow(
    plan,
    pump_list,
    rate_list,
    *,
    syringe_list,
    target_vol_list,
    set_target_list,
    syringe_mater_list,
    rate_unit=RATE_UNIT,
    mixer_lengths_cm=MIXER_LENGTHS_CM,
    resident_t_ratio=RESIDENT_T_RATIO,
    post_dilute=False,
    dilute_pump=None,
    dilute_rate_ratio=POST_DILUTE_RATIO,
    dilute_wait_sec=POST_DILUTE_WAIT_SEC,
):
    """Wrap ``plan`` with pump setup before and pump stop after.

    On entry (defensive stop, configure, start, wait, optional dilute):

    1. ``stop_group(pump_list)`` — defensive, in case a previous run was
       killed without teardown running.
    2. ``set_group_infuse2(...)``
    3. ``start_group_infuse(pump_list, rate_list)`` — pumps with rate>0 are
       recorded for teardown.
    4. ``wait_equilibrium2(...)`` — hardware-read wait.
    5. Optional toluene dilution: ``set_group_infuse2`` (no explicit
       ``start_group_infuse``); the dilute pump is still recorded for teardown
       so it's stopped on exit.

    On exit (success **or** exception): ``stop_group`` over every pump that
    was started. Best-effort; failures are logged.
    """
    started_pumps = []

    def setup():
        # 1. Defensive stop in case a prior run left pumps running.
        yield from stop_group(pump_list)

        # 2. Configure synthesis pumps.
        yield from set_group_infuse2(
            syringe_list,
            pump_list,
            set_target_list=set_target_list,
            target_vol_list=target_vol_list,
            rate_list=rate_list,
            syringe_mater_list=syringe_mater_list,
            rate_unit=rate_unit,
        )

        # 3. Start; record which ones actually started.
        yield from start_group_infuse(pump_list, rate_list)
        started_pumps.extend(p for p, r in zip(pump_list, rate_list) if r > 0)

        # 4. Wait for flow equilibrium (hardware-read wait).
        mixer_pump_list = [[f"{mixer_lengths_cm[0]} cm", *pump_list]]
        yield from wait_equilibrium2(mixer_pump_list, ratio=resident_t_ratio)

        # 5. Optional toluene dilution.
        if post_dilute and dilute_pump is not None:
            toluene_rate = sum(r for r in rate_list if r > 0) * dilute_rate_ratio
            print(
                f"\nStarted toluene dilution at {toluene_rate:.1f} uL/min, "
                f"waiting {dilute_wait_sec}s"
            )
            yield from set_group_infuse2(
                [50],
                [dilute_pump],
                set_target_list=[True],
                target_vol_list=["30 ml"],
                rate_list=[toluene_rate],
                syringe_mater_list=["steel"],
                rate_unit=rate_unit,
            )
            # TODO: start group infuse??
            started_pumps.append(dilute_pump)
            yield from sleep_sec_q(dilute_wait_sec)

    def teardown():
        if not started_pumps:
            return
        try:
            yield from stop_group(started_pumps)
        except Exception as e:
            names = [p.name for p in started_pumps]
            print(f"Warning: failed to stop pumps {names}: {repr(e)}")

    def body():
        yield from setup()
        return (yield from plan)

    return (yield from bpp.finalize_wrapper(body(), teardown()))


# ---------------------------------------------------------------------------
# Acquisition plan
# ---------------------------------------------------------------------------


def halide_acquire(
    suggestions: list[dict],
    actuators,
    sensors=None,
    md=None,
    *,
    syringe_list=None,
    syringe_mater_list=None,
    target_vol_list=None,
    set_target_list=None,
    rate_unit=None,
    mixer_lengths_cm=None,
    resident_t_ratio=None,
    num_abs=None,
    num_flu=None,
    precursor_list=None,
    post_dilute=None,
    post_dilute_ratio=None,
    post_dilute_wait_sec=None,
    dof_to_pump=None,
    dilute_pump_name=None,
    use_good_bad=None,
    good_target=None,
    max_bad=None,
):
    """Acquire UV-Vis data for halide perovskite optimization.

    The Blop ``AcquisitionPlan`` for one optimizer suggestion:

    1. Configure + start pumps; wait for equilibrium (inside
       :func:`steady_state_flow`). Optionally configure toluene dilution.
    2. In a single Bluesky run, collect absorbance and fluorescence streams.
       With ``use_good_bad`` enabled, PL is gated by :class:`PLQualityMonitor`
       and additional batches are taken until good/bad termination.
    3. Stop all started pumps (guaranteed by ``bpp.finalize_wrapper``).
    4. Return the run UID.

    Parameters
    ----------
    suggestions : list[dict]
        List of suggestions from the optimizer. Each dict contains DOF names
        as keys with their suggested values. Typically len == 1.
    actuators : list[str]
        Pump device names (may be empty if DOFs have no actuator field).
    sensors : list[str] | None
        Sensor device names (e.g., ["QEPro"]).
    md : dict | None
        Metadata dict (contains blop_correlation_uid for tracking).
    syringe_list : list[float] | None
        Syringe sizes in mL for each pump, in DOF order.
    syringe_mater_list : list[str] | None
        Syringe materials for each pump.
    target_vol_list : list[str] | None
        Target volumes (e.g. ``["30 ml", ...]``) for each pump.
    set_target_list : list[bool] | None
        Whether to auto-set the target volume for each pump.
    rate_unit : str | None
        Flow-rate unit string passed to ``set_group_infuse2``.
    mixer_lengths_cm : list[float] | None
        Lengths of mixer tubing segments in cm.
    resident_t_ratio : float | None
        Multiplier of the residence time to wait for equilibrium.
    num_abs : int | None
        Number of absorbance spectra to collect.
    num_flu : int | None
        Number of fluorescence spectra to collect.
    precursor_list : list[str] | None
        Precursor names stored in run metadata.
    post_dilute : bool | None
        Whether to perform toluene post-dilution.
    post_dilute_ratio : float | None
        Toluene rate = sum(active_rates) * ratio.
    post_dilute_wait_sec : float | None
        Seconds to wait after starting the toluene pump.
    dof_to_pump : dict[str, str] | None
        Mapping of DOF name → pump device name in the queueserver namespace.
    dilute_pump_name : str | None
        Device name of the toluene dilution pump.
    use_good_bad : bool | None
        If ``True``, PL is gated by quality monitoring; additional
        batches are taken until good/bad termination.
    good_target : int | None
        Quality target threshold for good PL data
    max_bad : int | None
        Maximum number of bad PL data before aborting

    Returns
    -------
    str
        UID of the Bluesky run.
    """
    # Resolve all configuration, falling back to module-level defaults
    syringe_list = syringe_list if syringe_list is not None else SYRINGE_LIST
    syringe_mater_list = (
        syringe_mater_list if syringe_mater_list is not None else SYRINGE_MATER_LIST
    )
    target_vol_list = (
        target_vol_list if target_vol_list is not None else TARGET_VOL_LIST
    )
    set_target_list = (
        set_target_list if set_target_list is not None else SET_TARGET_LIST
    )
    rate_unit = rate_unit if rate_unit is not None else RATE_UNIT
    mixer_lengths_cm = (
        mixer_lengths_cm if mixer_lengths_cm is not None else MIXER_LENGTHS_CM
    )
    resident_t_ratio = (
        resident_t_ratio if resident_t_ratio is not None else RESIDENT_T_RATIO
    )
    num_abs = num_abs if num_abs is not None else NUM_ABS
    num_flu = num_flu if num_flu is not None else NUM_FLU
    precursor_list = precursor_list if precursor_list is not None else PRECURSOR_LIST
    post_dilute = post_dilute if post_dilute is not None else POST_DILUTE
    post_dilute_ratio = (
        post_dilute_ratio if post_dilute_ratio is not None else POST_DILUTE_RATIO
    )
    post_dilute_wait_sec = (
        post_dilute_wait_sec
        if post_dilute_wait_sec is not None
        else POST_DILUTE_WAIT_SEC
    )
    dof_to_pump = dof_to_pump if dof_to_pump is not None else DOF_TO_PUMP
    dilute_pump_name = (
        dilute_pump_name if dilute_pump_name is not None else DILUTE_PUMP_NAME
    )
    use_good_bad = use_good_bad if use_good_bad is not None else USE_GOOD_BAD
    good_target = good_target if good_target is not None else GOOD_TARGET
    max_bad = max_bad if max_bad is not None else MAX_BAD

    suggestion = suggestions[0]

    # Extract rates from suggestion — DOF names like "infusion_rate_CsPb"
    dof_names = sorted(k for k in suggestion.keys() if k.startswith("infusion_rate"))
    rate_list = [float(suggestion[name]) for name in dof_names]

    # Resolve pump devices
    pump_list = _resolve_pumps_from_dofs(dof_names, dof_to_pump)

    sample_type = _make_sample_name(rate_list)

    # Build metadata
    _md = {
        "sample_type": sample_type,
        "infuse_rates": rate_list,
        "dof_names": dof_names,
        "precursors": precursor_list[: len(pump_list)],
        "pumps": [p.name for p in pump_list],
        "pump_status": [p.status.get() for p in pump_list],
        "detectors": ["qepro"],
        "use_good_bad": use_good_bad,
    }
    _md.update(md or {})

    # Quality monitoring
    monitor = (
        PLQualityMonitor(qepro, stream_name="fluorescence") if use_good_bad else None
    )
    # Only need descriptor and event documents
    subs = {"descriptor": [monitor], "event": [monitor]} if monitor else {}

    # Acquisition plan
    @bpp.subs_decorator(subs)
    @bpp.stage_decorator([qepro])
    @bpp.run_decorator(md=_md)
    def acquisition():
        yield from measure_absorbance(qepro, num_abs)
        yield from _pl_with_quality_gate(qepro, monitor, num_flu, good_target, max_bad)
        yield from bps.mv(LED, "Low", UV_shutter, "Low")

    dilute_pump = _resolve_pumps([dilute_pump_name])[0] if post_dilute else None

    uid = yield from steady_state_flow(
        acquisition(),
        pump_list=pump_list,
        rate_list=rate_list,
        syringe_list=syringe_list[: len(pump_list)],
        target_vol_list=target_vol_list[: len(pump_list)],
        set_target_list=set_target_list[: len(pump_list)],
        syringe_mater_list=syringe_mater_list[: len(pump_list)],
        post_dilute=post_dilute,
        dilute_pump=dilute_pump,
    )
    return uid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_name(rate_list):
    """Generate a sample name from pump rates."""
    parts = [f"{r:.1f}" for r in rate_list]
    return "_".join(parts)


def _resolve_pumps(pump_names):
    """Resolve pump device objects from their string names in the startup namespace."""
    pumps = []
    for name in pump_names:
        device = globals().get(name)
        if device is None:
            raise ValueError(f"Pump device '{name}' not found in queueserver namespace")
        pumps.append(device)
    return pumps


def _resolve_pumps_from_dofs(dof_names, dof_to_pump=None):
    """Map DOF names to pump devices via DOF_TO_PUMP mapping."""
    if dof_to_pump is None:
        dof_to_pump = DOF_TO_PUMP
    pump_names = []
    for dof in dof_names:
        pname = dof_to_pump.get(dof)
        if pname is None:
            raise ValueError(f"No pump mapping for DOF '{dof}'")
        pump_names.append(pname)
    return _resolve_pumps(pump_names)
