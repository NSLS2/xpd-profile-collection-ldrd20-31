"""
Mock ophyd devices for smoke testing the halide perovskite QueueserverAgent.

Replaces real EPICS devices with ophyd.sim-based mocks that share the same
variable names, so the acquisition plan runs unchanged.
"""

import numpy as np
from ophyd.sim import SynSignal, NullStatus
from ophyd import Device, Signal, Component as Cpt
import bluesky.plan_stubs as bps

print("[TEST MODE] Loading mock devices...")

# ---------------------------------------------------------------------------
# Mock QEPro
# ---------------------------------------------------------------------------

_WAVELENGTHS = np.linspace(200, 1000, 2048)


def _make_spectrum():
    """Generate a synthetic fluorescence spectrum with a gaussian peak around 520 nm.

    Amplitude is set to 5000 so the peak satisfies the PLQualityMonitor
    classifier thresholds used by the ``use_good_bad`` acquisition mode
    (key_height=2000, integral_low=100_000).
    """
    noise = np.random.normal(0, 5, 2048)
    peak = 5000 * np.exp(-0.5 * ((_WAVELENGTHS - 520) / 15) ** 2)
    return peak + noise


def _make_absorbance_spectrum():
    """Generate a synthetic absorbance spectrum with absorption at 365 nm."""
    # Broad absorption band centered at 380 nm (typical for perovskite NCs)
    abs_band = 0.8 * np.exp(-0.5 * ((_WAVELENGTHS - 380) / 60) ** 2)
    # Small baseline offset
    baseline = 0.02
    noise = np.random.normal(0, 0.005, 2048)
    return abs_band + baseline + noise


class MockQEPro(Device):
    """Minimal mock of the QEPro spectrometer."""

    # Signals the acquisition plan sets via bps.mv
    correction = Cpt(Signal, value="Reference", kind="normal")
    spectrum_type = Cpt(Signal, value="Absorbtion", kind="normal")

    # Signals that get read during data collection
    x_axis = Cpt(Signal, value=_WAVELENGTHS, kind="normal")
    output = Cpt(Signal, value=_make_spectrum(), kind="normal")
    sample = Cpt(Signal, value=_make_spectrum(), kind="normal")
    dark = Cpt(Signal, value=np.zeros(2048), kind="normal")
    reference = Cpt(Signal, value=np.ones(2048) * 1000, kind="normal")

    # Config signals
    integration_time = Cpt(Signal, value=100, kind="config")
    num_spectra = Cpt(Signal, value=10, kind="config")
    buff_capacity = Cpt(Signal, value=3, kind="config")

    def trigger(self):
        """Generate fresh spectrum data on each trigger.

        Uses absorbance-like data when spectrum_type is 'Absorbtion',
        fluorescence-like data otherwise.
        """
        if self.spectrum_type.get() == "Absorbtion":
            spec = _make_absorbance_spectrum()
        else:
            spec = _make_spectrum()
        self.output.put(spec)
        self.sample.put(spec)
        return NullStatus()

    def stage(self):
        return [self]

    def unstage(self):
        return [self]

    def describe(self):
        res = {}
        for attr in [
            "x_axis",
            "output",
            "sample",
            "dark",
            "reference",
            "spectrum_type",
            "integration_time",
            "num_spectra",
            "buff_capacity",
        ]:
            sig = getattr(self, attr)
            val = sig.get()
            if isinstance(val, np.ndarray):
                res[f"{self.name}_{attr}"] = {
                    "source": "SIM",
                    "dtype": "array",
                    "shape": list(val.shape),
                }
            else:
                res[f"{self.name}_{attr}"] = {
                    "source": "SIM",
                    "dtype": "number" if isinstance(val, (int, float)) else "string",
                    "shape": [],
                }
        return res

    def read(self):
        res = {}
        import time as _time

        ts = _time.time()
        for attr in [
            "x_axis",
            "output",
            "sample",
            "dark",
            "reference",
            "spectrum_type",
            "integration_time",
            "num_spectra",
            "buff_capacity",
        ]:
            sig = getattr(self, attr)
            res[f"{self.name}_{attr}"] = {"value": sig.get(), "timestamp": ts}
        return res


# Replace the real qepro
qepro = MockQEPro(name="QEPro")

# ---------------------------------------------------------------------------
# Mock LED and UV_shutter
# ---------------------------------------------------------------------------


class MockBinarySignal(Signal):
    """Signal that accepts 'High'/'Low' string values."""

    def set(self, value, **kwargs):
        self.put(value)
        return NullStatus()


LED = MockBinarySignal(name="LED_M365LP1", value="Low")
UV_shutter = MockBinarySignal(name="UV_shutter", value="Low")

# ---------------------------------------------------------------------------
# Mock DDS pumps
# ---------------------------------------------------------------------------


class MockPump(Device):
    """Minimal mock of syrng_DDS_ax with set_infuse2, infuse_pump2, stop_pump2."""

    # Signals that the real pump has
    infuse_rate = Cpt(Signal, value=0.0, kind="hinted")
    read_infuse_rate = Cpt(Signal, value=0.0, kind="hinted")
    read_infuse_rate_unit = Cpt(Signal, value="ul/min", kind="hinted")
    status = Cpt(Signal, value="Idle", kind="normal")

    def set_infuse2(
        self,
        input_size,
        syringe_material="steel",
        clear=False,
        set_target=True,
        target_vol=20,
        target_unit="ml",
        infuse_rate=100,
        infuse_unit="ul/min",
    ):
        """Mock: just store the rate."""
        self.infuse_rate.put(infuse_rate)
        self.read_infuse_rate.put(infuse_rate)
        yield from bps.null()

    def infuse_pump2(self, clear=False):
        """Mock: set status to Infusing."""
        self.status.put("Infusing")
        yield from bps.null()

    def stop_pump2(self, clear=False):
        """Mock: set status to Idle."""
        self.status.put("Idle")
        yield from bps.null()


# Instantiate mock pumps with the same variable names as real ones
dds2_p1 = MockPump(name="DDS2_p1")
dds2_p2 = MockPump(name="DDS2_p2")
dds1_p1 = MockPump(name="DDS1_p1")
dds1_p2 = MockPump(name="DDS1_p2")
dds3_p1 = MockPump(name="DDS3_p1")

# ---------------------------------------------------------------------------
# Mock helper plans that the acquisition plan calls
# ---------------------------------------------------------------------------


def start_group_infuse(pump_list, rate_list):
    """Mock: start all pumps."""
    for pump, rate in zip(pump_list, rate_list):
        if rate != 0.0:
            yield from pump.infuse_pump2()


def stop_group(pump_list):
    """Mock: stop all pumps."""
    for pump in pump_list:
        yield from pump.stop_pump2()


def wait_equilibrium2(mixer_pump_list, ratio=1, tubing_ID_mm=1.016):
    """Mock: skip the residence-time wait so smoke tests finish quickly.

    The real implementation sleeps for one residence-time (typically several
    minutes). In test mode we just yield a no-op so the plan proceeds
    immediately.
    """
    yield from bps.null()


print(
    "[TEST MODE] Mock devices loaded: qepro, LED, UV_shutter, dds1_p1, dds1_p2, dds2_p1, dds2_p2"
)
