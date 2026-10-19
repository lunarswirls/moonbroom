"""LP frame pulse UTC from archived ERT and explicit receiver ephemerides"""

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import loaded_kernels


def ephemeris_to_utc(epochs):
    """Preserve leap seconds as ISO strings alongside pandas-compatible UTC"""
    import spiceypy as spice

    values = np.atleast_1d(np.asarray(epochs, dtype=float))
    unique, indices = np.unique(values, return_inverse=True)
    converted = [spice.et2utc(value, "ISOC", 6) + "Z" if np.isfinite(value) else None
                 for value in unique]
    strings = [converted[index] for index in indices]
    leap = np.array([value is not None and value[17:19] == "60" for value in strings])
    dates = pd.to_datetime([None if flag else value for value, flag in zip(strings, leap)], utc=True)
    return pd.DataFrame({"utc": dates, "utc_iso": strings, "utc_leap_second": leap})


def reconstruct_frame_times(frames, kernels, *, receiver, spacecraft="-25",
                            data_system_delay_s=2.0):
    """Reconstruct LP frame pulses using converged reception light time

    receiver must identify the receiving station or an explicitly selected
    approximation such as EARTH for the geocenter. Load the appropriate LP
    spacecraft SPK, planetary ephemeris, leap-second kernel and any receiver
    kernels. MAGERFMT section 1.5 specifies a two-second regular data-system
    delay. The merged archive has already removed the 3200-second playback
    delay, so it must not be subtracted again. This does not supply a mission
    oscillator drift model or recover the unidentified receiving station.
    """
    import spiceypy as spice
    from .level0 import correlate_frame_times

    if not isinstance(receiver, str) or not receiver.strip():
        raise ValueError("Supply the receiving station or an explicit receiver approximation")
    delay = float(data_system_delay_s)
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("The data-system delay must be finite and nonnegative")
    kernels = [kernels] if isinstance(kernels, (str, Path)) else list(kernels)
    out = correlate_frame_times(frames)
    strings = out["ert_iso"] if "ert_iso" in out else out["ert"].map(
        lambda value: value.isoformat() if pd.notna(value) else None)
    reception = np.full(len(out), np.nan)
    light_time = np.full(len(out), np.nan)
    with loaded_kernels(kernels):
        for index, value in enumerate(strings):
            if value is None or pd.isna(value):
                continue
            if any(column in out and not bool(out[column].iloc[index])
                   for column in ("header_valid", "ert_valid")):
                continue
            value = str(value)
            if value.endswith("+00:00"):
                value = value[:-6] + "Z"
            reception[index] = spice.str2et(value)
            light_time[index] = spice.spkpos(spacecraft, reception[index], "J2000", "CN", receiver)[1]
        epochs = reception - light_time - delay
        rejected = np.zeros(len(out), dtype=bool)
        ordered = np.flatnonzero(np.isfinite(epochs))
        bad = np.flatnonzero(np.diff(epochs[ordered]) <= 0)
        rejected[ordered[bad]] = True
        rejected[ordered[bad + 1]] = True
        epochs[rejected] = np.nan
        if (np.diff(epochs[np.isfinite(epochs)]) <= 0).any():
            raise ValueError("Reconstructed frame times must increase with the frame counter")
        dates = ephemeris_to_utc(epochs)
    out["ert_et"] = reception
    out["light_time_s"] = light_time
    out["data_system_delay_s"] = delay
    out["frame_et"] = epochs
    out["ert_anchor_rejected"] = rejected
    for column in dates:
        out["frame_" + column] = dates[column].to_numpy()
    out["receiver"] = receiver
    out["time_basis"] = "merged ERT minus SPICE reception light time and data-system delay"
    out.attrs.update({"spacecraft": spacecraft, "receiver": receiver,
                      "clock_model": "frame correlations, bounded interpolation in ephemeris seconds",
                      "timing_kernels": [str(path) for path in kernels]})
    return out


def reconstruct_product_times(products, frames, kernels):
    """Attach bounded UTC correlations to every decoded collection clock

    Uncovered burst collection times remain missing rather than receiving
    an extrapolation from their later playback time. ISO columns retain leap
    seconds that pandas cannot represent. Input tables are not modified.
    """
    if "frame_et" not in frames:
        raise ValueError("Frames require reconstructed ephemeris times")
    known = frames.loc[np.isfinite(frames["frame_et"])].sort_values("frame_clock_s")
    if (len(known) < 2 or (np.diff(known["frame_clock_s"]) <= 0).any()
            or (np.diff(known["frame_et"]) <= 0).any()):
        raise ValueError("At least two unique frame clock correlations are required")
    result = {}
    names = {"clock_s": "utc", "start_clock_s": "start_utc", "stop_clock_s": "stop_utc",
             "major_clock_s": "major_utc", "accumulation_clock_s": "accumulation_utc",
             "pulse_clock_s": "pulse_utc"}
    with loaded_kernels(kernels):
        for name, table in products.items():
            out = table.copy()
            if name not in ("frames", "input_files"):
                for clock, target in names.items():
                    if clock not in out:
                        continue
                    epochs = np.interp(out[clock].to_numpy(float), known["frame_clock_s"],
                                       known["frame_et"], left=np.nan, right=np.nan)
                    dates = ephemeris_to_utc(epochs)
                    out[target] = dates["utc"].array
                    out[target + "_iso"] = dates["utc_iso"].to_numpy()
                    out[target + "_leap_second"] = dates["utc_leap_second"].to_numpy()
                    out[target + "_valid"] = np.isfinite(epochs)
                out.attrs["time_basis"] = frames.attrs.get("clock_model")
            result[name] = out
    result["frames"] = frames.copy()
    return result


def reconstruct_spin_phase(samples, sunpulse, *, clock_col="clock_s", max_gap_s=6.0,
                           allow_estimated=True):
    """Interpolate spin phase between official measured or reconstructed sun pulses

    Repeated reports of a pulse are deduplicated, preferring measured values.
    Missing pulses, unsupported intervals and extrapolation remain invalid.
    max_gap_s limits intervals to one rotation; the caller must choose it for
    the applicable spin rate. Eclipse estimates retain their source flag.
    """
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Maximum pulse gap must be positive and finite")
    pulses = sunpulse.loc[sunpulse["timing_valid"]].copy()
    if not allow_estimated:
        pulses = pulses.loc[~pulses["estimated"]]
    pulses = pulses.sort_values(["pulse_clock_s", "estimated"]).drop_duplicates("pulse_clock_s")
    out = samples.copy()
    out["spin_phase_deg"] = np.nan
    out["spin_period_s"] = np.nan
    out["sunpulse_estimated"] = False
    out["sunpulse_uncertainty_s"] = np.nan
    out["spin_phase_valid"] = False
    if len(pulses) < 2:
        return out
    clocks = pulses["pulse_clock_s"].to_numpy(float)
    values = out[clock_col].to_numpy(float)
    index = np.searchsorted(clocks, values, side="right") - 1
    safe = np.clip(index, 0, len(clocks) - 2)
    period = clocks[safe + 1] - clocks[safe]
    valid = (np.isfinite(values) & (index >= 0) & (index < len(clocks) - 1)
             & (period > 0) & (period <= max_gap_s))
    phase = (values - clocks[safe]) / period * 360
    estimated = pulses["estimated"].to_numpy(bool)
    uncertainty = pulses["uncertainty_s"].to_numpy(float)
    out["spin_phase_deg"] = np.where(valid, phase, np.nan)
    out["spin_period_s"] = np.where(valid, period, np.nan)
    out["sunpulse_estimated"] = valid & (estimated[safe] | estimated[safe + 1])
    out["sunpulse_uncertainty_s"] = np.where(valid, np.maximum(uncertainty[safe], uncertainty[safe + 1]), np.nan)
    out["spin_phase_valid"] = valid
    out.attrs["spin_phase_source"] = "archived SUNPULSE, bounded single-rotation interpolation"
    return out
