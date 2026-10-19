"""Native-cadence MAG offset estimation and calibrated nine-hertz sample output"""

from pathlib import Path

import numpy as np
import pandas as pd

from .archive_calibration import apply_mag_archive_calibration


def mag_calibration_diagnostics(samples, coefficients):
    """Annotate native output with one primary failure and model availability

    Quality, UTC, phase and local-offset failures take precedence over model
    gaps. Existing flags remain available for overlapping-cause summaries.
    This function reports support and never changes fields or quality flags.
    """
    out = samples.copy()
    times = pd.to_datetime(out["utc"], utc=True, format="mixed")
    table = coefficients.copy()
    for name in ("valid_from", "valid_to"):
        table[name] = pd.to_datetime(table[name], utc=True, format="mixed")
    accepted = table["fit_valid"].fillna(False)
    if "validation_pass" in table:
        accepted &= table["validation_pass"].fillna(False)
    out["calibration_model_available"] = False
    for row in table.loc[accepted].itertuples():
        selected = out["range"].eq(row.range) & times.ge(row.valid_from) & times.lt(row.valid_to)
        out.loc[selected, "calibration_model_available"] = True
    out["calibration_failure_reason"] = np.where(out["calibration_valid"], "calibrated", "other")
    pending = ~out["calibration_valid"]
    conditions = []
    for column, reason, invert in (("calibration_mode", "diagnostic_mode", False),
                                   ("adc_at_limit", "adc_limit", False),
                                   ("mag_frame_matches", "frame_mismatch", True),
                                   ("range_changed", "range_transition", False),
                                   ("science_valid_original", "other_telemetry_quality", True)):
        if column in out:
            bad = ~out[column] if invert else out[column]
            conditions.append((reason, bad))
    conditions.append(("raw_counts_unavailable", ~np.isfinite(out[["b" + axis + "_signed" for axis in "xyz"]]).all(axis=1)))
    conditions.append(("utc_unavailable", times.isna()))
    phase_bad = pd.Series(False, index=out.index)
    if "spin_phase_deg" in out:
        phase_bad |= ~np.isfinite(out["spin_phase_deg"])
    if "spin_phase_valid" in out:
        phase_bad |= ~out["spin_phase_valid"]
    conditions.append(("spin_phase_unavailable", phase_bad))
    if "spin_offset_valid" in out:
        conditions.append(("local_offset_unavailable", ~out["spin_offset_valid"]))
    for reason, bad in conditions:
        out.loc[pending & bad, "calibration_failure_reason"] = reason
        pending &= ~bad
    for range_id in out.loc[pending, "range"].unique():
        group = table.loc[table["range"].eq(range_id)]
        selected = pending & out["range"].eq(range_id)
        out.loc[selected, "calibration_failure_reason"] = "range_not_trained" if group.empty else "outside_model_scope"
        for row in group.itertuples():
            matching = selected & times.ge(row.valid_from) & times.lt(row.valid_to)
            if not row.fit_valid:
                reason = "training_insufficient" if getattr(row, "n_windows", np.inf) < 20 else "training_unidentified"
            elif hasattr(row, "validation_pass") and not row.validation_pass:
                reason = "validation_insufficient" if getattr(row, "n_validation", 0) < 20 else "validation_error"
            else:
                reason = "other"
            out.loc[matching, "calibration_failure_reason"] = reason
    out.attrs["failure_precedence"] = "telemetry quality, raw values, UTC, spin phase, local offset, range model"
    return out


def fit_mag_spin_offsets(samples, *, window_s=60.0, min_samples=120, max_condition=1000.0):
    """Estimate transverse sensor offsets from spin modulation in local windows

    The despun field is allowed a linear trend within each window. A constant
    sensor offset is separated from that rotating signal by least squares.
    This is an empirical approximation to flight offset processing and can
    remove real signals synchronous with spin; independent native-rate truth
    is required to establish accuracy at those frequencies. Z is not inferred.
    """
    if not np.isfinite(window_s) or window_s <= 0 or min_samples < 6:
        raise ValueError("Use a positive window and at least six valid samples")
    raw = samples.loc[samples["mode"].eq("realtime") & samples["science_valid"]
                      & samples["spin_phase_valid"]].copy()
    raw["offset_window"] = np.floor(raw["clock_s"]/window_s)
    rows = []
    for (range_id, window), group in raw.groupby(["range", "offset_window"]):
        if len(group) < min_samples:
            continue
        clocks = group["clock_s"].to_numpy(float)
        duration = np.ptp(clocks)
        period = group["spin_period_s"].median() if "spin_period_s" in group else 5.
        if not np.isfinite(period) or period <= 0 or duration < 3.5*period:
            continue
        center = np.mean(clocks)
        time = (clocks - center)/window_s
        phase = np.deg2rad(group["spin_phase_deg"].to_numpy(float))
        design = np.column_stack([np.cos(phase), np.sin(phase), time*np.cos(phase), time*np.sin(phase), np.ones(len(group))])
        target = group[["bx_signed", "by_signed"]].to_numpy(float)
        if not np.isfinite(design).all() or not np.isfinite(target).all():
            continue
        weights = np.ones(len(group))
        for _ in range(5):
            coefficients, _, rank, singular = np.linalg.lstsq(design*weights[:, None], target*weights[:, None], rcond=None)
            residual = target - design@coefficients
            amplitude = np.linalg.norm(residual, axis=1)
            scale = max(float(np.median(amplitude)), 1e-6)
            weights = np.sqrt(np.minimum(1, 2*scale/np.maximum(amplitude, 1e-6)))
        condition = singular[0]/singular[-1] if singular[-1] > 0 else np.inf
        rows.append((range_id, window, center, clocks.min(), clocks.max(), len(group), rank, condition,
                     coefficients[-1, 0], coefficients[-1, 1], np.sqrt(np.mean(residual**2)),
                     rank == 5 and condition <= max_condition))
    return pd.DataFrame(rows, columns=["range", "offset_window", "clock_center_s", "clock_start_s", "clock_stop_s",
                                        "n_samples", "design_rank", "condition", "zero_x", "zero_y",
                                        "residual_rms_counts", "offset_valid"])


def correct_mag_spin_offsets(samples, offsets, *, max_distance_s=60.0):
    """Subtract supported transverse offsets while retaining original signed counts"""
    if not np.isfinite(max_distance_s) or max_distance_s <= 0:
        raise ValueError("Maximum offset distance must be positive and finite")
    out = samples.copy()
    out["science_valid_original"] = out["science_valid"]
    clocks = out["clock_s"].to_numpy(float)
    out["spin_offset_valid"] = False
    for axis in "xy":
        out["b" + axis + "_signed_original"] = out["b" + axis + "_signed"]
        out["spin_offset_" + axis + "_counts"] = np.nan
    for range_id, group in offsets.loc[offsets["offset_valid"]].groupby("range"):
        group = group.sort_values("clock_center_s")
        centers = group["clock_center_s"].to_numpy(float)
        if (np.diff(centers) <= 0).any():
            raise ValueError("Offset centers must be unique within each range")
        selected = np.flatnonzero(out["range"].eq(range_id).to_numpy())
        values = clocks[selected]
        position = np.searchsorted(centers, values)
        left = centers[np.clip(position - 1, 0, len(centers) - 1)]
        right = centers[np.clip(position, 0, len(centers) - 1)]
        valid = np.minimum(np.abs(values - left), np.abs(values - right)) <= max_distance_s
        out.iloc[selected[valid], out.columns.get_loc("spin_offset_valid")] = True
        for axis in "xy":
            correction = np.interp(values, centers, group["zero_" + axis])
            out.iloc[selected[valid], out.columns.get_loc("spin_offset_" + axis + "_counts")] = correction[valid]
    for axis in "xy":
        out["b" + axis + "_signed"] = out["b" + axis + "_signed_original"] - out["spin_offset_" + axis + "_counts"]
    out["science_valid"] &= out["spin_offset_valid"]
    out.attrs["transverse_offset_model"] = "local rotating-field plus linear-trend fit"
    return out


def calibrate_mag_9hz(samples, coefficients, *, offsets=None, max_offset_distance_s=60.0):
    """Return original real-time samples with calibrated fields and no resampling

    Coefficients must have been fit using the same offset-processing choice.
    If validation_pass is present, only independently accepted rows are used.
    Invalid samples remain rows with missing fields. Gaps are never filled.
    """
    native = samples.loc[samples["mode"].eq("realtime")].copy()
    usable = coefficients.copy()
    if "validation_pass" in usable:
        usable = usable.loc[usable["validation_pass"]]
    if "offset_model" in usable and usable["offset_model"].eq("local spin fit").any() and offsets is None:
        raise ValueError("These coefficients require independently estimated local spin offsets")
    if offsets is not None:
        native = correct_mag_spin_offsets(native, offsets, max_distance_s=max_offset_distance_s)
    if usable.loc[usable["fit_valid"]].empty:
        out = native.copy()
        for axis in "xyz":
            out["B" + axis + "_SCD"] = np.nan
        out["calibration_valid"] = False
        out["calibration_basis"] = "no independently accepted archive-derived model"
    else:
        out = apply_mag_archive_calibration(native, usable)
    for axis in "xy":
        original = "b" + axis + "_signed_original"
        if original in out:
            out["b" + axis + "_calibration_counts"] = out["b" + axis + "_signed"]
            out["b" + axis + "_signed"] = out[original]
    out["sample_rate_hz"] = 9.0
    out["native_sample_preserved"] = True
    out["high_frequency_validated"] = False
    if all(name in out for name in ("source_file", "record_index", "sample_index")):
        out["source_sample_id"] = (out["source_file"].map(lambda value: Path(value).name) + ":"
                                    + out["record_index"].astype(str) + ":" + out["sample_index"].astype(str))
        if out["source_sample_id"].duplicated().any():
            raise ValueError("Native source-sample identities must be unique")
    out.attrs.update({"sample_rate_hz": 9.0, "resampled": False, "high_frequency_validated": False})
    return mag_calibration_diagnostics(out, coefficients)
