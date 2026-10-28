"""Native-cadence MAG offset estimation and calibrated 9/18 Hz sample output"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .archive_calibration import apply_mag_archive_calibration
from .mag_state import (MAG_SCOPE_FIELDS, mag_model_modes, mag_model_mask,
                        mag_model_assignments, validate_mag_model_scopes)


def mag_calibration_diagnostics(samples, coefficients, *, allow_realtime_transfer=False):
    """Annotate native output with one primary failure and model availability

    Quality, UTC, phase and local-offset failures take precedence over model
    gaps. Existing flags remain available for overlapping-cause summaries.
    This function reports support and never changes fields or quality flags.
    """
    out = samples.copy()
    times = pd.to_datetime(out["utc"], utc=True, format="mixed")
    table = coefficients.copy()
    table["mode"] = mag_model_modes(table)
    for name in ("valid_from", "valid_to"):
        table[name] = pd.to_datetime(table[name], utc=True, format="mixed")
    accepted = table["fit_valid"].fillna(False)
    if "validation_pass" in table:
        accepted &= table["validation_pass"].fillna(False)
    out["calibration_model_available"] = False
    for selected in mag_model_assignments(out, table.loc[accepted],
                                           allow_realtime_transfer=allow_realtime_transfer):
        out.loc[selected, "calibration_model_available"] = True
    out["calibration_failure_reason"] = np.where(out["calibration_valid"], "calibrated", "other")
    pending = ~out["calibration_valid"]
    conditions = []
    for column, reason in (('archive_noisy', 'archive_noisy'), ('archive_gap', 'archive_gap'),
                           ('mager_command_transition', 'mager_command_transition'),
                           ('maneuver_candidate', 'maneuver_candidate'),
                           ('post_maneuver_settling', 'post_maneuver_settling')):
        if column in out:
            bad = out[column]
            if column == 'maneuver_candidate' and 'maneuver_timing_uncertain' in out:
                bad = bad & ~out['maneuver_timing_uncertain']
            conditions.append((reason, bad))
    for column, reason, invert in (("calibration_mode", "diagnostic_mode", False),
                                   ("adc_at_limit", "adc_limit", False),
                                   ("mag_frame_matches", "frame_mismatch", True),
                                   ("range_changed", "range_transition", False),
                                   ("science_valid_original", "other_telemetry_quality", True)):
        if column in out:
            bad = ~out[column] if invert else out[column]
            conditions.append((reason, bad))
    if "mag_hardware_transition" in out:
        conditions.append(("hardware_state_transition", out["mag_hardware_transition"]))
    if "mag_hardware_state_valid" in out:
        conditions.append(("hardware_state_unavailable", ~out["mag_hardware_state_valid"]))
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
        if "mode" in out:
            wanted = out.loc[selected, "mode"].unique()
            group = group.loc[mag_model_modes(group).isin(wanted)
                              | (allow_realtime_transfer & mag_model_modes(group).eq("realtime"))]
            if group.empty and table["range"].eq(range_id).any():
                out.loc[selected, "calibration_failure_reason"] = "mode_not_trained"
                continue
        out.loc[selected, "calibration_failure_reason"] = "range_not_trained" if group.empty else "outside_model_scope"
        scope_matches = pd.Series(False, index=out.index)
        for mask in mag_model_assignments(out, group, allow_realtime_transfer=allow_realtime_transfer):
            scope_matches |= mask
        for row in group.itertuples():
            in_time = selected & times.ge(row.valid_from) & times.lt(row.valid_to)
            matching = in_time & mag_model_mask(out, pd.Series(row._asdict()))
            if allow_realtime_transfer and getattr(row, "mode", "realtime") == "realtime":
                matching |= in_time & mag_model_mask(out, pd.Series(row._asdict()), mode="burst")
            if any(pd.notna(getattr(row, field, None)) for field in MAG_SCOPE_FIELDS):
                out.loc[in_time & ~scope_matches, "calibration_failure_reason"] = "hardware_state_outside_model_scope"
            if not row.fit_valid:
                reason = "training_insufficient" if getattr(row, "n_windows", np.inf) < 20 else "training_unidentified"
            elif hasattr(row, "validation_pass") and not row.validation_pass:
                reason = "validation_insufficient" if getattr(row, "n_validation", 0) < 20 else "validation_error"
            else:
                reason = "other"
            out.loc[matching, "calibration_failure_reason"] = reason
    out.attrs["failure_precedence"] = "telemetry quality, raw values, UTC, spin phase, local offset, range model"
    return out


def fit_mag_spin_offsets(samples, *, mode="realtime", window_s=60.0, min_samples=120, max_condition=1000.0):
    """Estimate transverse sensor offsets from spin modulation in local windows

    The despun field is allowed a linear trend within each window. A constant
    sensor offset is separated from that rotating signal by least squares.
    This is an empirical approximation to flight offset processing and can
    remove real signals synchronous with spin; independent native-rate truth
    is required to establish accuracy at those frequencies. Z is not inferred.
    """
    if not np.isfinite(window_s) or window_s <= 0 or min_samples < 6:
        raise ValueError("Use a positive window and at least six valid samples")
    if mode not in ("realtime", "burst"):
        raise ValueError("MAG offset mode must be realtime or burst")
    if mode == "burst" and "burst_id" not in samples:
        raise ValueError("Burst offset fits require collection burst_id values")
    valid = samples["science_valid"] & samples["spin_phase_valid"]
    if "mag_hardware_state_valid" in samples:
        valid &= samples["mag_hardware_state_valid"] & ~samples["mag_hardware_transition"]
    raw = samples.loc[samples["mode"].eq(mode) & valid].copy()
    if mode == "burst":
        raw = raw.loc[raw["burst_id"].notna()]
    raw["offset_window"] = np.floor(raw["clock_s"]/window_s)
    cohorts = [name for name in ("mager_command_cohort", "mag_hardware_cohort") if name in raw]
    keys = ["range", "offset_window"] + (["burst_id"] if mode == "burst" else [])
    rows = []
    for key, group in raw.groupby(keys, dropna=False):
        range_id, window = key[:2]
        if len(group) < min_samples:
            continue
        if any(group[name].nunique(dropna=False) != 1 for name in cohorts):
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
        row = (range_id, window, center, clocks.min(), clocks.max(), len(group), rank, condition,
                     coefficients[-1, 0], coefficients[-1, 1], np.sqrt(np.mean(residual**2)),
                     rank == 5 and condition <= max_condition)
        extras = tuple(group[name].iloc[0] for name in cohorts)
        rows.append(row + (mode,) + extras + ((key[2],) if mode == "burst" else ()))
    columns = ["range", "offset_window", "clock_center_s", "clock_start_s", "clock_stop_s",
                                        "n_samples", "design_rank", "condition", "zero_x", "zero_y",
                                        "residual_rms_counts", "offset_valid"]
    return pd.DataFrame(rows, columns=columns + ["mode"] + cohorts + (["burst_id"] if mode == "burst" else []))


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
    scoped = offsets.copy()
    if "mode" not in scoped:
        scoped["mode"] = "realtime"
    if scoped['mode'].eq('burst').any() and 'burst_id' not in scoped:
        raise ValueError("Burst offsets require burst_id scope")
    if 'mag_hardware_cohort' in out and 'mag_hardware_cohort' not in scoped:
        raise ValueError("Hardware-scoped samples require hardware-scoped offsets")
    scope_keys = ["mode"] + [name for name in ("mager_command_cohort", "mag_hardware_cohort", "burst_id") if name in scoped]
    for name in scope_keys:
        if name not in out:
            raise ValueError(f"Scoped offsets require sample {name} values")
    for key, group in scoped.loc[scoped["offset_valid"]].groupby(["range"] + scope_keys, dropna=False):
        range_id = key[0]
        group = group.sort_values("clock_center_s")
        centers = group["clock_center_s"].to_numpy(float)
        if (np.diff(centers) <= 0).any():
            raise ValueError("Offset centers must be unique within each range and collection context")
        matching = out["range"].eq(range_id)
        for name, value in zip(scope_keys, key[1:]):
            matching &= out[name].isna() if pd.isna(value) else out[name].eq(value)
        selected = np.flatnonzero(matching.to_numpy())
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


def _calibrate_mag_native(samples, coefficients, *, mode, sample_rate_hz, offsets=None,
                          max_offset_distance_s=60.0, allow_realtime_transfer=False):
    """Return original samples in one mode with calibrated fields and no resampling

    Coefficients must have been fit using the same offset-processing choice.
    If validation_pass is present, only independently accepted rows are used.
    Invalid samples remain rows with missing fields. Gaps are never filled.
    """
    if not isinstance(allow_realtime_transfer, bool):
        raise ValueError("allow_realtime_transfer must be an explicit Boolean")
    native = samples.loc[samples["mode"].eq(mode)].copy()
    usable = coefficients.loc[mag_model_modes(coefficients).eq(mode)
                              | (allow_realtime_transfer & mag_model_modes(coefficients).eq("realtime"))].copy()
    if mode == 'burst' and not usable.empty and 'validation_pass' not in usable:
        raise ValueError("Burst calibration requires independent validation_pass flags")
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
        out = apply_mag_archive_calibration(native, usable, allow_realtime_transfer=allow_realtime_transfer)
    for axis in "xy":
        original = "b" + axis + "_signed_original"
        if original in out:
            out["b" + axis + "_calibration_counts"] = out["b" + axis + "_signed"]
            out["b" + axis + "_signed"] = out[original]
    out["sample_rate_hz"] = sample_rate_hz
    out["calibration_mode_transfer"] = False
    out["calibration_hardware_scoped"] = False
    out["native_sample_preserved"] = True
    out["high_frequency_validated"] = False
    out["calibration_model_id"] = pd.Series(None, index=out.index, dtype=object)
    out["cohort"] = pd.Series(None, index=out.index, dtype=object)
    out["calibration_source"] = pd.Series(None, index=out.index, dtype=object)
    identified = usable.loc[usable["fit_valid"]]
    validate_mag_model_scopes(identified)
    masks = mag_model_assignments(out, identified, allow_realtime_transfer=allow_realtime_transfer)
    for (_, model), selected in zip(identified.iterrows(), masks):
        identity = hashlib.sha256(json.dumps(model.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
        out.loc[selected, "calibration_model_id"] = identity
        out.loc[selected, "cohort"] = model.get("cohort", identity)
        out.loc[selected, "calibration_source"] = model.get("source", identity)
        model_mode = model.get("mode", "realtime")
        model_mode = "realtime" if pd.isna(model_mode) else model_mode
        out.loc[selected, "calibration_mode_transfer"] = model_mode != mode
        out.loc[selected, "calibration_hardware_scoped"] = any(pd.notna(model.get(f)) for f in MAG_SCOPE_FIELDS)
    if all(name in out for name in ("source_file", "record_index", "sample_index")):
        out["source_sample_id"] = (out["source_file"].map(lambda value: Path(value).name) + ":"
                                    + out["record_index"].astype(str) + ":" + out["sample_index"].astype(str))
        if mode == "burst":
            slot = out.get("burst_packet_slot", pd.Series(0, index=out.index))
            slot_id = pd.to_numeric(slot, errors="raise").astype("Int64").astype(str)
            out["source_sample_id"] = "burst:" + out["source_sample_id"] + ":" + slot_id
        if out["source_sample_id"].duplicated().any():
            raise ValueError("Native source-sample identities must be unique")
    transferred = out["calibration_mode_transfer"]
    out.loc[transferred, "calibration_basis"] = "explicit real-time coefficient transfer to burst; native-frequency accuracy unvalidated"
    out.attrs.update({"sample_rate_hz": sample_rate_hz, "resampled": False,
                      "high_frequency_validated": False, "allow_realtime_transfer": allow_realtime_transfer})
    return mag_calibration_diagnostics(out, coefficients, allow_realtime_transfer=allow_realtime_transfer)


def calibrate_mag_9hz(samples, coefficients, *, offsets=None, max_offset_distance_s=60.0):
    """Preserve real-time 9 Hz samples with mode- and hardware-scoped calibration."""
    return _calibrate_mag_native(samples, coefficients, mode="realtime", sample_rate_hz=9.0,
                                 offsets=offsets, max_offset_distance_s=max_offset_distance_s)


def calibrate_mag_18hz(samples, coefficients, *, offsets=None, max_offset_distance_s=60.0,
                       allow_realtime_transfer=False):
    """Preserve burst 18 Hz samples and collection timing without resampling.

    Legacy models are real-time only. Burst models require mode='burst'; an
    explicit allow_realtime_transfer=True permits accepted real-time gains
    with transfer provenance and no claim of native-frequency validation.
    Offset tables must be fitted to the same burst, mode and hardware cohort.
    """
    return _calibrate_mag_native(samples, coefficients, mode="burst", sample_rate_hz=18.0,
                                 offsets=offsets, max_offset_distance_s=max_offset_distance_s,
                                 allow_realtime_transfer=allow_realtime_transfer)
