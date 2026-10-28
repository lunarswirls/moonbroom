"""Reusable preparation and cohort selection for LP mission calibration studies"""

from pathlib import Path

import numpy as np
import pandas as pd

from .archive_calibration import apply_er_archive_response, er_3d_reference, match_er_archive, mag_archive_windows
from .calibration import calibrate_er_telemetry
from .io import read_pds4
from .level0 import decode_level0
from .native_mag import fit_mag_spin_offsets, correct_mag_spin_offsets
from .pad_calibration import apply_pad_normalization, calibrate_er_pad
from .timing import reconstruct_spin_phase
from .archive_quality import read_sunpulse_products
from .mag_state import mag_model_modes, validate_mag_model_scopes, annotate_mag_hardware_state
from .mager import decode_subcom_frames


def prepare_mag_interval(paths, sunpulse_paths, *, kernels, receiver, offset_window_s=60.0,
                          overlap_policy="exact", sunpulse_strict=True):
    """Prepare native MAG and local offsets without requiring an ER reference"""
    products = decode_level0(paths, timing_kernels=kernels, receiver=receiver, mag_only=True,
                             overlap_policy=overlap_policy)
    pulses = read_sunpulse_products(sunpulse_paths, frames=products['frames'], strict=sunpulse_strict)
    raw = reconstruct_spin_phase(annotate_mag_hardware_state(products["mag"],
                                 decode_subcom_frames(products["frames"])), pulses)
    offsets = fit_mag_spin_offsets(raw, window_s=offset_window_s)
    corrected = correct_mag_spin_offsets(raw, offsets, max_distance_s=offset_window_s)
    timing = pd.DataFrame({"frames": [len(products["frames"])],
                           "rejected_ert_anchors": [products["frames"]["ert_anchor_rejected"].sum()],
                           "raw_rows": [len(raw)]})
    result = {"mag_raw": raw, "mag": corrected, "offsets": offsets, "timing": timing,
              "sunpulse": pulses}
    if "overlap_records" in products:
        result["overlap_records"] = products["overlap_records"]
    return result


def prepare_archive_interval(paths, sunpulse_paths, reference_path, *, kernels,
                              receiver, reference_source, offset_window_s=60.0,
                              telemetry_version=None):
    """Decode an interval and prepare native MAG, ER pairs and PAD geometry inputs"""
    products = decode_level0(paths, timing_kernels=kernels, receiver=receiver,
                             telemetry_version=telemetry_version)
    pulses = read_sunpulse_products(sunpulse_paths, frames=products['frames'])
    raw = reconstruct_spin_phase(annotate_mag_hardware_state(products["mag"], products["digital_subcom"],
                                 burst_markers=products["burst_markers"]), pulses)
    offsets = fit_mag_spin_offsets(raw, window_s=offset_window_s)
    corrected = correct_mag_spin_offsets(raw, offsets, max_distance_s=offset_window_s)
    summary = products["er_accumulations"]
    context = summary[["accumulation_id", "telemetry_version", "pitch_map_version"]].copy()
    counts = products["er_counts"].loc[products["er_counts"]["mode"].eq("realtime")].copy()
    counts = counts.merge(context, on="accumulation_id", validate="many_to_one")
    reference = er_3d_reference(read_pds4(reference_path), source=reference_source, drop_invalid=True)
    pairs = match_er_archive(counts, reference)
    windows = mag_archive_windows(corrected, pairs)
    timing = pd.DataFrame({"frames": [len(products["frames"])],
                           "rejected_ert_anchors": [products["frames"]["ert_anchor_rejected"].sum()],
                           "raw_rows": [len(raw)], "matched_distributions": [pairs.loc[pairs["match_valid"], "accumulation_id"].nunique()],
                           "excluded_reference_spectra": [reference.attrs["excluded_reference_spectra"]]})
    return {"mag_raw": raw, "mag": corrected, "offsets": offsets, "pairs": pairs,
            "windows": windows, "pad": counts.loc[counts["kind"].eq("pad")].copy(),
            "angles": products["er_angles"].merge(summary[["accumulation_id", "mode"]],
                                                   on="accumulation_id", validate="many_to_one"),
            "accumulations": summary,
            "digital_subcom": products["digital_subcom"], "timing": timing}


def save_interval_cache(tables, directory):
    """Cache locally produced tables without executing any downloaded content"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_pickle(directory / (name + ".pkl.gz"), compression="gzip")


def load_interval_cache(directory, *, names=None):
    """Read trusted locally produced interval caches, optionally selecting tables"""
    directory = Path(directory)
    if names is None:
        names = [path.name.removesuffix(".pkl.gz") for path in directory.glob("*.pkl.gz")]
    return {name: pd.read_pickle(directory / (name + ".pkl.gz"), compression="gzip") for name in names}


def select_mag_models(bank, utc, *, mode="realtime", hardware_state=None, require_validation=True):
    """Select a nonoverlapping cohort for one processing date, preserving unsupported ranges"""
    time = pd.Timestamp(utc)
    time = time.tz_localize("UTC") if time.tzinfo is None else time.tz_convert("UTC")
    start = pd.to_datetime(bank["valid_from"], utc=True, format="mixed")
    stop = pd.to_datetime(bank["valid_to"], utc=True, format="mixed")
    if mode not in ("realtime", "burst"):
        raise ValueError("MAG model mode must be realtime or burst")
    selected = bank.loc[start.le(time) & stop.gt(time) & mag_model_modes(bank).eq(mode)].copy()
    if require_validation:
        if "validation_pass" not in selected:
            raise ValueError("A mission bank requires independent validation flags")
        selected = selected.loc[selected["validation_pass"]]
    if hardware_state is not None:
        from .mag_state import MAG_SCOPE_FIELDS
        for field in MAG_SCOPE_FIELDS:
            if field in selected:
                selected = selected.loc[selected[field].isna() | selected[field].eq(hardware_state.get(field))]
    validate_mag_model_scopes(selected)
    return selected


def apply_er_mission_response(counts, bank):
    """Apply independently accepted ER cohorts across dates without extrapolation

    Rows must match mode, kind, energy, angular bin, software version and
    the entire collection interval. Overlapping accepted models are errors.
    Unmatched counts remain present with missing flux.
    """
    if "validation_pass" not in bank:
        raise ValueError("A mission bank requires independent validation flags")
    usable = bank.loc[bank["validation_pass"] & bank["fit_valid"]].copy()
    keys = ["mode", "kind", "energy_step", "angle_bin", "telemetry_version"]
    usable["valid_from"] = pd.to_datetime(usable["valid_from"], utc=True, format="mixed")
    usable["valid_to"] = pd.to_datetime(usable["valid_to"], utc=True, format="mixed")
    if (usable[["valid_from", "valid_to"]].isna().any().any()
            or usable["valid_from"].ge(usable["valid_to"]).any()):
        raise ValueError("ER mission scopes require ordered, finite UTC bounds")
    for _, group in usable.groupby(keys):
        group = group.sort_values("valid_from")
        if (group["valid_from"].iloc[1:].to_numpy() < group["valid_to"].iloc[:-1].to_numpy()).any():
            raise ValueError("Mission model validity overlaps for an ER response key")
    original_index = counts.index
    raw = counts.reset_index(drop=True)
    out = calibrate_er_telemetry(raw)
    out["empirical_response_supported"] = False
    out["response_basis"] = "no independently accepted archive-derived response"
    starts = pd.to_datetime(raw["start_utc"], utc=True, format="mixed")
    stops = pd.to_datetime(raw["stop_utc"], utc=True, format="mixed")
    for (start, stop), coefficients in usable.groupby(["valid_from", "valid_to"]):
        candidates = raw.loc[starts.ge(start) & stops.le(stop)].copy()
        candidates["mission_row"] = candidates.index
        selected = candidates.merge(coefficients[keys].drop_duplicates(), on=keys, validate="many_to_one")
        if selected.empty:
            continue
        positions = selected.pop("mission_row").to_numpy(int)
        calibrated = apply_er_archive_response(selected.reset_index(drop=True), coefficients)
        metadata = coefficients[keys].copy()
        metadata['response_valid_from'] = start.isoformat()
        metadata['response_valid_to'] = stop.isoformat()
        metadata['response_cohort'] = coefficients['cohort'].to_numpy() if 'cohort' in coefficients else start.isoformat()
        calibrated = calibrated.merge(metadata, on=keys, how='left', validate='many_to_one')
        for column in calibrated:
            if column not in out:
                dtype = calibrated[column].dtype
                out[column] = (False if dtype == bool else
                               pd.Series(None, index=out.index, dtype=object)
                               if pd.api.types.is_string_dtype(dtype) or dtype.kind in 'OUS' else np.nan)
            out.loc[positions, column] = calibrated[column].to_numpy()
    out.index = original_index
    out.attrs["calibration_kind"] = "independently validated archive-derived mission response"
    return out


def apply_pad_mission_response(counts, directions, er_bank, pad_bank):
    """Apply validated PAD aggregate models with their matching 3D response cohort

    Individual pitch-bin responses retain the approximate geometry label.
    Both banks must preserve the cohort identity used during validation.
    """
    if "validation_pass" not in pad_bank:
        raise ValueError("A PAD mission bank requires independent validation flags")
    usable = pad_bank.loc[pad_bank["validation_pass"]].copy()
    raw = counts.reset_index(drop=True)
    out = calibrate_er_telemetry(raw)
    out["pad_normalization_supported"] = False
    out["omnidirectional_model_validated"] = False
    out["pitch_bin_response_validated"] = False
    assigned = np.zeros(len(raw), dtype=bool)
    starts = pd.to_datetime(raw["start_utc"], utc=True, format="mixed")
    stops = pd.to_datetime(raw["stop_utc"], utc=True, format="mixed")
    usable["valid_from"] = pd.to_datetime(usable["valid_from"], utc=True, format="mixed")
    usable["valid_to"] = pd.to_datetime(usable["valid_to"], utc=True, format="mixed")
    if (usable[["valid_from", "valid_to"]].isna().any().any()
            or usable["valid_from"].ge(usable["valid_to"]).any()):
        raise ValueError("PAD mission scopes require ordered, finite UTC bounds")
    for (cohort, start, stop), models in usable.groupby(["cohort", "valid_from", "valid_to"]):
        selected = raw.loc[starts.ge(start) & stops.le(stop)].copy()
        selected["mission_row"] = selected.index
        keys = ["telemetry_version", "energy_step"]
        if models[keys].duplicated().any():
            raise ValueError("Mission model validity overlaps for a PAD response key")
        selected = selected.merge(models[keys], on=keys, validate="many_to_one")
        if selected.empty:
            continue
        positions = selected.pop("mission_row").to_numpy(int)
        if assigned[positions].any():
            raise ValueError("Mission model validity overlaps for a PAD response key")
        coefficients = er_bank.loc[er_bank["cohort"].eq(cohort)]
        calibrated = calibrate_er_pad(selected.reset_index(drop=True), directions, coefficients)
        calibrated = apply_pad_normalization(calibrated, models)
        assigned[positions] = True
        for column in calibrated:
            if column not in out:
                dtype = calibrated[column].dtype
                out[column] = (False if dtype == bool else
                               pd.Series(None, index=out.index, dtype=object)
                               if pd.api.types.is_string_dtype(dtype) or dtype.kind in "OUS" else np.nan)
            out.loc[positions, column] = calibrated[column].to_numpy()
    out.index = counts.index
    out.attrs["pitch_bin_response_validated"] = False
    return out
