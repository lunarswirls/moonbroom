"""Approximate ER PAD exposure geometry and independent omnidirectional validation"""

from functools import lru_cache

import numpy as np
import pandas as pd

from .archive_calibration import _utc
from .calibration import calibrate_er_telemetry
from .telemetry import er_angular_bins


@lru_cache(maxsize=1)
def _pitch_geometry():
    """Phase-averaged fractions for a uniform circular detector and default PAD map"""
    phase = (np.arange(128) + .5)*2*np.pi/128
    anode = (np.arange(256) + .5)*2*np.pi/256
    fractions = []
    for theta in np.arange(181.):
        angle = np.deg2rad(theta)
        dot = np.sin(angle)*np.cos(phase[:, None])*np.sin(anode)[None, :] + np.cos(angle)*np.cos(anode)[None, :]
        pitch = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        values = np.histogram(pitch, bins=np.linspace(0, 180, 17))[0]
        fractions.append(values/values.sum())
    return np.asarray(fractions)


def pad_field_directions(angles, mag):
    """Use downlinked polar angles or reconstruct earlier-version directions

    Versions 1/2 use the nearest parity-compatible raw MAG sample. This is
    approximate because the flight processor's internal correction arithmetic
    is unavailable. Later software carries explicit polar angles. Direction
    provenance is preserved and no azimuthal precision is invented.
    """
    out = angles.loc[np.isfinite(angles["clock_s"])].copy()
    if "mode" in out:
        out = out.loc[out["mode"].eq("realtime")]
    out = out.sort_values("clock_s").drop_duplicates("clock_s")
    if "theta_deg" not in out:
        out["theta_deg"] = np.nan
    raw = mag.loc[mag["mode"].eq("realtime")].sort_values("clock_s")
    observed = np.isfinite(out["theta_deg"]) & out["theta_deg"].between(0, 180)
    if raw.empty:
        out["theta_reconstruction_error_deg"] = np.nan
        out["direction_source"] = np.where(observed, "downlinked polar angle", "unavailable")
        out["direction_valid"] = observed
        return out
    clocks = raw["clock_s"].to_numpy(float)
    values = out["clock_s"].to_numpy(float)
    position = np.searchsorted(clocks, values)
    candidates = np.clip(position[:, None] + np.arange(-3, 4), 0, len(raw) - 1)
    distance = np.abs(clocks[candidates] - values[:, None])
    if "sample_parity" in out:
        parity = out["sample_parity"].to_numpy(float)
        compatible = raw["sample_index"].to_numpy()[candidates] % 2 == parity[:, None]
        distance = np.where(compatible, distance, np.inf)
    chosen = candidates[np.arange(len(out)), np.argmin(distance, axis=1)]
    field = raw[["bx_signed", "by_signed", "bz_signed"]].to_numpy(float)[chosen]
    norm = np.linalg.norm(field, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        reconstructed = np.degrees(np.arccos(np.clip(field[:, 2]/norm, -1, 1)))
    valid = np.min(distance, axis=1) <= .25
    out["theta_reconstruction_error_deg"] = np.where(observed & valid, reconstructed - out["theta_deg"], np.nan)
    out["direction_source"] = np.where(observed, "downlinked polar angle", "parity-selected raw MAG approximation")
    out["theta_deg"] = np.where(observed, out["theta_deg"], np.where(valid, reconstructed, np.nan))
    out["direction_valid"] = np.isfinite(out["theta_deg"]) & out["theta_deg"].between(0, 180)
    return out


def calibrate_er_pad(counts, directions, coefficients, *, min_anodes=8, max_theta_std_deg=10.):
    """Calibrate PAD using pooled recovered anode response and phase-averaged geometry

    The model assumes uniform response within/between anodes after pooling
    the recovered 3D factors, the default 11.25-degree pitch map, an energy
    duty fraction of 1/16 and slowly varying polar direction. It is an
    approximate angular response model, not recovered onboard arithmetic.
    Individual pitch-bin validation remains false without resolved references.
    """
    out = counts.copy()
    groups = out.groupby(["accumulation_id", "energy_step", "sample"], sort=False).agg(
        start=("start_clock_s", "first"), stop=("stop_clock_s", "first"))
    d = directions.loc[directions["direction_valid"] & np.isfinite(directions["clock_s"])].sort_values("clock_s")
    clocks = d["clock_s"].to_numpy(float)
    theta = d["theta_deg"].to_numpy(float)
    left = np.searchsorted(clocks, groups["start"])
    right = np.searchsorted(clocks, groups["stop"])
    number = right - left
    sums = np.r_[0, np.cumsum(theta)]
    squares = np.r_[0, np.cumsum(theta**2)]
    mean = (sums[right] - sums[left])/np.maximum(number, 1)
    spread = np.sqrt(np.maximum((squares[right] - squares[left])/np.maximum(number, 1) - mean**2, 0))
    spacing = np.median(np.diff(clocks)) if len(clocks) > 1 else np.nan
    required = (groups["stop"] - groups["start"])/spacing if np.isfinite(spacing) and spacing > 0 else np.inf
    groups["pad_theta_deg"], groups["pad_theta_std_deg"] = mean, spread
    groups["pad_direction_valid"] = (number >= np.maximum(required*.85, 1)) & (spread <= max_theta_std_deg)
    out = out.merge(groups[["pad_theta_deg", "pad_theta_std_deg", "pad_direction_valid"]].reset_index(),
                     on=["accumulation_id", "energy_step", "sample"], how="left", validate="many_to_one")
    source = coefficients.loc[coefficients["fit_valid"] & coefficients["kind"].eq("3d")].copy()
    bins = er_angular_bins().set_index("angle_bin")
    source["anode_bin"] = source["angle_bin"].map(bins["anode_bin"])
    keys = ["mode", "telemetry_version", "energy_step"]
    pooled = source.groupby(keys).agg(anode_response=("geometric_factor_cm2_sr", "median"),
                                      n_anodes=("anode_bin", "nunique"), response_energy_ev=("energy_ev", "median"),
                                      valid_from=("valid_from", "first"), valid_to=("valid_to", "first")).reset_index()
    out = out.merge(pooled, on=keys, how="left", validate="many_to_one")
    fraction = _pitch_geometry()
    polar = out["pad_theta_deg"].to_numpy(float)
    pitch = out["angle_bin"].to_numpy(int)
    lower = np.clip(np.floor(np.nan_to_num(polar)).astype(int), 0, 179)
    weight = np.clip(polar - lower, 0, 1)
    occupancy = fraction[lower, pitch]*(1 - weight) + fraction[lower + 1, pitch]*weight
    factor = out["anode_response"].to_numpy(float)*16*occupancy
    valid = (out["pad_direction_valid"] & out["pitch_map_version"].eq(0) & out["n_anodes"].ge(min_anodes)
             & np.isclose(out["energy_ev"], out["response_energy_ev"], atol=.01, rtol=0)
             & _utc(out["start_utc"]).ge(_utc(out["valid_from"]))
             & _utc(out["stop_utc"]).le(_utc(out["valid_to"])) & np.isfinite(factor) & (factor > 0))
    response_keys = ["accumulation_id", "kind", "energy_step", "sample", "angle_bin"]
    response = out[response_keys].copy()
    response["exposure_s"] = (out["stop_clock_s"] - out["start_clock_s"])/16
    response["geometric_factor_cm2_sr"] = factor
    response = response.loc[valid].copy()
    response.attrs["source"] = "recovered 3D response with pooled-anode, phase-averaged pitch geometry"
    calibrated = calibrate_er_telemetry(out, response=response)
    calibrated["response_basis"] = "approximate PAD geometry from recovered 3D anode response"
    calibrated["pitch_bin_response_validated"] = False
    calibrated["pad_geometry_supported"] = valid.to_numpy()
    calibrated.attrs["angular_assumption"] = "uniform circular anodes, phase average, default pitch map"
    return calibrated


def pad_omnidirectional(pad):
    """Integrate complete calibrated PADs with spherical pitch-angle weights"""
    out = pad.copy()
    edge = np.linspace(0, np.pi, 17)
    weights = (np.cos(edge[:-1]) - np.cos(edge[1:]))/2
    out["solid_angle_weight"] = weights[out["angle_bin"].to_numpy(int)]
    out["weighted_flux"] = out["flux_estimate"]*out["solid_angle_weight"]
    out["weighted_lower"] = out["flux_lower"]*out["solid_angle_weight"]
    out["weighted_upper"] = out["flux_upper"]*out["solid_angle_weight"]
    keys = ["accumulation_id", "telemetry_version", "energy_step", "sample"]
    result = out.groupby(keys).agg(utc_start=("start_utc", "first"), utc_stop=("stop_utc", "first"),
                                    flux=("weighted_flux", "sum"), flux_lower=("weighted_lower", "sum"),
                                    flux_upper=("weighted_upper", "sum"), n_valid=("calibration_valid", "sum"),
                                    n_angles=("angle_bin", "nunique"), energy_ev=("energy_ev", "first")).reset_index()
    result["omni_valid"] = result["n_valid"].eq(16) & result["n_angles"].eq(16)
    result.loc[~result["omni_valid"], ["flux", "flux_lower", "flux_upper"]] = np.nan
    result["utc"] = _utc(result["utc_start"] + (result["utc_stop"] - result["utc_start"])/2)
    return result


def er_omni_reference(high, low, *, software_version):
    """Normalize PDS high/low omnidirectional products without inventing PAD arrays"""
    rows = []
    for channel, step in enumerate([6, 9] if software_version == 1 else [9, 11]):
        group = pd.DataFrame({"utc": _utc(high["Time"]), "reference_flux": high["eflux" + str(channel + 1)],
                               "energy_step": step, "reference_window_s": 5.})
        rows.append(group)
    fast = [6, 9] if software_version == 1 else [9, 11]
    for step in range(15):
        if step in fast:
            continue
        rows.append(pd.DataFrame({"utc": _utc(low["Time"]), "reference_flux": low[f"eflux_bin[{step}]"],
                                  "energy_step": step, "reference_window_s": 80.}))
    return pd.concat(rows, ignore_index=True).sort_values("utc").reset_index(drop=True)


def compare_pad_omni(omni, reference, *, time_offset_s=0., min_coverage=.85):
    """Compare integrated PAD with explicit reference-window timing and coverage"""
    rows = []
    for step, group in omni.loc[omni["omni_valid"]].groupby("energy_step"):
        group = group.sort_values("utc_start")
        start = _utc(group["utc_start"]).astype("int64").to_numpy()/1e9
        stop = _utc(group["utc_stop"]).astype("int64").to_numpy()/1e9
        flux = group["flux"].to_numpy(float)
        for row in reference.loc[reference["energy_step"].eq(step)].itertuples():
            if pd.isna(row.utc):
                continue
            center = row.utc.timestamp() + time_offset_s
            left, right = center - row.reference_window_s/2, center + row.reference_window_s/2
            first = np.searchsorted(stop, left, side="right")
            last = np.searchsorted(start, right, side="left")
            overlap = np.maximum(0, np.minimum(stop[first:last], right) - np.maximum(start[first:last], left))
            coverage = overlap.sum()/row.reference_window_s
            if coverage < min_coverage or overlap.sum() <= 0 or not np.isfinite(row.reference_flux) or row.reference_flux <= 0:
                continue
            estimate = np.sum(flux[first:last]*overlap)/overlap.sum()
            rows.append((row.utc, step, row.reference_flux, estimate, coverage, time_offset_s))
    return pd.DataFrame(rows, columns=["utc", "energy_step", "reference_flux", "predicted_flux", "coverage", "time_offset_s"])


def fit_pad_normalization(omni, reference, *, offsets_s=(-42., -2., 38.), min_pairs=20):
    """Fit only aggregate normalization and timing using training-date references"""
    candidates = []
    for offset in offsets_s:
        comparison = compare_pad_omni(omni, reference, time_offset_s=offset)
        for step, group in comparison.groupby("energy_step"):
            ratio = group["reference_flux"]/group["predicted_flux"]
            ratio = ratio.loc[np.isfinite(ratio) & ratio.gt(0)]
            if len(ratio) < min_pairs:
                continue
            scale = ratio.median()
            error = (scale*group["predicted_flux"]/group["reference_flux"] - 1).abs()
            candidates.append((step, scale, offset, len(ratio), error.median(), error.quantile(.9)))
    table = pd.DataFrame(candidates, columns=["energy_step", "flux_scale", "time_offset_s", "n_training", "training_median_error", "training_p90_error"])
    if table.empty:
        return table
    return table.sort_values("training_median_error").drop_duplicates("energy_step").sort_values("energy_step").reset_index(drop=True)


def apply_pad_normalization(pad, normalization, *, require_validation=True):
    """Apply a cohort's aggregate normalization without claiming pitch-bin validation"""
    out = pad.copy()
    out["pad_normalization_supported"] = False
    out["omnidirectional_model_validated"] = False
    scale = np.full(len(out), np.nan)
    table = normalization.copy()
    if require_validation:
        if "validation_pass" not in table:
            raise ValueError("PAD normalization requires independent validation flags")
        table = table.loc[table["validation_pass"]]
    keys = ["energy_step"] + (["telemetry_version"] if "telemetry_version" in table else [])
    if table[keys].duplicated().any():
        raise ValueError("Select one normalization cohort per energy and software version")
    for _, row in table.iterrows():
        valid = out["energy_step"].eq(row["energy_step"])
        if "telemetry_version" in table:
            valid &= out["telemetry_version"].eq(row["telemetry_version"])
        if "valid_from" in table:
            valid &= _utc(out["start_utc"]).ge(pd.Timestamp(row["valid_from"]))
            valid &= _utc(out["stop_utc"]).le(pd.Timestamp(row["valid_to"]))
        if "energy_ev" in table:
            valid &= np.isclose(out["energy_ev"], row["energy_ev"], rtol=0, atol=.01)
        scale[valid.to_numpy()] = row["flux_scale"]
        out.loc[valid, "pad_normalization_supported"] = True
        out.loc[valid, "omnidirectional_model_validated"] = bool(row.get("validation_pass", False))
    for column in ("flux_estimate", "flux_lower", "flux_upper"):
        out[column] *= scale
    out["calibration_valid"] &= np.isfinite(scale) & (scale > 0)
    out["pitch_bin_response_validated"] = False
    out["response_basis"] = "approximate PAD geometry with PDS omnidirectional normalization"
    return out
