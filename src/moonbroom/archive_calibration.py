"""Empirical calibration transfer from paired raw and calibrated PDS products"""

import numpy as np
import pandas as pd

from .calibration import calibrate_er_telemetry, calibrate_mag_telemetry, despin_mag


def _utc(values):
    return pd.to_datetime(values, utc=True, format="mixed").astype("datetime64[ns, UTC]")


def er_3d_reference(table, *, source, drop_invalid=False):
    """Normalize a labelled ER 3D table using native spectrum time, not row time

    table is the wide output of read_pds4. The archive's row Time advances
    artificially between energies; Spectrum Time identifies the distribution.
    Field vectors are repeated to allow a separate MAG reference extraction.
    """
    if not source:
        raise ValueError("Identify the calibrated PDS reference product")
    out = table.copy()
    out["reference_id"] = source + ":" + out["Spectrum Number"].astype(str)
    dropped = 0
    if drop_invalid:
        check = out.groupby("reference_id").agg(rows=("Energy Bin", "size"), energies=("Energy Bin", "nunique"),
                                                  times=("Spectrum Time", "nunique"), time=("Spectrum Time", "min"),
                                                  lowest=("Energy Bin", "min"), highest=("Energy Bin", "max"))
        valid = (check["rows"].eq(15) & check["energies"].eq(15) & check["times"].eq(1)
                 & check["time"].gt(0) & np.isfinite(check["time"])
                 & check["lowest"].gt(0) & np.isfinite(check["highest"]))
        dropped = int((~valid).sum())
        out = out.loc[out["reference_id"].isin(check.index[valid])].copy()
    if not np.isfinite(out["Energy Bin"].to_numpy(float)).all() or out["Energy Bin"].le(0).any():
        raise ValueError("Reference center energies must be positive and finite")
    if out.groupby("reference_id")["Spectrum Time"].nunique().gt(1).any():
        raise ValueError("A reference spectrum must have one native time")
    out["energy_step"] = out.groupby("reference_id")["Energy Bin"].rank(ascending=False, method="first") - 1
    if not out.groupby("reference_id").size().eq(15).all():
        raise ValueError("Each ER reference spectrum must contain fifteen energies")
    if not out.groupby("reference_id")["Energy Bin"].nunique().eq(15).all():
        raise ValueError("Each ER reference spectrum must have fifteen distinct center energies")
    out["reference_utc"] = pd.to_datetime(out["Spectrum Time"].where(out["Spectrum Time"].gt(0)), unit="s", utc=True)
    size = len(out)
    result = pd.DataFrame({
        "reference_id": np.repeat(out["reference_id"].to_numpy(), 88),
        "reference_utc": np.repeat(out["reference_utc"].to_numpy(), 88),
        "energy_step": np.repeat(out["energy_step"].to_numpy(int), 88),
        "angle_bin": np.tile(np.arange(88), size),
        "reference_energy_ev": np.repeat(out["Energy Bin"].to_numpy(float), 88),
        "reference_flux": out[[f"Electron Flux[{index}]" for index in range(88)]].to_numpy(float).ravel(),
    })
    for axis in "xyz":
        result["reference_b" + axis] = np.repeat(out["B" + axis].to_numpy(float), 88)
    result["reference_utc"] = _utc(result["reference_utc"])
    result.attrs["source"] = source
    result.attrs["excluded_reference_spectra"] = dropped
    return result


def match_er_archive(counts, reference, *, tolerance_s=10.0, min_zero_agreement=0.9,
                     min_count_correlation=0.95):
    """Pair complete raw 3D distributions and unique nearest PDS spectra

    Timing proposes a match; zero/nonzero count patterns independently check
    its content. Counter rollover, decoding and absolute UTC must be handled
    upstream. Ambiguous reuse of a reference spectrum raises. Unmatched rows
    retain missing reference values. A pairing is not a clock correction.
    """
    if (not np.isfinite(tolerance_s) or tolerance_s <= 0 or not 0 <= min_zero_agreement <= 1
            or not 0 <= min_count_correlation <= 1):
        raise ValueError("Use a positive time tolerance and agreement in [0, 1]")
    raw = counts.loc[counts["kind"].eq("3d") & counts["complete"]].copy()
    keys = ["accumulation_id", "energy_step", "angle_bin"]
    if raw[keys].duplicated().any():
        raise ValueError("Raw accumulation keys must be unique")
    refkeys = ["reference_id", "energy_step", "angle_bin"]
    if reference[refkeys].duplicated().any():
        raise ValueError("Reference spectrum keys must be unique")
    times = raw.groupby("accumulation_id").agg(start_utc=("start_utc", "min"), stop_utc=("stop_utc", "max"))
    times["match_utc"] = _utc(times["start_utc"] + (times["stop_utc"] - times["start_utc"])/2)
    times = times.dropna(subset=["match_utc"]).reset_index().sort_values("match_utc")
    refs = reference[["reference_id", "reference_utc"]].drop_duplicates()
    refs = refs.dropna(subset=["reference_utc"])
    if refs["reference_id"].duplicated().any() or refs["reference_utc"].duplicated().any():
        raise ValueError("Reference spectra require unique identities and times")
    refs["reference_utc"] = _utc(refs["reference_utc"])
    refs = refs.dropna(subset=["reference_utc"]).sort_values("reference_utc")
    matched = pd.merge_asof(times, refs, left_on="match_utc", right_on="reference_utc",
                            tolerance=pd.Timedelta(seconds=tolerance_s), direction="nearest")
    if matched["reference_id"].dropna().duplicated().any():
        raise ValueError("Two raw accumulations matched the same reference spectrum")
    out = raw.merge(matched[["accumulation_id", "reference_id", "match_utc"]], on="accumulation_id", how="left")
    out = out.merge(reference, on=["reference_id", "energy_step", "angle_bin"], how="left", validate="many_to_one")
    comparable = (np.isfinite(out["counts_lower"]) & np.isfinite(out["reference_flux"])
                  & out["reference_flux"].ge(0))
    agreement = out.loc[comparable, "counts_lower"].eq(0).eq(out.loc[comparable, "reference_flux"].eq(0))
    check = pd.DataFrame({"accumulation_id": out.loc[comparable, "accumulation_id"], "agreement": agreement})
    check = check.groupby("accumulation_id")["agreement"].agg(["mean", "size"])
    out["zero_pattern_agreement"] = out["accumulation_id"].map(check["mean"])
    out["match_valid"] = (out["reference_id"].notna() & out["zero_pattern_agreement"].ge(min_zero_agreement)
                           & out["accumulation_id"].map(check["size"]).ge(88))
    out["match_offset_s"] = (out["reference_utc"] - out["match_utc"]).dt.total_seconds()
    out["energy_ev"] = out["reference_energy_ev"]
    out["energy_valid"] = out["match_valid"] & np.isfinite(out["energy_ev"]) & out["energy_ev"].gt(0)
    correlations = {}
    out["comparison_rate"] = out["counts_lower"] / out["nominal_integration_s"] / out["energy_ev"]
    for accumulation, group in out.loc[comparable].groupby("accumulation_id"):
        ranks = group[["comparison_rate", "reference_flux"]].replace([np.inf, -np.inf], np.nan).dropna().rank()
        correlations[accumulation] = ranks.corr().iloc[0, 1] if len(ranks) > 2 else np.nan
    out["count_pattern_correlation"] = out["accumulation_id"].map(correlations)
    out["match_valid"] &= out["count_pattern_correlation"].ge(min_count_correlation)
    out["energy_valid"] &= out["match_valid"]
    out = out.drop(columns="comparison_rate")
    out.attrs.update({"reference_source": reference.attrs.get("source"), "time_tolerance_s": tolerance_s,
                      "matching": "unique nearest spectrum with zero-pattern consistency"})
    return out


def mag_sse_reference_to_scd(reference, attitude, kernels, *, attitude_frame):
    """Rotate archived SSE MAG vectors to SCD using explicit spin-axis geometry

    attitude requires utc, longitude_deg and latitude_deg in the explicitly
    named inertial frame. Values apply from each row's UTC until changed.
    Archived attitude is an orbit-average approximation, not a full attitude
    kernel; record the frame choice when resolving inconsistent label prose.
    """
    import spiceypy as spice
    from .geometry import loaded_kernels

    out = reference.copy()
    out["utc"] = _utc(out["utc"])
    orientation = attitude.copy()
    orientation["utc"] = _utc(orientation["utc"])
    if (orientation["utc"].isna().any() or orientation["utc"].duplicated().any()
            or not np.isfinite(orientation[["longitude_deg", "latitude_deg"]].to_numpy()).all()):
        raise ValueError("Attitude requires unique UTC and finite spin-axis angles")
    out = pd.merge_asof(out.sort_values("utc"), orientation.sort_values("utc"), on="utc", direction="backward")
    longitude = np.deg2rad(out["longitude_deg"].to_numpy(float))
    latitude = np.deg2rad(out["latitude_deg"].to_numpy(float))
    z = np.column_stack([np.cos(longitude)*np.cos(latitude), np.sin(longitude)*np.cos(latitude), np.sin(latitude)])
    with loaded_kernels(kernels):
        epochs = np.array([spice.str2et(value.strftime("%Y-%m-%dT%H:%M:%S.%f")) for value in out["utc"]])
        if attitude_frame != "ECLIPJ2000":
            z = np.array([spice.pxform(attitude_frame, "ECLIPJ2000", epoch)@axis for epoch, axis in zip(epochs, z)])
        sun = np.asarray(spice.spkpos("SUN", epochs, "ECLIPJ2000", "NONE", "MOON")[0]).reshape(-1, 3)
    sun /= np.linalg.norm(sun, axis=1)[:, None]
    y = np.cross(np.tile([0., 0., 1.], (len(out), 1)), sun)
    y /= np.linalg.norm(y, axis=1)[:, None]
    north = np.cross(sun, y)
    field = (out["Bx_SSE"].to_numpy()[:, None]*sun + out["By_SSE"].to_numpy()[:, None]*y
             + out["Bz_SSE"].to_numpy()[:, None]*north)
    x = sun - np.sum(sun*z, axis=1)[:, None]*z
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(z, x)
    for name, axis in zip("xyz", (x, y, z)):
        out["reference_b" + name] = np.sum(field*axis, axis=1)
    out.attrs.update({"attitude_frame": attitude_frame, "rotation": "orbit-average spin axis and Sun projection"})
    return out


def mag_fixed_reference_windows(samples, reference, *, time_offset_s=0.0, window_s=5.0,
                                min_coverage=0.9):
    """Build fixed MAG reference windows with an explicit empirical time alignment

    Positive time_offset_s shifts the raw averaging window later than the
    published reference center. It never changes reconstructed sample UTC.
    Reference field vectors must already be in SCD coordinates.
    """
    if not np.isfinite(time_offset_s) or not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("Use a finite alignment offset and positive window width")
    raw = samples.loc[samples["mode"].eq("realtime") & samples["utc"].notna()].sort_values("clock_s")
    times = _utc(raw["utc"]).astype("int64").to_numpy()/1e9
    if (np.diff(times) <= 0).any():
        raise ValueError("UTC must be strictly increasing for reference-window mapping")
    windows = reference.copy()
    windows["start_utc"] = _utc(windows["utc"]) + pd.to_timedelta(time_offset_s - window_s/2, unit="s")
    windows["stop_utc"] = _utc(windows["utc"]) + pd.to_timedelta(time_offset_s + window_s/2, unit="s")
    for edge in ("start", "stop"):
        values = _utc(windows[edge + "_utc"]).astype("int64").to_numpy()/1e9
        windows[edge + "_clock_s"] = np.interp(values, times, raw["clock_s"], left=np.nan, right=np.nan)
    windows = windows.loc[windows["start_clock_s"].notna() & windows["stop_clock_s"].notna()].copy()
    windows["accumulation_id"] = np.arange(len(windows))
    windows["reference_id"] = _utc(windows["utc"]).astype(str)
    windows["match_valid"] = True
    result = mag_archive_windows(samples, windows, min_coverage=min_coverage)
    result["reference_time_offset_s"] = time_offset_s
    return result


def fit_er_archive_response(pairs, *, valid_from, valid_to, source, min_pairs=8,
                            dead_time_s=3e-7, max_deadtime_factor=1.25):
    """Estimate effective per-energy/angular response from independent raw/flux pairs

    Only positive unsaturated counts and flux constrain response. The count
    quantization interval propagates to lower/upper response observations.
    These factors reproduce the reference pipeline under the supplied exposure
    and deadtime assumptions; they are not independently measured efficiencies.
    """
    start, stop = _utc(pd.Series([valid_from, valid_to]))
    if start >= stop or not source or min_pairs < 2:
        raise ValueError("Supply a valid scope, reference source and at least two pairs")
    if pairs[["mode", "accumulation_id", "kind", "energy_step", "sample", "angle_bin"]].duplicated().any():
        raise ValueError("Training pairs must identify independent raw bins without duplicates")
    rates = calibrate_er_telemetry(pairs, dead_time_s=dead_time_s, max_deadtime_factor=max_deadtime_factor)
    valid = (rates["match_valid"] & rates["count_rate_valid"] & rates["counts_lower"].gt(0)
             & np.isfinite(rates["reference_flux"]) & rates["reference_flux"].gt(0)
             & _utc(rates["start_utc"]).ge(start) & _utc(rates["stop_utc"]).le(stop))
    selected = rates.loc[valid].copy()
    if selected.empty:
        raise ValueError("No positive, matched, unsaturated ER observations constrain response")
    for name in ("lower", "upper", "estimate"):
        selected["response_" + name] = selected["count_rate_" + name] / selected["reference_flux"] / selected["energy_ev"]
    keys = ["mode", "kind", "energy_step", "angle_bin"]
    if "telemetry_version" in selected:
        keys.append("telemetry_version")
    rows = []
    for values, group in selected.groupby(keys):
        median = group["response_estimate"].median()
        row = dict(zip(keys, values))
        row.update({"geometric_factor_cm2_sr": median,
                    "response_lower_median": group["response_lower"].median(),
                    "response_upper_median": group["response_upper"].median(),
                    "response_mad": (group["response_estimate"] - median).abs().median(),
                    "energy_ev": group["energy_ev"].median(), "n_pairs": len(group),
                    "n_accumulations": group["accumulation_id"].nunique(),
                    "valid_from": start, "valid_to": stop, "source": source,
                    "training_start": group["start_utc"].min(), "training_stop": group["stop_utc"].max(),
                    "dead_time_s": dead_time_s, "max_deadtime_factor": max_deadtime_factor,
                    "calibration_kind": "archive-derived effective response",
                    "fit_valid": len(group) >= min_pairs and np.isfinite(median) and median > 0})
        if group["energy_ev"].max() - group["energy_ev"].min() > 0.01:
            raise ValueError("Split training data at an ER energy-sweep change")
        rows.append(row)
    return pd.DataFrame(rows)


def apply_er_archive_response(counts, coefficients):
    """Apply derived response only to supported bins, energies, modes and time scope"""
    if coefficients.empty:
        raise ValueError("No derived ER response coefficients supplied")
    keys = ["mode", "kind", "energy_step", "angle_bin"]
    if "telemetry_version" in coefficients:
        keys.append("telemetry_version")
    usable = coefficients.loc[coefficients["fit_valid"]].copy()
    if "validation_pass" in usable:
        usable = usable.loc[usable["validation_pass"]]
    if usable[keys].duplicated().any():
        raise ValueError("Use one response cohort per key; split time-dependent models before applying")
    if coefficients[["dead_time_s", "max_deadtime_factor"]].drop_duplicates().shape[0] != 1:
        raise ValueError("All derived response rows must use the same deadtime convention")
    out = counts.copy()
    out["empirical_row"] = np.arange(len(out))
    renamed = usable.rename(columns={name: "derived_" + name for name in usable if name not in keys})
    joined = out.merge(renamed, on=keys, how="left", sort=False,
                        validate="many_to_one").sort_values("empirical_row")
    energy = joined["derived_energy_ev"]
    valid = (np.isclose(out["energy_ev"].to_numpy(), energy.to_numpy(), rtol=0, atol=0.01)
             & _utc(joined["start_utc"]).ge(_utc(joined["derived_valid_from"]))
             & _utc(joined["stop_utc"]).le(_utc(joined["derived_valid_to"])))
    response = out[["accumulation_id", "kind", "energy_step", "sample", "angle_bin"]].copy()
    response["exposure_s"] = out["nominal_integration_s"].to_numpy()
    response["geometric_factor_cm2_sr"] = joined["derived_geometric_factor_cm2_sr"].to_numpy()
    response = response.loc[np.asarray(valid) & response["exposure_s"].gt(0).to_numpy()]
    response.attrs["source"] = "; ".join(usable["source"].dropna().unique())
    convention = coefficients.iloc[0]
    result = calibrate_er_telemetry(out.drop(columns="empirical_row"), response=response,
                                    dead_time_s=convention["dead_time_s"],
                                    max_deadtime_factor=convention["max_deadtime_factor"])
    result["response_basis"] = "archive-derived effective response"
    result["empirical_response_supported"] = np.asarray(valid)
    result.attrs["calibration_kind"] = "archive-derived effective response"
    return result


def mag_archive_windows(samples, pairs, *, min_coverage=0.9):
    """Average raw spin features over the matched ER reference MAG windows

    Reference Bx/By/Bz are calibrated SCD field vectors accompanying ER 3D
    data. Collection-clock bounds identify the paired raw interval directly.
    Mixed ranges, missing samples and invalid phase are rejected. Features
    constrain an effective planar gain/rotation, z gain and three offsets.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("Coverage must be in (0, 1]")
    raw = samples.loc[samples["mode"].eq("realtime")].sort_values("clock_s")
    clocks = raw["clock_s"].to_numpy(float)
    if not np.isfinite(clocks).all() or (np.diff(clocks) <= 0).any():
        raise ValueError("Raw real-time MAG clocks must be finite, unique and ordered")
    windows = pairs.loc[pairs["match_valid"]].groupby("accumulation_id").agg(
        start_clock_s=("start_clock_s", "min"), stop_clock_s=("stop_clock_s", "max"),
        start_utc=("start_utc", "min"), stop_utc=("stop_utc", "max"),
        reference_id=("reference_id", "first"), reference_bx=("reference_bx", "first"),
        reference_by=("reference_by", "first"), reference_bz=("reference_bz", "first")).reset_index()
    phase = np.deg2rad(raw["spin_phase_deg"].to_numpy(float))
    x, y, z = (raw["b" + axis + "_signed"].to_numpy(float) for axis in "xyz")
    features = np.column_stack([np.cos(phase)*x - np.sin(phase)*y,
                                np.sin(phase)*x + np.cos(phase)*y, z, np.cos(phase), np.sin(phase)])
    valid = (raw["science_valid"].to_numpy(bool) & raw["spin_phase_valid"].to_numpy(bool)
             & np.isfinite(features).all(axis=1))
    sums = np.vstack([np.zeros(5), np.cumsum(np.where(valid[:, None], features, 0), axis=0)])
    bad = np.r_[0, np.cumsum(~valid)]
    changed = np.r_[False, np.diff(raw["range"].to_numpy()) != 0]
    changes = np.r_[0, np.cumsum(changed)]
    left = np.searchsorted(clocks, windows["start_clock_s"])
    right = np.searchsorted(clocks, windows["stop_clock_s"])
    count = right - left
    expected = (windows["stop_clock_s"] - windows["start_clock_s"]).to_numpy()*9
    averages = (sums[right] - sums[left])/np.maximum(count[:, None], 1)
    for index, name in enumerate(("raw_u", "raw_v", "raw_z", "phase_cos", "phase_sin")):
        windows[name] = averages[:, index]
    ranges = raw["range"].to_numpy()
    windows["range"] = ranges[np.minimum(left, max(len(raw) - 1, 0))] if len(raw) else np.nan
    windows["n_samples"] = count
    windows["coverage"] = count / np.maximum(expected, 1)
    windows["window_s"] = windows["stop_clock_s"] - windows["start_clock_s"]
    windows["window_valid"] = ((count >= 3) & (count >= expected*min_coverage)
                                & (bad[right] == bad[left]) & (changes[right] == changes[np.minimum(left + 1, len(raw))])
                                & np.isfinite(windows[["reference_b" + axis for axis in "xyz"]]).all(axis=1))
    return windows


def _mag_design(windows):
    size = len(windows)
    design = np.zeros((size, 3, 6))
    u, v, z, c, s = (windows[name].to_numpy(float) for name in ("raw_u", "raw_v", "raw_z", "phase_cos", "phase_sin"))
    design[:, 0, 0], design[:, 0, 1], design[:, 0, 3], design[:, 0, 4] = u, -v, c, -s
    design[:, 1, 0], design[:, 1, 1], design[:, 1, 3], design[:, 1, 4] = v, u, s, c
    design[:, 2, 2], design[:, 2, 5] = z, 1
    return design.reshape(-1, 6)


def fit_mag_archive_calibration(windows, *, valid_from, valid_to, source,
                                min_windows=20, max_condition=1e5, fit_xy_offsets=True):
    """Fit effective MAG gain/phase/offset coefficients with identifiability checks

    Fits a common planar gain and rotation, an independent z gain, and
    offsets in sensor coordinates. It cannot identify a general alignment
    matrix or reproduce minute-dependent flight offsets. Validity is an
    explicit proposed application scope, not an assertion of validation.
    """
    start, stop = _utc(pd.Series([valid_from, valid_to]))
    if start >= stop or not source or min_windows < 3 or max_condition <= 1:
        raise ValueError("Supply a valid scope, source and sufficient fitting constraints")
    if "reference_id" in windows and windows["reference_id"].duplicated().any():
        raise ValueError("Training windows require independent reference identities")
    valid = (windows["window_valid"] & _utc(windows["start_utc"]).ge(start)
             & _utc(windows["stop_utc"]).le(stop))
    rows = []
    for range_id, group in windows.loc[valid].groupby("range"):
        row = {"range": range_id, "valid_from": start, "valid_to": stop,
               "n_windows": len(group), "source": source, "fit_valid": False,
               "calibration_kind": "archive-derived effective MAG correction",
               "high_frequency_validated": False,
               "training_start": group["start_utc"].min(), "training_stop": group["stop_utc"].max(),
               "reference_window_s": group["window_s"].median()}
        row["fit_xy_offsets"] = fit_xy_offsets
        if len(group) < min_windows:
            row["fit_reason"] = "insufficient reference windows"
            rows.append(row)
            continue
        columns = np.arange(6) if fit_xy_offsets else np.array([0, 1, 2, 5])
        design = _mag_design(group)[:, columns]
        target = group[["reference_b" + axis for axis in "xyz"]].to_numpy(float).ravel()
        scale = np.sqrt(np.mean(design**2, axis=0))
        scale = np.where(scale > 0, scale, 1)
        normalized = design/scale
        weights = np.ones(len(target))
        for _ in range(6):
            solution, _, rank, singular = np.linalg.lstsq(normalized*weights[:, None], target*weights, rcond=None)
            residual = normalized@solution - target
            scatter = max(float(np.median(np.abs(residual - np.median(residual)))*1.4826), 1e-10)
            weights = np.sqrt(np.minimum(1, 1.5*scatter/np.maximum(np.abs(residual), 1e-10)))
        condition = singular[0]/singular[-1] if singular[-1] > 0 else np.inf
        parameters = np.zeros(6)
        parameters[columns] = solution/scale
        a, b, gz, ox, oy, oz = parameters
        gain = np.hypot(a, b)
        row.update({"design_rank": rank, "condition": condition,
                    "training_rmse_nt": np.sqrt(np.mean(residual**2)),
                    "a": a, "b": b, "gz": gz, "ox": ox, "oy": oy, "oz": oz})
        if rank < len(columns) or condition > max_condition or gain <= 0 or gz <= 0:
            row["fit_reason"] = "unidentified or nonphysical gain/offset solution"
            rows.append(row)
            continue
        matrix = np.array([[a, -b, 0], [b, a, 0], [0, 0, gz]])
        zero = -np.linalg.solve(matrix, np.array([ox, oy, oz]))
        rotation = matrix/np.array([gain, gain, gz])[None, :]
        for index, axis in enumerate("xyz"):
            row["gain_" + axis] = [gain, gain, gz][index]
            row["zero_" + axis] = zero[index]
            for j, other in enumerate("xyz"):
                row["align_" + axis + other] = rotation[index, j]
        row.update({"fit_valid": True, "fit_reason": "identified effective planar model"})
        rows.append(row)
    if not rows:
        raise ValueError("No valid raw/reference MAG windows in the requested scope")
    return pd.DataFrame(rows)


def evaluate_mag_archive_calibration(windows, coefficients):
    """Predict reference-window averages and retain errors for independent validation"""
    out = windows.copy()
    for axis in "xyz":
        out["predicted_b" + axis] = np.nan
    out["empirical_calibration_supported"] = False
    usable = coefficients.loc[coefficients["fit_valid"]]
    if usable["range"].duplicated().any():
        raise ValueError("Select one coefficient cohort per MAG range")
    for _, row in usable.iterrows():
        valid = (out["window_valid"] & out["range"].eq(row["range"])
                 & _utc(out["start_utc"]).ge(pd.Timestamp(row["valid_from"]))
                 & _utc(out["stop_utc"]).le(pd.Timestamp(row["valid_to"])))
        group = out.loc[valid]
        parameters = row[["a", "b", "gz", "ox", "oy", "oz"]].to_numpy(float)
        predicted = (_mag_design(group)@parameters).reshape(-1, 3)
        out.loc[valid, ["predicted_b" + axis for axis in "xyz"]] = predicted
        out.loc[valid, "empirical_calibration_supported"] = True
    for axis in "xyz":
        out["error_b" + axis] = out["predicted_b" + axis] - out["reference_b" + axis]
    out.attrs["calibration_kind"] = "archive-derived effective MAG correction"
    return out


def apply_mag_archive_calibration(samples, coefficients):
    """Apply an effective correction to native samples without claiming native-rate validation"""
    usable = coefficients.loc[coefficients["fit_valid"]].copy()
    if usable.empty:
        raise ValueError("No identified MAG coefficients are available")
    source = "; ".join(usable["source"].unique())
    out = calibrate_mag_telemetry(samples, usable, source=source)
    out = despin_mag(out, phase_offset_deg=0)
    out["calibration_valid"] &= np.isfinite(out["spin_phase_deg"])
    if "spin_phase_valid" in out:
        out["calibration_valid"] &= out["spin_phase_valid"].to_numpy(bool)
    out.loc[~out["calibration_valid"], ["B" + axis + "_SCD" for axis in "xyz"]] = np.nan
    out["calibration_basis"] = "archive-derived effective MAG correction"
    out["high_frequency_validated"] = False
    out.attrs.update({"calibration_kind": "archive-derived effective MAG correction",
                      "high_frequency_validated": False})
    return out
