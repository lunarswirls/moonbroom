"""Explicit flight-coefficient MAG calibration and documented PDS ER response"""

from pathlib import Path

import numpy as np
import pandas as pd


def read_er_energy_bins(path, *, time_basis):
    """Read official highres or lowres E_BINS, retaining inconsistent time tags

    time_basis is explicitly unix or utc because the archive's two time
    columns disagree in some records. No silent correction is made. The
    lowres table contains all 15 energy steps; highres contains two channels
    whose selected steps are identified only when a telemetry version is
    supplied to assign_er_energies.
    """
    if time_basis not in ("unix", "utc"):
        raise ValueError("Choose time_basis='unix' or 'utc' explicitly")
    path = Path(path).expanduser()
    if path.suffix.lower() == ".xml":
        from .io import read_pds4

        table = read_pds4(path)
        unix = table.iloc[:, 0].to_numpy(float)
        utc = pd.to_datetime(table.iloc[:, 1], utc=True)
        energies = table.iloc[:, 2:].to_numpy(float)
    else:
        table = pd.read_csv(path, sep=r"\s+", header=None)
        unix = table.iloc[:, 0].to_numpy(float)
        utc = pd.to_datetime(table.iloc[:, 1], utc=True)
        energies = table.iloc[:, 2:].to_numpy(float)
    if energies.shape[1] not in (2, 15) or not np.isfinite(energies).all() or (energies <= 0).any():
        raise ValueError("Expected two or fifteen positive finite ER center energies")
    seconds = pd.to_datetime(unix, unit="s", utc=True)
    difference = (utc.reset_index(drop=True) - seconds).dt.total_seconds().to_numpy()
    selected = seconds if time_basis == "unix" else utc
    if pd.isna(selected).any() or (np.diff(pd.DatetimeIndex(selected).asi8) <= 0).any():
        raise ValueError("Energy change times must be valid and strictly increasing")
    count = energies.shape[1]
    out = pd.DataFrame({"valid_from": np.repeat(pd.DatetimeIndex(selected), count),
                        "channel": np.tile(np.arange(count), len(table)),
                        "energy_ev": energies.ravel(),
                        "unix_utc": np.repeat(seconds, count),
                        "text_utc": np.repeat(pd.DatetimeIndex(utc), count),
                        "time_disagreement_s": np.repeat(difference, count)})
    if count == 15:
        out["energy_step"] = out["channel"]
    out.attrs = {"source": str(path.resolve()), "time_basis": time_basis,
                 "energy_units": "eV", "channels": count}
    return out


def assign_er_energies(counts, energy_bins, *, telemetry_version=None):
    """Apply center energies in half-open validity intervals without extrapolating backwards"""
    bins = energy_bins.copy()
    if "energy_step" not in bins:
        if telemetry_version not in (1, 2, 3, 4, 5) or bins["channel"].nunique() != 2:
            raise ValueError("Two-channel energy tables require telemetry_version 1 through 5")
        bins["energy_step"] = bins["channel"].map({0: 6 if telemetry_version == 1 else 9,
                                                   1: 9 if telemetry_version == 1 else 11})
    bins["valid_from"] = pd.to_datetime(bins["valid_from"], utc=True, format="mixed")
    if bins[["valid_from", "energy_step"]].duplicated().any() or bins["valid_from"].isna().any():
        raise ValueError("Energy validity times and steps must be unique and nonmissing")
    if (not bins["energy_step"].isin(range(15)).all()
            or not np.isfinite(bins["energy_ev"].to_numpy(float)).all()
            or (bins["energy_ev"] <= 0).any()):
        raise ValueError("Energy steps must be 0 through 14 with positive finite energies")
    out = counts.copy()
    start = pd.to_datetime(out["start_utc"], utc=True, format="mixed")
    stop = pd.to_datetime(out["stop_utc"], utc=True, format="mixed")
    out["energy_ev"] = np.nan
    out["energy_valid"] = False
    out["energy_change_within_accumulation"] = False
    for step, group in bins.groupby("energy_step"):
        group = group.sort_values("valid_from")
        times = pd.DatetimeIndex(group["valid_from"]).as_unit("ns").asi8
        chosen = np.flatnonzero(out["energy_step"].eq(step).to_numpy())
        starts = pd.DatetimeIndex(start.iloc[chosen]).as_unit("ns").asi8
        stops = pd.DatetimeIndex(stop.iloc[chosen]).as_unit("ns").asi8
        indices = np.searchsorted(times, starts, side="right") - 1
        last = np.searchsorted(times, stops, side="left") - 1
        valid = (indices >= 0) & (last == indices) & (stops > starts) & start.iloc[chosen].notna().to_numpy()
        changed = (last > indices) & (indices >= 0)
        out.iloc[chosen, out.columns.get_loc("energy_change_within_accumulation")] = changed
        out.iloc[chosen[valid], out.columns.get_loc("energy_ev")] = group["energy_ev"].to_numpy()[indices[valid]]
        out.iloc[chosen[valid], out.columns.get_loc("energy_valid")] = True
    out.attrs["energy_source"] = energy_bins.attrs.get("source", "caller-supplied center energies")
    out.attrs["energy_time_basis"] = energy_bins.attrs.get("time_basis")
    return out


def calibrate_er_telemetry(counts, *, response=None, dead_time_s=3e-7,
                           max_deadtime_factor=1.25):
    """Convert decompressed ER counts to flux with the PDS response convention

    Input requires energy_ev from assign_er_energies. Without a supplied
    response, 3d and cut bins yield count rates using their documented
    integration times; flux remains missing. The collection's approximate
    full-instrument geometric factor cannot be used for individual angular
    bins. PAD bins require an effective exposure/response table because
    pitch-angle occupancy is not integration duration. response joins on accumulation_id, kind,
    energy_step, sample and angle_bin, with exposure_s and
    geometric_factor_cm2_sr. Flux divides by center energy, not bin width.
    Quantization bounds are propagated, saturated or excessive-deadtime
    values are masked, and background is retained as in the PDS archive.
    """
    if not np.isfinite(dead_time_s) or dead_time_s < 0:
        raise ValueError("Dead time must be finite and nonnegative")
    if not np.isfinite(max_deadtime_factor) or max_deadtime_factor < 1:
        raise ValueError("Maximum deadtime factor must be finite and at least one")
    out = counts.copy()
    if response is None:
        exposure = out["nominal_integration_s"].to_numpy(float)
        supported = out["kind"].isin(["3d", "cut"]).to_numpy()
        exposure = np.where(supported, exposure, np.nan)
        factor = np.full(len(out), np.nan)
        out["response_basis"] = "missing effective response"
    else:
        keys = ["accumulation_id", "kind", "energy_step", "sample", "angle_bin"]
        if response[keys].duplicated().any() or response[keys].isna().any().any():
            raise ValueError("Response keys must be unique and nonmissing")
        values = response[["exposure_s", "geometric_factor_cm2_sr"]].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Supplied exposure and geometric factors must be positive and finite")
        matched = out[keys].merge(response[keys + ["exposure_s", "geometric_factor_cm2_sr"]],
                                  how="left", on=keys, validate="many_to_one", sort=False)
        exposure = matched["exposure_s"].to_numpy(float)
        factor = matched["geometric_factor_cm2_sr"].to_numpy(float)
        out["response_basis"] = "caller-supplied effective response"
    out["exposure_s"], out["geometric_factor_cm2_sr"] = exposure, factor
    energy = out["energy_ev"].to_numpy(float)
    lower = out["counts_lower"].to_numpy(float)
    upper = out["counts_upper"].to_numpy(float)
    base = (np.isfinite(exposure) & (exposure > 0) & np.isfinite(lower)
            & (lower >= 0) & np.isfinite(upper) & (upper >= lower))
    if 'science_valid' in out:
        base &= out['science_valid'].fillna(False).to_numpy(bool)
    if "prescale_known" in out:
        base &= out["prescale_known"].to_numpy(bool)
    midpoint = lower + (upper - lower) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = 1 / (1 - upper / exposure * dead_time_s)
        rate_valid = base & (correction >= 1) & (correction <= max_deadtime_factor)
        valid = rate_valid & np.isfinite(energy) & (energy > 0) & np.isfinite(factor) & (factor > 0)
        if "energy_valid" in out:
            valid &= out["energy_valid"].to_numpy(bool)
        for name, values in (("lower", lower), ("upper", upper), ("estimate", midpoint)):
            rate = values / exposure
            corrected = rate / (1 - rate * dead_time_s)
            out["count_rate_" + name] = np.where(rate_valid, corrected, np.nan)
            flux = corrected / (factor * energy)
            out["flux_" + name] = np.where(valid, flux, np.nan)
        out["deadtime_factor_upper"] = np.where(base, correction, np.nan)
    out["calibration_valid"] = valid
    out["count_rate_valid"] = rate_valid
    out["saturation_possible"] = np.isposinf(upper)
    out.attrs.update({"flux_units": "electrons cm^-2 s^-1 sr^-1 eV^-1",
                      "dead_time_s": dead_time_s, "max_deadtime_factor": max_deadtime_factor,
                      "background_subtracted": False, "flux_estimate": "midpoint of count quantization interval",
                      "response_source": None if response is None else response.attrs.get("source"),
                      "calibration_model": "PDS deadtime and center-energy convention with explicit angular response"})
    return out


def calibrate_mag_telemetry(samples, calibration, *, source, allow_realtime_transfer=False):
    """Apply time-dependent flight gains, offsets and optional alignment matrix

    calibration requires range, valid_from, valid_to, zero_x/y/z in signed
    ADC counts and gain_x/y/z in nT/count. Optional align_xx through align_zz
    form a sensor-to-spacecraft matrix applied after gain/offset correction.
    source identifies the supplied flight coefficient product. No nominal
    table is substituted for missing mission coefficients. Diagnostic, ADC
    limit and range-transition samples are masked using decoder flags.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Identify the source of the flight calibration coefficients")
    coefficients = calibration.copy()
    for column in ("valid_from", "valid_to"):
        coefficients[column] = pd.to_datetime(coefficients[column], utc=True, format="mixed")
    if (coefficients[["valid_from", "valid_to", "range"]].isna().any().any()
            or (coefficients["valid_to"] <= coefficients["valid_from"]).any()
            or not coefficients["range"].isin(range(8)).all()):
        raise ValueError("MAG coefficients require valid intervals and ranges 0 through 7")
    columns = [kind + "_" + axis for kind in ("zero", "gain") for axis in "xyz"]
    matrix_cols = ["align_" + axis + other for axis in "xyz" for other in "xyz"]
    has_matrix = any(column in coefficients for column in matrix_cols)
    if has_matrix and not all(column in coefficients for column in matrix_cols):
        raise ValueError("Supply all nine alignment coefficients")
    required = columns + (matrix_cols if has_matrix else [])
    if not np.isfinite(coefficients[required].to_numpy(float)).all():
        raise ValueError("MAG calibration coefficients must be finite")
    if (coefficients[["gain_" + axis for axis in "xyz"]] <= 0).any().any():
        raise ValueError("MAG gains must be positive")
    from .mag_state import validate_mag_model_scopes, mag_model_assignments, MAG_SCOPE_FIELDS
    validate_mag_model_scopes(coefficients)
    out = samples.copy()
    values = out[["b" + axis + "_signed" for axis in "xyz"]].to_numpy(float)
    field = np.full((len(out), 3), np.nan)
    assigned = np.zeros(len(out), dtype=bool)
    out["calibration_mode_transfer"] = False
    out["calibration_hardware_scoped"] = False
    masks = mag_model_assignments(out, coefficients, allow_realtime_transfer=allow_realtime_transfer)
    for (_, coefficient), mask in zip(coefficients.iterrows(), masks):
        chosen = mask.to_numpy()
        model_mode = coefficient.get("mode", "realtime")
        model_mode = "realtime" if pd.isna(model_mode) else model_mode
        if "mode" in out:
            out.loc[mask, "calibration_mode_transfer"] = out.loc[mask, "mode"].ne(model_mode)
        out.loc[mask, "calibration_hardware_scoped"] = any(pd.notna(coefficient.get(f)) for f in MAG_SCOPE_FIELDS)
        zeros = coefficient[["zero_" + axis for axis in "xyz"]].to_numpy(float)
        gains = coefficient[["gain_" + axis for axis in "xyz"]].to_numpy(float)
        calibrated = (values[chosen] - zeros) * gains
        if has_matrix:
            matrix = coefficient[matrix_cols].to_numpy(float).reshape(3, 3)
            if not np.allclose(matrix @ matrix.T, np.eye(3), atol=0.05) or np.linalg.det(matrix) <= 0:
                raise ValueError("Alignment must preserve a right-handed, approximately orthonormal basis")
            calibrated = calibrated @ matrix.T
        field[chosen] = calibrated
        assigned[chosen] = True
    valid = assigned & np.isfinite(values).all(axis=1)
    if "science_valid" in out:
        valid &= out["science_valid"].to_numpy(bool)
    for column in ("calibration_mode", "adc_at_limit", "range_changed"):
        if column in out:
            valid &= ~out[column].to_numpy(bool)
    if "mag_hardware_state_valid" in out:
        valid &= out["mag_hardware_state_valid"].fillna(False).to_numpy(bool)
    if "mag_hardware_transition" in out:
        valid &= ~out["mag_hardware_transition"].fillna(True).to_numpy(bool)
    field[~valid] = np.nan
    frame = "SC" if has_matrix else "SEN"
    out[["B" + axis + "_" + frame for axis in "xyz"]] = field
    out["calibration_available"] = assigned
    out["calibration_valid"] = valid
    out["calibration_source"] = source
    out.attrs.update({"calibration_source": source, "field_units": "nT", "field_frame": frame})
    return out


def despin_mag(samples, *, phase_offset_deg, phase_col="spin_phase_deg", frame="SC"):
    """Rotate spacecraft field vectors into the sun-referenced despun SCD frame

    phase must be the rotation of spacecraft x from the sunward SCD x axis,
    increasing about spin-axis z. phase_offset_deg contains the instrument
    mounting/sun-sensor correction and must come from the applicable geometry.
    """
    if not np.isfinite(phase_offset_deg):
        raise ValueError("The phase offset must be finite")
    out = samples.copy()
    phase = np.deg2rad(out[phase_col].to_numpy(float) + phase_offset_deg)
    field = out[["B" + axis + "_" + frame for axis in "xyz"]].to_numpy(float)
    out["Bx_SCD"] = np.cos(phase)*field[:, 0] - np.sin(phase)*field[:, 1]
    out["By_SCD"] = np.sin(phase)*field[:, 0] + np.cos(phase)*field[:, 1]
    out["Bz_SCD"] = np.where(np.isfinite(phase), field[:, 2], np.nan)
    out.attrs["field_frame"] = "SCD"
    return out


def calibrate_level0(products, *, mag_calibration=None, mag_source=None,
                     er_energy_bins=None, er_response=None, sunpulse=None,
                     phase_offset_deg=None, max_pulse_gap_s=6.0,
                     er_telemetry_version=None):
    """Calibrate selected decoded products without assuming files or event settings

    Missing MAG coefficients leave raw MAG available and record the omission
    in calibration_info. ER outputs retain an explicit response basis;
    flux stays missing without an effective response. Despinning requires
    supplied sun pulses, an alignment matrix and a mounting phase offset.
    """
    from .timing import reconstruct_spin_phase

    result = {name: table.copy() for name, table in products.items()}
    info = []
    if sunpulse is not None:
        result["sunpulse"] = sunpulse.copy()
    if mag_calibration is None:
        info.append(("mag", "not calibrated", "missing flight gains, offsets and alignment"))
    else:
        mag = calibrate_mag_telemetry(result["mag"], mag_calibration, source=mag_source)
        if sunpulse is not None:
            mag = reconstruct_spin_phase(mag, sunpulse, max_gap_s=max_pulse_gap_s)
        if phase_offset_deg is not None:
            if sunpulse is None or mag.attrs["field_frame"] != "SC":
                raise ValueError("Despinning requires sun pulses and sensor-to-spacecraft alignment")
            mag = despin_mag(mag, phase_offset_deg=phase_offset_deg)
        result["mag_calibrated"] = mag
        info.append(("mag", "supplied flight coefficients", mag_source))
    if er_energy_bins is None:
        info.append(("er", "not calibrated", "missing official center-energy table"))
    elif result["er_counts"].empty:
        info.append(("er", "not calibrated", "no decoded ER counts"))
    else:
        counts = assign_er_energies(result["er_counts"], er_energy_bins,
                                    telemetry_version=er_telemetry_version)
        result["er_flux"] = calibrate_er_telemetry(counts, response=er_response)
        info.append(("er_energy", er_energy_bins.attrs.get("time_basis"), er_energy_bins.attrs.get("source")))
        info.append(("er_response", "count rates only, missing effective response" if er_response is None
                     else "supplied effective response", None if er_response is None else er_response.attrs.get("source")))
    result["calibration_info"] = pd.DataFrame(info, columns=["instrument", "basis", "source"])
    return result
