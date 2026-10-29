from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    calibrate_mag_9hz, correct_mag_spin_offsets, fit_mag_spin_offsets,
    fit_mag_archive_calibration, select_mag_models,
)
from moonbroom.archive_calibration import _mag_design
from moonbroom.pad_calibration import (
    _pitch_geometry, apply_pad_normalization, calibrate_er_pad, compare_pad_omni,
    fit_pad_normalization, pad_field_directions, pad_omnidirectional,
)
from moonbroom import er_angular_bins


def native_samples():
    clock = np.r_[np.arange(540)/9, 180 + np.arange(540)/9]
    phase = 2*np.pi*clock/5
    x, y = 100 + .05*clock, -50 + .03*clock
    return pd.DataFrame({"clock_s": clock, "utc": pd.Timestamp("1999-06-01T00:00Z") + pd.to_timedelta(clock, unit="s"),
                         "mode": "realtime", "range": 1, "spin_phase_deg": np.degrees(phase),
                         "spin_phase_valid": True, "spin_period_s": 5., "science_valid": True,
                         "bx_signed": x*np.cos(phase) + y*np.sin(phase) + 42,
                         "by_signed": -x*np.sin(phase) + y*np.cos(phase) - 13, "bz_signed": 20.,
                         "source_file": "test.B", "record_index": np.arange(len(clock))//18,
                         "sample_index": np.arange(len(clock))%18})


def native_coefficients():
    out = pd.DataFrame({"range": [1], "fit_valid": True, "validation_pass": True,
                         "valid_from": "1999-06-01T00:00Z", "valid_to": "1999-06-02T00:00Z",
                         "source": "synthetic test", "gain_x": .1, "gain_y": .1, "gain_z": .1,
                         "zero_x": 0., "zero_y": 0., "zero_z": 0., "offset_model": "local spin fit"})
    for i, axis in enumerate("xyz"):
        for j, other in enumerate("xyz"):
            out["align_" + axis + other] = float(i == j)
    return out


def test_local_offsets_recover_sensor_zero_without_filling_gap():
    raw = native_samples()
    offsets = fit_mag_spin_offsets(raw)
    assert offsets["offset_valid"].all()
    np.testing.assert_allclose(offsets[["zero_x", "zero_y"]], [[42, -13]]*2, atol=1e-11)
    gap = raw.iloc[[0]].copy()
    gap["clock_s"] = 120.
    corrected = correct_mag_spin_offsets(gap, offsets, max_distance_s=30)
    assert not corrected["spin_offset_valid"].any()
    assert not corrected["science_valid"].any()
    assert corrected["bx_signed"].isna().all()


def test_native_calibration_preserves_samples_cadence_gaps_counts_and_invalid_rows():
    raw = native_samples()
    raw.loc[10, "science_valid"] = False
    offsets = fit_mag_spin_offsets(raw)
    out = calibrate_mag_9hz(raw, native_coefficients(), offsets=offsets)
    assert len(out) == len(raw)
    pd.testing.assert_series_equal(out["clock_s"], raw["clock_s"])
    pd.testing.assert_series_equal(out["bx_signed"], raw["bx_signed"])
    assert out["source_sample_id"].is_unique
    assert out["calibration_valid"].sum() == len(raw) - 1
    assert out.loc[10, ["Bx_SCD", "By_SCD", "Bz_SCD"]].isna().all()
    valid = out["calibration_valid"]
    np.testing.assert_allclose(out.loc[valid, "Bx_SCD"], (100 + .05*raw.loc[valid, "clock_s"])*.1, atol=1e-10)
    assert out["sample_rate_hz"].eq(9).all()
    assert not out["high_frequency_validated"].any()
    assert out.attrs["resampled"] is False
    assert np.max(np.diff(out["clock_s"])) > 120
    with pytest.raises(ValueError, match="require"):
        calibrate_mag_9hz(raw, native_coefficients())
    rejected = native_coefficients().assign(validation_pass=False)
    assert not calibrate_mag_9hz(raw, rejected)["calibration_valid"].any()


def test_model_selection_rejects_overlaps_and_extrapolation():
    bank = native_coefficients()
    assert len(select_mag_models(bank, "1999-06-01")) == 1
    assert select_mag_models(bank, "1999-06-02").empty
    with pytest.raises(ValueError, match="overlaps"):
        select_mag_models(pd.concat([bank, bank]), "1999-06-01")
    with pytest.raises(ValueError, match="validation"):
        select_mag_models(bank.drop(columns="validation_pass"), "1999-06-01")


def test_mag_fit_can_fix_transverse_offsets_after_local_correction():
    rng = np.random.default_rng(7)
    windows = pd.DataFrame(rng.normal(size=(50, 5)), columns=["raw_u", "raw_v", "raw_z", "phase_cos", "phase_sin"])
    windows["range"], windows["window_valid"], windows["window_s"] = 1, True, 5.
    windows["start_utc"] = pd.Timestamp("1999-06-01T01:00Z") + pd.to_timedelta(np.arange(50)*5, unit="s")
    windows["stop_utc"] = windows["start_utc"] + pd.Timedelta(seconds=5)
    parameters = [.01, -.02, .03, 0, 0, -.1]
    windows[["reference_b" + axis for axis in "xyz"]] = (_mag_design(windows)@parameters).reshape(-1, 3)
    model = fit_mag_archive_calibration(windows, valid_from="1999-06-01", valid_to="1999-06-02", source="test", fit_xy_offsets=False)
    assert model["design_rank"].eq(4).all()
    np.testing.assert_allclose(model[["a", "b", "gz", "ox", "oy", "oz"]], [parameters], atol=1e-12)


def pad_inputs():
    raw = native_samples()
    angles = pd.DataFrame({"clock_s": [0, 1.25, np.nan, 2.5], "theta_deg": [90, 90, 90, 0],
                           "mode": ["realtime", "realtime", "realtime", "burst"]})
    directions = pad_field_directions(angles, raw)
    counts = pd.DataFrame({"accumulation_id": 0, "energy_step": 9, "sample": 0,
                          "angle_bin": np.arange(16), "start_clock_s": 0., "stop_clock_s": 2.5,
                          "start_utc": pd.Timestamp("1999-06-01T00:00Z"),
                          "stop_utc": pd.Timestamp("1999-06-01T00:00:02.5Z"),
                          "telemetry_version": 3, "mode": "realtime", "kind": "pad", "pitch_map_version": 0,
                          "energy_ev": 337., "energy_valid": True, "counts_lower": 1., "counts_upper": 1.,
                          "prescale_known": True, "nominal_integration_s": np.nan})
    coefficients = er_angular_bins()[["angle_bin"]].assign(mode="realtime", kind="3d", telemetry_version=3,
                          energy_step=9, energy_ev=337., geometric_factor_cm2_sr=.0013, fit_valid=True,
                          valid_from="1999-06-01T00:00Z", valid_to="1999-06-02T00:00Z")
    return counts, directions, coefficients


def test_pad_geometry_filters_untimed_and_burst_directions_and_masks_unknown_map():
    counts, directions, coefficients = pad_inputs()
    assert directions["clock_s"].tolist() == [0., 1.25]
    np.testing.assert_allclose(_pitch_geometry().sum(axis=1), 1)
    result = calibrate_er_pad(counts, directions, coefficients)
    assert result["calibration_valid"].all()
    assert result["pad_direction_valid"].all()
    assert not result["pitch_bin_response_validated"].any()
    assert pad_omnidirectional(result)["omni_valid"].all()
    unknown = calibrate_er_pad(counts.assign(pitch_map_version=1), directions, coefficients)
    assert not unknown["calibration_valid"].any()
    missing = calibrate_er_pad(counts, directions.iloc[:0], coefficients)
    assert not missing["calibration_valid"].any()


def test_pad_omni_weights_and_normalization_require_validation():
    counts, directions, coefficients = pad_inputs()
    pad = calibrate_er_pad(counts, directions, coefficients)
    for name in ["flux_lower", "flux_estimate", "flux_upper"]:
        pad[name] = 123.
    np.testing.assert_allclose(pad_omnidirectional(pad)["flux"], 123.)
    model = pd.DataFrame({"energy_step": [9], "flux_scale": [2.], "validation_pass": [True]})
    result = apply_pad_normalization(pad, model)
    np.testing.assert_allclose(pad_omnidirectional(result)["flux"], 246.)
    assert result["omnidirectional_model_validated"].all()
    assert not result["pitch_bin_response_validated"].any()
    assert not apply_pad_normalization(pad, model.assign(validation_pass=False))["calibration_valid"].any()
    with pytest.raises(ValueError, match="validation"):
        apply_pad_normalization(pad, model.drop(columns="validation_pass"))


def test_pad_training_alignment_and_scale_are_frozen_for_other_date():
    start = pd.Timestamp("1999-06-01T00:00Z") + pd.to_timedelta(np.arange(50)*5, unit="s")
    omni = pd.DataFrame({"energy_step": 9, "utc_start": start, "utc_stop": start + pd.Timedelta(seconds=5),
                         "flux": np.arange(50) + 10., "omni_valid": True})
    reference = pd.DataFrame({"energy_step": 9, "utc": start + pd.Timedelta(seconds=2.5),
                              "reference_window_s": 5., "reference_flux": 2*omni["flux"]})
    reference.loc[49, "utc"] = pd.NaT
    model = fit_pad_normalization(omni, reference, offsets_s=[0])
    assert model["flux_scale"].iloc[0] == 2
    assert model["n_training"].iloc[0] == 49
    for column in ["utc_start", "utc_stop"]:
        omni[column] += pd.Timedelta(days=2)
    reference["utc"] += pd.Timedelta(days=2)
    comparison = compare_pad_omni(omni, reference, time_offset_s=model["time_offset_s"].iloc[0])
    np.testing.assert_allclose(comparison["predicted_flux"]*model["flux_scale"].iloc[0], comparison["reference_flux"])


@pytest.mark.parametrize("cohort", ["1998_01", "1998_01_format2", "1998_06", "1999_03", "1999_03_sweep34"])
def test_pds_response_regression_on_independent_dates_and_software_versions(cohort):
    from moonbroom import apply_er_archive_response, fit_er_archive_response

    fixtures = Path(__file__).parent / "fixtures" / "mission_fit"
    train = pd.read_csv(fixtures / (cohort + "_training.csv"))
    test = pd.read_csv(fixtures / (cohort + "_validation.csv"))
    assert not set(train["reference_id"]) & set(test["reference_id"])
    assert set(pd.to_datetime(train["start_utc"], utc=True).dt.date).isdisjoint(
        pd.to_datetime(test["start_utc"], utc=True).dt.date)
    start = pd.to_datetime(train["start_utc"], utc=True).min().floor("D")
    stop = pd.to_datetime(test["stop_utc"], utc=True).max().ceil("D")
    models = fit_er_archive_response(train, valid_from=start, valid_to=stop, source="public PDS regression")
    out = apply_er_archive_response(test, models)
    assert out["calibration_valid"].all()
    error = (out["flux_estimate"]/out["reference_flux"] - 1).abs()
    assert error.median() < .02
    assert error.max() < .1


def test_er_mission_application_preserves_unmatched_rows_and_rejects_overlaps():
    from moonbroom import apply_er_mission_response, fit_er_archive_response

    fixtures = Path(__file__).parent / "fixtures" / "mission_fit"
    train = pd.read_csv(fixtures / "1998_06_training.csv")
    test = pd.read_csv(fixtures / "1998_06_validation.csv")
    model = fit_er_archive_response(train, valid_from="1998-06-01", valid_to="1998-07-01", source="PDS")
    model["validation_pass"] = True
    counts = pd.concat([test, test.assign(telemetry_version=2)], ignore_index=True)
    out = apply_er_mission_response(counts, model)
    assert len(out) == len(counts)
    assert out["calibration_valid"].iloc[:len(test)].all()
    assert not out["calibration_valid"].iloc[len(test):].any()
    with pytest.raises(ValueError, match="overlaps"):
        apply_er_mission_response(counts, pd.concat([model, model]))
    assert not apply_er_mission_response(counts, model.assign(validation_pass=False))["calibration_valid"].any()


def test_pad_mission_application_keeps_cohorts_separate():
    from moonbroom import apply_pad_mission_response

    counts, directions, er_model = pad_inputs()
    er_model["cohort"] = "test"
    model = pd.DataFrame({"cohort": ["test"], "telemetry_version": [3], "energy_step": [9], "flux_scale": [2.],
                          "validation_pass": [True], "valid_from": ["1999-06-01T00:00Z"], "valid_to": ["1999-06-02T00:00Z"]})
    out = apply_pad_mission_response(counts, directions, er_model, model)
    assert out["calibration_valid"].all()
    assert out["omnidirectional_model_validated"].all()
    assert not out["pitch_bin_response_validated"].any()
    with pytest.raises(ValueError, match="overlaps"):
        apply_pad_mission_response(counts, directions, er_model, pd.concat([model, model]))


def test_public_native_mag_preserves_9hz_and_matches_other_date_reference_averages():
    fixtures = Path(__file__).parent / "fixtures" / "mission_fit"
    raw = pd.read_csv(fixtures / "mag_native_validation.csv")
    model = pd.read_csv(fixtures / "mag_native_model.csv")
    windows = pd.read_csv(fixtures / "mag_native_validation_windows.csv")
    assert model["training_date"].iloc[0] != model["validation_date"].iloc[0]
    out = calibrate_mag_9hz(raw, model, offsets=fit_mag_spin_offsets(raw))
    assert len(out) == len(raw)
    pd.testing.assert_series_equal(out["clock_s"], raw["clock_s"])
    np.testing.assert_allclose(np.diff(out["clock_s"]), 1/9, atol=1e-8, rtol=0)
    errors = []
    for row in windows.itertuples():
        field = out.loc[out["clock_s"].ge(row.start_clock_s) & out["clock_s"].lt(row.stop_clock_s),
                        ["Bx_SCD", "By_SCD", "Bz_SCD"]].mean().to_numpy()
        errors.extend(field - [row.reference_bx, row.reference_by, row.reference_bz])
    assert len(windows) >= 8
    assert np.sqrt(np.mean(np.array(errors)**2)) < .1
    assert not out["high_frequency_validated"].any()


def test_pad_downlinked_direction_does_not_require_raw_mag():
    angles = pd.DataFrame({"clock_s": [0., 1.25], "theta_deg": [90., np.nan]})
    directions = pad_field_directions(angles, pd.DataFrame(columns=["mode", "clock_s"]))
    assert directions["direction_valid"].tolist() == [True, False]


def test_corrupt_reference_spectrum_can_be_excluded_with_a_count():
    from moonbroom import er_3d_reference

    wide = pd.DataFrame({"Spectrum Number": np.repeat([1, 2], 15),
                         "Spectrum Time": np.repeat([920250000, 920250080], 15),
                         "Energy Bin": np.tile(np.arange(15, 0, -1), 2), "Bx": 1., "By": 2., "Bz": 3.})
    flux = pd.DataFrame(np.ones((30, 88)), columns=[f"Electron Flux[{i}]" for i in range(88)])
    wide = pd.concat([wide, flux], axis=1)
    wide.loc[16, "Energy Bin"] = 0
    with pytest.raises(ValueError):
        er_3d_reference(wide, source="test")
    out = er_3d_reference(wide, source="test", drop_invalid=True)
    assert len(out) == 1320
    assert out.attrs["excluded_reference_spectra"] == 1


@pytest.mark.parametrize("version", [4, 5])
def test_real_firmware_packets_decode_to_matched_pds_distributions(version):
    from moonbroom import decode_er_accumulation

    fixtures = Path(__file__).parent / "fixtures" / "mission_fit"
    data = (fixtures / ("firmware" + str(version) + "_packets.bin")).read_bytes()
    packets = {index: data[index*83:(index + 1)*83] for index in range(36)}
    decoded = decode_er_accumulation(packets, version=version)
    counts = decoded["counts"].loc[decoded["counts"]["kind"].eq("3d")].sort_values(["energy_step", "angle_bin"])
    reference = pd.read_csv(fixtures / ("firmware" + str(version) + "_reference.csv.gz"))
    np.testing.assert_allclose(counts["counts_lower"], reference["counts_lower"])
    np.testing.assert_allclose(counts["counts_upper"], reference["counts_upper"])
    intensity = counts["counts_lower"].to_numpy()/reference["nominal_integration_s"]/reference["reference_energy_ev"]
    valid = intensity.gt(0) & reference["reference_flux"].gt(0) & np.isfinite(intensity)
    rank_correlation = intensity.loc[valid].rank().corr(reference.loc[valid, "reference_flux"].rank())
    assert rank_correlation > .99
    assert decoded["summary"]["layout_basis"] == "empirical format-3 compatibility"


def test_native_failure_diagnostics_separate_quality_timing_offsets_and_models():
    from moonbroom import mag_calibration_diagnostics

    out = native_samples().iloc[:8].copy().reset_index(drop=True)
    out["calibration_valid"] = False
    out["science_valid_original"] = True
    out["spin_offset_valid"] = True
    out["adc_at_limit"] = False
    out.loc[0, "adc_at_limit"] = True
    out.loc[1, "utc"] = pd.NaT
    out.loc[2, "spin_phase_valid"] = False
    out.loc[3, "spin_offset_valid"] = False
    out.loc[4, "range"] = 2
    out.loc[7, "calibration_valid"] = True
    model = native_coefficients().assign(validation_pass=False, n_validation=4)
    result = mag_calibration_diagnostics(out, model)
    assert result["calibration_failure_reason"].tolist() == [
        "adc_limit", "utc_unavailable", "spin_phase_unavailable", "local_offset_unavailable",
        "range_not_trained", "validation_insufficient", "validation_insufficient", "calibrated"]
    assert not result["calibration_model_available"].any()
    assert out.columns.equals(result[out.columns].columns)
    pd.testing.assert_frame_equal(result[out.columns], out)
    accepted = mag_calibration_diagnostics(out, native_coefficients())
    assert accepted["calibration_model_available"].iloc[0]
    assert accepted["calibration_failure_reason"].iloc[0] == "adc_limit"


def test_native_failure_diagnostics_accept_exported_phase_flag_without_angle():
    from moonbroom import mag_calibration_diagnostics

    out = native_samples().iloc[:1].drop(columns="spin_phase_deg").assign(calibration_valid=False)
    result = mag_calibration_diagnostics(out, native_coefficients().assign(fit_valid=False, n_windows=2))
    assert result["calibration_failure_reason"].iloc[0] == "training_insufficient"


@pytest.mark.parametrize("range_id", [0, 1, 2])
def test_additional_pds_dates_recover_range_models_without_same_date_leakage(range_id):
    from moonbroom import evaluate_mag_archive_calibration

    fixtures = Path(__file__).parent / "fixtures" / "gap_recovery"
    train = pd.read_csv(fixtures / ("range" + str(range_id) + "_training.csv"))
    test = pd.read_csv(fixtures / ("range" + str(range_id) + "_validation.csv"))
    assert set(pd.to_datetime(train["start_utc"], utc=True).dt.date).isdisjoint(
        pd.to_datetime(test["start_utc"], utc=True).dt.date)
    start = min(pd.to_datetime(train["start_utc"], utc=True).min(), pd.to_datetime(test["start_utc"], utc=True).min()).floor("D")
    stop = max(pd.to_datetime(train["stop_utc"], utc=True).max(), pd.to_datetime(test["stop_utc"], utc=True).max()).ceil("D")
    models = fit_mag_archive_calibration(train, valid_from=start, valid_to=stop,
                                        source="public PDS additional-date regression", fit_xy_offsets=False)
    assert models["fit_valid"].all()
    out = evaluate_mag_archive_calibration(test, models)
    assert out["empirical_calibration_supported"].all()
    error = out[["error_b" + axis for axis in "xyz"]].to_numpy(float)
    assert np.sqrt(np.mean(error**2)) < .1
