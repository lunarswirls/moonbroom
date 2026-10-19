from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    apply_er_archive_response, apply_mag_archive_calibration, er_3d_reference,
    evaluate_mag_archive_calibration, fit_er_archive_response, fit_mag_archive_calibration,
    mag_archive_windows, match_er_archive,
)
from moonbroom.archive_calibration import _mag_design


fixtures = Path(__file__).parent / "fixtures" / "archive_fit"


def er_pairs(size=12):
    out = pd.DataFrame({"accumulation_id": np.repeat(np.arange(size), 2),
                        "energy_step": 9, "angle_bin": np.tile([0, 1], size),
                        "mode": "realtime", "kind": "3d", "sample": 0,
                        "counts_lower": np.repeat(np.arange(size) + 10., 2),
                        "nominal_integration_s": 0.02, "energy_ev": 337.421,
                        "energy_valid": True, "match_valid": True,
                        "complete": True, "prescale_known": True,
                        "telemetry_version": 3})
    out["counts_upper"] = out["counts_lower"]
    out["start_utc"] = pd.Timestamp("1999-03-01T01:00Z") + pd.to_timedelta(out["accumulation_id"]*80, unit="s")
    out["stop_utc"] = out["start_utc"] + pd.Timedelta(seconds=2.5)
    rate = out["counts_lower"]/out["nominal_integration_s"]
    response = np.where(out["angle_bin"].eq(0), 0.001325, 0.0014)
    out["reference_flux"] = rate/(1-rate*3e-7)/response/out["energy_ev"]
    return out


def test_er_recovers_effective_response_and_predicts_independent_counts():
    train = er_pairs()
    coefficients = fit_er_archive_response(train, valid_from="1999-03-01", valid_to="1999-03-02", source="synthetic reference")
    np.testing.assert_allclose(coefficients["geometric_factor_cm2_sr"], [0.001325, 0.0014], rtol=1e-12)
    test = er_pairs(8)
    test["counts_lower"] *= 2
    test["counts_upper"] = test["counts_lower"]
    rate = test["counts_lower"]/test["nominal_integration_s"]
    test["reference_flux"] = rate/(1-rate*3e-7)/np.where(test["angle_bin"].eq(0), .001325, .0014)/test["energy_ev"]
    test["geometric_factor_cm2_sr"] = np.nan
    result = apply_er_archive_response(test, coefficients)
    np.testing.assert_allclose(result["flux_estimate"], test["reference_flux"], rtol=1e-12)
    assert result["calibration_valid"].all()
    assert result["response_basis"].str.contains("archive-derived").all()


def test_er_scope_energy_format_and_missing_bins_are_not_extrapolated():
    counts = er_pairs()
    coefficients = fit_er_archive_response(counts, valid_from="1999-03-01", valid_to="1999-03-02", source="test")
    counts.loc[0, "start_utc"] = pd.Timestamp("1999-03-02T01:00Z")
    counts.loc[0, "stop_utc"] = pd.Timestamp("1999-03-02T01:00:02Z")
    counts.loc[1, "energy_ev"] = 339.609
    counts.loc[2, "telemetry_version"] = 2
    counts.loc[3, "angle_bin"] = 7
    counts.loc[4, "mode"] = "burst"
    result = apply_er_archive_response(counts, coefficients)
    assert not result["calibration_valid"].iloc[:5].any()
    assert result["calibration_valid"].iloc[5:].all()


def test_er_zeros_saturation_sparse_groups_and_sweep_changes_are_not_fit():
    pairs = er_pairs()
    pairs.loc[:9, "counts_lower"] = 0
    pairs.loc[:9, "counts_upper"] = 0
    pairs.loc[10, "counts_upper"] = np.inf
    coefficients = fit_er_archive_response(pairs, valid_from="1999-03-01", valid_to="1999-03-02", source="test")
    assert not coefficients["fit_valid"].any()
    pairs = er_pairs()
    pairs.loc[0, "energy_ev"] = 100
    with pytest.raises(ValueError, match="sweep"):
        fit_er_archive_response(pairs, valid_from="1999-03-01", valid_to="1999-03-02", source="test")


def mag_windows(size=60):
    rng = np.random.default_rng(32)
    out = pd.DataFrame(rng.normal(size=(size, 5)), columns=["raw_u", "raw_v", "raw_z", "phase_cos", "phase_sin"])
    out["range"] = 2
    out["window_valid"] = True
    out["window_s"] = 2.5
    out["start_utc"] = pd.Timestamp("1999-03-01T01:00Z") + pd.to_timedelta(np.arange(size)*80, unit="s")
    out["stop_utc"] = out["start_utc"] + pd.Timedelta(seconds=2.5)
    parameters = np.array([.015, -.028, .032, -.4, 3., -.6])
    out[["reference_b" + axis for axis in "xyz"]] = (_mag_design(out)@parameters).reshape(-1, 3)
    return out


def test_mag_recovers_physical_planar_model_and_validates_separate_windows():
    windows = mag_windows()
    coefficients = fit_mag_archive_calibration(windows.iloc[:40], valid_from="1999-03-01", valid_to="1999-03-02", source="synthetic reference")
    assert coefficients["fit_valid"].all()
    assert coefficients["design_rank"].eq(6).all()
    np.testing.assert_allclose(coefficients["gain_x"], np.hypot(.015, .028), atol=1e-12)
    np.testing.assert_allclose(coefficients["gain_z"], .032, atol=1e-12)
    result = evaluate_mag_archive_calibration(windows.iloc[40:], coefficients)
    np.testing.assert_allclose(result[["error_b" + axis for axis in "xyz"]], 0, atol=1e-12)
    samples = pd.DataFrame({"utc": ["1999-03-01T02:00Z"], "range": [2],
                            "bx_signed": [10], "by_signed": [20], "bz_signed": [30],
                            "spin_phase_deg": [0], "science_valid": [True]})
    calibrated = apply_mag_archive_calibration(samples, coefficients)
    np.testing.assert_allclose(calibrated[["Bx_SCD", "By_SCD", "Bz_SCD"]], [[.31, 3.02, .36]], atol=1e-12)
    assert not calibrated["high_frequency_validated"].any()


def test_mag_unidentified_offsets_and_sparse_ranges_remain_unsupported():
    windows = mag_windows()
    windows["phase_cos"] = 0
    windows["phase_sin"] = 0
    coefficients = fit_mag_archive_calibration(windows, valid_from="1999-03-01", valid_to="1999-03-02", source="test")
    assert not coefficients["fit_valid"].any()
    assert coefficients["design_rank"].iloc[0] < 6
    with pytest.raises(ValueError, match="No identified"):
        apply_mag_archive_calibration(pd.DataFrame(), coefficients)
    coefficients = fit_mag_archive_calibration(mag_windows(5), valid_from="1999-03-01", valid_to="1999-03-02", source="test")
    assert not coefficients["fit_valid"].any()


def test_reference_reader_uses_native_time_and_rejects_incomplete_spectra():
    wide = pd.DataFrame({"Spectrum Number": [1]*15, "Spectrum Time": [920250000]*15,
                         "Time": ["wrong row times"]*15, "Energy Bin": np.arange(15, 0, -1),
                         "Bx": 1., "By": 2., "Bz": 3.})
    flux = pd.DataFrame(np.tile(np.arange(88), (15, 1)), columns=[f"Electron Flux[{i}]" for i in range(88)])
    wide = pd.concat([wide, flux], axis=1)
    result = er_3d_reference(wide, source="synthetic archive")
    assert len(result) == 1320
    assert result["reference_utc"].nunique() == 1
    assert result["reference_utc"].iloc[0] == pd.Timestamp(920250000, unit="s", tz="UTC")
    wide["Spectrum Time"] = 0
    assert er_3d_reference(wide, source="test")["reference_utc"].isna().all()
    with pytest.raises(ValueError, match="fifteen"):
        er_3d_reference(wide.iloc[:14], source="test")


def test_reference_matching_checks_content_and_preserves_unmatched_counts():
    size = 1320
    counts = pd.DataFrame({"accumulation_id": 0, "energy_step": np.repeat(np.arange(15), 88),
                           "angle_bin": np.tile(np.arange(88), 15), "kind": "3d", "complete": True,
                           "counts_lower": np.arange(size, dtype=float), "counts_upper": np.arange(size, dtype=float),
                           "nominal_integration_s": .02, "start_utc": pd.Timestamp("1999-03-01T01:00Z"),
                           "stop_utc": pd.Timestamp("1999-03-01T01:00:02.5Z")})
    reference = counts[["energy_step", "angle_bin"]].copy()
    reference["reference_id"] = "test"
    reference["reference_utc"] = pd.Timestamp("1999-03-01T01:00:02Z")
    reference["reference_energy_ev"] = 100.
    reference["reference_flux"] = counts["counts_lower"]*2
    result = match_er_archive(counts, reference)
    assert result["match_valid"].all()
    assert len(result) == len(counts)
    reference["reference_flux"] = reference["reference_flux"].to_numpy()[::-1]
    assert not match_er_archive(counts, reference)["match_valid"].any()


def test_mag_window_features_reject_bad_samples_and_mixed_ranges():
    clocks = np.arange(180)/9
    samples = pd.DataFrame({"clock_s": clocks, "mode": "realtime", "range": 2,
                            "bx_signed": 1., "by_signed": 0., "bz_signed": 2.,
                            "spin_phase_deg": clocks*72, "spin_phase_valid": True, "science_valid": True})
    pairs = pd.DataFrame({"accumulation_id": [0, 1], "match_valid": True,
                          "start_clock_s": [0., 10.], "stop_clock_s": [2.5, 12.5],
                          "start_utc": pd.to_datetime(["1999-03-01T00:00:00Z", "1999-03-01T00:00:10Z"]),
                          "stop_utc": pd.to_datetime(["1999-03-01T00:00:02.5Z", "1999-03-01T00:00:12.5Z"]),
                          "reference_id": ["a", "b"], "reference_bx": 1., "reference_by": 1., "reference_bz": 1.})
    result = mag_archive_windows(samples, pairs)
    assert result["window_valid"].all()
    assert result["raw_z"].eq(2).all()
    samples.loc[1, "science_valid"] = False
    samples.loc[95, "range"] = 1
    assert not mag_archive_windows(samples, pairs)["window_valid"].any()


def test_public_pds_mag_regression_with_separate_telemetry_intervals():
    train = pd.read_csv(fixtures / "mag_training_windows.csv")
    test = pd.read_csv(fixtures / "mag_validation_windows.csv")
    assert not set(train["reference_id"]) & set(test["reference_id"])
    coefficients = fit_mag_archive_calibration(train, valid_from="1999-03-01", valid_to="1999-03-02", source="public PDS regression")
    assert coefficients.loc[coefficients["fit_valid"], "range"].tolist() == [2]
    out = evaluate_mag_archive_calibration(test, coefficients)
    error = out[["error_b" + axis for axis in "xyz"]].to_numpy(float)
    assert out["empirical_calibration_supported"].sum() == 166
    assert np.sqrt(np.nanmean(error**2)) < 0.28


def test_public_pds_er_regression_with_separate_spectra():
    train = pd.read_csv(fixtures / "er_training_pairs.csv")
    test = pd.read_csv(fixtures / "er_validation_pairs.csv")
    assert not set(train["reference_id"]) & set(test["reference_id"])
    coefficients = fit_er_archive_response(train, valid_from="1999-03-01", valid_to="1999-03-02", source="public PDS regression")
    assert coefficients["fit_valid"].all()
    result = apply_er_archive_response(test, coefficients)
    valid = result["calibration_valid"] & result["reference_flux"].gt(0)
    error = (result.loc[valid, "flux_estimate"]/result.loc[valid, "reference_flux"] - 1).abs()
    assert len(error) > 100
    assert error.median() < 0.01
    assert error.quantile(.9) < 0.03
