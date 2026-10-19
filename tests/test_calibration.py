from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    assign_er_energies, calibrate_er_telemetry, calibrate_level0,
    calibrate_mag_telemetry, despin_mag, read_er_energy_bins,
)


fixtures = Path(__file__).parent / "fixtures"


def test_official_energy_tables_retain_time_conflicts_and_label_values():
    bins = read_er_energy_bins(fixtures / "er_energy_low/e_bins.xml", time_basis="unix")
    assert len(bins) == 75
    np.testing.assert_allclose(bins["energy_ev"].iloc[:3], [16680.334, 10813.814, 7010.566])
    assert bins["time_disagreement_s"].iloc[-1] == -6
    high = read_er_energy_bins(fixtures / "er_energy_high/e_bins.xml", time_basis="utc")
    assert high["valid_from"].iloc[0] == pd.Timestamp("1998-01-01", tz="UTC")
    assert high["unix_utc"].iloc[0] == pd.Timestamp("1998-01-16", tz="UTC")
    assert high["time_disagreement_s"].iloc[0] == -15 * 86400
    with pytest.raises(ValueError, match="time_basis"):
        read_er_energy_bins(fixtures / "er_energy_low/e_bins.tab", time_basis="auto")


def test_energy_assignment_does_not_cross_command_changes_or_backfill():
    bins = pd.DataFrame({"valid_from": ["1999-03-01T00:00:00Z", "1999-03-01T00:01:00Z"],
                         "energy_step": [9, 9], "energy_ev": [337.421, 339.609]})
    counts = pd.DataFrame({"energy_step": [9]*5,
                           "start_utc": pd.to_datetime(["1999-02-28T23:59:58Z", "1999-03-01T00:00:58Z",
                                                        "1999-03-01T00:00:59Z", "1999-03-01T00:01:00Z", None]),
                           "stop_utc": pd.to_datetime(["1999-03-01T00:00:00Z", "1999-03-01T00:01:00Z",
                                                       "1999-03-01T00:01:01Z", "1999-03-01T00:01:02Z", None])})
    out = assign_er_energies(counts, bins)
    assert out["energy_valid"].tolist() == [False, True, False, True, False]
    assert out["energy_change_within_accumulation"].tolist() == [False, False, True, False, False]
    np.testing.assert_allclose(out["energy_ev"].iloc[[1, 3]], [337.421, 339.609])
    assert "energy_ev" not in counts


def test_highres_channels_follow_supplied_telemetry_version():
    bins = read_er_energy_bins(fixtures / "er_energy_high/e_bins.tab", time_basis="unix")
    counts = pd.DataFrame({"energy_step": [9, 11, 6],
                           "start_utc": ["1999-03-01T00:00:00Z"]*3,
                           "stop_utc": ["1999-03-01T00:00:02Z"]*3})
    out = assign_er_energies(counts, bins, telemetry_version=3)
    np.testing.assert_allclose(out["energy_ev"].iloc[:2], [337.421, 141.815])
    assert not out["energy_valid"].iloc[2]
    with pytest.raises(ValueError, match="telemetry_version"):
        assign_er_energies(counts, bins)


def test_energy_assignment_handles_mixed_pandas_timestamp_resolutions():
    bins = pd.DataFrame({"valid_from": pd.to_datetime(["1999-03-01"], utc=True).as_unit("s"),
                         "energy_step": [9], "energy_ev": [337.421]})
    counts = pd.DataFrame({"energy_step": [9],
                           "start_utc": pd.to_datetime(["1999-03-01T00:00:00.001Z"], utc=True).as_unit("ns"),
                           "stop_utc": pd.to_datetime(["1999-03-01T00:00:02.501Z"], utc=True).as_unit("ns")})
    assert assign_er_energies(counts, bins)["energy_valid"].all()


def er_counts():
    return pd.DataFrame({"accumulation_id": [0]*4, "kind": ["3d", "pad", "cut", "3d"],
                         "energy_step": [9]*4, "sample": [0]*4, "angle_bin": [0, 1, 2, 3],
                         "nominal_integration_s": [0.02, np.nan, 0.02, 0.02],
                         "energy_ev": [337.421]*4, "energy_valid": [True]*4,
                         "counts_lower": [32, 16, 0, 507904],
                         "counts_upper": [33, 16, 0, np.inf], "prescale_known": [True]*4})


def er_response(counts):
    out = counts[["accumulation_id", "kind", "energy_step", "sample", "angle_bin"]].copy()
    out["exposure_s"] = 0.02
    out["geometric_factor_cm2_sr"] = 0.001
    out.attrs["source"] = "synthetic response for arithmetic tests"
    return out


def test_er_response_deadtime_center_energy_and_lossy_bounds():
    counts = er_counts()
    out = calibrate_er_telemetry(counts, response=er_response(counts))
    rate = 32 / 0.02
    expected = rate / (1 - rate*3e-7) / (0.001*337.421)
    assert out["flux_lower"].iloc[0] == pytest.approx(expected)
    assert out["flux_lower"].iloc[0] < out["flux_estimate"].iloc[0] < out["flux_upper"].iloc[0]
    assert out["calibration_valid"].tolist() == [True, True, True, False]
    assert out["flux_estimate"].iloc[2] == 0
    assert out["saturation_possible"].iloc[3]
    assert out.attrs["background_subtracted"] is False
    assert "flux_estimate" not in counts


def test_er_missing_response_yields_rates_without_claiming_flux():
    out = calibrate_er_telemetry(er_counts())
    assert out["count_rate_valid"].tolist() == [True, False, True, False]
    assert out["flux_estimate"].isna().all()
    assert not out["calibration_valid"].any()


def test_er_deadtime_threshold_and_duplicate_response_are_rejected():
    counts = er_counts()
    counts.loc[0, ["counts_lower", "counts_upper"]] = [14000, 14001]
    response = er_response(counts)
    out = calibrate_er_telemetry(counts, response=response)
    assert not out["calibration_valid"].iloc[0]
    with pytest.raises(ValueError, match="unique"):
        calibrate_er_telemetry(counts, response=pd.concat([response, response]))
    response.loc[0, "exposure_s"] = 0
    with pytest.raises(ValueError, match="positive"):
        calibrate_er_telemetry(counts, response=response)


def mag_coefficients():
    return pd.DataFrame({"range": [2, 2], "valid_from": ["1999-03-01", "1999-03-02"],
                         "valid_to": ["1999-03-02", "1999-03-03"],
                         "zero_x": [1, 2], "zero_y": [2, 3], "zero_z": [3, 4],
                         "gain_x": [0.1, 0.2], "gain_y": [0.2, 0.3], "gain_z": [0.3, 0.4]})


def mag_samples():
    return pd.DataFrame({"utc": pd.to_datetime(["1999-03-01", "1999-03-02", "1999-03-03"], utc=True),
                         "range": [2]*3, "bx_signed": [11]*3, "by_signed": [12]*3,
                         "bz_signed": [13]*3, "science_valid": [True]*3})


def test_mag_signed_offsets_time_validity_and_missing_coefficients():
    out = calibrate_mag_telemetry(mag_samples(), mag_coefficients(), source="synthetic test coefficients")
    np.testing.assert_allclose(out[["Bx_SEN", "By_SEN", "Bz_SEN"]].iloc[:2], [[1, 2, 3], [1.8, 2.7, 3.6]])
    assert out["calibration_available"].tolist() == [True, True, False]
    assert out["Bx_SEN"].isna().iloc[2]
    samples = mag_samples()
    samples["calibration_mode"] = True
    assert calibrate_mag_telemetry(samples, mag_coefficients(), source="test")["Bx_SEN"].isna().all()
    with pytest.raises(ValueError, match="source"):
        calibrate_mag_telemetry(samples, mag_coefficients(), source="")


def test_mag_alignment_despin_and_overlapping_coefficients():
    calibration = mag_coefficients()
    matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for i, axis in enumerate("xyz"):
        for j, other in enumerate("xyz"):
            calibration["align_" + axis + other] = matrix[i, j]
    out = calibrate_mag_telemetry(mag_samples(), calibration, source="synthetic test coefficients")
    np.testing.assert_allclose(out[["Bx_SC", "By_SC", "Bz_SC"]].iloc[0], [-2, 1, 3])
    out["spin_phase_deg"] = 90
    despun = despin_mag(out, phase_offset_deg=0)
    np.testing.assert_allclose(despun[["Bx_SCD", "By_SCD", "Bz_SCD"]].iloc[0], [-1, -2, 3], atol=1e-14)
    calibration.loc[1, "valid_from"] = "1999-03-01T12:00:00Z"
    with pytest.raises(ValueError, match="overlap"):
        calibrate_mag_telemetry(mag_samples(), calibration, source="test")


def test_calibration_manifest_reports_missing_inputs_without_nominal_substitutes():
    products = {"mag": mag_samples(), "er_counts": er_counts()}
    result = calibrate_level0(products)
    assert result["calibration_info"]["basis"].eq("not calibrated").all()
    assert "mag_calibrated" not in result
    assert "er_flux" not in result
