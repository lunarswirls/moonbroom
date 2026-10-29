import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    bayesian_trend_fit, detrend_track, filter_altitude_brms, load_lp_csvs,
    segment_tracks_by_time, select_time_window, sel_to_spherical, write_csv,
)


def test_spherical_signs_and_altitude_preserved():
    df = pd.DataFrame({"X_SEL": [1800, 0], "Y_SEL": [0, 1800], "Z_SEL": [0, 0],
                       "Bx_SEL": [3, 3], "By_SEL": [4, 4], "Bz_SEL": [5, 5], "altitude_km": [62.6, 62.6]})
    original = df.copy(deep=True)
    out = sel_to_spherical(df)
    np.testing.assert_allclose(out[["Br", "Btheta", "Bphi"]], [[3, -5, 4], [4, -5, -3]], atol=1e-12)
    np.testing.assert_allclose(out["r_km"], 1800)
    pd.testing.assert_series_equal(out["altitude_km"], original["altitude_km"])
    pd.testing.assert_frame_equal(df, original)


def test_zero_radius_has_no_spherical_direction():
    df = pd.DataFrame({"X_SEL": [0], "Y_SEL": [0], "Z_SEL": [0],
                       "Bx_SEL": [1], "By_SEL": [2], "Bz_SEL": [3]})
    assert sel_to_spherical(df)[["Br", "Btheta", "Bphi"]].isna().all().all()


def test_filter_quantile_is_computed_after_altitude_cut():
    df = pd.DataFrame({"altitude_km": [10, 20, 30, 100], "Brms": [1, 2, 10, 1000]})
    out = filter_altitude_brms(df, brms_quantile=0.5)
    assert out["Brms"].tolist() == [1, 2]
    assert len(df) == 4


@pytest.mark.parametrize("quantile", [-0.1, 1.1, np.nan])
def test_invalid_quantile(quantile):
    with pytest.raises(ValueError):
        filter_altitude_brms(pd.DataFrame(), brms_quantile=quantile)


def test_time_window_inclusive_and_offsets():
    df = pd.DataFrame({"utc": ["1999-03-01T00:00:00Z", "1999-03-01T02:00:00+01:00", "1999-03-01T02:00:00Z"]})
    original = df.copy(deep=True)
    assert len(select_time_window(df, "1999-03-01", "1999-03-01T01:00:00Z")) == 2
    pd.testing.assert_frame_equal(df, original)
    with pytest.raises(ValueError):
        select_time_window(df, "1999-03-02", "1999-03-01")


def test_track_sort_and_gap():
    df = pd.DataFrame({"utc": ["1999-03-01T02:00:00Z", "1999-03-01T00:00:00Z", "1999-03-01T00:00:05Z"]})
    out = segment_tracks_by_time(df, gap_sec=60)
    assert out["track_id"].tolist() == [0, 0, 1]


def test_altitude_weighted_polynomial_recovers_trend():
    s = np.linspace(0, 1, 50)
    y = 2 + 3*s + 4*s*s
    trend, coefficients = bayesian_trend_fit(s, y, np.linspace(20, 80, 50), tau=1e8)
    np.testing.assert_allclose(trend, y, atol=1e-9)
    np.testing.assert_allclose(coefficients, [2, 3, 4], atol=1e-8)


def test_prior_strength_scales_with_noise():
    trend, _ = bayesian_trend_fit([0, 1], [10, 10], [0, 0], degree=0, sigma0=2, tau=1)
    np.testing.assert_allclose(trend, 10 / 3)


def test_detrending_missing_components_and_short_track():
    df = pd.DataFrame({"X_SEL": [1800, 1801, 1802], "Y_SEL": [0, 0, 0], "Z_SEL": [0, 0, 0],
                       "Br": [1, 2, 3], "Btheta": [np.nan, 2, 3], "Bphi": [1, 1, 1], "altitude_km": [30]*3})
    out = detrend_track(df, degree=2, tau=1e8)
    assert out["Btheta_trend"].isna().all()
    np.testing.assert_allclose(out["Br_crust"], 0, atol=1e-10)


def test_csv_roundtrip_preserves_utc_and_sources(tmp_path):
    df = pd.DataFrame({"utc": ["1999-03-01T00:00:00Z"], "value": [2]})
    output = write_csv(df, tmp_path / "nested" / "a_nightside.csv")
    loaded = load_lp_csvs(tmp_path, recursive=True)
    assert loaded["utc"].iloc[0] == pd.Timestamp("1999-03-01T00:00:00Z")
    assert loaded["source_file"].iloc[0] == str(output.resolve())
    assert "+00:00" in output.read_text()
