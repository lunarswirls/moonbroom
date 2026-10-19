import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    classify_shadow_alt_sza, loaded_kernels, postprocess_mag,
    process_mag_directory, select_wake_from_nightside, sel_to_sse,
)


def test_wake_models_exclude_inside_moon_and_sunward_points():
    df = pd.DataFrame({"X_SSE": [-1800, -1800, 1800, -1000],
                       "Y_SSE": [0, 2000, 0, 0], "Z_SSE": [0]*4})
    cone = select_wake_from_nightside(df, aberr_deg=0)
    cylinder = select_wake_from_nightside(df, aberr_deg=0, model="cylinder", wake_radius_km=100)
    assert cone.index.tolist() == [0, 1]
    assert cylinder.index.tolist() == [0]
    assert cylinder["wake_t_km"].iloc[0] == 1800


def test_empty_wake_schema():
    df = pd.DataFrame(columns=["X_SSE", "Y_SSE", "Z_SSE"])
    out = select_wake_from_nightside(df)
    assert out.empty
    assert "wake_t_km" in out


def test_shadow_altitude_and_sza(synthetic_kernels):
    spice = pytest.importorskip("spiceypy")
    with loaded_kernels(synthetic_kernels):
        epoch = spice.str2et("1999-03-01T00:00:00")
        sun = np.asarray(spice.spkpos("SUN", epoch, "IAU_MOON", "NONE", "MOON")[0])
        axis = sun / np.linalg.norm(sun)
        df = pd.DataFrame([1800*axis, -1800*axis, [np.nan, 0, 0]], columns=["X_SEL", "Y_SEL", "Z_SEL"])
        df["utc"] = pd.to_datetime(["1999-03-01"]*3, utc=True)
        state, altitude, sza = classify_shadow_alt_sza(df, abcorr="NONE", chunk_size=1)
        assert state.iloc[:2].tolist() == ["sunlit", "umbra"]
        assert pd.isna(state.iloc[2])
        np.testing.assert_allclose(altitude[:2], [62.6, 62.6], atol=1e-8)
        np.testing.assert_allclose(sza[:2], [0, 180], atol=1e-6)


def test_sel_sse_rotation_preserves_norm(synthetic_kernels):
    with loaded_kernels(synthetic_kernels):
        df = pd.DataFrame({"utc": ["1999-03-01"], "X_SEL": [1800], "Y_SEL": [10], "Z_SEL": [2],
                           "Bx_SEL": [3], "By_SEL": [4], "Bz_SEL": [0]})
        out = sel_to_sse(df)
        np.testing.assert_allclose(np.linalg.norm(out[["X_SSE", "Y_SSE", "Z_SSE"]], axis=1),
                                   np.linalg.norm(df[["X_SEL", "Y_SEL", "Z_SEL"]], axis=1))
        np.testing.assert_allclose(np.linalg.norm(out[["Bx_SSE", "By_SSE", "Bz_SSE"]], axis=1), [5])


def test_kernel_pool_preserves_preloaded_kernels_and_cleans_up(synthetic_kernels):
    spice = pytest.importorskip("spiceypy")
    before = spice.ktotal("ALL")
    with loaded_kernels(synthetic_kernels[:1]):
        existing = spice.ktotal("ALL")
        with loaded_kernels(synthetic_kernels):
            assert spice.ktotal("ALL") == existing + 1
        assert spice.ktotal("ALL") == existing
    assert spice.ktotal("ALL") == before


def test_overlapping_metakernel_preserves_existing_pool(tmp_path, synthetic_kernels, monkeypatch):
    spice = pytest.importorskip("spiceypy")
    monkeypatch.chdir(tmp_path)
    meta = tmp_path / "shared.tm"
    meta.write_text("KPL/MK\n\\begindata\nKERNELS_TO_LOAD = (\n" +
                    "\n".join(f"'{path.name}'" for path in synthetic_kernels) + "\n)\n\\begintext\n")
    with loaded_kernels(synthetic_kernels[:1]):
        before = spice.ktotal("ALL")
        with loaded_kernels([meta]):
            assert spice.ktotal("ALL") > before
        assert spice.ktotal("ALL") == before
        np.testing.assert_allclose(spice.bodvrd("MOON", "RADII", 3)[1], [1737.4]*3)


def test_failed_metakernel_load_does_not_leak(tmp_path, synthetic_kernels, monkeypatch):
    spice = pytest.importorskip("spiceypy")
    monkeypatch.chdir(tmp_path)
    meta = tmp_path / "broken.tm"
    meta.write_text("KPL/MK\n\\begindata\nKERNELS_TO_LOAD = ( 'geometry.tpc' 'missing.bsp' )\n\\begintext\n")
    with loaded_kernels(synthetic_kernels[:1]):
        before = spice.ktotal("ALL")
        with pytest.raises(spice.utils.exceptions.SpiceyError):
            with loaded_kernels([meta]):
                pass
        assert spice.ktotal("ALL") == before
        np.testing.assert_allclose(spice.bodvrd("MOON", "RADII", 3)[1], [1737.4]*3)


def test_directory_outputs_and_missing_sse_conversion(tmp_path, make_label, synthetic_kernels):
    names = ["Time", "Bx_SEL", "By_SEL", "Bz_SEL", "X_SEL", "Y_SEL", "Z_SEL", "B_RMS"]
    rows = [["1999-03-01T00:00:00", 3, 4, 0, 0, -1800, 0, 0.2],
            ["1999-03-01T00:00:05", 3, 4, 0, 0, 1800, 0, 0.2]]
    make_label(names, ["ASCII_Date_Time_YMD"] + ["ASCII_Real"] * 7, rows)
    with loaded_kernels(synthetic_kernels):
        manifest = process_mag_directory(tmp_path, tmp_path / "output")
    assert set(manifest["subset"]) == {"all", "nightside", "nightside_wake"}
    all_rows = pd.read_csv(tmp_path / "output" / "sample_all.csv")
    assert len(all_rows) == 2
    assert all_rows["is_wake"].sum() == 1
    assert all_rows[["X_SSE", "Y_SSE", "Z_SSE"]].notna().all().all()
    night = pd.read_csv(tmp_path / "output" / "sample_nightside.csv")
    wake = pd.read_csv(tmp_path / "output" / "sample_nightside_wake.csv")
    assert len(night) == len(wake) == 1
    assert night["wake_t_km"].notna().all()


def test_empty_window_produces_empty_annotated_frame(synthetic_kernels):
    df = pd.DataFrame({"utc": ["1999-03-01"], "X_SEL": [1800], "Y_SEL": [0], "Z_SEL": [0]})
    with loaded_kernels(synthetic_kernels):
        out = postprocess_mag(df, start="1999-03-02")
    assert out.empty
    assert "is_wake" in out
