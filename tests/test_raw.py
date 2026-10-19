import struct

import numpy as np
import pandas as pd
import pytest

from moonbroom import calibrate_er, calibrate_mag, read_raw_records


def test_explicit_binary_endianness_and_truncation(tmp_path):
    path = tmp_path / "records.bin"
    path.write_bytes(struct.pack(">BHHH", 2, 2049, 2050, 2051))
    fields = pd.DataFrame([("range", "u1", 0), ("bx", ">u2", 1), ("by", ">u2", 3), ("bz", ">u2", 5)],
                          columns=["name", "dtype", "offset_bytes"])
    out = read_raw_records(path, fields, record_bytes=7)
    assert out.iloc[0].tolist() == [2, 2049, 2050, 2051]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        read_raw_records(path, fields, record_bytes=7)


def test_mag_supplied_gains_offsets_and_range_changes():
    raw = pd.DataFrame({"range": [0, 0, 1, 1], "bx": [2050]*4, "by": [2052]*4, "bz": [2054]*4})
    calibration = pd.DataFrame({"range": [0, 1], "gain_x": [0.5, 1], "gain_y": [0.5, 1],
                                "gain_z": [0.5, 1], "zero_x": [2048]*2, "zero_y": [2048]*2, "zero_z": [2048]*2})
    out = calibrate_mag(raw, calibration)
    np.testing.assert_allclose(out.loc[0, ["Bx_SEN", "By_SEN", "Bz_SEN"]].to_numpy(float), [1, 2, 3])
    assert out.loc[2, ["Bx_SEN", "By_SEN", "Bz_SEN"]].isna().all()
    assert out["Bx_SEN"].iloc[3] == 2
    assert "Bx_SEN" not in raw
    with pytest.raises(ValueError, match="Missing calibration"):
        calibrate_mag(raw, calibration.iloc[:1])


def test_er_supplied_response_and_invalid_live_time():
    raw = pd.DataFrame({"counts": [100, -1], "accumulation_s": [2, 2]})
    channels = pd.DataFrame([("counts", "flux", 0.1, 10, 0.5)],
                            columns=["count_col", "flux_col", "geometric_factor_cm2_sr", "energy_width_ev", "efficiency"])
    out = calibrate_er(raw, channels)
    assert out["flux"].iloc[0] == 100
    assert np.isnan(out["flux"].iloc[1])
    raw["accumulation_s"] = 0
    with pytest.raises(ValueError):
        calibrate_er(raw, channels)
