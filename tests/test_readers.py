import numpy as np
import pandas as pd
import pytest
import struct

from moonbroom import read_er, read_lper_txt, read_lpmag_l1b_tab, read_mag, read_mag_l2, read_pds4


def test_mag_masks_archive_flags_and_preserves_rms(make_label):
    names = ["Time", "Bx_SEL", "By_SEL", "Bz_SEL", "B_RMS"]
    rows = [["1999-03-01T00:00:00", 3, 4, 0, 0.2],
            ["1999-03-01T00:00:05", 999.999, 4, 0, 0.3]]
    label = make_label(names, ["ASCII_Date_Time_YMD"] + ["ASCII_Real"] * 4,
                       rows, constants={"Bx_SEL": {"missing_constant": 999.999}})
    df = read_mag(label)
    assert df["Bmag_SEL"].iloc[0] == 5
    assert np.isnan(df["Bmag_SEL"].iloc[1])
    assert df["Brms"].tolist() == [0.2, 0.3]
    assert str(df["utc"].dt.tz) == "UTC"
    assert df.attrs["units"]["Bx_SEL"] == "nT"
    pd.testing.assert_frame_equal(read_pds4(label, table="test_table"), read_pds4(label))


def test_er_flags_are_exact_not_near_zero(make_label):
    rows = [["1999-03-01T00:00:00", 9.999e-99, 1e-98],
            ["1999-03-01T00:00:05", -9.99e-99, 1.2]]
    label = make_label(["Time", "eflux1", "eflux2"], ["ASCII_Date_Time_YMD", "ASCII_Real", "ASCII_Real"],
                       rows, constants={"eflux1": {"missing_constant": 9.999e-99,
                                                    "error_constant": -9.99e-99}})
    df = read_er(label)
    assert df["eflux1"].isna().all()
    assert df["eflux2"].iloc[0] == 1e-98


def test_legacy_mag_datetime_and_column_count(tmp_path):
    path = tmp_path / "ma990301.tab"
    path.write_text("1999-03-01T00:00:02.5 60.000029 3 4 0 3 4 0 0.2 1800 0 0 1800 0 0 0\n")
    df = read_lpmag_l1b_tab(path)
    assert df["Bmag_SEL"].iloc[0] == 5
    assert df["Brms"].iloc[0] == 0.2
    assert str(df["utc"].dt.tz) == "UTC"
    path.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="16"):
        read_lpmag_l1b_tab(path)


def test_legacy_er_mask_and_utc(tmp_path):
    path = tmp_path / "er.tab"
    path.write_text("1999-03-01T00:00:00 9.999E-99 -9.99E-99\n")
    df = read_lper_txt(path)
    assert df[["f1_340ev", "f2_140ev"]].isna().all().all()
    assert str(df["utc"].dt.tz) == "UTC"


def test_l2_midnight_rollover_no_sse_or_fake_rms(make_label):
    names = ["Hour", "Bx_SG", "By_SG", "Bz_SG", "Latitude", "Longitude", "Altitude"]
    rows = [[23.9, 3, 4, 0, 0, 0, 30], [0.1, 3, 4, 0, 0, 90, 30]]
    label = make_label(names, ["ASCII_Real"] * len(names), rows)
    df = read_mag_l2(label, radius_km=1700)
    assert df["utc"].iloc[1] == pd.Timestamp("1999-03-02T00:06:00Z")
    assert df["X_SEL"].iloc[0] == 1730
    assert df["Bmag_SEL"].iloc[0] == 5
    assert "Brms" not in df
    assert "X_SSE" not in df
    assert df["altitude_km"].tolist() == [30, 30]


def test_l2_extended_hours(make_label):
    names = ["Hour", "Bx_SG", "By_SG", "Bz_SG", "Latitude", "Longitude", "Altitude"]
    label = make_label(names, ["ASCII_Real"] * len(names), [[25, 1, 0, 0, 0, 0, 30]])
    assert read_mag_l2(label)["utc"].iloc[0] == pd.Timestamp("1999-03-02T01:00:00Z")


def test_binary_repeated_er_fields_scaling_and_mask(tmp_path):
    (tmp_path / "flux.bin").write_bytes(struct.pack(">fff", 2, -999, 4))
    label = tmp_path / "flux.xml"
    label.write_text('''<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
      <File_Area_Observational><File><file_name>flux.bin</file_name></File>
      <Table_Binary><name>flux</name><offset unit="byte">0</offset><records>1</records>
      <Record_Binary><fields>0</fields><groups>1</groups><record_length unit="byte">12</record_length>
      <Group_Field_Binary><name>energy</name><repetitions>3</repetitions>
      <group_location unit="byte">1</group_location><group_length unit="byte">12</group_length>
      <fields>1</fields><groups>0</groups><Field_Binary><name>flux</name>
      <field_location unit="byte">1</field_location><data_type>IEEE754MSBSingle</data_type>
      <field_length unit="byte">4</field_length><scaling_factor>2</scaling_factor><value_offset>1</value_offset>
      <Special_Constants><missing_constant>-999</missing_constant></Special_Constants>
      </Field_Binary></Group_Field_Binary></Record_Binary></Table_Binary>
      </File_Area_Observational></Product_Observational>''')
    df = read_er(label)
    assert df.columns.tolist() == ["flux[0]", "flux[1]", "flux[2]"]
    assert df["flux[0]"].iloc[0] == 5
    assert np.isnan(df["flux[1]"].iloc[0])
    assert df["flux[2]"].iloc[0] == 9
    stored = read_pds4(label, scale=False)
    assert stored["flux[0]"].iloc[0] == 2
