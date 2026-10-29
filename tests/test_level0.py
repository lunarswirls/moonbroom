from pathlib import Path
import struct

import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    correlate_frame_times, decode_digital_subcom, decode_er_accumulation, decode_level0,
    decode_mag_block, decompress_er_counts, er_angular_bins, er_energy_bins,
    er_frame_layout, read_merged_telemetry,
    read_sunpulse, write_level0_tables,
)
from moonbroom.er_telemetry import er_accumulation_clock


def mag_block(frame=0, calibration=False, gain=2):
    values = [0, 1, 2047, 2048, 4094, 4095]
    pair = sum(value << (12*i) for i, value in enumerate(values)).to_bytes(9, "little")
    return bytes([(frame << 4) | (8 if calibration else 0) | gain]) + pair*9


def digital_subcom(version=3, phase=1024, period=5000):
    data = bytearray(32)
    data[0:3] = bytes.fromhex("563412")
    data[5] = 0xa5
    data[6] = version << 4 | 0b1011
    data[7:10] = bytes([0x23, 0x81, 0x45])
    data[22], data[23] = phase & 255, (phase >> 8) | 0xf0
    data[24], data[25] = period & 255, (period >> 8) | 0x80
    data[26:28] = (48000).to_bytes(2, "little")
    data[28], data[29] = 0xc0 | 27, 100
    return bytes(data)


def merged_record(vcdu, *, code=255, data=None, day=60, milliseconds=None):
    record = bytearray(472)
    record[:4] = bytes.fromhex("1acffc1d")
    record[4:6] = bytes.fromhex("66c5")
    record[6:9] = (vcdu % 2**24).to_bytes(3, "big")
    frame = (vcdu - 1) % 16
    payload = bytearray(168)
    payload[0] = code
    payload[1:3] = digital_subcom()[frame*2:frame*2 + 2]
    payload[3:85] = mag_block(frame)
    if data is not None:
        payload[85:168] = data
    record[286:454] = payload
    milliseconds = (vcdu*2000) % 86400000 if milliseconds is None else milliseconds
    record[456:462] = struct.pack(">HI", day, milliseconds)
    record[462:470] = struct.pack(">ff", -159.6, 15.6)
    record[470:472] = b"\x00\x11"
    return bytes(record)


def test_mag_reference_bit_pattern_and_diagnostic_flag():
    df = decode_mag_block(mag_block(15, calibration=True, gain=7))
    assert df[["bx", "by", "bz"]].iloc[:2].values.tolist() == [[0, 1, 2047], [2048, 4094, 4095]]
    assert df[["bx_signed", "by_signed", "bz_signed"]].iloc[:2].values.tolist() == [[-2048, -2047, -1], [0, 2046, 2047]]
    assert df["range"].eq(7).all()
    assert df["mag_frame"].eq(15).all()
    assert df["calibration_mode"].all()
    with pytest.raises(ValueError):
        decode_mag_block(bytes(81))


def test_compression_published_transition_values_and_prescale_bounds():
    codes = [0, 15, 16, 31, 32, 47, 48, 128, 240, 255, np.nan]
    df = decompress_er_counts(codes)
    np.testing.assert_allclose(df["counts_lower"].iloc[:-1], [0, 15, 16, 31, 32, 62, 64, 2048, 262144, 507904])
    assert df["counts_upper"].iloc[4] == 33
    assert np.isinf(df["counts_upper"].iloc[-2])
    scaled = decompress_er_counts([32], prescale=8)
    assert scaled["counts_lower"].iloc[0] == 256
    assert scaled["counts_upper"].iloc[0] == 271
    assert df["counts_lower"].isna().iloc[-1]


@pytest.mark.parametrize("code", [-1, 256, 3.5, np.inf])
def test_invalid_compression_code_rejected(code):
    with pytest.raises(ValueError):
        decompress_er_counts([code])


def test_digital_bitfields_and_partial_words():
    df = decode_digital_subcom(digital_subcom())
    assert df["command"].iloc[0] == 0x123456
    assert df["reset_count"].iloc[0] == 10
    assert df["error_count"].iloc[0] == 5
    assert df["telemetry_version"].iloc[0] == 3
    assert df["imon_counts"].iloc[0] == 0x123
    assert df["sweep_monitor_counts"].iloc[0] == 0x458
    assert df["phase"].iloc[0] == 1024
    assert df["period_s"].iloc[0] == 5.12
    partial = decode_digital_subcom(digital_subcom(), word_mask=65535 ^ (1 << 12))
    assert np.isnan(partial["period_s"].iloc[0])
    assert partial["phase"].iloc[0] == 1024
    assert not partial["complete"].iloc[0]


@pytest.mark.parametrize("version, angular_bytes", [(1, 1664), (2, 1664), (3, 1536), (4, 1536), (5, 1536)])
def test_real_time_layout_byte_coverage(version, angular_bytes):
    layout = er_frame_layout(version)
    assert layout.loc[layout["kind"].eq("angular"), "length"].sum() == angular_bytes
    for _, frame in layout.groupby("frame_type"):
        assigned = []
        for row in frame.itertuples():
            assigned += list(range(row.offset, row.offset + row.length))
        assert assigned == list(range(83))
    volume = layout.loc[layout["kind"].eq("3d")]
    assert volume["length"].sum() == 1320
    assert volume["stream_offset"].iloc[0] == 0


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_real_time_counts_follow_official_boundaries(version):
    packets = {i: bytes([i])*83 for i in range(36)}
    result = decode_er_accumulation(packets, version=version)
    counts = result["counts"]
    volume = counts.loc[counts["kind"].eq("3d")]
    assert volume["code"].iloc[:6].tolist() == [1, 1, 2, 2, 2, 3]
    assert volume["code"].iloc[-1] == (33 if version >= 3 else 35)
    fast = 6 if version == 1 else 11
    assert len(counts.loc[counts["energy_step"].eq(fast) & counts["kind"].eq("pad")]) == 32*16
    assert result["summary"]["complete"]
    assert len(result["angles"]) == (64 if version >= 3 else 0)
    assert result["summary"]["telemetry_layout"] == min(version, 3)
    assert result["summary"]["layout_basis"] == ("empirical format-3 compatibility" if version >= 4 else "MAGERFMT")


def test_prescale_applies_only_to_long_pitch_angle_accumulations():
    packets = {i: bytearray([i]*83) for i in range(36)}
    packets[0][81] = 255
    result = decode_er_accumulation({key: bytes(value) for key, value in packets.items()}, version=3)
    df = result["counts"]
    assert df.loc[df["energy_step"].eq(6) & df["kind"].eq("pad"), "prescale"].eq(8).all()
    assert df.loc[df["energy_step"].eq(9) & df["kind"].eq("pad"), "prescale"].eq(1).all()
    assert df.loc[df["kind"].isin(["3d", "cut"]), "prescale"].eq(1).all()


def test_missing_packet_masks_counts_and_does_not_assume_prescale():
    packets = {i: bytes([i])*83 for i in range(1, 36)}
    result = decode_er_accumulation(packets, version=3)
    assert not result["summary"]["complete"]
    df = result["counts"]
    assert df.loc[df["energy_step"].eq(0) & df["sample"].eq(0) & df["kind"].eq("pad"), "code"].isna().all()
    assert df.loc[df["energy_step"].eq(6) & df["kind"].eq("pad"), "counts_lower"].isna().all()
    assert df.loc[df["energy_step"].eq(9) & df["kind"].eq("pad"), "counts_lower"].notna().all()


def test_burst_contiguous_reassembly_and_direction_byte_order():
    stream = bytearray(1640)
    stream[:4] = bytes([0, 255, 32, 32])
    stream[4:1324] = bytes([32])*1320
    stream[1324:1564] = bytes([33])*240
    stream[1564:1568] = bytes([64, 128, 32, 192])
    stream[1568:] = bytes(range(72))
    packets = {36+i: bytes(stream[i*82:(i+1)*82]) for i in range(20)}
    result = decode_er_accumulation(packets, version=3, mode="burst")
    assert len(result["counts"]) == 1560
    assert result["counts"]["prescale"].eq(1).all()
    assert result["angles"]["theta_deg"].tolist() == [90, 45]
    assert result["angles"]["phi_deg"].tolist() == [180, 270]
    assert result["control"].tolist() == list(range(72))
    assert result["summary"]["total_counts_lower"] == 512


def test_er_timing_uses_dimensionally_correct_phase_and_previous_major():
    context = {"major_clock_s": 64, "period_s": 5.12, "phase": 1024}
    previous = {**context, "major_clock_s": 32}
    clock, selected = er_accumulation_clock([0, 0, 0, 0], context)
    assert clock == pytest.approx(65.2824)
    clock, selected = er_accumulation_clock([0, 0, 0, 0], context, first_clock_s=65, previous=previous)
    assert clock == pytest.approx(33.2824)
    assert selected is previous


def test_er_off_status_prevents_stale_phase_timing():
    data = bytearray(digital_subcom())
    data[28] &= 127
    context = decode_digital_subcom(bytes(data)).iloc[0].to_dict()
    context["major_clock_s"] = 64
    assert not context["er_parameters_valid"]
    clock, selected = er_accumulation_clock([0, 0, 0, 0], context)
    assert np.isnan(clock)
    assert selected is None


def test_official_angular_map_and_nominal_sweep():
    bins = er_angular_bins()
    assert len(bins) == 88
    assert bins.groupby("anode_bin")["phase_bins"].sum().eq(8).all()
    assert bins.loc[0, "phase_bins"] == 4
    assert bins.loc[6, "phase_bins"] == 1
    energy = er_energy_bins(48000, 27)
    assert energy["energy_high_ev_nominal"].iloc[0] == pytest.approx(20008.667, rel=1e-3)
    assert energy["energy_low_ev_nominal"].iloc[-1] == pytest.approx(30.03, rel=0.1)


def test_merged_endianness_ert_quality_and_corruption(tmp_path):
    path = tmp_path / "M9906003.B"
    path.write_bytes(merged_record(17))
    df = read_merged_telemetry(path)
    assert df["vcdu"].iloc[0] == 17
    assert df["ert"].iloc[0] == pd.Timestamp("1999-03-01T00:00:34Z")
    assert df["agc"].iloc[0] == pytest.approx(-159.6)
    assert df["selected_stream"].iloc[0] == 1
    assert len(df["mag_er"].iloc[0]) == 168
    path.write_bytes(b"\x00" + path.read_bytes()[1:])
    with pytest.raises(ValueError, match="header"):
        read_merged_telemetry(path)
    assert not read_merged_telemetry(path, strict=False)["header_valid"].iloc[0]
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="Incomplete"):
        read_merged_telemetry(path)


@pytest.mark.parametrize('tail_bytes', [10, 44, 471])
def test_labelled_partial_final_record_requires_opt_in_and_retains_tail(tmp_path, tail_bytes):
    path = tmp_path / 'M9906003.B'
    tail = merged_record(19)[:tail_bytes]
    original = merged_record(17) + merged_record(18) + tail
    path.write_bytes(original)
    label = path.with_suffix('.LBL')
    label.write_text('RECORD_TYPE = FIXED_LENGTH\nRECORD_BYTES = 472\n'
                     'FILE_RECORDS = 3\n^TABLE = "M9906003.B"\n')
    with pytest.raises(ValueError, match='Incomplete'):
        read_merged_telemetry(label)
    with pytest.raises(ValueError, match='FILE_RECORDS'):
        read_merged_telemetry(label, strict=False)
    full = read_merged_telemetry(label, allow_incomplete_final_record=True)
    assert full['vcdu'].tolist() == [17, 18]
    assert full['header_valid'].all()
    assert full.attrs['label_records'] == 3
    assert full.attrs['complete_records'] == 2
    assert full.attrs['incomplete_final_record']
    assert full.attrs['trailing_hex'] == tail.hex()
    assert full.attrs['trailing_byte_offset'] == 944
    assert full.attrs['missing_bytes'] == 472 - tail_bytes
    selected = read_merged_telemetry(label, allow_incomplete_final_record=True, record_slice=(1, 2))
    pd.testing.assert_frame_equal(selected, full.iloc[1:2].reset_index(drop=True))
    assert path.read_bytes() == original
    # Tail recovery never disables strict checks on the complete science frames.
    path.write_bytes(b'\x00' + original[1:])
    with pytest.raises(ValueError, match='header'):
        read_merged_telemetry(label, allow_incomplete_final_record=True)


@pytest.mark.parametrize('case', ['missing_whole_frame', 'extra_tail', 'garbage', 'no_label', 'short_header'])
def test_partial_record_opt_in_rejects_other_structural_corruption(tmp_path, case):
    path = tmp_path / 'M9906003.B'
    tail = merged_record(19)[:44]
    if case == 'garbage':
        tail = bytes(44)
    elif case == 'short_header':
        tail = tail[:9]
    path.write_bytes(merged_record(17) + merged_record(18) + tail)
    if case != 'no_label':
        count = {'missing_whole_frame': 4, 'extra_tail': 2}.get(case, 3)
        path.with_suffix('.LBL').write_text('RECORD_TYPE = FIXED_LENGTH\nRECORD_BYTES = 472\n'
                                          f'FILE_RECORDS = {count}\n^TABLE = "M9906003.B"\n')
    with pytest.raises(ValueError, match='incomplete final merged record'):
        read_merged_telemetry(path, allow_incomplete_final_record=True)


def test_counter_and_calendar_rollover_without_fake_utc(tmp_path):
    path = tmp_path / "M9936523.B"
    path.write_bytes(merged_record(2**24-1, day=365, milliseconds=86398000)
                     + merged_record(0, day=1, milliseconds=0))
    out = correlate_frame_times(read_merged_telemetry(path))
    assert out["vcdu_unwrapped"].tolist() == [2**24-1, 2**24]
    assert out["ert"].iloc[-1] == pd.Timestamp("2000-01-01T00:00:00Z")
    assert out["frame_utc"].isna().all()


def test_ert_leap_second_is_retained_without_shifting_to_midnight(tmp_path):
    path = tmp_path / "M9836523.B"
    path.write_bytes(merged_record(17, day=365, milliseconds=86400500))
    out = read_merged_telemetry(path)
    assert out["ert_iso"].iloc[0] == "1998-12-31T23:59:60.500Z"
    assert out["ert_leap_second"].iloc[0]
    assert pd.isna(out["ert"].iloc[0])


def test_clock_correlation_and_supplied_light_time(tmp_path):
    path = tmp_path / "M9906003.B"
    path.write_bytes(b"".join(merged_record(vcdu) for vcdu in range(17, 21)))
    frames = read_merged_telemetry(path)
    correlation = pd.DataFrame({"vcdu_unwrapped": [17, 19], "utc": ["1999-03-01T01:00:00Z", "1999-03-01T01:00:04Z"]})
    out = correlate_frame_times(frames, clock_correlation=correlation)
    assert out["frame_utc"].iloc[1] == pd.Timestamp("1999-03-01T01:00:02Z")
    assert pd.isna(out["frame_utc"].iloc[3])
    out = correlate_frame_times(frames, light_time_s=1.3)
    assert (out["ert"] - out["frame_utc"]).dt.total_seconds().eq(1.3).all()


def test_cross_file_er_reassembly_and_boundary_duplicates(tmp_path):
    records = []
    for vcdu in range(17, 97):
        kind = vcdu-50
        records.append(merged_record(vcdu, code=kind if 0 <= kind < 36 else 255,
                                      data=bytes([kind])*83 if 0 <= kind < 36 else None))
    first, second = tmp_path / "first.b", tmp_path / "second.b"
    first.write_bytes(b"".join(records[:50]))
    second.write_bytes(b"".join(records[49:]))
    result = decode_level0([first, second], year=1999, light_time_s=1.3)
    assert result["frames"].attrs["duplicate_records_removed"] == 1
    assert result["er_accumulations"]["complete"].tolist() == [True]
    assert result["er_accumulations"]["timing_valid"].all()
    assert len(result["mag"]) == 80*18
    assert len(result["er_counts"]) == 2856
    assert result["mag"]["utc"].notna().all()
    assert result["er_counts"]["stop_utc"].notna().all()
    assert result["mag"]["clock_s"].iloc[0] == pytest.approx(34 - 4 + 1/36)
    assert result["memory_dumps"]["payload"].map(len).eq(81).all()


def test_full_burst_marker_timing_and_half_second_code(tmp_path):
    records = []
    marker = bytearray(82)
    marker[:5] = struct.pack("<HHB", 100, 101, 7)
    marker[50:] = digital_subcom()
    for vcdu in range(17, 33):
        record = bytearray(merged_record(vcdu))
        if vcdu == 25:
            record[286] = 57 | 64
            record[371] = 0
            record[372:454] = marker
        elif vcdu == 26:
            record[286] = 56 | 128
            record[289:371] = mag_block(0)
            record[371] = 56 | 64
            record[372:454] = mag_block(0)
        records.append(record)
    path = tmp_path / "synthetic.b"
    path.write_bytes(b"".join(records))
    result = decode_level0(path, year=1999, light_time_s=1.3)
    burst = result["mag"].loc[result["mag"]["mode"].eq("burst")]
    assert len(burst) == 36
    np.testing.assert_allclose(burst["clock_s"].iloc[[0, 18]], [0, 1])
    assert burst["utc"].notna().all()
    assert result["burst_markers"]["age_major_frames"].iloc[0] == 1
    assert len(result["control_tables"]) == 3


def test_public_archive_regression_excerpt():
    path = Path(__file__).parent / "fixtures" / "M9906003_excerpt.LBL"
    result = decode_level0(path)
    assert len(result["frames"]) == 128
    mag = result["mag"].loc[result["mag"]["mode"].eq("realtime")]
    assert len(mag) == 128*18
    np.testing.assert_array_equal(mag[["bx", "by", "bz"]].iloc[:2], [[2491, 1390, 2061], [2401, 1351, 2058]])
    np.testing.assert_array_equal(mag[["bx", "by", "bz"]].iloc[32*18], [2857, 2117, 2100])
    assert mag["mag_frame_matches"].all()
    complete = result["er_accumulations"].loc[result["er_accumulations"]["complete"]]
    assert complete["first_frame_id"].tolist() == [22, 62]
    first = complete.iloc[0]
    assert first["nhs"] == 11
    assert first["telemetry_version"] == 3
    selected = result["er_counts"].loc[
        result["er_counts"]["accumulation_id"].eq(first["accumulation_id"])
        & result["er_counts"]["energy_step"].eq(9)
        & result["er_counts"]["sample"].eq(0)
        & result["er_counts"]["kind"].eq("pad")]
    assert selected["code"].iloc[:2].tolist() == [8, 42]
    assert selected["counts_lower"].iloc[:2].tolist() == [8, 52]
    assert result["mag"]["utc"].isna().all()


def test_sunpulse_previous_major_fade_and_estimate_uncertainty(tmp_path):
    path = tmp_path / "sun.b"
    path.write_bytes(b"".join(counter.to_bytes(3, "big") + struct.pack(">HHB", pulse, error, source)
                             for counter, pulse, error, source in [(16, 50400, 0, 0), (18, 1800, 9, 1), (20, 0xfade, 0, 0)]))
    out = read_sunpulse(path)
    assert out["pulse_clock_s"].iloc[:2].tolist() == [28, 33]
    assert out["estimated"].iloc[1]
    assert out["uncertainty_s"].iloc[1] == 0.005
    assert np.isnan(out["pulse_clock_s"].iloc[2])


def test_burst_er_pipeline_reassembles_all_twenty_packets(tmp_path):
    records = []
    marker = bytearray(82)
    marker[:5] = struct.pack("<HHB", 100, 101, 7)
    marker[50:] = digital_subcom()
    for vcdu in range(17, 65):
        record = bytearray(merged_record(vcdu))
        if vcdu == 25:
            record[286], record[371] = 57 | 64, 0
            record[372:454] = marker
        elif 26 <= vcdu <= 45:
            kind = vcdu + 10
            record[286], record[371] = kind, vcdu - 25
            record[372:454] = bytes([kind])*82
        records.append(record)
    path = tmp_path / "burst.b"
    path.write_bytes(b"".join(records))
    result = decode_level0(path, year=1999, light_time_s=1.3)
    assert result["er_accumulations"]["complete"].tolist() == [True]
    assert result["er_accumulations"]["timing_valid"].all()
    assert len(result["er_counts"]) == 1560
    assert result["er_counts"]["stop_utc"].notna().all()
    assert len(result["control_tables"]) == 4


def test_missing_burst_marker_never_uses_transmit_time(tmp_path):
    record = bytearray(merged_record(20))
    record[286], record[371] = 56, 0
    record[372:454] = mag_block(0)
    path = tmp_path / "missing_marker.b"
    path.write_bytes(record)
    result = decode_level0(path, year=1999, light_time_s=1.3)
    burst = result["mag"].loc[result["mag"]["mode"].eq("burst")]
    assert burst["clock_s"].isna().all()
    assert burst["utc"].isna().all()
    assert "missing_burst_mag_timing" in result["issues"]["code"].tolist()


def test_conflicting_overlapping_records_are_not_silently_chosen(tmp_path):
    path = tmp_path / "conflicting.b"
    path.write_bytes(merged_record(17) + merged_record(17, data=bytes([1])*83))
    with pytest.raises(ValueError, match="overlapping"):
        decode_level0(path, year=1999)


def test_permissive_trailing_bytes_remain_visible_in_manifest(tmp_path):
    path = tmp_path / "trailing.b"
    path.write_bytes(merged_record(17) + b"abc")
    result = decode_level0(path, year=1999, strict=False)
    assert result["input_files"]["trailing_bytes"].tolist() == [3]


def test_level0_table_export_encodes_raw_bytes(tmp_path):
    products = {"frames": pd.DataFrame({"payload": [b"\x1a\xcf"], "utc": pd.to_datetime(["1999-03-01"], utc=True)})}
    manifest = write_level0_tables(products, tmp_path)
    assert manifest["rows"].tolist() == [1]
    assert pd.read_csv(tmp_path / "frames.csv")["payload"].iloc[0] == "1acf"
    assert products["frames"]["payload"].iloc[0] == b"\x1a\xcf"


def test_archive_sunpulse_independently_validates_subcom_phase():
    root = Path(__file__).parent / "fixtures"
    result = decode_level0(root / "M9906003_excerpt.LBL")
    pulses = read_sunpulse(root / "S9906003_excerpt.LBL")
    clock = pulses.loc[pulses["timing_valid"] & ~pulses["estimated"], "pulse_clock_s"].to_numpy()
    subcom = result["digital_subcom"].loc[result["digital_subcom"]["complete"]]
    times = subcom["major_clock_s"].to_numpy()
    indices = np.searchsorted(clock, times, side="right") - 1
    valid = indices >= 0
    phase = ((times[valid] - clock[indices[valid]]) / subcom["period_s"].to_numpy()[valid]) % 1
    error = ((phase - subcom["phase"].to_numpy()[valid]/4096 + 0.5) % 1 - 0.5) * 360
    assert len(error) >= 5
    assert np.median(np.abs(error)) < 1.0


def test_batched_mag_path_matches_complete_decoder_with_mixed_playback(tmp_path):
    path = tmp_path / "M9906003.B"
    path.write_bytes(b"".join(merged_record(index, code=code) for index, code in
                              zip(range(17, 21), [255, 0, 164, 36])))
    complete = decode_level0(path, light_time_s=1.3)["mag"]
    expected = complete.loc[complete["mode"].eq("realtime")].reset_index(drop=True)
    fast = decode_level0(path, light_time_s=1.3, mag_only=True)["mag"]
    assert len(fast) == 54
    for column in fast:
        if column != "spin_phase_deg":
            pd.testing.assert_series_equal(fast[column], expected[column], check_dtype=False)
    assert fast["spin_phase_deg"].isna().all()


def test_batched_mag_path_matches_original_pds_sample_grid():
    path = Path(__file__).parent / "fixtures" / "M9906003_excerpt.LBL"
    complete = decode_level0(path, light_time_s=1.3)["mag"]
    expected = complete.loc[complete["mode"].eq("realtime")].reset_index(drop=True)
    fast = decode_level0(path, light_time_s=1.3, mag_only=True)["mag"]
    for column in fast:
        if column != "spin_phase_deg":
            pd.testing.assert_series_equal(fast[column], expected[column], check_dtype=False)


def test_mag_overlap_policy_preserves_provenance_and_rejects_payload_conflicts(tmp_path):
    first = tmp_path / "M9906000.B"
    second = tmp_path / "M9906001.B"
    a = merged_record(17)
    overlap = bytearray(a)
    overlap[10] = 1
    overlap[461] += 10
    first.write_bytes(a)
    second.write_bytes(bytes(overlap) + merged_record(18))
    with pytest.raises(ValueError, match="ordered and unique"):
        decode_level0([first, second], light_time_s=1.3, mag_only=True)
    with pytest.raises(ValueError, match="disagree"):
        decode_level0([first, second], light_time_s=1.3, mag_only=True, overlap_policy="identical_payload")
    with pytest.raises(ValueError, match="requires mag_only"):
        decode_level0([first, second], light_time_s=1.3, overlap_policy="identical_mag")
    result = decode_level0([first, second], light_time_s=1.3, mag_only=True, overlap_policy="identical_mag")
    assert len(result["mag"]) == 36
    assert len(result["overlap_records"]) == 1
    assert result["overlap_records"]["ert_difference_s"].iloc[0] == pytest.approx(.01)
    assert result["overlap_records"]["kept_source_file"].iloc[0] == str(first)
    assert result["overlap_records"]["removed_source_file"].iloc[0] == str(second)
    full = decode_level0([first, second], light_time_s=1.3, overlap_policy="identical_mager", telemetry_version=3)
    assert len(full['mag']) == 36
    assert full['overlap_records']['other_payload_differs'].all()
    assert full['overlap_records']['removed_record_sha256'].str.len().eq(64).all()
    overlap[290] ^= 1
    second.write_bytes(bytes(overlap) + merged_record(18))
    with pytest.raises(ValueError, match="disagree"):
        decode_level0([first, second], light_time_s=1.3, mag_only=True, overlap_policy="identical_mag")
    with pytest.raises(ValueError, match="disagree"):
        decode_level0([first, second], light_time_s=1.3, overlap_policy="identical_mager", telemetry_version=3)


def test_non_strict_sunpulse_retains_bad_counter_with_invalid_timing(tmp_path):
    path = tmp_path / "S9906000.B"
    records = [(18, 3600, 0, 0), (19, 0xfade, 0, 1), (22, 10800, 0, 0)]
    path.write_bytes(b"".join(counter.to_bytes(3, "big") + struct.pack(">HHB", pulse, error, flag)
                              for counter, pulse, error, flag in records))
    with pytest.raises(ValueError, match="even"):
        read_sunpulse(path)
    out = read_sunpulse(path, strict=False)
    assert out["counter_valid"].tolist() == [True, False, True]
    assert not out["timing_valid"].iloc[1]
    assert np.isnan(out["pulse_clock_s"].iloc[1])
    assert out["source_flag"].iloc[1] == 1
    assert out["record_index"].tolist() == [0, 1, 2]
    assert out["source_file"].eq(str(path)).all()
