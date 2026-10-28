from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    decode_level0, loaded_kernels, read_merged_telemetry, read_sunpulse,
    reconstruct_frame_times, reconstruct_product_times, reconstruct_spin_phase,
)
from moonbroom.timing import ephemeris_to_utc


fixtures = Path(__file__).parent / "fixtures"


@pytest.fixture
def timing_kernels(tmp_path):
    spice = pytest.importorskip("spiceypy")
    lsk = fixtures / "naif0012.tls"
    with loaded_kernels([lsk]):
        first = spice.str2et("1998-12-31T23:59:00Z")
        last = spice.str2et("1999-03-02T00:00:00Z")
    spk = tmp_path / "synthetic_trajectory.bsp"
    handle = spice.spkopn(str(spk), "synthetic timing test", 0)
    epochs = np.array([first, last])
    state = np.zeros((2, 6))
    spice.spkw09(handle, 399, 0, "J2000", first, last, "synthetic earth", 1, 2, state, epochs)
    state[:, 0] = spice.clight() * 1.5
    spice.spkw09(handle, -25, 0, "J2000", first, last, "synthetic spacecraft", 1, 2, state, epochs)
    spice.spkcls(handle)
    return [lsk, spk]


def test_ert_to_frame_pulse_includes_light_time_and_two_second_system_delay(timing_kernels):
    frames = read_merged_telemetry(fixtures / "M9906003_excerpt.LBL")
    out = reconstruct_frame_times(frames, timing_kernels, receiver="EARTH")
    np.testing.assert_allclose(out["light_time_s"], 1.5, atol=1e-8)
    np.testing.assert_allclose((out["ert"] - out["frame_utc"]).dt.total_seconds(), 3.5, atol=1e-6)
    assert "frame_utc" not in frames
    with pytest.raises(ValueError, match="receiver"):
        reconstruct_frame_times(frames, timing_kernels, receiver=None)


def test_duplicate_ert_pair_is_flagged_without_stopping_real_telemetry(timing_kernels):
    frames = read_merged_telemetry(fixtures / "M9906003_excerpt.LBL").iloc[:8].copy()
    frames.loc[2, "ert_iso"] = frames.loc[3, "ert_iso"]
    out = reconstruct_frame_times(frames, timing_kernels, receiver="EARTH")
    assert out["ert_anchor_rejected"].tolist() == [False, False, True, True, False, False, False, False]
    assert out["frame_et"].isna().sum() == 2


def test_leap_second_preserved_and_interpolated_in_ephemeris_seconds(timing_kernels):
    spice = pytest.importorskip("spiceypy")
    with loaded_kernels(timing_kernels):
        epoch = spice.str2et("1998-12-31T23:59:60.25Z")
        frames = pd.DataFrame({"frame_clock_s": [0, 2], "frame_et": [epoch - 1, epoch + 1]})
        dates = ephemeris_to_utc([epoch, np.nan])
        assert dates["utc"].isna().all()
        assert dates["utc_iso"].iloc[0] == "1998-12-31T23:59:60.250000Z"
        assert dates["utc_leap_second"].tolist() == [True, False]
    result = reconstruct_product_times({"mag": pd.DataFrame({"clock_s": [-1, 1, 3]})}, frames, timing_kernels)
    assert result["mag"]["utc_valid"].tolist() == [False, True, False]
    assert result["mag"]["utc_iso"].iloc[1] == "1998-12-31T23:59:60.250000Z"


def test_integrated_decoder_uses_collection_time_coverage(timing_kernels):
    result = decode_level0(fixtures / "M9906003_excerpt.LBL", timing_kernels=timing_kernels, receiver="EARTH")
    mag = result["mag"]
    assert not mag["utc_valid"].iloc[0]
    assert mag["utc_valid"].any()
    assert "start_utc_valid" in result["er_counts"]
    assert result["frames"]["receiver"].eq("EARTH").all()
    sample = mag.loc[mag['record_index'].eq(10) & mag['sample_index'].eq(0)].iloc[0]
    ert = result['frames'].loc[result['frames']['record_index'].eq(10), 'ert'].iloc[0]
    # 1.5 s propagation + 2 s C&DH + 4 s MAG, with pair-average midpoint.
    assert (ert - sample.utc).total_seconds() == pytest.approx(1.5 + 2 + 4 - 1/36, abs=1e-6)
    with pytest.raises(ValueError, match="Choose"):
        decode_level0(fixtures / "M9906003_excerpt.LBL", timing_kernels=timing_kernels,
                      receiver="EARTH", light_time_s=1)


def test_real_sun_pulses_phase_support_and_estimate_flags():
    pulse = read_sunpulse(fixtures / "S9906003_excerpt.LBL")
    clocks = np.sort(pulse.loc[pulse["timing_valid"], "pulse_clock_s"].unique())
    selected = np.flatnonzero((np.diff(clocks) > 4) & (np.diff(clocks) < 6))[0]
    left, right = clocks[selected:selected + 2]
    out = reconstruct_spin_phase(pd.DataFrame({"clock_s": [left - 10, left, (left + right)/2]}), pulse)
    assert out["spin_phase_valid"].tolist() == [False, True, True]
    np.testing.assert_allclose(out["spin_phase_deg"].iloc[1:], [0, 180], atol=1e-6)
    assert out["sunpulse_uncertainty_s"].iloc[1:].ge(0).all()


def test_missing_sun_pulses_never_create_a_slow_artificial_spin():
    pulses = pd.DataFrame({"pulse_clock_s": [0, 5, 15], "timing_valid": [True]*3,
                          "estimated": [False, True, False], "uncertainty_s": [0.001]*3})
    samples = pd.DataFrame({"clock_s": [2.5, 7.5, 20]})
    result = reconstruct_spin_phase(samples, pulses)
    assert result["spin_phase_valid"].tolist() == [True, False, False]
    assert result["sunpulse_estimated"].tolist() == [True, False, False]
    assert not reconstruct_spin_phase(samples, pulses, allow_estimated=False)["spin_phase_valid"].any()


def test_command_arrival_adds_uplink_without_any_telemetry_buffering(timing_kernels):
    from moonbroom import annotate_command_actions, reconstruct_command_times
    commands = pd.DataFrame(dict(utc=['1999-03-01T03:00:00Z', '1999-03-01T03:00:01Z'],
                                  command=['MRESET', 'MERBON'], status=['CONFIRM', 'SKIP'], station=[42, 42]))
    requested = annotate_command_actions(commands)
    arrived = reconstruct_command_times(requested, timing_kernels, transmitter='EARTH')
    assert arrived['arrival_scet_valid'].tolist() == [True, False]
    assert arrived['arrival_scet_utc'].iloc[0] == pd.Timestamp('1999-03-01T03:00:01.5Z')
    assert arrived['uplink_light_time_s'].iloc[0] == pytest.approx(1.5)
    assert arrived['transmit_utc'].equals(requested['transmit_utc'])
    assert arrived['execution_evidence'].iloc[0] == 'not established by command log'
    # Coverage failure does not invent a spacecraft arrival time.
    outside = annotate_command_actions(commands.iloc[:1].assign(utc='1998-01-07T06:00:00Z'))
    missing = reconstruct_command_times(outside, timing_kernels, transmitter='EARTH')
    assert not missing['arrival_scet_valid'].any() and missing['command_timing_error'].notna().all()


def test_delayed_merged_frames_do_not_receive_a_second_playback_correction(timing_kernels):
    frames = read_merged_telemetry(fixtures / 'M9906003_excerpt.LBL').iloc[:2].copy()
    frames['selected_stream'] = [1, 2]
    result = reconstruct_frame_times(frames, timing_kernels, receiver='EARTH')
    np.testing.assert_allclose((result['ert'] - result['frame_utc']).dt.total_seconds(), 3.5, atol=1e-6)
