"""Burst preservation and calibration applicability across observed hardware states."""

import numpy as np
import pandas as pd
import pytest

from moonbroom import (
    annotate_mag_hardware_state, calibrate_mag_9hz, calibrate_mag_18hz,
    correct_mag_spin_offsets, fit_mag_spin_offsets, mag_hardware_calibration_checks,
    select_mag_models, validate_mag_pseudocalibration,
)
from moonbroom.archive_calibration import _mag_design, mag_archive_windows, apply_mag_archive_calibration
from moonbroom.mag_state import validate_mag_model_scopes


EPOCH = pd.Timestamp('1999-06-01T00:00Z')


def samples(mode='burst', *, zero_x=42., burst_id='a'):
    rate = 18 if mode == 'burst' else 9
    clocks = np.r_[np.arange(60*rate)/rate, 180 + np.arange(60*rate)/rate]
    phase = 2*np.pi*clocks/5
    x, y = 100 + .05*clocks, -50 + .03*clocks
    return pd.DataFrame(dict(clock_s=clocks, utc=EPOCH + pd.to_timedelta(clocks, unit='s'),
        mode=mode, range=1, spin_phase_deg=np.degrees(phase), spin_phase_valid=True,
        spin_period_s=5., science_valid=True, burst_id=burst_id, burst_packet_slot=0,
        bx_signed=x*np.cos(phase)+y*np.sin(phase)+zero_x,
        by_signed=-x*np.sin(phase)+y*np.cos(phase)-13, bz_signed=20.,
        source_file=burst_id+'.B', record_index=np.arange(len(clocks))//18,
        sample_index=np.arange(len(clocks)) % 18))


def coefficients(mode='burst', **scope):
    row = dict(range=1, fit_valid=True, validation_pass=True, valid_from=EPOCH,
        valid_to=EPOCH+pd.Timedelta(days=3), source='synthetic independent test',
        gain_x=.1, gain_y=.1, gain_z=.1, zero_x=0., zero_y=0., zero_z=0.,
        offset_model='local spin fit')
    if mode is not None:
        row['mode'] = mode
    for a in 'xyz':
        for b in 'xyz':
            row['align_'+a+b] = float(a == b)
    row.update(scope)
    return pd.DataFrame([row])


def status(clocks, sides, **extra):
    table = pd.DataFrame(dict(major_clock_s=clocks, electronics_side=sides,
        boom_deployed=1, telemetry_version=3, reset_count=0))
    for field, value in extra.items():
        table[field] = value
    return table


def test_18hz_calibration_preserves_samples_counts_gaps_and_quality():
    raw = samples()
    raw.loc[10, 'science_valid'] = False
    offsets = fit_mag_spin_offsets(raw, mode='burst')
    out = calibrate_mag_18hz(raw, coefficients(), offsets=offsets)
    assert len(out) == len(raw)
    for field in ['clock_s', 'utc', 'bx_signed', 'by_signed', 'bz_signed']:
        pd.testing.assert_series_equal(out[field], raw[field])
    assert out['source_sample_id'].is_unique
    assert out['sample_rate_hz'].eq(18).all()
    assert out['calibration_valid'].sum() == len(raw)-1
    assert out.loc[10, ['Bx_SCD', 'By_SCD', 'Bz_SCD']].isna().all()
    valid = out['calibration_valid']
    np.testing.assert_allclose(out.loc[valid, 'Bx_SCD'], (100+.05*raw.loc[valid, 'clock_s'])*.1, atol=1e-10)
    assert np.max(np.diff(out['clock_s'])) > 120
    assert not out['high_frequency_validated'].any()
    assert not out['calibration_mode_transfer'].any()
    assert out.attrs['resampled'] is False


def test_legacy_realtime_models_require_explicit_burst_transfer_even_at_low_level():
    raw = samples()
    offsets = fit_mag_spin_offsets(raw, mode='burst')
    models = coefficients(None)
    unsupported = calibrate_mag_18hz(raw, models, offsets=offsets)
    assert not unsupported['calibration_valid'].any()
    assert unsupported['calibration_failure_reason'].eq('mode_not_trained').all()
    assert not apply_mag_archive_calibration(raw, models)['calibration_valid'].any()
    transferred = calibrate_mag_18hz(raw, models, offsets=offsets, allow_realtime_transfer=True)
    assert transferred['calibration_valid'].all()
    assert transferred['calibration_mode_transfer'].all()
    assert transferred['calibration_basis'].str.contains('transfer').all()
    assert not transferred['high_frequency_validated'].any()
    with pytest.raises(ValueError, match='Boolean'):
        calibrate_mag_18hz(raw, models, offsets=offsets, allow_realtime_transfer='true')
    with pytest.raises(ValueError, match='validation_pass'):
        calibrate_mag_18hz(raw, coefficients().drop(columns='validation_pass'), offsets=offsets)


def test_explicit_burst_model_takes_priority_over_realtime_transfer():
    raw = samples().iloc[:1080].copy()
    realtime = coefficients('realtime', offset_model='none', gain_z=.2)
    burst = coefficients('burst', offset_model='none', gain_z=.3,
                         valid_from=EPOCH+pd.Timedelta(seconds=30))
    models = pd.concat([realtime, burst], ignore_index=True)
    out = calibrate_mag_18hz(raw, models, allow_realtime_transfer=True)
    early = raw['clock_s'].lt(30)
    np.testing.assert_allclose(out.loc[early, 'Bz_SCD'], 4)
    np.testing.assert_allclose(out.loc[~early, 'Bz_SCD'], 6)
    assert out.loc[early, 'calibration_mode_transfer'].all()
    assert not out.loc[~early, 'calibration_mode_transfer'].any()
    assert len(select_mag_models(models, EPOCH, mode='realtime')) == 1


def test_burst_offsets_never_mix_overlapping_bursts_or_realtime_samples():
    first, second = samples(burst_id='a'), samples(zero_x=90., burst_id='b')
    mixed = pd.concat([first, second, samples('realtime')], ignore_index=True)
    offsets = fit_mag_spin_offsets(mixed, mode='burst')
    np.testing.assert_allclose(offsets.loc[offsets['burst_id'].eq('a'), 'zero_x'], 42, atol=1e-10)
    np.testing.assert_allclose(offsets.loc[offsets['burst_id'].eq('b'), 'zero_x'], 90, atol=1e-10)
    corrected = correct_mag_spin_offsets(mixed, offsets)
    assert corrected.loc[corrected['mode'].eq('burst'), 'spin_offset_valid'].all()
    assert not corrected.loc[corrected['mode'].eq('realtime'), 'spin_offset_valid'].any()
    with pytest.raises(ValueError, match='burst_id'):
        correct_mag_spin_offsets(first, offsets.drop(columns='burst_id'))


def test_burst_hardware_comes_from_collection_markers_not_playback_subcom():
    raw = pd.DataFrame(dict(mode=['realtime', 'burst', 'burst'], clock_s=[1001., 1., 1.],
                            burst_id=[None, 'a', 'missing']))
    playback = status([1000.], [1])
    markers = status([0.], [0], burst_id='a')
    out = annotate_mag_hardware_state(raw, playback, burst_markers=markers)
    assert out['electronics_side'].iloc[:2].tolist() == [1, 0]
    assert out['mag_hardware_state_source'].tolist() == ['digital_subcom', 'burst_marker', 'unavailable']
    assert out['mag_hardware_state_valid'].tolist() == [True, True, False]


def test_hardware_changes_are_bracketed_and_returned_states_get_new_offset_cohorts():
    raw = pd.DataFrame(dict(mode='realtime', clock_s=[0., 1., 31., 32., 33., 63., 64., 65., 96., 128.]))
    out = annotate_mag_hardware_state(raw, status([0., 32., 64., 96.], [0, 1, 0, 0]))
    assert out['mag_hardware_transition'].tolist() == [False, True, True, False, True, True, False, False, False, False]
    assert out['mag_hardware_state_valid'].iloc[-1] == False
    assert out['mag_hardware_state_id'].iloc[0] == out['mag_hardware_state_id'].iloc[6]
    assert out['mag_hardware_cohort'].iloc[0] != out['mag_hardware_cohort'].iloc[6]
    partial = status([0., 32.], [0, 0], reset_count=[0., np.nan])
    assert not annotate_mag_hardware_state(raw.iloc[[3]], partial)['mag_hardware_state_valid'].any()
    with pytest.raises(ValueError, match='Conflicting'):
        annotate_mag_hardware_state(raw, status([0., 0.], [0, 1]))


def test_hardware_scoped_models_select_different_gains_and_reject_unknown_states():
    raw = samples('realtime').iloc[:3].copy()
    raw['electronics_side'] = [0, 1, np.nan]
    model = pd.concat([coefficients('realtime', electronics_side=0, gain_z=.1, offset_model='none'),
                       coefficients('realtime', electronics_side=1, gain_z=.2, offset_model='none')], ignore_index=True)
    out = calibrate_mag_9hz(raw, model)
    np.testing.assert_allclose(out['Bz_SCD'].iloc[:2], [2, 4])
    assert not out['calibration_valid'].iloc[2]
    assert out['calibration_failure_reason'].iloc[2] == 'hardware_state_outside_model_scope'
    assert len(select_mag_models(model, EPOCH, hardware_state={'electronics_side': 1})) == 1
    assert not select_mag_models(model, EPOCH, hardware_state={}).shape[0]
    with pytest.raises(ValueError, match='overlap'):
        validate_mag_model_scopes(pd.concat([model, coefficients('realtime', offset_model='none')], ignore_index=True))
    rejected = model.copy()
    rejected.loc[0, 'validation_pass'] = False
    rejected['n_validation'] = 4
    rejected_out = calibrate_mag_9hz(raw, rejected)
    assert rejected_out['calibration_failure_reason'].iloc[0] == 'validation_insufficient'


def test_transition_masks_and_hardware_offset_scopes_are_enforced():
    raw = samples('realtime').iloc[:540].copy()
    annotated = annotate_mag_hardware_state(raw, status([0., 32., 64.], [0, 1, 1]))
    model = coefficients('realtime', offset_model='none')
    out = calibrate_mag_9hz(annotated, model)
    assert not out.loc[out['mag_hardware_transition'], 'calibration_valid'].any()
    assert out.loc[out['mag_hardware_transition'], 'calibration_failure_reason'].eq('hardware_state_transition').all()
    old_offsets = fit_mag_spin_offsets(raw)
    with pytest.raises(ValueError, match='Hardware-scoped'):
        correct_mag_spin_offsets(annotated, old_offsets)
    checks = mag_hardware_calibration_checks(out)
    assert checks['n_samples'].sum() == len(out)
    assert checks['n_transition'].sum() == out['mag_hardware_transition'].sum()
    assert checks['review_required'].all()


def test_offset_support_does_not_cross_a_nearby_hardware_cohort_boundary():
    raw = samples().iloc[:1080].copy()
    raw['mag_hardware_cohort'] = np.where(raw['clock_s'].lt(30), 'before', 'after')
    assert fit_mag_spin_offsets(raw, mode='burst').empty
    offsets = fit_mag_spin_offsets(raw.loc[raw['clock_s'].lt(30)], mode='burst')
    assert offsets['offset_valid'].all()
    corrected = correct_mag_spin_offsets(raw, offsets)
    assert corrected.loc[raw['clock_s'].lt(30), 'spin_offset_valid'].all()
    assert not corrected.loc[raw['clock_s'].ge(30), 'spin_offset_valid'].any()


def test_burst_reference_windows_use_18hz_coverage_and_preserve_mode():
    raw = samples().iloc[:90]
    pairs = pd.DataFrame(dict(accumulation_id=[0], match_valid=True,
        start_clock_s=0., stop_clock_s=2.5, start_utc=EPOCH,
        stop_utc=EPOCH+pd.Timedelta(seconds=2.5), reference_id='independent',
        reference_bx=1., reference_by=2., reference_bz=3.))
    windows = mag_archive_windows(raw, pairs, mode='burst')
    assert windows['n_samples'].iloc[0] == 45
    assert windows['coverage'].iloc[0] == 1
    assert windows['mode'].iloc[0] == 'burst'
    assert windows['window_valid'].all()
    raw = raw.copy()
    raw['electronics_side'] = np.where(raw['clock_s'].lt(1), 0, 1)
    assert not mag_archive_windows(raw, pairs, mode='burst')['window_valid'].any()


def test_independent_validation_keeps_hardware_states_separate():
    rng = np.random.default_rng(54)
    def windows(date, count):
        frame = pd.DataFrame(rng.normal(size=(2*count, 5)), columns=['raw_u', 'raw_v', 'raw_z', 'phase_cos', 'phase_sin'])
        frame['range'], frame['mode'], frame['window_valid'], frame['window_s'] = 1, 'burst', True, 5.
        frame['electronics_side'] = np.repeat([0, 1], count)
        frame['start_utc'] = date + pd.to_timedelta(np.arange(2*count)*10, unit='s')
        frame['stop_utc'] = frame['start_utc'] + pd.Timedelta(seconds=5)
        for side, group in frame.groupby('electronics_side'):
            values = _mag_design(group) @ np.array([.01, -.02, .03+side*.01, .3, -.2, .1])
            frame.loc[group.index, ['reference_b'+a for a in 'xyz']] = values.reshape(-1, 3)
        return frame
    training, validation = windows(EPOCH, 40), windows(EPOCH+pd.Timedelta(days=1), 20)
    validation.loc[validation['electronics_side'].eq(1), 'reference_bz'] += 1
    result = validate_mag_pseudocalibration(training, validation, valid_from=EPOCH,
        valid_to=EPOCH+pd.Timedelta(days=3), source='synthetic held-out state test',
        max_rmse_nt=.01, max_bias_nt=.01)
    models = result['coefficients'].set_index('electronics_side')
    assert models['mode'].eq('burst').all()
    assert models['validation_pass'].to_dict() == {0: True, 1: False}
    assert result['metrics']['n_validation'].tolist() == [20, 20]


def test_full_burst_packet_slots_have_unique_identity_and_reach_mission_output(tmp_path):
    import struct
    from test_level0 import merged_record, digital_subcom
    from moonbroom import decode_level0
    from moonbroom.mission_pipeline import process_shard

    marker = bytearray(82)
    marker[:5] = struct.pack('<HHB', 100, 101, 7)
    marker[50:] = digital_subcom()
    pair = sum(value << (12*i) for i, value in enumerate([2058, 2068, 2078, 2059, 2069, 2079]))
    block = bytes([2]) + pair.to_bytes(9, 'little')*9
    records = []
    for vcdu in range(17, 33):
        record = bytearray(merged_record(vcdu))
        if vcdu == 25:
            record[286], record[371] = 57 | 64, 0
            record[372:454] = marker
        elif vcdu == 26:
            record[286], record[371] = 56 | 128, 56 | 64
            record[289:371] = block
            record[372:454] = block
        records.append(record)
    path = tmp_path / 'M9906003.B'
    path.write_bytes(b''.join(records))
    products = decode_level0(path, light_time_s=1.3)
    raw = products['mag']
    burst = raw.loc[raw['mode'].eq('burst')]
    assert burst['spin_phase_valid'].all()
    assert burst['burst_packet_slot'].unique().tolist() == [0, 1]
    start = burst['utc'].min().floor('D')
    model = coefficients('burst', range=2, offset_model='none', valid_from=start,
                          valid_to=burst['utc'].max().ceil('D')+pd.Timedelta(days=1))
    calibrated = calibrate_mag_18hz(raw, model)
    assert len(calibrated) == 36 and calibrated['source_sample_id'].is_unique
    assert calibrated['calibration_valid'].all()
    integer_slots = raw.copy()
    integer_slots['burst_packet_slot'] = integer_slots['burst_packet_slot'].astype('Int64')
    pd.testing.assert_series_equal(calibrate_mag_18hz(integer_slots, model)['source_sample_id'],
                                    calibrated['source_sample_id'])
    np.testing.assert_allclose(np.diff(calibrated['clock_s']), 1/18)
    output, stats = process_shard([path], {}, str(path), 0, 16, [],
                                  {'clock_only': True}, {}, None)
    assert len(output['native_burst_mag']) == stats['burst_mag_rows'] == 36
    assert output['native_burst_mag']['source_sample_id'].is_unique
    assert not output['native_burst_mag']['calibration_valid'].any()
    assert output['native_burst_mag']['mag_hardware_state_source'].eq('burst_marker').all()
    assert output['mag_hardware_checks']['n_samples'].sum() == len(output['native_mag'])+36
