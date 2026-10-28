"""Scientific regressions for official ancillary formats and quality propagation."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moonbroom.archive_quality import (
    read_sunpulse_products, select_sunpulse_paths, read_outages, read_commands,
    read_attitude, maneuver_intervals, load_archive_quality, flag_archive_frames,
    flag_science_quality,
)
from moonbroom import decode_level0, reconstruct_spin_phase
from moonbroom.calibration import calibrate_er_telemetry
from moonbroom.mission_pipeline import process_shard


FIXTURES = Path(__file__).parent / 'fixtures'


def table(tmp_path, name, rows):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='ascii') as stream:
        csv.writer(stream).writerows(rows)
    return path


def test_official_replacement_wins_before_phase_interpolation_regardless_of_input_order(tmp_path):
    old = tmp_path / 'LP_0005/SUNPULSE/D12X/S9812114.B'
    new = tmp_path / 'lp_0019/sunpulse/dxxx/s9812114.b'
    for path, shift in ((old, 0), (new, 180)):
        path.parent.mkdir(parents=True)
        # Pulses at 10 and 15 seconds, replacement shifts both by 0.1 s.
        path.write_bytes(b''.join(counter.to_bytes(3, 'big') + (counts + shift).to_bytes(2, 'big')
                                  + b'\x00\x00\x00' for counter, counts in ((6, 18000), (8, 27000))))
    for paths in ([old, new], [new, old]):
        pulses = read_sunpulse_products(paths)
        assert pulses['source_file'].eq(str(new.resolve())).all()
        assert pulses.attrs['product_selection'][0]['superseded'] == [str(old.resolve())]
        phase = reconstruct_spin_phase(pd.DataFrame({'clock_s': [12.6]}), pulses)
        np.testing.assert_allclose(phase['spin_phase_deg'], 180)
    new.rename(new.with_name('s9812115.b'))
    other = old.with_name('S9812115.B')
    other.write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match='Conflicting SUNPULSE'):
        select_sunpulse_paths([other, new.with_name('s9812115.b')])


def test_identical_pulse_copies_and_label_binary_pair_are_one_product(tmp_path):
    paths = []
    for volume in ('lp_0001', 'lp_0019'):
        path = tmp_path / volume / 'S9800100.B'
        path.parent.mkdir()
        path.write_bytes(b'12345678')
        paths.append(path)
    label = paths[1].with_suffix('.LBL')
    label.write_text('^TABLE = "S9800100.B"\n')
    chosen, log = select_sunpulse_paths([*paths, label])
    assert chosen == [str(label.resolve())]
    assert len(log[0]['superseded']) == 1


def test_outages_use_ert_even_without_spacecraft_utc_and_keep_repaired_status(tmp_path):
    path = table(tmp_path, 'OUTAGES.TAB', [
        ['LP_0011', 'example.MERGE', '1998-11-08T06:00', '1998-11-08T06:01', 'NOISY DATA', 'NO CORRECTION'],
        ['LP_0011', 'example.MERGE', '1998-11-08T07:00', '1998-11-08T07:01', 'TRACKING GAP', 'NOT RECOVERABLE'],
        ['LP_0011', 'example.MERGE', '1998-11-08T08:00', '1998-11-08T08:01', 'NOISY DATA', 'RECOVERED']])
    quality = load_archive_quality()
    quality.outages = read_outages(path)
    quality.manifest['outages_source'] = str(path)
    ert = pd.to_datetime(['1998-11-08T05:59:58Z', '1998-11-08T06:00Z', '1998-11-08T06:01:59Z',
                          '1998-11-08T06:02Z', '1998-11-08T07:00Z', '1998-11-08T08:00Z'], format='mixed')
    raw = pd.DataFrame({'ert': ert, 'frame_utc': pd.NaT})
    result = flag_archive_frames(raw, quality)
    assert result['archive_noisy'].tolist() == [False, True, True, False, False, False]
    assert result['archive_gap'].tolist() == [False, False, False, False, True, False]
    assert result['archive_quality_assessed'].all()
    assert result['archive_interval_ids'].iloc[-1] == 'outage:2;'
    assert 'archive_noisy' not in raw


def test_command_candidates_ignore_heaters_aborts_and_duplicates_and_track_open_burns(tmp_path):
    rows = [
        ['1998-11-08T06:00:00', 'A1HTR', 'CONFIRM', 42],
        ['1998-11-08T06:00:32', 'EXEC', 'CONFIRM', 42],
        ['1998-11-08T06:01:00', 'A14REOR', 'ABORT', 42],
        ['1998-11-08T06:01:32', 'EXEC', 'CONFIRM', 42],
        ['1998-11-08T06:02:00', 'A14REOR', 'CONFIRM', 42],
        ['1998-11-08T06:02:32', 'EXEC', 'CONFIRM', 42],
        ['1998-11-08T06:02:32', 'EXEC', 'CONFIRM', 42],
        ['1998-11-08T06:12:00', 'STOPFIRE', 'CONFIRM', 42],
        ['1998-11-09T06:00:00', 'T1D', 'CONFIRM', 42],
        ['1998-11-09T06:00:32', 'EXEC', 'CONFIRM', 42]]
    commands = read_commands([table(tmp_path, 'C9831242.TAB', rows)])
    intervals = maneuver_intervals(commands)
    assert len(commands) == 9 and len(intervals) == 2
    assert intervals['start_utc'].iloc[0] == pd.Timestamp('1998-11-08T06:02:27Z')
    assert intervals['settled_utc'].iloc[0] == pd.Timestamp('1998-11-08T14:12:05Z')
    assert intervals['command'].iloc[0] == 'A14REOR'
    assert pd.isna(intervals['stop_utc'].iloc[-1])
    quality = load_archive_quality()
    quality.maneuvers = intervals
    quality.manifest['command_sources'] = ['commands']
    raw = pd.DataFrame({'utc': pd.to_datetime(['1998-11-08T06:00Z', '1998-11-08T06:03Z',
                                              '1998-11-08T14:00Z', '1998-11-08T15:00Z',
                                              '1998-11-10T06:00Z'])})
    flagged = flag_science_quality(raw, quality)
    assert flagged['maneuver_candidate'].tolist() == [False, True, False, False, True]
    assert flagged['post_maneuver_settling'].tolist() == [False, False, True, False, False]
    assert flagged['science_valid'].tolist() == [True, False, False, True, True]
    assert flagged['maneuver_timing_uncertain'].tolist() == [False, False, False, False, True]
    # An accumulation ending exactly at a maneuver boundary does not overlap.
    bounds = pd.DataFrame({'start_utc': pd.to_datetime(['1998-11-08T06:02:00Z']),
                           'stop_utc': pd.to_datetime(['1998-11-08T06:02:27Z'])})
    assert not flag_science_quality(bounds, quality, start_col='start_utc', stop_col='stop_utc')['maneuver_candidate'].any()


def test_attitude_support_staleness_high_spin_and_missing_context_are_explicit(tmp_path):
    quality = load_archive_quality(settings={'max_attitude_age_s': 3600})
    quality.attitude = read_attitude(table(tmp_path, 'ATTITUDE.TAB', [
        ['1998-11-08T06:00', 12, 90, 85], ['1999-07-29T06:00', 24, 90, 85]]))
    assert quality.attitude.attrs['attitude_frame'] == 'ECLIPJ2000'
    raw = pd.DataFrame({'utc': pd.to_datetime(['1998-11-08T05:00Z', '1998-11-08T06:30Z',
                                              '1998-11-08T07:01Z', '1999-07-29T06:30Z', None])})
    result = flag_science_quality(raw, quality)
    assert result['attitude_quality_valid'].tolist() == [False, True, False, True, False]
    assert result['sunpulse_spin_supported'].tolist() == [False, True, False, False, False]
    assert result['attitude_sun_sensor_only'].iloc[1]
    assert result['attitude_uncertainty_deg'].iloc[1] == 1
    assert not result['science_quality_assessed'].any()
    assert result['science_valid'].all()  # No invented fault from absent evidence.


def test_latest_cumulative_tables_override_stale_snapshots(tmp_path):
    downloads = []
    for volume, status in ((1, 'NO CORRECTION'), (19, 'RECOVERED')):
        rel = f'lp_{volume:04d}/index/OUTAGES.TAB'
        path = table(tmp_path, rel, [['LP_0001', 'example.MERGE', '1998-01-08T06:00',
                                     '1998-01-08T06:01', 'NOISY DATA', status]])
        label = path.with_suffix('.LBL')
        label.write_text('FILE_RECORDS = 1\n^TABLE = "OUTAGES.TAB"\n')
        downloads.append(dict(relative_path=str(label.relative_to(tmp_path)), path=str(label)))
    quality = load_archive_quality(downloads)
    assert quality.manifest['outages_source'].startswith('lp_0019/')
    assert not quality.outages['unresolved'].any()
    with pytest.raises(ValueError, match='record count'):
        label.write_text('FILE_RECORDS = 2\n^TABLE = "OUTAGES.TAB"\n')
        read_outages(label)


def test_noise_in_later_er_packet_propagates_across_owning_shard_and_masks_rates(tmp_path, monkeypatch):
    import moonbroom.mission_pipeline as pipeline
    path = str((FIXTURES / 'M9906003_excerpt.LBL').resolve())
    source = FIXTURES / 'M9906003_excerpt.B'
    decoded = decode_level0(path, light_time_s=3.28)
    group = next(group for _, group in decoded['er_packets'].groupby('accumulation_id') if len(group) > 20)
    affected = int(group['record_index'].iloc[-2])
    first = int(group['record_index'].iloc[0])
    assert affected > first
    time = decoded['frames'].loc[decoded['frames']['record_index'].eq(affected), 'ert'].iloc[0]
    quality = load_archive_quality(settings={'outage_stop_padding_s': 0})
    quality.outages = read_outages(table(tmp_path, 'OUTAGES.TAB', [
        ['LP_0015', 'example.MERGE', time.isoformat(), (time + pd.Timedelta(seconds=2)).isoformat(),
         'NOISY DATA', 'NO CORRECTION']]))
    quality.manifest['outages_source'] = 'test outage'
    original = pipeline.decode_level0
    monkeypatch.setattr(pipeline, 'decode_level0', lambda paths, **kw: original(paths, light_time_s=3.28, **kw))
    result, stats = process_shard([path], {path: (0, 128)}, source, 0, affected, [],
                                 {'clock_only': True}, {}, None, quality=quality)
    counts = result['er_counts']
    selected = counts['accumulation_id'].eq(group['accumulation_id'].iloc[0])
    assert selected.any() and counts.loc[selected, 'archive_noisy'].all()
    assert not result['pseudo_er'].loc[selected, 'count_rate_valid'].any()
    assert counts.loc[selected, 'counts_lower'].notna().any()
    assert len(result['native_mag']) == affected * 18
    assert stats['payload_audit']['audit_pass']


def test_quality_failure_masks_er_flux_without_changing_digital_counts():
    raw = pd.DataFrame(dict(accumulation_id=[1, 1], kind=['3d'] * 2, energy_step=[0, 0],
                            sample=[0, 0], angle_bin=[0, 1], counts_lower=[10., 10.],
                            counts_upper=[11., 11.], nominal_integration_s=[1., 1.],
                            energy_ev=[100., 100.], science_valid=[True, False]))
    response = raw[['accumulation_id', 'kind', 'energy_step', 'sample', 'angle_bin']].copy()
    response['exposure_s'], response['geometric_factor_cm2_sr'] = 1., .01
    result = calibrate_er_telemetry(raw, response=response)
    assert result['calibration_valid'].tolist() == [True, False]
    assert pd.isna(result['flux_estimate'].iloc[1])
    np.testing.assert_array_equal(result['counts_lower'], raw['counts_lower'])


def test_actual_mager_requests_keep_parameters_and_failed_heater_command_is_not_applied(tmp_path):
    from moonbroom import annotate_command_actions
    path = table(tmp_path, 'C9802342.TAB', [
        ['1998-01-23T00:49:53.200', 'MERHTROFF', 'CONFIRM', 42],
        ['1998-01-23T01:09:11.700', 'MERHTROFF', 'CONFIRM', 42],
        ['1998-01-23T01:10:23.500', 'EXEC', 'CONFIRM', 42],
        ['1998-01-23T01:13:24.400', 'MTLOADW, 0, 2, 350', 'CONFIRM', 42],
        ['1998-01-23T01:16:23.000', 'MLADDR, 54503', 'CONFIRM', 42],
        ['1998-01-23T01:23:53.200', 'MBCOMMIT, 218, 24', 'CONFIRM', 42]])
    commands = annotate_command_actions(read_commands([path]))
    assert commands['known_execution_failure'].tolist() == [True, False, False, False, False, False]
    assert commands['command'].iloc[3] == 'MTLOADW, 0, 2, 350'
    assert commands['may_change_science'].tolist() == [True, True, False, True, False, True]
    assert commands['following_exec_event_id'].iloc[1] == commands['command_event_id'].iloc[2]
    assert commands['requested_action'].iloc[3] == 'write table word'
    quality = load_archive_quality()
    quality.commands = commands
    samples = pd.DataFrame({'utc': pd.to_datetime(['1998-01-23T00:49:53.200Z', '1998-01-23T01:09:50Z'], format='mixed')})
    flags = flag_science_quality(samples, quality, instrument='MAG')
    assert flags['mager_command_transition'].tolist() == [False, True]
    assert flags['science_valid'].tolist() == [True, False]


def test_mager_action_flags_use_arrival_scet_and_respect_instrument_scope():
    from moonbroom import annotate_command_actions
    quality = load_archive_quality(settings={'mager_command_padding_s': 0})
    commands = annotate_command_actions(pd.DataFrame(dict(utc=['1998-11-08T06:00:00Z'],
        command=['MERLVOFF'], status=['CONFIRM'], station=[42])))
    commands['arrival_scet_utc'] = pd.to_datetime(['1998-11-08T06:00:01.5Z'])
    commands['arrival_scet_valid'] = True
    quality.commands = commands
    raw = pd.DataFrame({'utc': pd.to_datetime(['1998-11-08T06:00:00Z', '1998-11-08T06:00:01.5Z'], format='mixed')})
    er = flag_science_quality(raw, quality, instrument='ER')
    mag = flag_science_quality(raw, quality, instrument='MAG')
    assert er['mager_command_transition'].tolist() == [False, True]
    assert not mag['mager_command_transition'].any()
    assert er['mager_command_time_basis'].iloc[1] == 'estimated spacecraft arrival SCET'
    assert er['mager_command_cohort'].iloc[1] == commands['command_event_id'].iloc[0]
    # An ER collection interval spanning the request is also flagged.
    bounded = pd.DataFrame({'start_utc': pd.to_datetime(['1998-11-08T05:59:59Z']),
                            'stop_utc': pd.to_datetime(['1998-11-08T06:00:02Z'])})
    assert flag_science_quality(bounded, quality, start_col='start_utc', stop_col='stop_utc', instrument='ER')['mager_command_transition'].all()


def test_parameter_register_reset_does_not_prove_firing_stopped_and_maneuvers_use_scet(tmp_path):
    from moonbroom import annotate_command_actions
    commands = annotate_command_actions(read_commands([table(tmp_path, 'commands.TAB', [
        ['1998-11-08T06:00:00Z', 'A14REOR', 'CONFIRM', 42],
        ['1998-11-08T06:00:32Z', 'EXEC', 'CONFIRM', 42],
        ['1998-11-08T06:00:40Z', 'SETCMDREG', 'CONFIRM', 42],
        ['1998-11-08T06:01:00Z', 'STOPFIRE', 'CONFIRM', 42]])]))
    commands['arrival_scet_utc'] = commands['transmit_utc'] + pd.Timedelta(seconds=1.5)
    commands['arrival_scet_valid'] = True
    intervals = maneuver_intervals(commands, padding_s=0)
    assert len(intervals) == 1
    assert intervals['start_utc'].iloc[0] == pd.Timestamp('1998-11-08T06:00:33.5Z')
    assert intervals['stop_utc'].iloc[0] == pd.Timestamp('1998-11-08T06:01:01.5Z')
    assert 'arrival SCET' in intervals['time_basis'].iloc[0]
    assert pd.isna(maneuver_intervals(commands.iloc[:3])['stop_utc'].iloc[0])


def test_local_offsets_never_mix_or_interpolate_across_command_cohorts():
    from moonbroom.native_mag import fit_mag_spin_offsets, correct_mag_spin_offsets
    before = pd.read_csv(FIXTURES / 'mission_fit/mag_native_validation.csv')
    before['mager_command_cohort'] = 'before'
    after = before.copy()
    after['clock_s'] += 60
    after['bx_signed'] += 100
    after['mager_command_cohort'] = 'after'
    combined = pd.concat([before, after], ignore_index=True)
    offsets = fit_mag_spin_offsets(combined)
    assert offsets['offset_valid'].all() and len(offsets) == 2
    corrected = correct_mag_spin_offsets(combined, offsets)
    assert corrected['spin_offset_valid'].all()
    # The second interval has a 100-count offset jump, not a gradual ramp.
    np.testing.assert_allclose(corrected['spin_offset_x_counts'].iloc[len(before):].to_numpy()
                               - corrected['spin_offset_x_counts'].iloc[:len(before)].to_numpy(), 100, atol=1e-6)
    mixed = before.copy()
    mixed.loc[mixed.index >= len(mixed)//2, 'mager_command_cohort'] = 'after'
    assert fit_mag_spin_offsets(mixed).empty
