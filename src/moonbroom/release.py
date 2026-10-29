"""Resumable native-count and pseudo-calibrated LP MAG release builder.

Release status remains incomplete without a complete official source inventory.
Native digital recovery does not imply validated native-frequency calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .level0 import decode_level0
from .timing import reconstruct_spin_phase, reconstruct_product_times
from .native_mag import fit_mag_spin_offsets, calibrate_mag_9hz
from .mission_calibration import select_mag_models
from .mager import decode_subcom_frames
from .archive_quality import (quality_from_paths, flag_archive_frames, flag_science_quality,
                              select_sunpulse_paths, read_sunpulse_products)
from .quality_events import telemetry_events, ancillary_events, combine_events, SCHEMA_VERSION
from .mag_state import annotate_mag_hardware_state, mag_hardware_calibration_checks, MAG_HARDWARE_FIELDS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory_local_mag(roots):
    """Find original official binaries and their detached labels, preserving case."""
    rows = []
    for root in roots:
        for path in Path(root).rglob('*'):
            if not path.is_file() or not re.fullmatch(r'm\d{7}\.b', path.name, re.I):
                continue
            labels = [p for p in path.parent.iterdir() if p.stem.lower() == path.stem.lower()
                      and p.suffix.lower() == '.lbl']
            stamp = pd.to_datetime(path.stem[1:6], format='%y%j')
            rows.append(dict(date=str(stamp.date()), filename=path.name, path=str(path.resolve()),
                             label_path=str(labels[0].resolve()) if len(labels) == 1 else None,
                             bytes=path.stat().st_size, sha256=sha256(path),
                             label_sha256=sha256(labels[0]) if len(labels)==1 else None))
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError('No official merged binaries found')
    # Duplicate copies must agree before one is selected.
    out['_name'] = out['filename'].str.lower()
    if out.groupby('_name')['sha256'].nunique().gt(1).any():
        raise ValueError('Conflicting copies of a source binary')
    return out.drop_duplicates('_name').drop(columns='_name').sort_values(['date','filename']).reset_index(drop=True)


def audit_native_payloads(frames, samples):
    """Vectorized independent bit expansion for release-scale exact verification."""
    selected = frames.loc[frames['header_valid']].copy()
    payload = np.frombuffer(b''.join(selected['mag_er']), dtype=np.uint8).reshape(-1,168)
    types = payload[:,0] & 63
    use = ~((types >= 36) & (types <= 62) & ((payload[:,0] & 128) != 0))
    selected = selected.loc[use]
    blocks = payload[use,3:85]
    bits = np.unpackbits(blocks[:,1:], axis=1, bitorder='little').reshape(-1,54,12)
    codes = (bits.astype(np.uint16)@(2**np.arange(12, dtype=np.uint16))).reshape(-1,3)
    expected = pd.DataFrame(dict(source_file=np.repeat(selected['source_file'].to_numpy(),18),
                    record_index=np.repeat(selected['record_index'].to_numpy(),18),
                    sample_index=np.tile(np.arange(18),len(selected)),
                    expected_clock=np.repeat(selected['frame_clock_s'].to_numpy(),18)+np.tile(np.arange(18)/9+1/36-4,len(selected)),
                    expected_range=np.repeat(blocks[:,0] & 7,18),
                    expected_mode=np.repeat((blocks[:,0] & 8)!=0,18),
                    expected_frame=np.repeat(blocks[:,0] >> 4,18)))
    for i,a in enumerate('xyz'):
        expected['expected_'+a] = codes[:,i]
    keys = ['source_file','record_index','sample_index']
    actual = samples.loc[samples['mode'].eq('realtime')]
    paired = expected.merge(actual, on=keys, how='outer', validate='one_to_one', indicator=True)
    valid = paired['_merge'].eq('both').to_numpy(copy=True)
    valid &= np.isclose(paired['clock_s'], paired['expected_clock'], rtol=0, atol=1e-8)
    valid &= paired['range'].eq(paired['expected_range']).to_numpy()
    valid &= paired['calibration_mode'].eq(paired['expected_mode']).to_numpy()
    valid &= paired['mag_frame'].eq(paired['expected_frame']).to_numpy()
    for a in 'xyz':
        valid &= paired['b'+a].eq(paired['expected_'+a]).to_numpy()
        valid &= paired['b'+a+'_signed'].eq(paired['expected_'+a].astype(np.int64)-2048).to_numpy()
    return dict(expected_samples=len(expected), actual_samples=len(actual),
                exact_samples=int(valid.sum()), audit_pass=bool(len(expected) and valid.all()))


def _csv(table, path):
    temporary = path.with_name(path.name+'.partial')
    compression = dict(method='gzip', compresslevel=1, mtime=0) if path.name.endswith('.gz') else None
    table.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def build_native_release(config):
    """Process every inventoried local source date, retaining missing mission dates.

    This does not promote the existing monthly calibration scopes into claims
    of continuous independent validation. The supplied bank and all hashes are
    included. Records outside scope remain unsupported. No public upload occurs.
    """
    cfg = dict(config)
    root = Path(cfg['output_directory'])
    root.mkdir(parents=True, exist_ok=True)
    config_text = json.dumps(cfg, sort_keys=True, indent=2)
    fingerprint = hashlib.sha256(config_text.encode()).hexdigest()
    existing = root/'configuration.json'
    if existing.exists() and existing.read_text() != config_text:
        if list((root/'data').glob('*/complete.json')):
            raise ValueError('Resume requires the identical configuration')
        # An interrupted attempt with no committed products may be restarted;
        # retain its exact status/configuration rather than overwriting evidence.
        history = root/'attempt_history'
        history.mkdir(exist_ok=True)
        for name in ('configuration.json', 'processing_status.json', 'release_status.json'):
            original = root/name
            if original.exists():
                (history/(sha256(original)[:12]+'_'+name)).write_bytes(original.read_bytes())
    existing.write_text(config_text)
    inventory = inventory_local_mag(cfg['input_roots'])
    inventory.to_csv(root/'source_inventory.csv',index=False)
    bank = pd.read_csv(cfg['coefficient_bank'])
    if 'validation_pass' not in bank:
        raise ValueError('An independently evaluated coefficient bank is required')
    command_paths = sorted(str(path.resolve()) for folder in cfg.get('command_roots', [])
                           for path in Path(folder).rglob('*') if path.is_file()
                           and re.fullmatch(r'c\d{7}\.tab', path.name, re.I))
    quality = quality_from_paths(command_paths=command_paths, outage_path=cfg.get('outage_path'),
        attitude_path=cfg.get('attitude_path'), settings=cfg.get('archive_quality'),
        timing_kernels=() if cfg.get('clock_only', False) else cfg.get('kernels', ()),
        transmitter=cfg.get('command_transmitter', 'EARTH'))
    ancillary_manifest = json.loads(Path(cfg['ancillary_manifest']).read_text()) if cfg.get('ancillary_manifest') else {}
    quality.manifest['command_history_complete'] = ancillary_manifest.get('command_history_complete', False)
    pulse_paths = [str(path.resolve()) for folder in cfg['input_roots'] for path in Path(folder).rglob('*')
                   if path.is_file() and re.fullmatch(r's\d{7}\.lbl', path.name, re.I)]
    pulse_paths, pulse_selection = select_sunpulse_paths(pulse_paths)
    auxiliary_paths = list(cfg.get('kernels', [])) + [cfg['coefficient_bank']] + command_paths + pulse_paths
    auxiliary_paths += [cfg[key] for key in ('outage_path', 'attitude_path', 'ancillary_manifest') if cfg.get(key)]
    provenance = [{'path':str(Path(p).resolve()), 'sha256':sha256(p)} for p in dict.fromkeys(auxiliary_paths)]
    fingerprint = hashlib.sha256(json.dumps(dict(config=cfg, inputs=inventory.to_dict('records'),
                          auxiliary=provenance, code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}),sort_keys=True).encode()).hexdigest()
    pd.DataFrame(provenance).to_csv(root/'calibration_kernel_inventory.csv',index=False)
    pd.DataFrame(bank).to_csv(root/'coefficient_bank.csv',index=False)
    (root/'code_sha256.json').write_text(json.dumps({p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},indent=2))
    _csv(quality.commands, root/'archive_commands.csv')
    _csv(quality.maneuvers, root/'maneuver_intervals.csv')
    _csv(quality.outages, root/'archive_outages.csv')
    _csv(ancillary_events(quality), root/'ancillary_quality_events.csv')
    records = []
    for date, group in inventory.groupby('date', sort=True):
        target = root/'data'/date
        target.mkdir(parents=True,exist_ok=True)
        done = target/'complete.json'
        if done.exists():
            record = json.loads(done.read_text())
            if record.get('configuration_sha256') != fingerprint:
                raise ValueError('Resume fingerprint differs')
            for product in record['products']:
                if sha256(root/product['path']) != product['sha256']:
                    raise ValueError('Completed shard hash mismatch')
            records.append(record)
            continue
        started = time.monotonic()
        record = dict(date=date, status='failed', configuration_sha256=fingerprint, error=None)
        try:
            labels = group['label_path'].tolist()
            if any(pd.isna(p) for p in labels):
                raise ValueError('Detached source label missing')
            timing_options = {} if cfg.get('clock_only', False) else dict(timing_kernels=cfg['kernels'], receiver=cfg['receiver'])
            timing_error = None
            try:
                products = decode_level0(labels, mag_only=True, **timing_options,
                                          overlap_policy=cfg.get('overlap_policy','exact'))
            except Exception as error:
                if type(error).__name__ != 'SpiceSPKINSUFFDATA' or not cfg.get('retain_clock_only', True):
                    raise
                timing_error = str(error)
                products = decode_level0(labels, mag_only=True, overlap_policy=cfg.get('overlap_policy','exact'))
            raw_original = products['mag']
            frames = flag_archive_frames(products['frames'], quality)
            raw = raw_original.copy()
            raw['decoder_science_valid'] = raw['science_valid']
            frame_flags = [name for name in frames if name.startswith('archive_')]
            raw = raw.merge(frames[['source_file', 'record_index']+frame_flags],
                            on=['source_file', 'record_index'], validate='many_to_one')
            raw = flag_science_quality(raw, quality, instrument='MAG')
            subcom = decode_subcom_frames(frames)
            raw = annotate_mag_hardware_state(raw, subcom)
            audit = audit_native_payloads(products['frames'],raw)
            if not audit['audit_pass']:
                raise ValueError('Independent native payload audit failed')
            selected_pulses = [path for path in pulse_paths
                if abs((pd.to_datetime(Path(path).stem[1:6], format='%y%j') - pd.Timestamp(date)).days) <= 1]
            pulses = read_sunpulse_products(selected_pulses, frames=frames, strict=False)
            sunpulse_inputs = [dict(path=path, sha256=sha256(path)) for path in selected_pulses]
            phase = reconstruct_spin_phase(raw, pulses) if not pulses.empty else raw.assign(spin_phase_valid=False)
            high_spin = phase['attitude_available'] & ~phase['attitude_stale'] & phase['attitude_spin_rate_rpm'].ge(15)
            phase.loc[high_spin, 'spin_phase_valid'] = False
            if 'frame_et' in frames:
                subcom = reconstruct_product_times({'digital_subcom':subcom}, frames, cfg['kernels'])['digital_subcom']
            events = telemetry_events(frames, raw_original, subcom, phase=phase, partition=date)
            offsets = fit_mag_spin_offsets(phase)
            models = select_mag_models(bank, date)
            calibrated = calibrate_mag_9hz(phase,models,offsets=offsets)
            keys=['source_file','record_index','sample_index']
            if not calibrated['clock_s'].equals(raw['clock_s']):
                raise ValueError('Native clocks changed during calibration')
            if not calibrated[keys].equals(raw[keys]):
                raise ValueError('Native identity/order changed during calibration')
            for a in 'xyz':
                if not np.array_equal(calibrated['b'+a+'_signed'],raw['b'+a+'_signed']):
                    raise ValueError('Original signed counts changed')
            raw['source_sample_id'] = calibrated['source_sample_id']
            raw['clock_partition'] = date
            raw['source_file'] = raw['source_file'].map(lambda p:Path(p).name)
            raw_columns = ['source_sample_id','source_file','record_index','sample_index','clock_partition',
                           'clock_s','utc','utc_iso','utc_valid','utc_leap_second','bx','by','bz',
                           'bx_signed','by_signed','bz_signed','range','science_valid','calibration_mode',
                           'adc_at_limit','range_changed','mag_frame_matches']
            pseudo_columns = ['source_sample_id','utc','utc_iso','clock_s','range','Bx_SCD','By_SCD','Bz_SCD',
                              'calibration_valid','calibration_model_available','calibration_failure_reason',
                              'spin_phase_valid','spin_phase_deg','spin_period_s','spin_offset_valid','sunpulse_estimated','sunpulse_uncertainty_s',
                              'calibration_basis','high_frequency_validated']
            extra_columns = ['decoder_science_valid', 'archive_noisy', 'archive_gap', 'archive_unplaceable_interval', 'archive_interval_ids',
                'archive_quality_assessed', 'archive_quality_valid', 'maneuver_candidate', 'maneuver_timing_uncertain', 'post_maneuver_settling',
                'maneuver_quality_assessed', 'mager_command_transition', 'mager_command_ids', 'mager_command_cohort',
                'mager_command_time_basis', 'attitude_quality_valid', 'attitude_stale', 'attitude_sun_sensor_only',
                'science_quality_valid', 'science_quality_assessed']
            extra_columns += list(MAG_HARDWARE_FIELDS) + ['mag_hardware_state_id', 'mag_hardware_cohort',
                'mag_hardware_state_source', 'mag_hardware_state_valid', 'mag_hardware_transition',
                'calibration_hardware_scoped']
            raw_columns += extra_columns
            pseudo_columns += extra_columns + ['calibration_model_id', 'calibration_source', 'cohort']
            _csv(raw[[c for c in raw_columns if c in raw]],target/'raw_mag_9hz.csv.gz')
            _csv(calibrated[[c for c in pseudo_columns if c in calibrated]],target/'pseudo_mag_9hz.csv.gz')
            _csv(offsets,target/'local_offsets.csv')
            _csv(mag_hardware_calibration_checks(calibrated),target/'mag_hardware_checks.csv')
            _csv(events,target/'quality_events.csv')
            if not subcom.empty:
                _csv(subcom.drop(columns=['raw_dsc'], errors='ignore'),target/'digital_subcom.csv')
            if 'overlap_records' in products:
                _csv(products['overlap_records'],target/'overlap_records.csv')
            files = []
            for path in sorted(target.glob('*.csv*')):
                files.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha256(path)))
            record.update(status='processed', raw_rows=len(raw), calibrated_rows=int(calibrated['calibration_valid'].sum()),
                          utc_valid_rows=int(raw['utc'].notna().sum()), payload_audit=audit, products=files, sunpulse_inputs=sunpulse_inputs,
                          utc_start=str(raw['utc'].min()),utc_stop=str(raw['utc'].max()),
                          ert_start=str(frames['ert'].min()), ert_stop=str(frames['ert'].max()),
                          quality_event_rows=len(events), timing_error=timing_error,
                          elapsed_s=round(time.monotonic()-started,3))
            done.write_text(json.dumps(record,indent=2))
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        records.append(record)
        (root/'processing_status.json').write_text(json.dumps(records,indent=2))
        print(date,record['status'],record.get('raw_rows',0),record.get('calibrated_rows',0),record.get('error') or '',flush=True)
    dates = pd.date_range(cfg['mission_start'],cfg['mission_stop'],freq='D')
    calendar = pd.DataFrame({'date':dates.strftime('%Y-%m-%d')})
    count = inventory.groupby('date').size().rename('local_source_files').reset_index()
    calendar = calendar.merge(count,on='date',how='left').fillna({'local_source_files':0})
    calendar['status'] = calendar['date'].map({r['date']:r['status'] for r in records}).fillna('missing_local_inputs')
    calendar.to_csv(root/'mission_coverage.csv',index=False)
    event_tables = [ancillary_events(quality)]
    event_tables += [pd.read_csv(root/'data'/r['date']/'quality_events.csv') for r in records if r['status']=='processed']
    events = combine_events(event_tables)
    _csv(events, root/'quality_events.csv')
    ancillary_manifest = json.loads(Path(cfg['ancillary_manifest']).read_text()) if cfg.get('ancillary_manifest') else {}
    event_manifest = dict(schema_version=SCHEMA_VERSION, configuration_sha256=fingerprint,
        event_count=len(events), source_type_counts=events['source_type'].value_counts().to_dict(),
        event_type_counts=events['event_type'].value_counts().to_dict(),
        telemetry_dates=sum(r['status']=='processed' for r in records),
        mission_calendar_days=len(calendar), command_tables=len(command_paths),
        ancillary_coverage=ancillary_manifest, settings=quality.settings,
        sunpulse_selection=pulse_selection, event_csv_sha256=sha256(root/'quality_events.csv'),
        time_basis_note='SCET, snapshot brackets, packet SCET, ERT and DSN transmit proxies must not be conflated',
        interpretation='Potential review intervals, not confirmed instrumental faults or plasma events',
        input_hashes=provenance)
    (root/'quality_events_manifest.json').write_text(json.dumps(event_manifest, indent=2))
    summary = dict(release_status='LOCAL_SUBSET_COMPLETE' if all(r['status']=='processed' for r in records) else 'LOCAL_SUBSET_PARTIAL', publication_ready=False,
                   mission_start=cfg['mission_start'],mission_stop=cfg['mission_stop'],
                   calendar_days=len(calendar),local_source_files=len(inventory),
                   processed_dates=sum(r['status']=='processed' for r in records),
                   missing_input_dates=int(calendar['local_source_files'].eq(0).sum()),
                   failed_dates=sum(r['status']=='failed' for r in records),
                   raw_rows=sum(r.get('raw_rows',0) for r in records),
                   pseudo_calibrated_rows=sum(r.get('calibrated_rows',0) for r in records),
                   official_inventory_complete=False, high_frequency_validated=False,
                   quality_events=len(events), quality_event_catalog='quality_events.csv')
    (root/'release_status.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    args=parser.parse_args()
    print(json.dumps(build_native_release(json.loads(args.config.read_text())),indent=2))


if __name__=='__main__':
    main()
