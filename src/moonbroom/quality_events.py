"""Reproducible MAG/ER review intervals from observed telemetry and uplink logs.

Catalog rows describe evidence, including bracketed changes and unproven command
requests. They are not a list of confirmed instrumental faults or a plasma-event
catalog. SCET, packet-frame SCET, ERT, and transmit-time proxies stay distinct.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ['event_id', 'source_type', 'event_type', 'instruments', 'start_utc', 'stop_utc',
           'time_basis', 'clock_start_s', 'clock_stop_s', 'clock_partition',
           'evidence', 'recommendation', 'selection_rule', 'source_file',
           'record_start', 'record_stop', 'command_event_id', 'open_end', 'details_json']
SCHEMA_VERSION = 1


def _iso(value):
    return pd.Timestamp(value).isoformat() if pd.notna(value) else None


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def event_row(kind, *, instruments='MAG,ER', start=pd.NaT, stop=pd.NaT,
              basis='SCET', source_type='telemetry', source_file='', record_start=None,
              record_stop=None, clock_start=np.nan, clock_stop=np.nan, partition='',
              evidence='observed flag', recommendation='review', rule='', command_id='',
              open_end=False, details=None):
    row = dict(source_type=source_type, event_type=kind, instruments=instruments,
               start_utc=_iso(start), stop_utc=_iso(stop), time_basis=basis,
               clock_start_s=float(clock_start), clock_stop_s=float(clock_stop),
               clock_partition=partition, evidence=evidence, recommendation=recommendation,
               selection_rule=rule, source_file=Path(source_file).name if source_file else '',
               record_start=record_start, record_stop=record_stop,
               command_event_id=command_id, open_end=bool(open_end),
               details_json=json.dumps(_clean(details or {}), sort_keys=True, default=str, allow_nan=False))
    row['event_id'] = hashlib.sha256(json.dumps(_clean(row), sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()[:24]
    return row


def _runs(table, mask, *, kind, time_col, clock_col, interval_s, partition='', **options):
    """Coalesce adjacent flagged observations within one original source file."""
    indices = np.flatnonzero(pd.Series(mask, index=table.index).fillna(False).to_numpy(bool))
    if not len(indices):
        return []
    clocks = table[clock_col].to_numpy(float)
    sources = table['source_file'].astype(str).to_numpy()
    breaks = (np.diff(indices) != 1) | (np.diff(clocks[indices]) > interval_s * 1.01)
    breaks |= sources[indices[1:]] != sources[indices[:-1]]
    groups = np.split(indices, np.flatnonzero(breaks) + 1)
    rows = []
    for group in groups:
        first, last = table.iloc[group[0]], table.iloc[group[-1]]
        start, stop = first[time_col], last[time_col]
        stop = stop + pd.Timedelta(seconds=interval_s) if pd.notna(stop) else pd.NaT
        basis = options.get('basis', 'SCET') if pd.notna(start) and pd.notna(stop) else 'SPACECRAFT_CLOCK_ONLY'
        kwargs = dict(options, basis=basis)
        rows.append(event_row(kind, start=start, stop=stop, partition=partition,
                              source_file=first.source_file, record_start=int(first.record_index),
                              record_stop=int(last.record_index), clock_start=clocks[group[0]],
                              clock_stop=clocks[group[-1]] + interval_s, **kwargs))
    return rows


def telemetry_events(frames, mag, subcom, *, phase=None, partition=''):
    """Select digital MAG flags, packet diagnostics and sampled MAG/ER state changes.

    Counter/setting changes are bracketed by consecutive available valid
    snapshots, including gaps. No zero is substituted for a missing subcom word.
    A latched error code does not establish when the original error occurred.
    """
    rows = []
    for field, kind in [('calibration_mode', 'mag_diagnostic_mode'), ('adc_at_limit', 'mag_adc_endpoint'),
                         ('range_changed', 'mag_range_transition'), ('mag_frame_matches', 'mag_frame_mismatch')]:
        if field in mag:
            mask = ~mag[field] if field == 'mag_frame_matches' else mag[field]
            rows.extend(_runs(mag, mask, kind=kind, instruments='MAG', time_col='utc', clock_col='clock_s',
                              interval_s=1/9, partition=partition, recommendation='exclude_flagged_samples',
                              rule=field + (' == false' if field == 'mag_frame_matches' else ' == true')))
    if phase is not None:
        for column, kind, invert in [('spin_phase_valid', 'spin_phase_unsupported', True),
                                      ('sunpulse_estimated', 'estimated_sunpulse_phase', False)]:
            if column in phase:
                mask = ~phase[column] if invert else phase[column]
                rows.extend(_runs(phase, mask, kind=kind, instruments='MAG', time_col='utc', clock_col='clock_s',
                                  interval_s=1/9, partition=partition,
                                  recommendation='exclude_despun_field' if invert else 'review_phase_uncertainty',
                                  rule=column + (' == false' if invert else ' == true')))
    for mask, kind, recommendation in [
        (~frames['header_valid'], 'invalid_telemetry_header', 'exclude_record'),
        (frames['stream_comparison'].eq(2), 'telemetry_stream_mismatch', 'review_selected_stream'),
        (frames.get('ert_anchor_rejected', pd.Series(False, index=frames.index)), 'rejected_ert_anchor', 'review_timing')]:
        # Packet diagnostics are anchored to receipt time, not burst collection.
        rows.extend(_runs(frames, mask, kind=kind, time_col='ert', clock_col='frame_clock_s', interval_s=2,
                          partition=partition, basis='ERT', recommendation=recommendation,
                          rule=kind, evidence='packet metadata'))
    payload = np.frombuffer(b''.join(frames['mag_er']), dtype=np.uint8).reshape(-1, 168)
    types = payload[:, 0] & 63
    full_burst = (types >= 36) & (types <= 62) & ((payload[:, 0] & 128) != 0)
    rows.extend(_runs(frames, full_burst, kind='full_burst_playback', time_col='frame_utc',
                      clock_col='frame_clock_s', interval_s=2, partition=partition,
                      basis='PACKET_FRAME_SCET', evidence='packet type', recommendation='review_realtime_coverage',
                      rule='full burst frame has no real-time MAG payload; burst collection is earlier'))
    for index in np.flatnonzero(frames['gap_frames'].gt(0)):
        if index == 0:
            continue
        previous, current = frames.iloc[index-1], frames.iloc[index]
        start = previous.frame_utc + pd.Timedelta(seconds=2) if pd.notna(previous.frame_utc) else pd.NaT
        rows.append(event_row('counter_coverage_gap', start=start, stop=current.frame_utc,
                    basis='PACKET_FRAME_SCET' if pd.notna(start) and pd.notna(current.frame_utc) else 'SPACECRAFT_CLOCK_ONLY',
                    clock_start=previous.frame_clock_s + 2, clock_stop=current.frame_clock_s, partition=partition,
                    source_file=current.source_file, record_start=int(current.record_index),
                    record_stop=int(current.record_index), evidence='missing counter interval in supplied inputs',
                    recommendation='coverage_gap', rule='gap_frames > 0',
                    details=dict(missing_frames=int(current.gap_frames), previous_file=Path(previous.source_file).name,
                                 previous_record=int(previous.record_index),
                                 caveat='May represent missing local files rather than a spacecraft outage')))
    if not subcom.empty:
        state = subcom.copy().reset_index(drop=True)
        fields = {
            'command_count': 'MAG,ER', 'reset_count': 'MAG,ER', 'error_count': 'MAG,ER',
            'error_code': 'MAG,ER', 'telemetry_version': 'MAG,ER', 'electronics_side': 'MAG',
            'autoranging': 'MAG', 'boom_deployed': 'MAG', 'boot_prom': 'MAG,ER',
            'eeprom_write_enabled': 'MAG,ER', 'er_power': 'ER', 'sweep_max': 'ER',
            'sweep_rate': 'ER', 'mcp_setting': 'ER', 'mcp_enabled': 'ER',
            'sweep_enabled': 'ER', 'er_cover_closed': 'ER', 'hv_plug_enabled': 'ER', 'pll_status': 'ER'}
        for field, instruments in fields.items():
            if field not in state:
                continue
            valid = state[field].notna()
            if instruments == 'ER' and field != 'er_power':
                valid &= state['er_power'].eq(1)
            selected = state.loc[valid]
            previous = None
            for row in selected.itertuples():
                value = getattr(row, field)
                if previous is not None and value != getattr(previous, field):
                    before = getattr(previous, field)
                    details = dict(field=field, previous=before, current=value,
                                   previous_file=Path(previous.source_file).name,
                                   previous_record=int(previous.record_index),
                                   snapshot_gap_s=float(row.major_clock_s - previous.major_clock_s))
                    if field in {'command_count', 'error_count', 'reset_count'}:
                        modulo = 256 if field == 'command_count' else 16
                        details['minimum_modulo_increment'] = int((value - before) % modulo)
                        details['caveat'] = 'Multiple wraps and exact occurrence times cannot be recovered from these snapshots'
                    rows.append(event_row('subcom_' + field + '_change', instruments=instruments,
                        start=previous.major_utc, stop=row.major_utc,
                        basis='SCET_BRACKET' if pd.notna(previous.major_utc) and pd.notna(row.major_utc) else 'SPACECRAFT_CLOCK_ONLY',
                        source_file=row.source_file, record_start=int(row.record_index), record_stop=int(row.record_index),
                        clock_start=previous.major_clock_s, clock_stop=row.major_clock_s, partition=partition,
                        evidence='change between available valid subcom snapshots',
                        rule=field + ' differs between consecutive available valid snapshots', details=details))
                previous = row
        for field, bad_value in [('er_power', 0), ('mcp_enabled', 0), ('sweep_enabled', 0), ('er_cover_closed', 1)]:
            valid = state[field].eq(bad_value)
            if field != 'er_power':
                valid &= state['er_power'].eq(1)
            rows.extend(_runs(state, valid, kind='subcom_' + field + '_state', instruments='ER',
                              time_col='major_utc', clock_col='major_clock_s', interval_s=32, partition=partition,
                              evidence='sampled status; held for nominal snapshot interval',
                              rule=f'{field} == {bad_value}; other ER status requires er_power == 1'))
        rows.extend(_runs(state, ~state['complete'], kind='partial_status_snapshot', time_col='major_utc',
                          clock_col='major_clock_s', interval_s=32, partition=partition,
                          evidence='subcom word mask', recommendation='context_only', rule='word_mask != 65535'))
        rows.extend(_runs(state, state['error_code'].notna() & state['error_code'].ne(0),
                          kind='latched_instrument_error', time_col='major_utc', clock_col='major_clock_s',
                          interval_s=32, partition=partition, rule='most recent error code != 0',
                          evidence='latched status; original error may precede these snapshots'))
    return pd.DataFrame(rows, columns=COLUMNS)


def ancillary_events(quality):
    """Preserve archive outages and command evidence, including failed requests."""
    rows = []
    for row in quality.outages.itertuples():
        if not getattr(row, 'interval_valid', True):
            rows.append(event_row('archive_interval_unplaceable', source_type='archive',
                basis='UNRESOLVED_ARCHIVE_TIME', source_file=row.source_file,
                record_start=int(row.record_index), record_stop=int(row.record_index),
                evidence='invalid bounds in official archive table', recommendation='review_source_record',
                rule='retain original bounds; do not guess corrected dates',
                details=dict(start_original=row.start_utc_original, stop_original=row.stop_utc_original,
                             product_id=row.product_id, description=row.description, status=row.status)))
            continue
        rows.append(event_row('archive_noise' if row.noisy else 'archive_outage', source_type='archive',
            start=row.start_utc, stop=row.stop_utc + pd.Timedelta(seconds=quality.settings['outage_stop_padding_s']),
            basis='ERT', source_file=row.source_file, record_start=int(row.record_index), record_stop=int(row.record_index),
            evidence='official cumulative OUTAGES table', recommendation='exclude_received_records' if row.unresolved else 'context_only',
            rule='archived interval, including stated stop minute',
            details=dict(description=row.description, status=row.status, product_id=row.product_id,
                         volume_id=row.volume_id, unresolved=bool(row.unresolved))))
    commands = quality.commands
    if not commands.empty:
        arrival = commands['arrival_scet_valid'] & commands['arrival_scet_utc'].notna()
        epochs = commands['arrival_scet_utc'].where(arrival, commands['transmit_utc'])
        by_id = pd.Series(epochs.array, index=commands['command_event_id']).groupby(level=0).last()
        additional = {'SUNOFF', 'SUNON', 'EMOFF', 'EMON', 'SETTIM', 'T300', 'T3600', 'TLMOFF', 'TLMON'}
        selected = commands['may_change_science'] | commands['mager_action_unmapped'] | commands['mnemonic'].isin(additional)
        for index, row in commands.loc[selected].iterrows():
            epoch = epochs.loc[index]
            end = by_id.get(row.following_exec_event_id, epoch)
            if pd.isna(end):
                end = epoch
            failed = bool(row.known_execution_failure)
            kind = 'known_command_failure' if failed else 'commanded_action_candidate'
            pad = quality.settings['mager_command_padding_s']
            rows.append(event_row(kind, source_type='command', instruments=row.affected_instruments or 'MAG,ER',
                start=epoch - pd.Timedelta(seconds=pad), stop=max(epoch, end) + pd.Timedelta(seconds=pad),
                basis='ESTIMATED_ARRIVAL_SCET' if arrival.loc[index] else 'DSN_TRANSMIT_UTC_PROXY',
                source_file=row.source_file, record_start=int(row.record_index), record_stop=int(row.record_index),
                command_id=row.command_event_id, evidence=row.execution_evidence,
                recommendation='review_failed_request' if failed or not row.transmission_confirmed else 'review_transition',
                rule='science-changing MAG/ER or spacecraft timing/sensor request; no inferred persistent state',
                details=dict(command=row.command, requested_action=row.requested_action, transmission_status=row.status,
                             station=str(row.station), transmit_utc=_iso(row.transmit_utc),
                             arrival_scet_utc=_iso(row.arrival_scet_utc), following_exec_event_id=row.following_exec_event_id)))
    for row in quality.maneuvers.itertuples():
        uncertain = pd.isna(row.stop_utc) or (row.stop_utc - row.start_utc).total_seconds() > quality.settings['max_maneuver_exclusion_s']
        rows.append(event_row('maneuver_and_settling_candidate', source_type='command', start=row.start_utc,
            stop=row.settled_utc, basis=row.time_basis, source_file=row.source_file, open_end=pd.isna(row.settled_utc),
            evidence='confirmed thruster selection and EXEC, or STOPFIRE',
            recommendation='review_wide_or_open_bracket' if uncertain else 'review_maneuver_and_settling',
            rule='candidate firing plus configured settling interval; SETCMDREG does not end firing',
            details=dict(command=row.command, firing_stop_utc=_iso(row.stop_utc),
                         settling_s=quality.settings['maneuver_settling_s'], timing_uncertain=uncertain)))
    return pd.DataFrame(rows, columns=COLUMNS)


def combine_events(tables):
    tables = [table for table in tables if not table.empty]
    if not tables:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(tables, ignore_index=True).drop_duplicates('event_id')
    return out.sort_values(['start_utc', 'source_type', 'event_type', 'event_id'], na_position='last').reset_index(drop=True)
