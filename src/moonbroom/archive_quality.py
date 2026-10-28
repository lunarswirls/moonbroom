"""LP Level-0 ancillary selection and traceable science-quality annotations.

Formats: VOLSIS, INDEX/OUTAGES.LBL, GEOMETRY/ATTITUDE.LBL and COMMAND/*.LBL.
Policy choices (minute endpoint padding, attitude age and settling time) are
recorded explicitly. Command transmission is evidence for a candidate maneuver,
not proof of execution. No rows or original digital measurements are removed.
"""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .command_history import annotate_command_actions, flag_command_transitions


OUTAGE_COLUMNS = ['volume_id', 'product_id', 'start_utc', 'stop_utc', 'description', 'status']
ATTITUDE_COLUMNS = ['utc', 'spin_rate_rpm', 'longitude_deg', 'latitude_deg']
MANEUVER_COLUMNS = ['start_utc', 'stop_utc', 'settled_utc', 'command', 'source_file', 'time_basis']
THRUSTER_COMMANDS = {'A1', 'A2', 'A3', 'A4', 'A1234REOR', 'A12LOI', 'A14REOR',
                     'A23REOR', 'A34VEC', 'T12VEC', 'T1D', 'T2UP'}
DEFAULTS = dict(outage_stop_padding_s=60.0, maneuver_padding_s=5.0,
                maneuver_settling_s=28800.0, max_attitude_age_s=172800.0,
                mager_command_padding_s=5.0, max_maneuver_exclusion_s=7200.0)


def _volume(path):
    matches = [int(m.group(1)) for part in Path(path).parts
               if (m := re.fullmatch(r'lp_(\d{4})', part, re.I))]
    return matches[-1] if matches else 0


def _data_path(path):
    path = Path(path)
    if path.suffix.lower() != '.lbl':
        return path
    match = re.search(r'(?mi)^\^TABLE\s*=\s*"([^"]+)"', path.read_text(encoding='ascii'))
    if not match or Path(match[1]).name != match[1]:
        raise ValueError(f'Expected a local detached TABLE pointer: {path}')
    choices = [item for item in path.parent.iterdir() if item.name.lower() == match[1].lower()]
    if len(choices) != 1:
        raise ValueError(f'Expected one TABLE file for {path}')
    return choices[0]


def select_sunpulse_paths(paths):
    """Prefer the explicitly superseding LP_0019 S9812114; audit every choice.

    Other duplicate products must have identical binary bytes. A later volume
    number alone does not authorize silently replacing a conflicting product.
    A label/binary pair supplied together is one product, not a conflict.
    """
    groups = {}
    for supplied in paths:
        path = Path(supplied).resolve()
        data = _data_path(path)
        key = data.stem.lower()
        if not re.fullmatch(r's\d{7}(?:_excerpt)?', key):
            raise ValueError(f'Unrecognized SUNPULSE product: {path}')
        groups.setdefault(key, {})[str(data.resolve())] = (
            path if path.suffix.lower() == '.lbl' else
            groups.get(key, {}).get(str(data.resolve()), path))
    selected, decisions = [], []
    for product, copies in sorted(groups.items()):
        candidates = sorted(copies.values(), key=lambda p: (_volume(p), str(p)))
        replacement = [p for p in candidates if product == 's9812114' and _volume(p) == 19]
        eligible = replacement or candidates
        hashes = {hashlib.sha256(_data_path(p).read_bytes()).hexdigest() for p in eligible}
        if len(hashes) != 1:
            raise ValueError(f'Conflicting SUNPULSE copies without replacement precedence: {product}')
        chosen = eligible[-1]
        selected.append(str(chosen))
        decisions.append(dict(product=product.upper(), selected=str(chosen),
                              superseded=[str(p) for p in candidates if p != chosen],
                              rule='LP_0019 errata replacement' if replacement else 'identical copies or unique product',
                              binary_sha256=next(iter(hashes))))
    return selected, decisions


def read_sunpulse_products(paths, *, frames=None, strict=True):
    """Select before decoding, so obsolete pulses cannot survive deduplication."""
    from .level0 import read_sunpulse
    selected, decisions = select_sunpulse_paths(paths)
    tables = [read_sunpulse(path, frames=frames, strict=strict) for path in selected]
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=['timing_valid', 'pulse_clock_s', 'estimated', 'uncertainty_s'])
    result.attrs['product_selection'] = decisions
    return result


def _read_table(path, columns, *, fixed_spans=None):
    source = Path(path)
    data = _data_path(source)
    with data.open(encoding='ascii', newline='') as stream:
        rows = [[value.strip() for value in row] for row in csv.reader(stream) if row]
    if fixed_spans and any(len(row) != len(columns) for row in rows):
        # The official OUTAGES.TAB has a missing comma in one 168-character
        # record. Its labelled byte fields remain intact and authoritative.
        lines = data.read_text(encoding='ascii').splitlines()
        if all(len(line) == 168 for line in lines):
            rows = [[line[start:stop].strip() for start, stop in fixed_spans] for line in lines]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f'Expected {len(columns)} columns in {data}')
    if source.suffix.lower() == '.lbl':
        match = re.search(r'(?m)^FILE_RECORDS\s*=\s*(\d+)', source.read_text(encoding='ascii'))
        if match and len(rows) != int(match[1]):
            raise ValueError(f'TABLE record count differs from label: {source}')
    result = pd.DataFrame(rows, columns=columns)
    result['source_file'] = str(data.resolve())
    result['record_index'] = np.arange(len(result))
    return result


def read_outages(path, *, strict=True):
    """Read the official six-column cumulative OUTAGES table without filtering."""
    out = _read_table(path, OUTAGE_COLUMNS, fixed_spans=[(1, 8), (11, 33), (36, 52),
                                                       (55, 71), (74, 140), (143, 167)])
    for name in ('start_utc', 'stop_utc'):
        out[name + '_original'] = out[name]
        out[name] = pd.to_datetime(out[name], utc=True, errors='coerce')
    out['interval_valid'] = out['start_utc'].notna() & out['stop_utc'].notna() & out['stop_utc'].ge(out['start_utc'])
    if strict and not out['interval_valid'].all():
        raise ValueError('Outage intervals require ordered finite UTC bounds')
    out['noisy'] = out['description'].str.contains(r'NOIS[YE]', case=False, regex=True)
    # Preserve the status verbatim; only explicit repair statuses clear a mask.
    out['unresolved'] = ~out['status'].str.fullmatch(r'(?:RECOVERED|CORRECTED|FILLED)', case=False)
    out['interval_id'] = ['outage:' + str(i) for i in range(len(out))]
    return out


def read_attitude(path):
    out = _read_table(path, ATTITUDE_COLUMNS)
    out['utc'] = pd.to_datetime(out['utc'], utc=True, errors='raise')
    for name in ATTITUDE_COLUMNS[1:]:
        out[name] = pd.to_numeric(out[name], errors='raise')
    if (out['utc'].isna().any() or out['utc'].duplicated().any()
            or not np.isfinite(out[ATTITUDE_COLUMNS[1:]]).all().all()
            or out['spin_rate_rpm'].le(0).any() or out['latitude_deg'].abs().gt(90).any()):
        raise ValueError('Attitude requires unique UTC and finite physical angles/spin rates')
    out = out.sort_values('utc').reset_index(drop=True)
    out.attrs['attitude_frame'] = 'ECLIPJ2000'
    return out


def read_commands(paths):
    tables = [_read_table(path, ['utc', 'command', 'status', 'station']) for path in paths]
    if not tables:
        return pd.DataFrame(columns=['utc', 'command', 'status', 'station', 'source_file'])
    out = pd.concat(tables, ignore_index=True)
    out['utc'] = pd.to_datetime(out['utc'], utc=True, errors='raise')
    out['status'] = out['status'].str.upper()
    if out['utc'].isna().any() or not out['status'].isin(['CONFIRM', 'ABORT', 'SKIP']).all():
        raise ValueError('Commands require finite UTC and recognized transmission status')
    out['mnemonic'] = out['command'].str.split(',').str[0].str.strip().str.upper()
    # ERRATA: identical commands at the same time may be listed twice.
    return out.sort_values('utc', kind='stable').drop_duplicates(['utc', 'command', 'status', 'station'])


def maneuver_intervals(commands, *, settling_s=28800.0, padding_s=5.0):
    """Candidate burns require a confirmed thruster selection followed by EXEC.

    Heater EXEC commands do not trigger burns. STOPFIRE also supplies evidence
    of a settling interval when the start is absent. Unclosed burns stay open;
    transmission status is never promoted to proof of successful execution.
    """
    if not np.isfinite([settling_s, padding_s]).all() or min(settling_s, padding_s) < 0:
        raise ValueError('Maneuver padding and settling must be finite and nonnegative')
    pending, active, active_command, rows = None, None, None, []
    active_basis = None
    confirmed = commands['status'].eq('CONFIRM')
    if 'known_execution_failure' in commands:
        confirmed &= ~commands['known_execution_failure']
    for row in commands.loc[confirmed].itertuples():
        arrival = getattr(row, 'arrival_scet_valid', False) and pd.notna(row.arrival_scet_utc)
        event_time = row.arrival_scet_utc if arrival else row.utc
        basis = 'estimated spacecraft arrival SCET' if arrival else 'DSN transmit UTC proxy'
        if row.mnemonic in THRUSTER_COMMANDS:
            pending = row
        elif row.mnemonic == 'SETCMDREG':
            # Appendix B: resets parameter registers, not thruster execution.
            pending = None
        elif row.mnemonic == 'STOPFIRE':
            if active is not None or row.mnemonic == 'STOPFIRE':
                event = active if active is not None else row._replace(utc=event_time)
                rows.append([event.utc - pd.Timedelta(seconds=padding_s),
                             event_time + pd.Timedelta(seconds=padding_s),
                             event_time + pd.Timedelta(seconds=padding_s + settling_s),
                             active_command if active is not None else event.command, event.source_file,
                             f'start: {active_basis if active is not None else basis}; stop: {basis}'])
            pending, active = None, None
        elif row.mnemonic == 'EXEC':
            if pending is not None and active is None:
                active = row._replace(utc=event_time)
                active_basis = basis
                active_command = pending.command
            pending = None
        elif row.mnemonic.endswith('HTR'):
            pending = None
    if active is not None:
        rows.append([active.utc - pd.Timedelta(seconds=padding_s), pd.NaT, pd.NaT,
                     active_command, active.source_file, f'start: {active_basis}; stop unavailable'])
    return pd.DataFrame(rows, columns=MANEUVER_COLUMNS)


@dataclass
class ArchiveQuality:
    outages: pd.DataFrame
    commands: pd.DataFrame
    maneuvers: pd.DataFrame
    attitude: pd.DataFrame
    settings: dict
    manifest: dict


def load_archive_quality(downloads=(), settings=None, *, timing_kernels=(), transmitter=None):
    options = dict(DEFAULTS, **(settings or {}))
    if (set(options) != set(DEFAULTS)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in options.values())
            or not np.isfinite(list(options.values())).all() or min(options.values()) < 0):
        raise ValueError('Unknown archive quality setting or nonfinite/negative duration')
    rows = list(downloads)
    def latest(name):
        candidates = [row for row in rows if Path(row['relative_path']).name.lower() == name]
        return max(candidates, key=lambda row: (_volume(row['relative_path']), row['relative_path'])) if candidates else None
    outage, attitude = latest('outages.lbl'), latest('attitude.lbl')
    command_paths = [row['path'] for row in rows if re.search(r'/command/c\d{7}\.lbl$', row['relative_path'], re.I)]
    commands = annotate_command_actions(read_commands(command_paths))
    if timing_kernels and transmitter and not commands.empty:
        from .timing import reconstruct_command_times
        commands = reconstruct_command_times(commands, timing_kernels, transmitter=transmitter)
    maneuvers = maneuver_intervals(commands, settling_s=options['maneuver_settling_s'], padding_s=options['maneuver_padding_s'])
    return ArchiveQuality(
        read_outages(outage['path'], strict=False) if outage else pd.DataFrame(columns=OUTAGE_COLUMNS),
        commands, maneuvers,
        read_attitude(attitude['path']) if attitude else pd.DataFrame(columns=ATTITUDE_COLUMNS),
        options, dict(settings=options, outages_source=outage['relative_path'] if outage else None,
                      attitude_source=attitude['relative_path'] if attitude else None,
                      command_sources=sorted(command_paths), attitude_frame='ECLIPJ2000',
                      outage_time_basis='merged telemetry ERT; two-second frame overlap',
                      command_transmitter=transmitter, command_scet_valid_rows=int(commands['arrival_scet_valid'].sum()),
                      command_execution_evidence='transmission only, except documented failures',
                      maneuver_time_basis='SPICE uplink arrival SCET where available; padded DSN transmit proxy otherwise'))


def quality_from_paths(*, command_paths=(), outage_path=None, attitude_path=None,
                       settings=None, timing_kernels=(), transmitter='EARTH'):
    """Load explicitly identified ancillary files for a local release subset."""
    quality = load_archive_quality(settings=settings)
    quality.commands = annotate_command_actions(read_commands(command_paths))
    if timing_kernels and not quality.commands.empty:
        from .timing import reconstruct_command_times
        quality.commands = reconstruct_command_times(quality.commands, timing_kernels, transmitter=transmitter)
    quality.maneuvers = maneuver_intervals(quality.commands,
        settling_s=quality.settings['maneuver_settling_s'], padding_s=quality.settings['maneuver_padding_s'])
    if outage_path:
        quality.outages = read_outages(outage_path, strict=False)
    if attitude_path:
        quality.attitude = read_attitude(attitude_path)
    quality.manifest.update(command_sources=[str(Path(path).resolve()) for path in command_paths],
                            outages_source=str(Path(outage_path).resolve()) if outage_path else None,
                            attitude_source=str(Path(attitude_path).resolve()) if attitude_path else None,
                            command_transmitter=transmitter,
                            command_scet_valid_rows=int(quality.commands['arrival_scet_valid'].sum()))
    return quality


def flag_archive_frames(frames, quality):
    """Compare archive intervals with reception times, including clock-only runs."""
    out = frames.copy()
    start = pd.to_datetime(out['ert'], utc=True, format='mixed')
    stop = start + pd.Timedelta(seconds=2)
    out['archive_quality_assessed'] = start.notna() & (quality.manifest['outages_source'] is not None)
    out['archive_noisy'] = False
    out['archive_gap'] = False
    out['archive_unplaceable_interval'] = False
    out['archive_interval_ids'] = ''
    earliest, latest = start.min(), stop.max()
    for row in quality.outages.itertuples():
        if not getattr(row, 'interval_valid', True):
            # Associate an unresolved record with its named telemetry product,
            # without correcting its dates or assigning an exclusion interval.
            match = re.match(r'(\d{2})_(\d{3})_(\d{2})\d{2}_', row.product_id)
            if match:
                name = 'm' + ''.join(match.groups()) + '.b'
                selected = out['source_file'].map(lambda value: Path(value).name.lower()).eq(name)
                out.loc[selected, 'archive_unplaceable_interval'] = True
                out.loc[selected, 'archive_quality_assessed'] = False
                out.loc[selected, 'archive_interval_ids'] += row.interval_id + ';'
            continue
        end = row.stop_utc + pd.Timedelta(seconds=quality.settings['outage_stop_padding_s'])
        if row.start_utc >= latest or end <= earliest:
            continue
        selected = start.lt(end) & stop.gt(row.start_utc)
        out.loc[selected, 'archive_interval_ids'] += row.interval_id + ';'
        if row.unresolved:
            out.loc[selected, 'archive_noisy' if row.noisy else 'archive_gap'] = True
    out['archive_quality_valid'] = ~(out['archive_noisy'] | out['archive_gap'])
    return out


def flag_science_quality(samples, quality, *, start_col='utc', stop_col=None, instrument=None):
    """Flag collection-time maneuvers and bounded orbit-average attitude support.

    Known archive or maneuver failures gate science_valid. Missing ancillary
    evidence is explicit in assessed/support flags, never claimed clean.
    Attitude support separately gates inertial rotations, not SCD counts.
    """
    out = samples.copy()
    start = pd.to_datetime(out[start_col], utc=True, format='mixed')
    stop = pd.to_datetime(out[stop_col], utc=True, format='mixed') if stop_col else start
    timed = start.notna() & stop.notna() & stop.ge(start)
    out['maneuver_quality_assessed'] = (timed & bool(quality.manifest['command_sources'])
                                       & quality.manifest.get('command_history_complete', True))
    out['maneuver_candidate'] = False
    out['maneuver_timing_uncertain'] = False
    out['post_maneuver_settling'] = False
    earliest, latest, has_time = start[timed].min(), stop[timed].max(), bool(timed.any())
    for row in quality.maneuvers.itertuples():
        if not has_time or row.start_utc > latest or (pd.notna(row.settled_utc) and row.settled_utc <= earliest):
            continue
        end = row.stop_utc if pd.notna(row.stop_utc) else pd.Timestamp.max.tz_localize('UTC')
        # Half-open collection intervals; equal bounds represent a point.
        overlap = timed & (stop.gt(row.start_utc) | (stop.eq(start) & start.eq(row.start_utc))) & start.lt(end)
        out.loc[overlap, 'maneuver_candidate'] = True
        uncertain = pd.isna(row.stop_utc) or (end - row.start_utc).total_seconds() > quality.settings['max_maneuver_exclusion_s']
        if uncertain:
            out.loc[overlap, 'maneuver_timing_uncertain'] = True
        if pd.notna(row.settled_utc):
            overlap = timed & (stop.gt(end) | (stop.eq(start) & start.eq(end))) & start.lt(row.settled_utc)
            out.loc[overlap, 'post_maneuver_settling'] = True
    out['attitude_utc'] = pd.Series(pd.NaT, index=out.index, dtype='datetime64[ns, UTC]')
    out['attitude_age_s'] = np.nan
    out['attitude_spin_rate_rpm'] = np.nan
    if not quality.attitude.empty:
        att = quality.attitude
        epochs = pd.DatetimeIndex(att['utc']).as_unit('ns').asi8
        indices = np.searchsorted(epochs, pd.DatetimeIndex(start).as_unit('ns').asi8, side='right') - 1
        selected = np.flatnonzero(timed.to_numpy() & (indices >= 0))
        chosen = att.iloc[indices[selected]]
        out.iloc[selected, out.columns.get_loc('attitude_utc')] = chosen['utc'].to_numpy()
        out.iloc[selected, out.columns.get_loc('attitude_spin_rate_rpm')] = chosen['spin_rate_rpm'].to_numpy()
        out.loc[:, 'attitude_age_s'] = (stop - out['attitude_utc']).dt.total_seconds()
    out['attitude_available'] = out['attitude_utc'].notna()
    out['attitude_stale'] = out['attitude_available'] & out['attitude_age_s'].gt(quality.settings['max_attitude_age_s'])
    out['attitude_sun_sensor_only'] = (
        (start.ge('1998-01-07') & start.lt('1998-01-16'))
        | (start.ge('1998-03-04') & start.lt('1998-04-13'))
        | (start.ge('1998-04-16') & start.lt('1998-04-17'))
        | (start.ge('1998-05-31') & start.lt('1999-08-01')))
    out['attitude_uncertainty_deg'] = np.where(out['attitude_available'], 1.0, np.nan)
    out['sunpulse_spin_supported'] = out['attitude_available'] & ~out['attitude_stale'] & out['attitude_spin_rate_rpm'].lt(15)
    maneuver_excluded = out['maneuver_candidate'] & ~out['maneuver_timing_uncertain']
    out['attitude_quality_valid'] = (out['attitude_available'] & ~out['attitude_stale']
                                     & ~maneuver_excluded & ~out['post_maneuver_settling'])
    if 'archive_quality_valid' not in out:
        out['archive_quality_valid'] = True
        out['archive_quality_assessed'] = False
    out = flag_command_transitions(out, quality.commands, start, stop,
                                   padding_s=quality.settings['mager_command_padding_s'], instrument=instrument)
    out['science_quality_valid'] = (out['archive_quality_valid'] & ~maneuver_excluded
                                    & ~out['post_maneuver_settling'] & ~out['mager_command_transition'])
    out['science_quality_assessed'] = (out['archive_quality_assessed'] & out['maneuver_quality_assessed']
                                      & ~out['maneuver_timing_uncertain'])
    previous = out['science_valid'] if 'science_valid' in out else True
    out['science_valid'] = previous & out['science_quality_valid']
    return out
