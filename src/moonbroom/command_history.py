"""MAG/ER command requests from LVL0SIS Appendix B, without inventing execution.

The separate MAG/ER Commanding document referenced by MAGERFMT is not supplied
here. Raw memory/table parameters therefore remain uninterpreted.
"""

import hashlib

import numpy as np
import pandas as pd


# mnemonic: (documented action, affected instruments, possible science change)
COMMAND_ACTIONS = {
    'MERABOFF': ('both electronics power off', 'MAG,ER', True),
    'MERAON': ('electronics A on, B off', 'MAG,ER', True),
    'MERBON': ('electronics B on, A off', 'MAG,ER', True),
    'MERHTROFF': ('heater power off', 'MAG,ER', True),
    'MERHTRON': ('heater power on', 'MAG,ER', True),
    'MERLVOFF': ('ER low voltage off', 'ER', True),
    'MERLVON': ('ER low voltage on', 'ER', True),
    'MRESET': ('processor reset', 'MAG,ER', True),
    'MRRST': ('master circuitry reset', 'MAG,ER', True),
    'ERCOV': ('ER cover open', 'ER', True),
    'MAGBOOM': ('MAG boom release', 'MAG', True),
    'SHVDIS': ('all high voltage supplies disabled', 'ER', True),
    'MBPROM': ('select PROM for next reset', 'MAG,ER', False),
    'MEEPROM': ('select EEPROM for next reset', 'MAG,ER', False),
    'MEEWEON': ('EEPROM write enable', 'MAG,ER', False),
    'MEEWEOFF': ('EEPROM write disable', 'MAG,ER', False),
    'MEECKSM': ('update EEPROM checksum', 'MAG,ER', False),
    'MBBYTE': ('load byte into load buffer', 'MAG,ER', False),
    'MBWORD': ('load word into load buffer', 'MAG,ER', False),
    'MBCOMMIT': ('commit load buffer to memory', 'MAG,ER', True),
    'MDBYTE': ('write memory byte', 'MAG,ER', True),
    'MDWORD': ('write memory word', 'MAG,ER', True),
    'MTLOADB': ('write table byte', 'MAG,ER', True),
    'MTLOADW': ('write table word', 'MAG,ER', True),
    'MEXEC': ('jump to memory address', 'MAG,ER', True),
    'MLADDR': ('set memory load address', 'MAG,ER', False),
    'MDADDR': ('set memory dump address', 'MAG,ER', False),
    'MTADDR': ('set table address', 'MAG,ER', False),
    'MDUMP': ('dump memory', 'MAG,ER', False),
    'MDUMPR': ('dump memory repeatedly', 'MAG,ER', False),
    'MTDUMP': ('dump table', 'MAG,ER', False),
    'MTDUMPR': ('dump table repeatedly', 'MAG,ER', False),
}


def annotate_command_actions(commands):
    """Preserve transmit UTC, classify requests, and identify a documented failure.

    A confirmed transmission does not demonstrate reception or execution.
    Selecting a boot image for the next reset is not an immediate image change.
    Unknown MAG/ER-like mnemonics are retained without guessed semantics.
    """
    out = commands.copy()
    if 'mnemonic' not in out:
        out['mnemonic'] = out['command'].str.split(',').str[0].str.strip().str.upper()
    out['transmit_utc'] = pd.to_datetime(out['utc'], utc=True, format='mixed')
    values = [COMMAND_ACTIONS.get(name, ('unmapped', '', False)) for name in out['mnemonic']]
    out['requested_action'] = [value[0] for value in values]
    out['affected_instruments'] = [value[1] for value in values]
    out['may_change_science'] = pd.Series([value[2] for value in values], index=out.index, dtype=bool)
    out['mager_action_known'] = out['mnemonic'].isin(COMMAND_ACTIONS)
    out['mager_action_unmapped'] = ~out['mager_action_known'] & out['mnemonic'].str.match(r'^(?:MER|MAG)')
    out['transmission_confirmed'] = out['status'].eq('CONFIRM')
    # ERRATA, LP_0001: the heater-off request did not execute (bad format).
    out['known_execution_failure'] = (out['mnemonic'].eq('MERHTROFF')
                                       & out['transmit_utc'].eq(pd.Timestamp('1998-01-23T00:49:53.200Z')))
    out['execution_evidence'] = 'not established by command log'
    out.loc[out['known_execution_failure'], 'execution_evidence'] = 'ERRATA: command did not execute'
    out.loc[~out['transmission_confirmed'], 'execution_evidence'] = 'transmission incomplete or skipped'
    out['command_event_id'] = [hashlib.sha256(
        (str(row.transmit_utc) + '|' + row.command + '|' + str(row.station) + '|' + row.status).encode()).hexdigest()[:20]
        for row in out.itertuples()]
    # Hardware requests may be followed by EXEC (e.g. C9802342's heater retry).
    # Record adjacency as evidence; it is not a proven execution association.
    hardware = {'MERABOFF', 'MERAON', 'MERBON', 'MERHTROFF', 'MERHTRON',
                'MERLVOFF', 'MERLVON', 'MRESET', 'MRRST', 'ERCOV', 'MAGBOOM', 'SHVDIS'}
    following = out['mnemonic'].shift(-1).eq('EXEC') & out['transmission_confirmed'].shift(-1, fill_value=False)
    out['following_exec_event_id'] = out['command_event_id'].shift(-1).where(
        following & out['mnemonic'].isin(hardware) & out['transmission_confirmed'])
    out['arrival_scet_utc'] = pd.Series(pd.NaT, index=out.index, dtype='datetime64[ns, UTC]')
    out['arrival_scet_utc_iso'] = pd.Series(None, index=out.index, dtype=object)
    out['arrival_scet_valid'] = False
    out['uplink_light_time_s'] = float('nan')
    out['command_time_basis'] = 'DSN transmission UTC; spacecraft arrival unavailable'
    return out


def flag_command_transitions(samples, commands, start, stop, *, padding_s=5.0, instrument=None):
    """Flag candidate science changes; retain requests and timing provenance.

    No persistent hardware state or new calibration is inferred from a request.
    Without uplink reconstruction the padded DSN time is an explicit proxy.
    """
    out = samples.copy()
    out['mager_command_transition'] = False
    out['mager_command_ids'] = ''
    out['mager_command_cohort'] = pd.Series(None, index=out.index, dtype=object)
    out['mager_command_time_basis'] = 'no prior recorded science-changing MAG/ER request'
    if commands.empty:
        return out
    events = commands if 'may_change_science' in commands else annotate_command_actions(commands)
    selected = events['may_change_science'] & events['transmission_confirmed'] & ~events['known_execution_failure']
    if instrument is not None:
        selected &= events['affected_instruments'].str.split(',').map(lambda values: instrument in values)
    events = events.loc[selected].copy()
    if events.empty:
        return out
    arrival = events['arrival_scet_valid'] & events['arrival_scet_utc'].notna()
    events['event_utc'] = events['arrival_scet_utc'].where(arrival, events['transmit_utc'])
    events['event_time_basis'] = np.where(arrival, 'estimated spacecraft arrival SCET', 'DSN transmit UTC proxy')
    all_events = commands if 'command_event_id' in commands else annotate_command_actions(commands)
    all_arrival = all_events['arrival_scet_valid'] & all_events['arrival_scet_utc'].notna()
    all_times = all_events['arrival_scet_utc'].where(all_arrival, all_events['transmit_utc'])
    by_id = pd.Series(all_times.array, index=all_events['command_event_id']).groupby(level=0).last()
    events['candidate_until'] = events['following_exec_event_id'].map(by_id).fillna(events['event_utc'])
    events = events.sort_values(['event_utc', 'command_event_id']).reset_index(drop=True)
    timed = start.notna() & stop.notna() & stop.ge(start)
    epochs = pd.DatetimeIndex(events['event_utc']).as_unit('ns').asi8
    indices = np.searchsorted(epochs, pd.DatetimeIndex(start).as_unit('ns').asi8, side='right') - 1
    valid = np.flatnonzero(timed.to_numpy() & (indices >= 0))
    chosen = events.iloc[indices[valid]]
    out.iloc[valid, out.columns.get_loc('mager_command_cohort')] = chosen['command_event_id'].to_numpy()
    out.iloc[valid, out.columns.get_loc('mager_command_time_basis')] = chosen['event_time_basis'].to_numpy()
    earliest, latest = start[timed].min(), stop[timed].max()
    padding = pd.Timedelta(seconds=padding_s)
    for row in events.itertuples():
        left, right = row.event_utc - padding, max(row.event_utc, row.candidate_until) + padding
        if left > latest or right < earliest:
            continue
        # Include the action instant even for a zero-width transition policy.
        overlap = timed & stop.ge(left) & start.le(right)
        out.loc[overlap, 'mager_command_transition'] = True
        out.loc[overlap, 'mager_command_ids'] += row.command_event_id + ';'
    return out
