"""Download the complete PDS LP archive and decode/pseudo-calibrate MAG and ER.

Usage: python -m moonbroom.mission_pipeline CONFIG.json [--stage all]
Processing uses bounded record shards with context on both sides. Every
unsupported row remains present. Independent coefficient banks are inputs,
never inferred by fitting the same mission rows that are being evaluated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import assign_er_energies, calibrate_er_telemetry, read_er_energy_bins
from .level0 import decode_level0, inspect_merged_telemetry
from .mission_archive import (VOLUMES, discover_mission, download_mission, file_hash,
                              validate_plan, write_json)
from .mission_calibration import apply_er_mission_response, apply_pad_mission_response
from .native_mag import calibrate_mag_9hz, calibrate_mag_18hz, fit_mag_spin_offsets
from .mag_state import (annotate_mag_hardware_state, mag_hardware_calibration_checks,
                        validate_mag_model_scopes)
from .pad_calibration import pad_field_directions
from .release import audit_native_payloads
from .timing import reconstruct_spin_phase
from .archive_quality import (load_archive_quality, flag_archive_frames,
                              flag_science_quality, select_sunpulse_paths,
                              read_sunpulse_products)


def _read_bank(path, *, require_fit=True):
    bank = pd.read_csv(path)
    for name in (("validation_pass", "fit_valid") if require_fit else ("validation_pass",)):
        if name not in bank:
            raise ValueError(f"Coefficient bank requires {name}: {path}")
        values = bank[name].fillna(False).astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"{name} must contain explicit Boolean flags: {path}")
        bank[name] = values.eq("true")
    return bank


def _empty_mag_bank():
    return pd.DataFrame(columns=["range", "fit_valid", "validation_pass", "valid_from", "valid_to"])


def _check_bank(bank, kind):
    accepted = bank.loc[bank['validation_pass'] & bank['fit_valid']] if kind != 'pad' else bank.loc[bank['validation_pass']]
    if accepted.empty:
        return
    keys = {'mag': ['range'], 'er': ['mode', 'kind', 'energy_step', 'angle_bin', 'telemetry_version'],
            'pad': ['energy_step', 'telemetry_version']}[kind]
    required = keys + ['valid_from', 'valid_to']
    numeric = []
    if kind == 'mag':
        numeric = [f'{name}_{axis}' for name in ('gain', 'zero') for axis in 'xyz']
        numeric += ['align_' + a + b for a in 'xyz' for b in 'xyz']
        required += numeric + ['source']
    elif kind == 'er':
        numeric = ['energy_ev', 'geometric_factor_cm2_sr', 'dead_time_s', 'max_deadtime_factor']
        required += numeric + ['source']
    else:
        required += ['cohort', 'flux_scale', 'time_offset_s']
    missing = set(required) - set(accepted)
    if missing:
        raise ValueError(f'{kind.upper()} bank lacks required columns: {sorted(missing)}')
    if accepted[required].isna().any().any() or (numeric and not np.isfinite(accepted[numeric].to_numpy(float)).all()):
        raise ValueError(f'{kind.upper()} accepted coefficients must be finite and complete')
    if kind == 'mag':
        validate_mag_model_scopes(accepted)
        return
    scopes = accepted[keys].copy()
    scopes['start'] = pd.to_datetime(accepted['valid_from'], utc=True, format='mixed')
    scopes['stop'] = pd.to_datetime(accepted['valid_to'], utc=True, format='mixed')
    if scopes['start'].isna().any() or scopes['stop'].isna().any() or scopes['start'].ge(scopes['stop']).any():
        raise ValueError(f'{kind.upper()} bank requires finite ordered UTC scopes')
    scopes = scopes.sort_values(keys + ['start'])
    previous = scopes.groupby(keys)['stop'].shift()
    if scopes['start'].lt(previous).any():
        raise ValueError(f'{kind.upper()} bank has overlapping accepted model scopes')


def preflight(config):
    """Check scientific prerequisites before starting a full archive transfer."""
    cfg = dict(config)
    if not isinstance(cfg.get('allow_realtime_burst_transfer', False), bool):
        raise ValueError('allow_realtime_burst_transfer must be an explicit Boolean')
    if not isinstance(cfg.get('allow_incomplete_final_record', True), bool):
        raise ValueError('allow_incomplete_final_record must be an explicit Boolean')
    load_archive_quality(settings=cfg.get('archive_quality'))
    for key in ("archive_directory", "output_directory"):
        if not cfg.get(key):
            raise ValueError(f"Specify {key}")
    if Path(cfg["archive_directory"]).resolve() == Path(cfg["output_directory"]).resolve():
        raise ValueError("Use distinct archive and processed-output directories")
    size, halo = cfg.get("records_per_shard", 1800), cfg.get("halo_records", 128)
    if not isinstance(size, int) or size < 1 or not isinstance(halo, int) or halo < 128:
        raise ValueError("Use positive records_per_shard and at least 128 context records")
    if not cfg.get("clock_only", False):
        if not cfg.get("kernels") or not cfg.get("receiver"):
            raise ValueError("Supply timing kernels and a named receiver, or explicitly choose clock_only")
        import spiceypy  # Verify the geometry extra before a large download.
        for path in cfg["kernels"]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    for key in ("mag_bank", "er_bank", "energy_bins"):
        if not cfg.get(key):
            if not cfg.get("allow_uncalibrated", False):
                raise ValueError(f"Supply {key}, or explicitly choose allow_uncalibrated")
        elif not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    banks = {key: _read_bank(cfg[key], require_fit=key != 'pad_bank')
             for key in ("mag_bank", "er_bank", "pad_bank") if cfg.get(key)}
    for key, bank in banks.items():
        _check_bank(bank, key.removesuffix('_bank'))
    local = cfg.get("local_spin_offsets", True)
    if "mag_bank" in banks and "offset_model" in banks["mag_bank"]:
        bank = banks["mag_bank"]
        accepted = bank.loc[bank["fit_valid"] & bank["validation_pass"]]
        if not accepted.empty and accepted["offset_model"].eq("local spin fit").ne(local).any():
            raise ValueError("local_spin_offsets must match every accepted MAG model's offset procedure")
    if cfg.get("energy_bins") and cfg.get("energy_time_basis") not in ("unix", "utc"):
        raise ValueError("Explicitly choose energy_time_basis='unix' or 'utc'")
    if cfg.get('energy_bins'):
        energies = read_er_energy_bins(cfg['energy_bins'], time_basis=cfg['energy_time_basis'])
        if energies['energy_step'].nunique() != 15:
            raise ValueError('Mission ER processing requires the full 15-step low-resolution energy table')
    if "pad_bank" in banks and "er_bank" not in banks:
        raise ValueError("PAD normalization requires its independently accepted ER response bank")
    return banks


def _write_table(table, path):
    out = table.copy()
    for column in out:
        if out[column].dtype != object:
            continue
        out[column] = out[column].map(lambda value: value.hex() if isinstance(value, bytes) else value)
    temporary = path.with_name(path.name + ".partial")
    out.to_csv(temporary, index=False, compression=dict(method="gzip", compresslevel=1, mtime=0))
    temporary.replace(path)


def _owned_frames(frames, source, start, stop):
    return frames["source_file"].eq(str(Path(source).resolve())) & frames["record_index"].ge(start) & frames["record_index"].lt(stop)


def _mission_offsets(phase, *, mode="realtime"):
    """Keep the existing bank's 60-s counter grid after global clock unwrapping.

    Existing calibration intervals used the local 24-bit counter epoch. Adding
    a mission rollover epoch must not shift their offset windows by 32 seconds.
    Fit on the local grid and return centers/bounds in the global mission clock.
    """
    rows = []
    finite = phase.loc[np.isfinite(phase['clock_s'])]
    cycles = np.floor(finite['clock_s'] / 2**25).astype('int64')
    for cycle, group in finite.groupby(cycles):
        local = group.copy()
        local['clock_s'] -= int(cycle) * 2**25
        offsets = fit_mag_spin_offsets(local, mode=mode)
        for name in ('clock_center_s', 'clock_start_s', 'clock_stop_s'):
            offsets[name] += int(cycle) * 2**25
        offsets['clock_cycle'] = int(cycle)
        rows.append(offsets)
    return pd.concat(rows, ignore_index=True) if rows else fit_mag_spin_offsets(phase, mode=mode)


def process_shard(paths, slices, source, start, stop, sunpulse_paths, config, banks, energies, *, clock_cycle=0, quality=None):
    """Return owned products only; surrounding records provide decoding context.

    ER accumulations belong to the shard containing their first retained packet.
    The stable identity uses that packet's original source file and record.
    Burst times can precede playback, so ownership never uses collection UTC.
    """
    options = dict(record_slices=slices, clock_cycle=clock_cycle, overlap_policy=config.get("overlap_policy", "exact"),
                   allow_incomplete_final_record=config.get('allow_incomplete_final_record', False),
                   strict=config.get("strict", True), telemetry_version=config.get("telemetry_version"))
    timing_error = None
    if not config.get("clock_only", False):
        options.update(timing_kernels=config["kernels"], receiver=config["receiver"],
                       data_system_delay_s=config.get("data_system_delay_s", 2.0))
    try:
        products = decode_level0(paths, **options)
    except Exception as error:
        # Retain digital recovery when supplied ephemerides do not span a shard.
        # Other decoding/timing errors fail the shard and are never hidden.
        if type(error).__name__ != "SpiceSPKINSUFFDATA" or not config.get("retain_clock_only", True):
            raise
        timing_error = f"{type(error).__name__}: {error}"
        for key in ("timing_kernels", "receiver", "data_system_delay_s"):
            options.pop(key, None)
        products = decode_level0(paths, **options)
    quality = quality if quality is not None else load_archive_quality(settings=config.get('archive_quality'))
    frames = flag_archive_frames(products['frames'], quality)
    archive_columns = [name for name in frames if name.startswith('archive_')]
    frame_quality = frames[['source_file', 'record_index'] + archive_columns]
    raw = products['mag'].merge(frame_quality, on=['source_file', 'record_index'], validate='many_to_one')
    raw = annotate_mag_hardware_state(raw, products['digital_subcom'],
                                      burst_markers=products['burst_markers'])
    raw = flag_science_quality(raw, quality, instrument='MAG')
    frames = flag_science_quality(frames, quality, start_col='frame_utc')
    frames['frame_id'] = np.arange(len(frames))
    audit = audit_native_payloads(frames, raw)
    if not audit["audit_pass"]:
        raise ValueError("Independent native MAG bit audit failed")
    owned = frames.loc[_owned_frames(frames, source, start, stop)]
    # Decoder frame IDs are local to this contextual shard. Source identities
    # remain the global keys and are written alongside all local frame IDs.
    frame_keys = owned[["source_file", "record_index"]].drop_duplicates()
    output = {"frames": owned.copy(), "input_files": products["input_files"]}
    raw_owned = raw.merge(frame_keys, on=["source_file", "record_index"], how="inner", validate="many_to_one")
    output["mag_counts"] = raw_owned
    realtime = raw.loc[raw["mode"].eq("realtime")].copy()
    pulses = read_sunpulse_products(sunpulse_paths, frames=frames, strict=False)
    if not pulses.empty:
        phase = reconstruct_spin_phase(realtime, pulses)
    else:
        phase = realtime.assign(spin_phase_valid=False, spin_phase_deg=np.nan, spin_period_s=np.nan)
    # ERRATA: pulse timing is unsupported at >=15 rpm. Missing attitude does
    # not discard independently measured pulses; its support flag stays false.
    high_spin = phase['attitude_available'] & ~phase['attitude_stale'] & phase['attitude_spin_rate_rpm'].ge(15)
    phase.loc[high_spin, 'spin_phase_valid'] = False
    phase.loc[high_spin, ['spin_phase_deg', 'spin_period_s']] = np.nan
    offsets = _mission_offsets(phase) if config.get("local_spin_offsets", True) else None
    native = calibrate_mag_9hz(phase, banks.get("mag_bank", _empty_mag_bank()), offsets=offsets)
    native = native.merge(frame_keys, on=["source_file", "record_index"], how="inner", validate="many_to_one")
    expected = raw_owned.loc[raw_owned["mode"].eq("realtime")]
    keys = ["source_file", "record_index", "sample_index", "clock_s", "bx_signed", "by_signed", "bz_signed"]
    pd.testing.assert_frame_equal(native[keys].reset_index(drop=True), expected[keys].reset_index(drop=True), check_dtype=False)
    output["native_mag"] = native
    if offsets is not None:
        output["local_offsets"] = offsets
    burst = raw.loc[raw['mode'].eq('burst')].copy()
    burst_offsets = _mission_offsets(burst, mode='burst') if config.get('local_spin_offsets', True) else None
    calibrated_burst = calibrate_mag_18hz(
        burst, banks.get('mag_bank', _empty_mag_bank()), offsets=burst_offsets,
        allow_realtime_transfer=config.get('allow_realtime_burst_transfer', False))
    calibrated_burst = calibrated_burst.merge(frame_keys, on=['source_file', 'record_index'],
                                               how='inner', validate='many_to_one')
    output['native_burst_mag'] = calibrated_burst
    if burst_offsets is not None:
        output['burst_local_offsets'] = burst_offsets
    output['mag_hardware_checks'] = mag_hardware_calibration_checks(
        pd.concat([native, calibrated_burst], ignore_index=True))
    packets = products["er_packets"]
    if not packets.empty:
        # Any affected packet contaminates the assembled accumulation, even
        # when its first packet and owning shard are outside the outage.
        packet_quality = packets.merge(frame_quality, on=['source_file', 'record_index'], validate='many_to_one')
        accumulation_quality = packet_quality.groupby('accumulation_id').agg(
            archive_noisy=('archive_noisy', 'any'), archive_gap=('archive_gap', 'any'),
            archive_quality_valid=('archive_quality_valid', 'all'),
            archive_quality_assessed=('archive_quality_assessed', 'all'))
        accumulation_quality['archive_interval_ids'] = packet_quality.groupby('accumulation_id')['archive_interval_ids'].agg(
            lambda values: ';'.join(sorted({item for value in values for item in value.split(';') if item})))
        first = packets.drop_duplicates("accumulation_id").copy()
        first = first.merge(frame_keys, on=["source_file", "record_index"], how="inner", validate="many_to_one")
        first["source_accumulation_id"] = (first["mode"] + ":" + first["source_file"].map(lambda p: Path(p).name)
                                            + ":" + first["record_index"].astype(str))
        identities = first[["accumulation_id", "source_accumulation_id"]]
        for name in ("er_counts", "er_accumulations", "er_angles", "er_bits", "er_packets"):
            table = products[name]
            if not table.empty:
                table = table.merge(accumulation_quality, on='accumulation_id', validate='many_to_one')
                if name == 'er_counts':
                    table = flag_science_quality(table, quality, start_col='start_utc', stop_col='stop_utc', instrument='ER')
                output[name] = table.merge(identities, on="accumulation_id", how="inner", validate="many_to_one")
        counts = output.get("er_counts", pd.DataFrame())
        if not counts.empty:
            context = products["er_accumulations"][["accumulation_id", "telemetry_version", "pitch_map_version"]]
            counts = counts.merge(context, on="accumulation_id", validate="many_to_one")
            if energies is None:
                counts = counts.assign(energy_ev=np.nan, energy_valid=False)
            else:
                counts = assign_er_energies(counts, energies)
            if "er_bank" in banks:
                er = apply_er_mission_response(counts, banks["er_bank"])
            else:
                er = calibrate_er_telemetry(counts)
                er["empirical_response_supported"] = False
            if "pad_bank" in banks:
                angles = products["er_angles"].merge(products["er_accumulations"][["accumulation_id", "mode"]],
                                                     on="accumulation_id", validate="many_to_one")
                directions = pad_field_directions(angles, raw)
                pad = counts.loc[counts["kind"].eq("pad") & counts["mode"].eq("realtime")]
                if not pad.empty:
                    calibrated_pad = apply_pad_mission_response(pad, directions, banks["er_bank"], banks["pad_bank"])
                    output["pseudo_er_pad"] = calibrated_pad
            er["pseudo_calibrated"] = er["calibration_valid"]
            output["pseudo_er"] = er
    for name in ("issues", "digital_subcom", "burst_packets", "burst_markers", "burst_controls", "memory_dumps", "overlap_records"):
        if name in products and not products[name].empty:
            # Context diagnostics are explicitly shard-local, never summed as
            # independently owned science records in the mission denominator.
            output["context_" + name] = products[name]
    stats = dict(payload_audit=audit, owned_frames=len(owned), native_mag_rows=len(native),
                 calibrated_mag_rows=int(native["calibration_valid"].sum()),
                 er_count_rows=len(output.get("er_counts", [])),
                 calibrated_er_rows=int(output["pseudo_er"]["calibration_valid"].sum()) if "pseudo_er" in output else 0,
                 timing_error=timing_error, clock_only=bool(config.get("clock_only", False) or timing_error))
    stats['initial_clock_cycle'] = clock_cycle
    stats['burst_mag_rows'] = len(calibrated_burst)
    stats['calibrated_burst_mag_rows'] = int(calibrated_burst['calibration_valid'].sum())
    stats['transferred_burst_mag_rows'] = int(calibrated_burst['calibration_mode_transfer'].sum())
    stats['archive_noisy_mag_rows'] = int(native['archive_noisy'].sum())
    stats['maneuver_mag_rows'] = int((native['maneuver_candidate'] | native['post_maneuver_settling']).sum())
    stats['attitude_supported_mag_rows'] = int(native['attitude_quality_valid'].sum())
    stats['mager_command_transition_mag_rows'] = int(native['mager_command_transition'].sum())
    stats['sunpulse_selection'] = pulses.attrs.get('product_selection', [])
    return output, stats


def _counter_at(entry, position):
    with Path(entry['path']).open('rb') as stream:
        stream.seek(position * 472 + 6)
        return int.from_bytes(stream.read(3), 'big')


def _assign_clock_cycles(entries):
    cycle, previous = 0, None
    for entry in entries:
        first = _counter_at(entry, 0)
        last = _counter_at(entry, entry['records'] - 1)
        if previous is not None and first - previous < -(2**23):
            cycle += 1
        entry['clock_cycle'] = cycle
        entry['first_counter'] = first
        if last - first < -(2**23):
            cycle += 1
        previous = last


def _shard_inputs(entries, position, start, stop, halo):
    central = entries[position]
    paths, slices = [], {}
    left, right = max(0, start - halo), min(central["records"], stop + halo)
    if start < halo and position:
        previous = entries[position - 1]
        paths.append(previous["label"])
        slices[previous["label"]] = (max(0, previous["records"] - halo), previous["records"])
    paths.append(central["label"])
    slices[central["label"]] = (left, right)
    if stop + halo > central["records"] and position + 1 < len(entries):
        following = entries[position + 1]
        paths.append(following["label"])
        slices[following["label"]] = (0, min(halo, following["records"]))
    initial = next(entry for entry in entries if entry['label'] == paths[0])
    first_counter = _counter_at(initial, slices[paths[0]][0])
    cycle = initial['clock_cycle'] + int(first_counter - initial['first_counter'] < -(2**23))
    return paths, slices, cycle


def process_mission(plan, downloads, config):
    """Process verified local files; resume only matching inputs/code/products."""
    config = {'allow_incomplete_final_record': True, **config}
    validate_plan(plan)
    banks = preflight(config)
    available = {row["relative_path"].lower(): row for row in downloads if row["status"] == "available"}
    inventoried = {row["relative_path"].lower(): row for row in plan["products"]}
    if set(available) != set(inventoried):
        raise ValueError("All inventoried files must be downloaded before mission processing")
    # Recheck local inputs even on processing-only runs, including SUNPULSE.
    for key, row in available.items():
        if row['md5'] != inventoried[key]['md5']:
            raise ValueError(f"Source checksum differs from official inventory: {row['relative_path']}")
        if file_hash(row["path"]) != row["sha256"] or file_hash(row["path"], "md5") != row["md5"]:
            raise ValueError(f"Source hash mismatch: {row['relative_path']}")
    quality = load_archive_quality(available.values(), config.get('archive_quality'),
                                   timing_kernels=() if config.get('clock_only', False) else config.get('kernels', ()),
                                   transmitter=config.get('command_transmitter', 'EARTH'))
    entries = []
    sun = []
    for key, row in available.items():
        name = Path(key).name
        if "/merged/" in key and name.startswith("m") and name.endswith(".b"):
            label = available[key[:-2] + ".lbl"]["path"]
            layout = inspect_merged_telemetry(
                row['path'], label_path=label,
                allow_incomplete_final_record=config['allow_incomplete_final_record'])
            count = layout['complete_records']
            if count < 1:
                raise ValueError(f"Invalid official merged file size: {row['path']}")
            entries.append(dict(row, **layout, records=count, date=pd.to_datetime(name[1:6], format="%y%j")))
        elif "/sunpulse/" in key and name.endswith(".lbl"):
            sun.append(dict(row, date=pd.to_datetime(name[1:6], format="%y%j")))
    selected_pulses, pulse_decisions = select_sunpulse_paths([row['path'] for row in sun])
    selected_pulses = set(selected_pulses)
    sun = [row for row in sun if str(Path(row['path']).resolve()) in selected_pulses]
    entries.sort(key=lambda row: Path(row["path"]).name.lower())
    _assign_clock_cycles(entries)
    anomaly_columns = ['relative_path', 'source_file', 'label', 'md5', 'sha256', 'file_bytes',
                       'label_records', 'complete_records', 'incomplete_final_record',
                       'trailing_byte_offset', 'trailing_bytes', 'missing_bytes', 'trailing_hex']
    anomalies = [{key: entry[key] for key in anomaly_columns}
                 for entry in entries if entry['incomplete_final_record']]
    aux = [path for key in ("mag_bank", "er_bank", "pad_bank", "energy_bins") if (path := config.get(key))]
    aux += config.get("kernels", [])
    provenance = dict(config=config, archive_quality=quality.manifest, sunpulse_selection=pulse_decisions,
                      archive_record_anomalies=anomalies,
                      inputs=[dict(path=row["relative_path"], sha256=row["sha256"]) for row in available.values()],
                      auxiliary=[dict(path=str(Path(path).resolve()), sha256=file_hash(path)) for path in aux],
                      code={path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))})
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Inputs, code or configuration changed. Use a new output directory")
    write_json(manifest, dict(fingerprint=fingerprint, **provenance))
    pd.DataFrame(anomalies, columns=anomaly_columns).to_csv(output / 'archive_record_anomalies.csv', index=False)
    for name, table in (('archive_outages', quality.outages), ('maneuver_intervals', quality.maneuvers),
                        ('archive_commands', quality.commands), ('archive_attitude', quality.attitude),
                        ('mager_command_actions', quality.commands.loc[quality.commands['mager_action_known']
                                                                      | quality.commands['mager_action_unmapped']])):
        table.to_csv(output / (name + '.csv'), index=False)
    energies = read_er_energy_bins(config["energy_bins"], time_basis=config["energy_time_basis"]) if config.get("energy_bins") else None
    statuses = []
    size, halo = config.get("records_per_shard", 1800), config.get("halo_records", 128)
    for position, entry in enumerate(entries):
        for start in range(0, entry["records"], size):
            stop = min(start + size, entry["records"])
            shard = Path(entry["volume"]) / (Path(entry["path"]).stem + f"_{start:06d}")
            target = output / "shards" / shard
            target.mkdir(parents=True, exist_ok=True)
            complete = target / "complete.json"
            if complete.exists():
                status = json.loads(complete.read_text())
                if status["fingerprint"] != fingerprint or any(file_hash(output / row["path"]) != row["sha256"] for row in status["products"]):
                    raise ValueError(f"Completed shard changed: {shard}")
            else:
                status = dict(shard=str(shard), source=entry["relative_path"], record_start=start,
                              record_stop=stop, fingerprint=fingerprint, status="failed")
                try:
                    paths, slices, clock_cycle = _shard_inputs(entries, position, start, stop, halo)
                    dates = [e["date"] for e in entries if e["label"] in paths]
                    pulses = [row["path"] for row in sun if min(dates) - pd.Timedelta(days=1) <= row["date"] <= max(dates) + pd.Timedelta(days=1)]
                    tables, statistics = process_shard(paths, slices, entry["path"], start, stop, pulses, config, banks, energies,
                                                       clock_cycle=clock_cycle, quality=quality)
                    files = []
                    for name, table in tables.items():
                        path = target / (name + ".csv.gz")
                        _write_table(table, path)
                        files.append(dict(path=str(path.relative_to(output)), rows=len(table), sha256=file_hash(path)))
                    status.update(status="processed", products=files, **statistics)
                    write_json(complete, status)
                except Exception as error:
                    status["error"] = f"{type(error).__name__}: {error}"
                    write_json(target / "failure.json", status)
            statuses.append(status)
            write_json(output / "processing_status.json", statuses)
            print(f"{shard}: {status['status']}", flush=True)
    success = [row for row in statuses if row["status"] == "processed"]
    summary = dict(official_inventory_complete=set(plan["volumes"]) == set(VOLUMES),
                   archive_download_complete=True, processing_complete=len(success) == len(statuses) and bool(statuses),
                   source_files=len(entries), shards=len(statuses), failed_shards=len(statuses) - len(success),
                   source_records_complete=not bool(anomalies), incomplete_source_records=len(anomalies),
                   source_trailing_bytes=sum(row['trailing_bytes'] for row in anomalies),
                   missing_source_bytes=sum(row['missing_bytes'] for row in anomalies),
                   native_mag_rows=sum(row["native_mag_rows"] for row in success),
                   pseudo_mag_valid_rows=sum(row["calibrated_mag_rows"] for row in success),
                   burst_mag_rows=sum(row["burst_mag_rows"] for row in success),
                   pseudo_burst_mag_valid_rows=sum(row["calibrated_burst_mag_rows"] for row in success),
                   transferred_burst_mag_rows=sum(row["transferred_burst_mag_rows"] for row in success),
                   er_count_rows=sum(row["er_count_rows"] for row in success),
                   pseudo_er_valid_rows=sum(row["calibrated_er_rows"] for row in success),
                   clock_only_shards=sum(row["clock_only"] for row in success),
                   high_frequency_validated=False, pseudo_calibration=True)
    summary["full_mission_processed"] = summary["official_inventory_complete"] and summary["processing_complete"]
    write_json(output / "mission_status.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("all", "auxiliary", "inventory", "download", "process"), default="all")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    # Paths in JSON are relative to the JSON file, independent of shell cwd.
    base = args.config.resolve().parent
    for key in ("archive_directory", "output_directory", "mag_bank", "er_bank", "pad_bank", "energy_bins", "auxiliary_manifest"):
        if cfg.get(key):
            cfg[key] = str((base / Path(cfg[key]).expanduser()).resolve())
    cfg["kernels"] = [str((base / Path(path).expanduser()).resolve()) for path in cfg.get("kernels", [])]
    if (args.stage == 'auxiliary' or (args.stage in ('all', 'process') and not cfg.get('clock_only', False))) and cfg.get('auxiliary_manifest'):
        from .auxiliary_data import prepare_auxiliary_data
        prepare_auxiliary_data(cfg['auxiliary_manifest'])
    if args.stage == 'auxiliary':
        if not cfg.get('auxiliary_manifest'):
            raise ValueError('Specify auxiliary_manifest to prepare mission kernels')
        return 0
    if args.stage in ("all", "process"):
        preflight(cfg)
    directory = Path(cfg["archive_directory"])
    inventory = directory / "inventory.json"
    plan = discover_mission(directory) if args.stage == "inventory" or (args.stage in ("all", "download") and not inventory.exists()) else json.loads(inventory.read_text())
    if args.stage == "inventory":
        return 0
    downloads = (download_mission(plan, directory, workers=cfg.get("download_workers", 4),
                                  timeout=cfg.get("download_timeout_s", 120), retries=cfg.get("download_retries", 4))
                 if args.stage in ("all", "download") else json.loads((directory / "downloads.json").read_text()))
    if any(row["status"] != "available" for row in downloads):
        print("Download incomplete. Rerun to retry. See downloads.json", flush=True)
        return 1
    if args.stage == "download":
        return 0
    summary = process_mission(plan, downloads, cfg)
    print(json.dumps(summary, indent=2))
    return 0 if summary["processing_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
