"""Stateful demultiplexing of LP MAG/ER telemetry across merged-file boundaries"""

import numpy as np
import pandas as pd

from .er_telemetry import decode_er_accumulation, er_accumulation_clock, time_er_counts
from .telemetry import decode_digital_subcom, decode_mag_block


def _subcom(frames):
    groups = {}
    for row in frames.itertuples(index=False):
        if not row.header_valid:
            continue
        index = (row.vcdu_unwrapped - 1) % 16
        clock = (row.vcdu_unwrapped - index - 1) * 2.0
        if clock not in groups:
            groups[clock] = {"data": bytearray(32), "mask": 0, "row": row}
        group = groups[clock]
        group["data"][index*2:index*2 + 2] = row.mag_er[1:3]
        group["mask"] |= 1 << index
    tables = []
    for clock, group in groups.items():
        out = decode_digital_subcom(group["data"], word_mask=group["mask"])
        row = group["row"]
        out["major_clock_s"] = clock
        out["major_utc"] = row.frame_utc + pd.to_timedelta(clock - row.frame_clock_s, unit="s")
        out['source_file'], out['record_index'] = row.source_file, row.record_index
        tables.append(out)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def decode_subcom_frames(frames):
    """Decode sampled MAG/ER status without expanding ER spectra or burst data."""
    return _subcom(frames)


def _mag(block, row, *, mode, context=None, start_clock_s=None, burst_number=np.nan,
         burst_packet_slot=np.nan):
    out = decode_mag_block(block)
    offset = np.arange(18) / (9 if mode == "realtime" else 18)
    if mode == "realtime":
        offset += 1/36 - 4
        clocks = row.frame_clock_s + offset
    else:
        clocks = np.full(18, np.nan) if start_clock_s is None else start_clock_s + offset
    out["clock_s"] = clocks
    dates = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    if pd.notna(row.frame_utc):
        valid = np.isfinite(clocks)
        dates.loc[valid] = row.frame_utc + pd.to_timedelta(clocks[valid] - row.frame_clock_s, unit="s")
    out["utc"] = dates
    out["mode"], out["burst_number"] = mode, burst_number
    out["frame_id"], out["source_file"], out["record_index"] = row.frame_id, row.source_file, row.record_index
    out["spin_phase_deg"] = np.nan
    out["spin_period_s"] = np.nan
    out["spin_phase_valid"] = False
    out["burst_id"] = context.get("burst_id") if context is not None and mode == "burst" else None
    out["burst_packet_slot"] = burst_packet_slot
    out["burst_marker_id"] = context.get("marker_id", np.nan) if context is not None and mode == "burst" else np.nan
    if (context is not None and context.get("er_power", 1) == 1
            and np.isfinite(context.get("phase", np.nan)) and context.get("period_s", 0) > 0):
        out["spin_phase_deg"] = ((context["phase"] / 4096
                                 + (clocks - context["major_clock_s"]) / context["period_s"]) * 360) % 360
        if mode == "burst":
            out["spin_period_s"] = context["period_s"]
            out["spin_phase_valid"] = np.isfinite(clocks)
    if mode == "burst":
        out["spin_phase_source"] = "burst collection marker phase/period; flywheel possible"
    return out


def decode_realtime_mag_frames(frames):
    """Unpack only original real-time MAG in batches, retaining decoder quality

    Full burst frames carry no real-time MAG and are excluded. Spin phase is
    intentionally unavailable until archived SUNPULSE is supplied.
    """
    frame_ids = np.flatnonzero(frames["header_valid"].to_numpy(bool))
    valid = frames.iloc[frame_ids]
    payload = np.frombuffer(b"".join(valid["mag_er"]), dtype=np.uint8).reshape(-1, 168)
    kind = payload[:, 0] & 63
    full = (kind >= 36) & (kind <= 62) & ((payload[:, 0] & 128) != 0)
    selected = valid.loc[~full]
    frame_ids = frame_ids[~full]
    data = payload[~full, 3:85]
    packed = data[:, 1:].reshape(-1, 3).astype(np.uint16)
    values = np.column_stack([packed[:, 0] | ((packed[:, 1] & 15) << 8),
                              (packed[:, 1] >> 4) | (packed[:, 2] << 4)]).reshape(-1, 3)
    out = pd.DataFrame(values, columns=["bx", "by", "bz"])
    for axis in "xyz":
        out["b" + axis + "_signed"] = out["b" + axis].astype(int) - 2048
    out["sample_index"] = np.tile(np.arange(18), len(selected))
    out["range"] = np.repeat(data[:, 0] & 7, 18)
    out["calibration_mode"] = np.repeat((data[:, 0] & 8) != 0, 18)
    out["mag_frame"] = np.repeat(data[:, 0] >> 4, 18)
    out["adc_at_limit"] = ((values == 0) | (values == 4095)).any(axis=1)
    offset = np.tile(np.arange(18)/9 + 1/36 - 4, len(selected))
    out["clock_s"] = np.repeat(selected["frame_clock_s"].to_numpy(), 18) + offset
    out["utc"] = pd.to_datetime(np.repeat(selected["frame_utc"].array, 18)) + pd.to_timedelta(offset, unit="s")
    out["mode"], out["burst_number"] = "realtime", np.nan
    out["frame_id"] = np.repeat(frame_ids, 18)
    for name in ("source_file", "record_index"):
        out[name] = np.repeat(selected[name].to_numpy(), 18)
    out["spin_phase_deg"] = np.nan
    changed = np.r_[False, np.diff(data[:, 0] & 7) != 0] if len(data) else np.array([], dtype=bool)
    out["range_changed"] = False
    out.loc[np.flatnonzero(changed)*18, "range_changed"] = True
    out["mag_frame_matches"] = np.repeat((data[:, 0] >> 4) == ((selected["vcdu"].to_numpy() - 1) % 16), 18)
    out["science_valid"] = (~out["calibration_mode"] & ~out["adc_at_limit"]
                             & ~out["range_changed"] & out["mag_frame_matches"])
    return out


def decode_mager_frames(frames, *, telemetry_version=None):
    """Decode correlated merged frames into MAG, ER, status and raw-packet tables

    frames comes from correlate_frame_times. An explicit telemetry_version
    override is useful for a file starting with incomplete digital subcom.
    Unknown versions and missing context are reported, never inferred by date.
    """
    if telemetry_version is not None and telemetry_version not in (1, 2, 3, 4, 5):
        raise ValueError("telemetry_version must be 1 through 5")
    frames = frames.reset_index(drop=True).copy()
    frames["frame_id"] = frames.index
    subcom = _subcom(frames)
    contexts = ({row["major_clock_s"]: row.to_dict() for _, row in subcom.iterrows()}
                if not subcom.empty else {})
    mag, counts, summaries, angles, bits, packets, markers, controls, dumps, issues = ([] for _ in range(10))
    realtime = []
    burst = []
    burst_er = []
    marker = None
    last_burst_counter = None
    last_burst_mag = None
    burst_mag_timing_valid = False
    burst_id = None
    last_ranges = {}

    def issue(row, code, message):
        issues.append((row.source_file, row.record_index, row.frame_id, code, message))

    def context_for(row):
        word = (row.vcdu_unwrapped - 1) % 16
        return contexts.get((row.vcdu_unwrapped - word - 1)*2.0)

    def append_mag(block, row, **kwargs):
        out = _mag(block, row, **kwargs)
        mode = kwargs["mode"]
        key = (mode, kwargs.get("burst_number") if mode == "burst" else 0)
        changed = key in last_ranges and last_ranges[key] != int(out["range"].iloc[0])
        out["range_changed"] = False
        out.loc[0, "range_changed"] = changed
        last_ranges[key] = int(out["range"].iloc[0])
        out["mag_frame_matches"] = True
        if mode == "realtime":
            matches = int(out["mag_frame"].iloc[0]) == (row.vcdu - 1) % 16
            out["mag_frame_matches"] = matches
            if not matches:
                issue(row, "mag_frame_mismatch", "MAG status frame differs from the documented VCDU minus one")
        out["science_valid"] = (~out["calibration_mode"] & ~out["adc_at_limit"]
                                & ~out["range_changed"] & out["mag_frame_matches"])
        mag.append(out)

    def finish_er(group, mode):
        if not group:
            return
        first = group[0]
        row, context = first["row"], first["context"]
        version = telemetry_version if telemetry_version is not None else (
            context.get("telemetry_version", np.nan) if context is not None else np.nan)
        accumulation = len(summaries)
        mapping = {item["type"]: item["payload"] for item in group}
        for item in group:
            packets.append((accumulation, mode, item["type"], item["row"].frame_id,
                            item["row"].source_file, item["row"].record_index, item["payload"]))
        if mode == "realtime" and version not in (1, 2, 3, 4, 5):
            issue(row, "unknown_er_version", "ER packets retained; no supported digital-subcom version or override")
            summaries.append({"accumulation_id": accumulation, "mode": mode, "telemetry_version": version,
                              "complete": False, "packets_received": len(mapping), "packets_expected": 36,
                              "timing_valid": False, "accumulation_clock_s": np.nan})
            return
        version = int(version) if version in (1, 2, 3, 4, 5) else 0
        decoded = decode_er_accumulation(mapping, version=version, mode=mode)
        previous = contexts.get(context["major_clock_s"] - 32) if context is not None and mode == "realtime" else None
        epoch, timing_context = er_accumulation_clock(
            decoded["header"], context, first_clock_s=row.frame_clock_s if mode == "realtime" else None,
            previous=previous)
        summary = decoded["summary"]
        summary["reported_software_version"] = context.get("telemetry_version", np.nan) if context is not None else np.nan
        summary["telemetry_version_overridden"] = telemetry_version is not None
        summary.update({"accumulation_id": accumulation, "accumulation_clock_s": epoch,
                        "timing_valid": np.isfinite(epoch), "first_frame_id": row.frame_id,
                        "last_frame_id": group[-1]["row"].frame_id,
                        "burst_number": first.get("burst_number", np.nan),
                        "source_file": row.source_file, "time_basis": row.time_basis})
        timed = time_er_counts(decoded["counts"], summary, context=timing_context,
                              reference_clock_s=row.frame_clock_s, reference_utc=row.frame_utc)
        timed["accumulation_id"], timed["mode"] = accumulation, mode
        timed["complete"] = summary["complete"]
        counts.append(timed)
        direction = decoded["angles"].copy()
        direction["accumulation_id"] = accumulation
        if timing_context is not None:
            period = timing_context["period_s"]
            direction["clock_s"] = epoch - period/2 + direction["quarter_spin"] * period/4
        else:
            direction["clock_s"] = np.nan
        if mode == "realtime":
            ancillary = np.full(72, np.nan)
            for item in group:
                start = item["type"]*2
                ancillary[start:start + 2] = [item["high"] & 1, item["high"] >> 1]
            bit_table = pd.DataFrame({"accumulation_id": accumulation, "bit_index": np.arange(72),
                                      "value": ancillary, "quarter_spin": np.arange(72) - 8})
            bits.append(bit_table)
            if version < 3:
                direction = pd.DataFrame({"accumulation_id": accumulation, "quarter_spin": np.arange(64),
                                          "sample_parity": ancillary[8:]})
                direction["clock_s"] = (epoch - timing_context["period_s"]/2
                                         + np.arange(64)*timing_context["period_s"]/4
                                         if timing_context is not None else np.nan)
        elif np.isfinite(decoded["control"]).all():
            controls.append(("accumulation", accumulation, "er", bytes(decoded["control"].astype(np.uint8))))
        angles.append(direction)
        if not summary["complete"]:
            issue(row, "partial_er_accumulation", f"{mode} ER missing packet types {summary['missing_packets']}")
        if not summary["timing_valid"]:
            issue(row, "missing_er_timing", "ER counts decoded with missing collection timing context")
        summaries.append(summary)

    for row in frames.itertuples(index=False):
        if not row.header_valid:
            issue(row, "invalid_header", "Raw record retained; invalid merged header excluded from science decoding")
            continue
        payload = row.mag_er
        if len(payload) != 168:
            raise ValueError("MAG/ER frames must have exactly 168 bytes")
        if row.gap_frames > 0:
            finish_er(realtime, "realtime")
            realtime = []
            issue(row, "merged_counter_gap", f"{row.gap_frames} merged frames are absent")
        code, kind = payload[0], payload[0] & 63
        full = 36 <= kind <= 62 and bool(code & 128)
        context = context_for(row)
        if not full:
            append_mag(payload[3:85], row, mode="realtime", context=context)
        if kind < 36:
            if realtime and (kind <= realtime[-1]["type"] or row.gap_frames > 0):
                finish_er(realtime, "realtime")
                realtime = []
            realtime.append({"row": row, "type": kind, "payload": payload[85:168],
                             "high": code >> 6, "context": context})
        elif kind == 63:
            dumps.append((row.frame_id, payload[85] | payload[86] << 8, payload[87:168]))
            if code >> 6 != 3:
                issue(row, "memory_dump_code", "Memory dump frame does not have both high code bits set")
        else:
            records = [(code & 127, payload[3:85], np.nan)] if full else []
            records.append((payload[85] & 127, payload[86:168], np.nan) if full
                           else (code & 127, payload[86:168], payload[85]))
            for slot, (burst_code, packet, counter) in enumerate(records):
                packet_type = burst_code & 63
                start_burst = packet_type in (57, 62) and bool(burst_code & 64)
                if start_burst:
                    finish_er(burst_er, "burst")
                    burst_er = []
                    marker, last_burst_mag, last_burst_counter = None, None, None
                    burst_mag_timing_valid = False
                    burst_id = None
                counter_gap = False
                if np.isfinite(counter):
                    counter_gap = last_burst_counter is not None and (counter - last_burst_counter) % 256 != 1
                    last_burst_counter = counter
                burst.append((row.frame_id, slot, burst_code, packet_type, counter, counter_gap, packet))
                if counter_gap:
                    issue(row, "burst_counter_gap", "One or more burst playback packets are absent")
                    finish_er(burst_er, "burst")
                    burst_er = []
                if packet_type == 57:
                    collected, transmitted = int.from_bytes(packet[:2], "little"), int.from_bytes(packet[2:4], "little")
                    age = (transmitted - collected) % 65536
                    clock = row.frame_clock_s - (row.vcdu % 16)*2 - age*32
                    if burst_id is None or (marker is not None and marker["burst_number"] != packet[4]):
                        burst_id = f"{packet[4]}:{clock:.9f}"
                    marker = decode_digital_subcom(packet[50:82]).iloc[0].to_dict()
                    marker.update({"major_clock_s": clock, "burst_number": packet[4],
                                   "major_utc": row.frame_utc + pd.to_timedelta(clock - row.frame_clock_s, unit="s"),
                                   "collected_major": collected, "transmitted_major": transmitted,
                                   "age_major_frames": age, "frame_id": row.frame_id,
                                   "marker_id": len(markers),
                                   "burst_id": burst_id,
                                   "new_burst": start_burst, "best_available": packet[5],
                                   "read_state": packet[6], "record_state": packet[7],
                                   "best_trigger": packet[8], "read_trigger": packet[9],
                                   "filtered_trigger": int.from_bytes(packet[10:12], "little") / 256,
                                   "raw_packet": packet})
                    markers.append(marker.copy())
                    for name, values in [("pll", packet[12:24]), ("mag", packet[24:34]), ("burst", packet[34:50])]:
                        controls.append(("marker", len(markers) - 1, name, values))
                    last_burst_mag = None
                    burst_mag_timing_valid = True
                elif packet_type == 56:
                    offset = ((packet[0] >> 4)*2 + ((burst_code >> 6) & 1))
                    valid = (marker is not None and burst_mag_timing_valid
                             and (last_burst_mag is None or offset > last_burst_mag))
                    if not valid:
                        burst_mag_timing_valid = False
                        issue(row, "missing_burst_mag_timing", "Burst MAG lacks a marker or crossed an unmarked major-frame boundary")
                    append_mag(packet, row, mode="burst", context=marker,
                               start_clock_s=marker["major_clock_s"] + offset if valid else None,
                               burst_number=marker["burst_number"] if marker is not None else np.nan,
                               burst_packet_slot=slot)
                    last_burst_mag = offset
                elif 36 <= packet_type <= 55:
                    if burst_er and packet_type <= burst_er[-1]["type"]:
                        finish_er(burst_er, "burst")
                        burst_er = []
                    burst_er.append({"row": row, "type": packet_type, "payload": packet,
                                     "context": marker.copy() if marker is not None else None,
                                     "burst_number": marker["burst_number"] if marker is not None else np.nan})
                elif packet_type != 62:
                    issue(row, "undefined_burst_type", f"Undefined burst packet type {packet_type}; bytes retained")
    finish_er(realtime, "realtime")
    finish_er(burst_er, "burst")

    def combined(tables):
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    return {
        "mag": combined(mag), "er_counts": combined(counts), "er_accumulations": pd.DataFrame(summaries),
        "er_angles": combined(angles), "er_bits": combined(bits), "digital_subcom": subcom,
        "er_packets": pd.DataFrame(packets, columns=["accumulation_id", "mode", "packet_type", "frame_id",
                                                    "source_file", "record_index", "payload"]),
        "burst_packets": pd.DataFrame(burst, columns=["frame_id", "slot", "code", "packet_type",
                                                      "playback_counter", "counter_gap", "payload"]),
        "burst_markers": pd.DataFrame(markers),
        "control_tables": pd.DataFrame(controls, columns=["owner_kind", "owner_id", "table", "payload"]),
        "memory_dumps": pd.DataFrame(dumps, columns=["frame_id", "address", "payload"]),
        "issues": pd.DataFrame(issues, columns=["source_file", "record_index", "frame_id", "code", "message"]),
    }
