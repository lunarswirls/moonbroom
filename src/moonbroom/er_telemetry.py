"""Reassembly of real-time and burst ER accumulations without science calibration"""

import numpy as np
import pandas as pd

from .telemetry import decompress_er_counts, er_angular_bins, er_frame_layout


def er_accumulation_clock(header, context, *, first_clock_s=None, previous=None):
    """End of first ER accumulation from phase/4096 and 1.024 ms period units"""
    if context is None or not np.isfinite(header[0]):
        return np.nan, None
    if context.get("er_power", 1) != 1:
        return np.nan, None
    for current in (context, previous):
        if current is None:
            continue
        if current.get("er_power", 1) != 1:
            continue
        period, phase = current.get("period_s", np.nan), current.get("phase", np.nan)
        if not np.isfinite(period) or period <= 0 or not np.isfinite(phase):
            continue
        nhs = int(header[0]) & 31
        clock = (current["major_clock_s"] + 0.0024 - (phase % 2048) / 4096 * period
                 + ((nhs + 1) % 32) * period / 2)
        if first_clock_s is None or clock <= first_clock_s:
            return clock, current
    return np.nan, None


def _count_table(codes, energy, sample, angle, kind, prescale):
    out = decompress_er_counts(codes, prescale=prescale)
    out["energy_step"] = energy
    out["sample"] = sample
    out["angle_bin"] = angle
    out["kind"] = kind
    return out


def decode_er_accumulation(packets, *, version, mode="realtime"):
    """Decode one ER accumulation from a mapping of packet types to raw bytes

    Missing packet bytes are masked. Real-time payloads have 83 bytes and
    keys 0..35; burst payloads have 82 bytes and keys 36..55. Burst control
    parameters are retained byte-for-byte because MAGERFMT does not define
    their commanding-table internals.
    """
    if mode not in ("realtime", "burst"):
        raise ValueError("mode must be realtime or burst")
    if mode == "realtime" and version not in (1, 2, 3, 4, 5):
        raise ValueError("Real-time ER requires software version 1 through 5")
    expected = set(range(36)) if mode == "realtime" else set(range(36, 56))
    length = 83 if mode == "realtime" else 82
    if not set(packets).issubset(expected) or any(len(packet) != length for packet in packets.values()):
        raise ValueError("ER packet type or length does not match the selected mode")
    header = np.full(4, np.nan)
    volume = np.full(1320, np.nan)
    directions = np.full(128 if mode == "realtime" else 4, np.nan)
    control = np.full(72, np.nan)
    samples = []
    if mode == "realtime":
        for row in er_frame_layout(version).itertuples(index=False):
            if row.frame_type not in packets:
                values = np.full(row.length, np.nan)
            else:
                values = np.frombuffer(packets[row.frame_type], dtype=np.uint8)[row.offset:row.offset + row.length]
            if row.kind == "header":
                header[row.stream_offset:row.stream_offset + row.length] = values
            elif row.kind == "3d":
                volume[row.stream_offset:row.stream_offset + row.length] = values
            elif row.kind == "mag_angles":
                directions[row.stream_offset:row.stream_offset + row.length] = values
            else:
                samples.append((row.energy_step, row.sample, values))
    else:
        stream = np.concatenate([np.frombuffer(packets[key], dtype=np.uint8) if key in packets
                                 else np.full(82, np.nan) for key in range(36, 56)])
        header, volume = stream[:4], stream[4:1324]
        angular = stream[1324:1564].reshape(15, 16)
        samples = [(energy, 0, angular[energy]) for energy in range(15)]
        directions, control = stream[1564:1568], stream[1568:1640]
    prescale_bits = int(header[1]) if np.isfinite(header[1]) else None
    fast = [6, 9] if version == 1 else [9, 11]
    cut_start = 10 if version == 1 else 12
    tables = []
    for energy, sample, codes in samples:
        cut = mode == "realtime" and energy >= cut_start
        scale = 1
        if mode == "realtime" and not cut and energy not in fast:
            scale = 8 if prescale_bits is not None and prescale_bits & (1 << (energy // 2)) else 1
            if prescale_bits is None:
                scale = None
        out = _count_table(codes, energy, sample, np.arange(16), "cut" if cut else "pad", scale or 1)
        out["prescale_known"] = scale is not None
        if scale is None:
            out[["prescale", "counts_lower", "counts_upper"]] = np.nan
        tables.append(out)
    volume_table = _count_table(volume, np.repeat(np.arange(15), 88), 0,
                                np.tile(np.arange(88), 15), "3d", 1)
    volume_table["prescale_known"] = True
    tables.append(volume_table)
    counts = pd.concat(tables, ignore_index=True)
    complete = set(packets) == expected
    summary = {"mode": mode, "telemetry_version": version, "complete": complete,
               "packets_received": len(packets), "packets_expected": len(expected),
               "missing_packets": ",".join(map(str, sorted(expected.difference(packets)))),
               "nhs": int(header[0]) & 31 if np.isfinite(header[0]) else np.nan,
               "pitch_map_version": int(header[0]) >> 5 if np.isfinite(header[0]) else np.nan,
               "prescale_bits": prescale_bits if prescale_bits is not None else np.nan}
    summary["telemetry_layout"] = min(version, 3)
    summary["layout_basis"] = "empirical format-3 compatibility" if version >= 4 else "MAGERFMT"
    header_counts = decompress_er_counts(header[2:4], prescale=16)
    for i, name in enumerate(("total", "rate")):
        summary[f"{name}_code"] = header[i + 2]
        summary[f"{name}_counts_lower"] = header_counts["counts_lower"].iloc[i]
        summary[f"{name}_counts_upper"] = header_counts["counts_upper"].iloc[i]
    if mode == "burst":
        angles = pd.DataFrame({"quarter_spin": [0, 1], "theta_code": directions[::2],
                               "phi_code": directions[1::2]})
    elif version >= 3:
        angles = pd.DataFrame({"quarter_spin": np.arange(64), "phi_code": directions[::2],
                               "theta_code": directions[1::2]})
    else:
        angles = pd.DataFrame(columns=["quarter_spin", "phi_code", "theta_code"])
    angles["phi_deg"] = angles["phi_code"].astype(float) * 360 / 256
    angles["theta_deg"] = angles["theta_code"].astype(float) * 360 / 256
    return {"counts": counts, "summary": summary, "header": header,
            "angles": angles, "control": control}


def time_er_counts(counts, summary, *, context=None, reference_clock_s=np.nan,
                   reference_utc=pd.NaT):
    """Attach accumulation bounds; these intervals are not detector live times"""
    out = counts.copy()
    period = context.get("period_s", np.nan) if context is not None else np.nan
    epoch = summary.get("accumulation_clock_s", np.nan)
    version, mode = summary["telemetry_version"], summary["mode"]
    energy, sample = out["energy_step"].to_numpy(), out["sample"].to_numpy()
    fast = np.isin(energy, [6, 9] if version == 1 else [9, 11])
    long = energy <= 5 if version < 3 else np.isin(energy, list(range(8)) + [13, 14])
    spins = np.where(fast, 0.5, np.where(long, 8, 4))
    if mode == "burst":
        spins[:] = 0.5
    interval = spins * period
    stop = epoch + sample * interval
    start = stop - interval
    volume = out["kind"].eq("3d").to_numpy()
    cut = out["kind"].eq("cut").to_numpy()
    start[cut] = stop[cut] - period/2
    start[volume], stop[volume] = epoch - period/2, epoch
    out["start_clock_s"], out["stop_clock_s"] = start, stop
    out["interval_s"] = stop - start
    out["nominal_integration_s"] = np.nan
    bins = er_angular_bins().set_index("angle_bin")
    widths = bins.loc[out.loc[volume, "angle_bin"], "phase_bins"].to_numpy()
    out.loc[volume, "nominal_integration_s"] = period / 256 * widths
    out.loc[cut, "nominal_integration_s"] = period / 256
    for name, values in (("start_utc", start), ("stop_utc", stop)):
        if pd.notna(reference_utc) and np.isfinite(reference_clock_s):
            valid = np.isfinite(values)
            dates = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
            dates.loc[valid] = reference_utc + pd.to_timedelta(values[valid] - reference_clock_s, unit="s")
            out[name] = dates
        else:
            out[name] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    return out
