"""MAG/ER instrument bit fields from MAGERFMT sections 2.2, 2.5 and 2.6"""

import numpy as np
import pandas as pd


def decompress_er_counts(codes, *, prescale=1):
    """Return the lower and upper count bounds of the lossy 8-bit ER code

    Compression and prescaling truncate. Code 255 may be saturated and has
    an unbounded upper value. Missing codes remain missing, not zero counts.
    """
    values = np.asarray(codes, dtype=float).reshape(-1)
    scale = np.broadcast_to(np.asarray(prescale, dtype=float), values.shape)
    valid = np.isfinite(values)
    if np.isinf(values).any() or ((values[valid] < 0) | (values[valid] > 255) | (values[valid] % 1 != 0)).any():
        raise ValueError("ER codes must be integers in [0, 255] or missing")
    if not np.isfinite(scale).all() or (scale < 1).any() or (scale % 1 != 0).any():
        raise ValueError("Prescale factors must be positive integers")
    code = np.where(valid, values, 0).astype(np.int64)
    exponent, mantissa = code >> 4, code & 15
    lower = np.where(exponent == 0, mantissa, (16 + mantissa) * 2.0**np.maximum(exponent - 1, 0))
    step = 2.0**np.maximum(exponent - 1, 0)
    upper = (lower + step) * scale - 1
    lower *= scale
    upper[code == 255] = np.inf
    lower[~valid] = np.nan
    upper[~valid] = np.nan
    return pd.DataFrame({"code": values, "prescale": scale, "counts_lower": lower,
                         "counts_upper": upper, "saturation_possible": valid & (code == 255)})


def decode_mag_block(block):
    """Unpack an 82-byte MAG block into eighteen excess-2048 vector samples"""
    if len(block) != 82:
        raise ValueError("A MAG block must contain exactly 82 bytes")
    data = np.frombuffer(block, dtype=np.uint8)
    packed = data[1:].reshape(-1, 3).astype(np.uint16)
    values = np.column_stack([packed[:, 0] | ((packed[:, 1] & 15) << 8),
                              (packed[:, 1] >> 4) | (packed[:, 2] << 4)]).reshape(18, 3)
    df = pd.DataFrame(values, columns=["bx", "by", "bz"])
    for axis in "xyz":
        df[f"b{axis}_signed"] = df[f"b{axis}"].astype(int) - 2048
    df["sample_index"] = np.arange(18)
    df["range"] = int(data[0] & 7)
    df["calibration_mode"] = bool(data[0] & 8)
    df["mag_frame"] = int(data[0] >> 4)
    df["adc_at_limit"] = ((values == 0) | (values == 4095)).any(axis=1)
    return df


def decode_digital_subcom(data, *, word_mask=65535):
    """Decode 32 digital subcom bytes, retaining missing words as missing fields

    word_mask bit n indicates that the two bytes for subcom frame n exist.
    This permits decoding partial major frames without inventing timing data.
    Housekeeping conversions are the nominal coefficients in section 2.5.
    """
    if len(data) != 32 or not 0 <= word_mask <= 65535:
        raise ValueError("Digital subcom requires 32 bytes and a 16-bit word mask")

    def value(indices, transform):
        if all(word_mask & (1 << (i // 2)) for i in indices):
            return transform(*[data[i] for i in indices])
        return np.nan

    fields = {
        "command": ([0, 1, 2], lambda a, b, c: a | b << 8 | c << 16),
        "command_count": ([3], lambda a: a),
        "error_code": ([4], lambda a: a),
        "reset_count": ([5], lambda a: a >> 4),
        "error_count": ([5], lambda a: a & 15),
        "telemetry_version": ([6], lambda a: a >> 4),
        "boom_deployed": ([6], lambda a: (a >> 3) & 1),
        "eeprom_write_enabled": ([6], lambda a: (a >> 2) & 1),
        "electronics_side": ([6], lambda a: (a >> 1) & 1),
        "boot_prom": ([6], lambda a: a & 1),
        "imon_counts": ([7, 8], lambda a, b: a | (b & 15) << 8),
        "sweep_monitor_counts": ([8, 9], lambda a, b: (a >> 4) | b << 4),
        "phase": ([22, 23], lambda a, b: a | (b & 15) << 8),
        "hv_plug_enabled": ([23], lambda a: (a >> 7) & 1),
        "er_cover_closed": ([23], lambda a: (a >> 6) & 1),
        "sweep_enabled": ([23], lambda a: (a >> 5) & 1),
        "mcp_enabled": ([23], lambda a: (a >> 4) & 1),
        "period": ([24, 25], lambda a, b: a | (b & 63) << 8),
        "pll_status": ([25], lambda a: a >> 6),
        "sweep_max": ([26, 27], lambda a, b: a | b << 8),
        "er_power": ([28], lambda a: a >> 7),
        "autoranging": ([28], lambda a: (a >> 6) & 1),
        "sweep_rate": ([28], lambda a: a & 63),
        "mcp_setting": ([29], lambda a: a),
        "best_trigger": ([30], lambda a: a),
        "burst_read_enabled": ([31], lambda a: a >> 7),
        "burst_read_state": ([31], lambda a: (a >> 6) & 1),
        "burst_record_state": ([31], lambda a: (a >> 4) & 3),
        "burst_read_fraction": ([31], lambda a: a & 15),
    }
    result = {name: value(indices, transform) for name, (indices, transform) in fields.items()}
    result.update({"raw_dsc": bytes(data), "word_mask": word_mask, "complete": word_mask == 65535,
                   "period_s": result["period"] * 0.001024,
                   "phase_deg": result["phase"] * 360 / 4096,
                   "er_parameters_valid": result["er_power"] == 1})
    coefficients = [("imon_ma_nominal", "imon_counts", 0.0724, 2006.6),
                    ("sweep_v_nominal", "sweep_monitor_counts", 2.475, 2047.1)]
    for name, source, gain, zero in coefficients:
        result[name] = (result[source] - zero) * gain
    for name, index, gain, zero in [
        ("mag_plus12_v", 11, 0.064, 0), ("mag_minus12_v", 13, -0.064, 0),
        ("mag_plus5_v", 14, 0.032, 0), ("er_mcp_ma", 15, 0.0988, 0),
        ("er_mcp_v", 16, 15.02, -0.5), ("er_sweep_ma", 17, 0.060, 0),
        ("er_minus5_v", 18, -0.0258, 0), ("er_plus5_v", 19, 0.0252, 0),
        ("er_plus12_v", 20, 0.0627, 0), ("er_plus28_v", 21, 0.137, 0),
    ]:
        result[name + "_nominal"] = (value([index], lambda a: a) - zero) * gain
    for name, index, coefficients in [("pc_temperature_c_nominal", 10, [32.33, 0.413, 1.84e-3, 1e-5]),
                                       ("mag_temperature_c_nominal", 12, [9.46, 0.332, 5.98e-5, 4.4e-6])]:
        counts = value([index], lambda a: a) - 128
        result[name] = sum(coefficient * counts**i for i, coefficient in enumerate(coefficients))
    return pd.DataFrame([result])


def er_frame_layout(version):
    """Return real-time ER byte assignments for software versions 1 through 5

    Sample indices are counted in transmission order, resolving inconsistent
    sample labels in the printed tables while preserving every byte boundary.
    Software 4/5 use the format-3 packet layout, established empirically from
    paired PDS products rather than published format-4/5 byte specifications.
    """
    if version not in (1, 2, 3, 4, 5):
        raise ValueError("Supported ER software versions are 1 through 5")
    fast = [6, 9] if version == 1 else [9, 11]
    schedule = [list(range(5)), list(range(5, 10)), list(range(10, 15)), fast*2]
    schedule += [fast] * 5
    schedule += [list(range(6, 11)) if version < 3 else [8, 9, 10],
                 list(range(11, 15)) if version < 3 else [11, 12]]
    schedule += [fast] * 7
    schedule += [list(range(5)), list(range(5, 10)), list(range(10, 15)), fast*2]
    schedule += [fast] * 5
    schedule += [list(range(6, 11)) if version < 3 else [8, 9, 10],
                 list(range(11, 15)) if version < 3 else [11, 12]]
    schedule += [fast] * 7
    seen = np.zeros(15, dtype=int)
    records = []
    stream = 0
    for frame_type, energies in enumerate(schedule):
        offset = 0
        for energy in energies:
            records.append((frame_type, offset, 16, "angular", energy, int(seen[energy]), -1))
            seen[energy] += 1
            offset += 16
        header = 3 if frame_type == 0 else 1 if frame_type == 1 else 0
        if header:
            records.append((frame_type, offset, header, "header", -1, -1, 0 if frame_type == 0 else 3))
            offset += header
        while offset < 83:
            kind = "3d" if stream < 1320 else "mag_angles"
            length = min(83 - offset, 1320 - stream if stream < 1320 else 1448 - stream)
            records.append((frame_type, offset, length, kind, -1, -1,
                            stream if kind == "3d" else stream - 1320))
            offset += length
            stream += length
    if stream != (1448 if version >= 3 else 1320):
        raise ValueError("Internal ER layout does not cover the expected byte stream")
    return pd.DataFrame(records, columns=["frame_type", "offset", "length", "kind", "energy_step", "sample", "stream_offset"])


def er_angular_bins():
    """Return the official 88-bin map to anode and 22.5-degree spin-phase bins"""
    rows = []
    for anode, count in enumerate([2, 4, 8, 8, 8, 8, 4, 2, 2, 4, 8, 8, 8, 8, 4, 2]):
        width = 8 // count
        for phase in range(0, 8, width):
            rows.append((len(rows), anode, phase, width, phase*22.5, (phase + width)*22.5))
    return pd.DataFrame(rows, columns=["angle_bin", "anode_bin", "phase_bin", "phase_bins", "phase_start_deg", "phase_stop_deg"])


def er_energy_bins(sweep_max, sweep_rate, *, analyzer_constant=5.65, max_voltage_v=4835.0):
    """Nominal energy edges from the programmable exponential sweep

    This evaluates the published recurrence without assuming undocumented DAC
    integer rounding or actual instrument response. Channel 0 is highest energy.
    """
    if not 0 < sweep_max <= 65535 or not 0 <= sweep_rate <= 47:
        raise ValueError("Sweep maximum must be in (0, 65535] and sweep rate in [0, 47]")
    top = sweep_max / 65535 * max_voltage_v * analyzer_constant
    edges = top * (1 - sweep_rate / 512)**(8 * np.arange(16))
    return pd.DataFrame({"energy_step": np.arange(15), "energy_low_ev_nominal": edges[1:],
                         "energy_high_ev_nominal": edges[:-1]})
