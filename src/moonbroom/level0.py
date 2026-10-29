"""Official LP 472-byte merged telemetry records and MAG/ER decoding entry points"""

from pathlib import Path
import re
import struct
import hashlib

import numpy as np
import pandas as pd


def _matching_file(directory, name):
    matches = [path for path in directory.iterdir() if path.name.lower() == name.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file named {name!r} in {directory}")
    return matches[0]


def _merged_source(path, label_path=None):
    """Resolve the binary and validate its optional detached label."""
    path = Path(path).expanduser()
    label = Path(label_path).expanduser() if label_path is not None else None
    if path.suffix.lower() == ".lbl":
        label = path
    elif label is None:
        matches = [item for item in path.parent.iterdir()
                   if item.stem.lower() == path.stem.lower() and item.suffix.lower() == ".lbl"]
        if len(matches) == 1:
            label = matches[0]
    metadata = {}
    if label is not None:
        text = label.read_text(encoding="ascii")
        root = re.split(r"(?m)^OBJECT\s*=", text, maxsplit=1)[0]
        metadata = {key: value.strip().strip('"') for key, value in
                    re.findall(r"(?m)^([\w^]+)\s*=\s*([^\r\n]+)", root)}
        if metadata.get("RECORD_BYTES") != "472" or metadata.get("RECORD_TYPE") != "FIXED_LENGTH":
            raise ValueError("Expected official fixed-length 472-byte LP merged telemetry")
        if not re.fullmatch(r'"?[^"(),]+"?', metadata.get("^TABLE", "")):
            raise ValueError("Expected a detached merged table starting at record one")
        labelled = _matching_file(label.parent, metadata["^TABLE"])
        if path.suffix.lower() != ".lbl" and path.resolve() != labelled.resolve():
            raise ValueError("The supplied label describes a different data file")
        path = labelled
    return path, label, metadata


def _merged_record_layout(path, metadata, *, strict=True, allow_incomplete_final_record=False):
    file_bytes = path.stat().st_size
    count, trailing = divmod(file_bytes, 472)
    label_records = int(metadata["FILE_RECORDS"]) if "FILE_RECORDS" in metadata else None
    recover = bool(trailing and allow_incomplete_final_record)
    if recover:
        # Only a labelled, partial final frame is recoverable. Do not excuse
        # missing whole frames, arbitrary garbage or an incomplete first frame.
        with path.open('rb') as stream:
            stream.seek(count * 472)
            header = stream.read(10)
        if (not count or label_records != count + 1 or len(header) < 10
                or header[:6] != bytes.fromhex('1acffc1d66c5') or header[9] != 0):
            raise ValueError(f"Not a labelled incomplete final merged record: {path.name}")
    elif trailing and strict:
        raise ValueError(f"Incomplete 472-byte record in {path.name}: {trailing} trailing bytes")
    if label_records is not None and label_records != count + int(recover):
        raise ValueError("Data length does not match the label FILE_RECORDS")
    with path.open('rb') as stream:
        stream.seek(count * 472)
        tail = stream.read()
    return dict(file_bytes=file_bytes, complete_records=count, label_records=label_records,
                trailing_bytes=trailing, trailing_byte_offset=count * 472,
                trailing_hex=tail.hex(), incomplete_final_record=recover,
                missing_bytes=472 - trailing if recover else 0)


def inspect_merged_telemetry(path, *, label_path=None, allow_incomplete_final_record=False):
    """Inspect a source layout without decoding or changing any archive bytes."""
    path, label, metadata = _merged_source(path, label_path)
    return dict(source_file=str(path.resolve()), label=str(label) if label else None,
                **_merged_record_layout(path, metadata,
                                        allow_incomplete_final_record=allow_incomplete_final_record))


def read_merged_telemetry(path, *, label_path=None, year=None, strict=True, record_slice=None,
                          allow_incomplete_final_record=False):
    """Read a Level-0 .B or detached .LBL file, preserving all instrument bytes.

    ERT is Earth receive time, already corrected for delayed playback by the
    archive. It is not spacecraft event time. No extra 3200-second correction
    is made. The label supplies year and record count; without a label, year
    must be supplied or present in the official Myydddhh.B filename.
    allow_incomplete_final_record explicitly recovers complete frames when the
    label counts one final frame whose valid header but incomplete bytes remain.
    The tail is retained as hex in attributes and is never padded or decoded.
    Complete-frame header checks still follow strict independently.
    """
    path, label, metadata = _merged_source(path, label_path)
    if year is None:
        start = metadata.get("EARTH_RECEIVED_START_TIME", metadata.get("START_TIME"))
        year = pd.Timestamp(start).year if start else None
    if year is None:
        match = re.fullmatch(r"m(\d{2})\d{5}\.b", path.name.lower())
        if match:
            year = 1900 + int(match[1]) if int(match[1]) >= 80 else 2000 + int(match[1])
        else:
            raise ValueError("Supply the ERT year or a matching detached PDS label")
    layout = _merged_record_layout(path, metadata, strict=strict,
                                   allow_incomplete_final_record=allow_incomplete_final_record)
    count = layout['complete_records']
    raw = path.read_bytes()
    if record_slice is None:
        start, stop = 0, count
    else:
        start, stop = record_slice
        start = 0 if start is None else start
        stop = count if stop is None else stop
        if not (isinstance(start, int) and isinstance(stop, int) and 0 <= start <= stop <= count):
            raise ValueError("record_slice must be ordered, nonnegative record bounds within the file")
    rows = []
    previous_day = None
    for index in range(count):
        record = raw[index*472:(index + 1)*472]
        day, milliseconds = struct.unpack(">HI", record[456:462])
        if previous_day is not None and previous_day - day > 180:
            year += 1
        previous_day = day
        if index < start or index >= stop:
            continue
        header = int.from_bytes(record[4:6], "big")
        version, spacecraft, rate = header >> 14, (header >> 6) & 255, header & 63
        vcdu = int.from_bytes(record[6:9], "big")
        days = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
        ert_valid = 1 <= day <= days and milliseconds < 86401000
        leap_second = ert_valid and milliseconds >= 86400000
        ert = (pd.Timestamp(year, 1, 1, tz="UTC") + pd.Timedelta(days=day - 1, milliseconds=milliseconds)
               if ert_valid and not leap_second else pd.NaT)
        ert_iso = ert.isoformat() if pd.notna(ert) else None
        if leap_second:
            date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=day - 1)
            ert_iso = date.strftime("%Y-%m-%d") + f"T23:59:60.{milliseconds - 86400000:03d}Z"
        flags = int.from_bytes(record[470:472], "big")
        valid = (record[:4] == bytes.fromhex("1acffc1d") and version == 1 and spacecraft == 155
                 and rate == 5 and record[9] == 0 and flags >> 6 == 0)
        if not valid and strict:
            raise ValueError(f"Invalid merged record header at {path.name}:{index}")
        agc, snr = struct.unpack(">ff", record[462:470])
        rows.append((str(path.resolve()), index, index*472, vcdu, ert, day, milliseconds,
                     agc, snr, (flags >> 4) & 3, (flags >> 2) & 3, flags & 3,
                     valid, ert_valid, record[10:54], record[54:286], record[286:454],
                     record[454:456], record, ert_iso, leap_second))
    columns = ["source_file", "record_index", "byte_offset", "vcdu", "ert", "ert_day", "ert_ms",
               "agc", "snr", "stream_count", "stream_comparison", "selected_stream",
               "header_valid", "ert_valid", "engineering", "spectrometer", "mag_er", "tailbytes", "raw_record",
               "ert_iso", "ert_leap_second"]
    df = pd.DataFrame(rows, columns=columns)
    df["ert"] = pd.to_datetime(df["ert"], utc=True)
    df.attrs = {"source_file": str(path.resolve()), "label": str(label) if label else None, **layout,
                "processing_level": "level0", "time_basis": "earth receive time"}
    return df


def correlate_frame_times(frames, *, clock_correlation=None, light_time_s=None):
    """Unwrap the 24-bit counter and optionally correlate header times to UTC

    A correlation DataFrame has vcdu_unwrapped and utc columns and at least two
    anchors. Interpolation is bounded by its anchors and allows clock drift.
    Alternatively light_time_s is a nonnegative scalar or per-record array;
    frame_utc then equals ERT minus that supplied one-way propagation delay.
    With neither option, clock-relative timing is available and UTC is missing.
    """
    if clock_correlation is not None and light_time_s is not None:
        raise ValueError("Choose clock correlation or light-time correction")
    out = frames.copy()
    counter = out["vcdu"].to_numpy(np.int64)
    differences = np.diff(counter)
    differences = np.where(differences < -(1 << 23), differences + (1 << 24), differences)
    if (differences <= 0).any():
        raise ValueError("Records must be ordered and unique; overlapping or conflicting counters found")
    unwrapped = np.r_[counter[0], counter[0] + np.cumsum(differences)] if len(counter) else counter
    out["vcdu_unwrapped"] = unwrapped
    out["frame_clock_s"] = unwrapped * 2.0
    out["gap_frames"] = np.r_[0, differences - 1] if len(counter) else np.array([], dtype=int)
    out["frame_utc"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["time_basis"] = "clock only"
    if light_time_s is not None:
        delay = np.broadcast_to(np.asarray(light_time_s, dtype=float), (len(out),))
        if not np.isfinite(delay).all() or (delay < 0).any():
            raise ValueError("One-way light time must be finite and nonnegative")
        out["frame_utc"] = out["ert"] - pd.to_timedelta(delay, unit="s")
        out["time_basis"] = "ert minus supplied one-way light time"
    elif clock_correlation is not None:
        correlation = clock_correlation.sort_values("vcdu_unwrapped")
        x = correlation["vcdu_unwrapped"].to_numpy(float)
        utc = pd.to_datetime(correlation["utc"], utc=True, format="mixed", errors="raise")
        if len(x) < 2 or not np.isfinite(x).all() or (np.diff(x) <= 0).any() or utc.isna().any():
            raise ValueError("Correlation requires at least two unique, finite ordered anchors")
        reference = utc.iloc[0]
        y = (utc - reference).dt.total_seconds().to_numpy()
        if (np.diff(y) <= 0).any():
            raise ValueError("UTC correlation anchors must increase with spacecraft clock")
        seconds = np.interp(unwrapped, x, y, left=np.nan, right=np.nan)
        valid = np.isfinite(seconds)
        out.loc[valid, "frame_utc"] = reference + pd.to_timedelta(seconds[valid], unit="s")
        out["time_basis"] = "supplied spacecraft clock correlation"
    return out


def clock_to_utc(clock_s, frames):
    """Interpolate UTC only within provided frame correlations, including gaps"""
    values = np.atleast_1d(np.asarray(clock_s, dtype=float))
    known = frames.loc[frames["frame_utc"].notna()]
    result = pd.Series(pd.NaT, index=range(len(values)), dtype="datetime64[ns, UTC]")
    if len(known) < 2:
        return result
    reference = known["frame_utc"].iloc[0]
    x = known["frame_clock_s"].to_numpy(float)
    y = (known["frame_utc"] - reference).dt.total_seconds().to_numpy()
    seconds = np.interp(values, x, y, left=np.nan, right=np.nan)
    valid = np.isfinite(seconds)
    result.loc[valid] = reference + pd.to_timedelta(seconds[valid], unit="s")
    return result


def read_sunpulse(path, *, frames=None, strict=True):
    """Read official eight-byte SUNPULSE records and reconstruct pulse clocks

    Pulse offsets and uncertainties use 1/1800 second. A downlinked pulse
    occurred within the preceding four seconds, so major-frame wrap is
    resolved against that interval. 0xFADE means no pulse. Source flag one
    indicates a reconstructed pulse, not a measured sun-sensor detection.
    strict=False retains odd-counter records with counter_valid=False and
    timing_valid=False; counter ordering violations still raise.
    """
    path = Path(path).expanduser()
    label = path if path.suffix.lower() == ".lbl" else None
    if label is None:
        labels = [item for item in path.parent.iterdir()
                  if item.stem.lower() == path.stem.lower() and item.suffix.lower() == ".lbl"]
        label = labels[0] if len(labels) == 1 else None
    metadata = {}
    if label is not None:
        text = re.split(r"(?m)^OBJECT\s*=", label.read_text(encoding="ascii"), maxsplit=1)[0]
        metadata = {key: value.strip().strip('"') for key, value in
                    re.findall(r"(?m)^([\w^]+)\s*=\s*([^\r\n]+)", text)}
        if metadata.get("RECORD_BYTES") != "8":
            raise ValueError("Expected eight-byte SUNPULSE records")
        labelled = _matching_file(label.parent, metadata["^TABLE"])
        if path.suffix.lower() != ".lbl" and path.resolve() != labelled.resolve():
            raise ValueError("The supplied sun-pulse label describes a different file")
        path = labelled
    raw = path.read_bytes()
    if len(raw) % 8:
        raise ValueError("Incomplete eight-byte SUNPULSE record")
    records = [raw[index:index + 8] for index in range(0, len(raw), 8)]
    if "FILE_RECORDS" in metadata and len(records) != int(metadata["FILE_RECORDS"]):
        raise ValueError("SUNPULSE file length does not match the PDS label")
    rows = [(int.from_bytes(record[:3], "big"), *struct.unpack(">HHB", record[3:])) for record in records]
    out = pd.DataFrame(rows, columns=["vcdu", "pulse_counts", "uncertainty_counts", "source_flag"])
    if not out.empty:
        counter = out["vcdu"].to_numpy(np.int64)
        differences = (np.diff(counter) + 2**24) % 2**24
        if ((strict and (counter % 2 != 0).any()) or (differences == 0).any()
                or (differences >= 2**23).any()):
            raise ValueError("SUNPULSE records require increasing even VCDU counters")
        base = counter[0]
        if frames is not None and not frames.empty:
            base += int(round((frames["vcdu_unwrapped"].iloc[0] - base) / 2**24)) * 2**24
        unwrapped = np.r_[base, base + np.cumsum(differences)]
    else:
        unwrapped = np.array([], dtype=np.int64)
    out["vcdu_unwrapped"] = unwrapped
    out["counter_valid"] = out["vcdu"].mod(2).eq(0)
    out["source_file"] = str(path.resolve())
    out["record_index"] = np.arange(len(out))
    frame_clock = unwrapped*2.0
    pulse = (unwrapped - unwrapped % 16)*2.0 + out["pulse_counts"].to_numpy(float) / 1800
    pulse = np.where(pulse > frame_clock, pulse - 32, pulse)
    valid = (out["counter_valid"].to_numpy() & out["pulse_counts"].ne(0xfade).to_numpy()
             & out["source_flag"].isin([0, 1]).to_numpy()
             & (pulse >= frame_clock - 4) & (pulse <= frame_clock))
    out["pulse_clock_s"] = np.where(valid, pulse, np.nan)
    out["uncertainty_s"] = out["uncertainty_counts"] / 1800
    out["estimated"] = out["source_flag"].eq(1)
    out["timing_valid"] = valid
    out["pulse_utc"] = (clock_to_utc(out["pulse_clock_s"], frames) if frames is not None
                         else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"))
    return out


def write_level0_tables(products, out_dir):
    """Write decoded tables as CSVs, encoding raw byte columns as hex strings"""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, df in products.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Product names must be lowercase identifiers")
        out = df.copy()
        for column in out.select_dtypes(include="object").columns:
            out[column] = out[column].map(lambda value: value.hex() if isinstance(value, (bytes, bytearray)) else value)
        path = out_dir / f"{name}.csv"
        out.to_csv(path, index=False)
        rows.append((name, str(path), len(out)))
    return pd.DataFrame(rows, columns=["product", "output", "rows"])


def decode_level0(paths, *, year=None, strict=True, clock_correlation=None,
                  light_time_s=None, telemetry_version=None, timing_kernels=None,
                  receiver=None, data_system_delay_s=2.0, mag_only=False,
                  overlap_policy="exact", record_slices=None, clock_cycle=0,
                  allow_incomplete_final_record=False):
    """Decode chronological merged files into a dictionary of reusable DataFrames

    Exact repeated boundary records are removed. Conflicting overlaps raise.
    Accumulations, digital subcom and burst state cross file boundaries.
    mag_only selects a batched real-time MAG path without ER or burst products;
    its spin phase requires archived SUNPULSE rather than digital subcom.
    identical_payload optionally deduplicates adjacent equal-counter physical
    records whose ground metadata differs and ERT agrees within 0.1 second.
    Every such removal is recorded; conflicting telemetry still raises.
    identical_mag applies that check only to MAG/ER payload and frame header,
    and is restricted to mag_only output so other instruments are not lost.
    identical_mager explicitly permits the same equality rule for full MAG/ER
    science decoding, recording differences and hashes of the other payloads.
    record_slices maps input paths to half-open source record bounds, preserving
    original indices. clock_cycle is the known mission rollover count at the
    first selected record, keeping clocks consistent between contextual shards.
    allow_incomplete_final_record retains a labelled partial final frame in
    input_files diagnostics and decodes only complete frames, even with strict.
    """
    from .mager import decode_mager_frames, decode_realtime_mag_frames

    if overlap_policy not in ("exact", "identical_payload", "identical_mag", "identical_mager"):
        raise ValueError("Choose exact, identical_payload, identical_mag or identical_mager overlaps")
    if overlap_policy == "identical_mag" and not mag_only:
        raise ValueError("identical_mag overlap handling requires mag_only output")
    if not isinstance(clock_cycle, int) or isinstance(clock_cycle, bool) or clock_cycle < 0:
        raise ValueError('clock_cycle must be a nonnegative integer rollover count')
    if isinstance(paths, (str, Path)):
        paths = [paths]
    record_slices = record_slices or {}
    tables = [read_merged_telemetry(path, year=year, strict=strict,
                                  allow_incomplete_final_record=allow_incomplete_final_record,
                                  record_slice=record_slices.get(str(path))) for path in paths]
    if not tables:
        raise ValueError("Supply at least one merged telemetry file")
    frames = pd.concat(tables, ignore_index=True)
    duplicates = int(frames["raw_record"].duplicated().sum())
    frames = frames.drop_duplicates("raw_record").reset_index(drop=True)
    overlaps = []
    if overlap_policy != "exact":
        remove = []
        for index in np.flatnonzero(np.diff(frames["vcdu"].to_numpy()) == 0) + 1:
            kept_index = index - 1
            while kept_index in remove:
                kept_index -= 1
            previous, current = frames.iloc[kept_index], frames.iloc[index]
            delta = (current.ert - previous.ert).total_seconds()
            same = previous.raw_record[:456] == current.raw_record[:456]
            if overlap_policy in ("identical_mag", "identical_mager"):
                same = (previous.raw_record[:10] == current.raw_record[:10]
                        and previous.raw_record[286:456] == current.raw_record[286:456])
            if not same or not np.isfinite(delta) or abs(delta) > .1:
                raise ValueError("Overlapping counters disagree in physical payload or ERT")
            remove.append(index)
            overlaps.append((previous.source_file, previous.record_index, current.source_file,
                             current.record_index, current.vcdu, previous.ert, current.ert, delta,
                             previous.raw_record[:456] != current.raw_record[:456],
                             hashlib.sha256(previous.raw_record).hexdigest(),
                             hashlib.sha256(current.raw_record).hexdigest()))
        frames = frames.drop(index=remove).reset_index(drop=True)
    if timing_kernels is not None:
        from .timing import reconstruct_frame_times

        if clock_correlation is not None or light_time_s is not None:
            raise ValueError("Choose SPICE reconstruction or supplied timing correlations")
        timing_kernels = ([timing_kernels] if isinstance(timing_kernels, (str, Path))
                          else list(timing_kernels))
        frames = reconstruct_frame_times(frames, timing_kernels, receiver=receiver,
                                         data_system_delay_s=data_system_delay_s)
    else:
        frames = correlate_frame_times(frames, clock_correlation=clock_correlation, light_time_s=light_time_s)
    if clock_cycle and clock_correlation is not None:
        raise ValueError('Explicit clock correlation already defines its counter epoch')
    frames['vcdu_unwrapped'] += clock_cycle * 2**24
    frames['frame_clock_s'] += clock_cycle * 2**25
    frames['clock_cycle'] = frames['vcdu_unwrapped'] // 2**24
    frames.attrs["duplicate_records_removed"] = duplicates
    frames.attrs["overlap_metadata_records_removed"] = len(overlaps)
    result = ({"mag": decode_realtime_mag_frames(frames)} if mag_only
              else decode_mager_frames(frames, telemetry_version=telemetry_version))
    if overlaps:
        result["overlap_records"] = pd.DataFrame(overlaps, columns=[
            "kept_source_file", "kept_record_index", "removed_source_file", "removed_record_index",
            "vcdu", "kept_ert", "removed_ert", "ert_difference_s", "other_payload_differs",
            "kept_record_sha256", "removed_record_sha256"])
        result["overlap_records"]["policy"] = overlap_policy
    result["frames"] = frames
    result["input_files"] = pd.DataFrame([
        {key: value for key, value in dict(table.attrs, records=len(table)).items()
         if key not in ('processing_level', 'time_basis')} for table in tables])
    if timing_kernels is not None:
        from .timing import reconstruct_product_times

        result = reconstruct_product_times(result, frames, timing_kernels)
        result["timing_kernels"] = pd.DataFrame({"path": [str(Path(path).expanduser().resolve())
                                                          for path in timing_kernels]})
    return result
