"""Explicit-layout raw records and caller-supplied instrument calibration"""

from pathlib import Path

import numpy as np
import pandas as pd


def read_raw_records(path, fields, *, record_bytes, offset_bytes=0):
    """Read fixed-size binary records using an explicit field-layout DataFrame

    fields requires name, dtype and offset_bytes columns. NumPy dtype strings
    specify byte order explicitly, for example >u2 or <f4. This is a record
    reader, not a Level-0 packet decoder: packet demultiplexing, bit unpacking
    and time reconstruction must follow the specific archive format first.
    """
    if record_bytes <= 0 or offset_bytes < 0:
        raise ValueError("record_bytes must be positive and offset_bytes nonnegative")
    names = fields["name"].tolist()
    formats = [np.dtype(value) for value in fields["dtype"]]
    offsets = fields["offset_bytes"].to_numpy(int)
    if len(set(names)) != len(names) or not names:
        raise ValueError("Field names must be unique and nonempty")
    if any(dtype.hasobject or dtype.itemsize == 0 for dtype in formats):
        raise ValueError("Raw record fields must have a fixed binary size")
    if any(start < 0 or start + dtype.itemsize > record_bytes
           for start, dtype in zip(offsets, formats)):
        raise ValueError("Field extends outside the record")
    path = Path(path).expanduser()
    size = path.stat().st_size - offset_bytes
    if size < 0 or size % record_bytes:
        raise ValueError("File contains an incomplete record or an invalid offset")
    dtype = np.dtype({"names": names, "formats": formats,
                      "offsets": offsets.tolist(), "itemsize": record_bytes})
    records = np.fromfile(path, dtype=dtype, offset=offset_bytes)
    df = pd.DataFrame({name: records[name].astype(records[name].dtype.newbyteorder("="))
                       for name in names})
    df.attrs = {"source": str(path), "processing_level": "raw"}
    return df


def calibrate_mag(df, calibration, *, range_col="range", count_cols=("bx", "by", "bz"),
                  frame="SEN", flag_range_changes=True):
    """Apply supplied per-range zero counts and nT/count gains to unpacked MAG data

    calibration requires range and gain_x/gain_y/gain_z, zero_x/zero_y/zero_z.
    Each output is (counts - zero) * gain. Gains and zero levels must come
    from the instrument calibration, with any time-dependent offsets already
    resolved by the caller. No despinning or attitude rotation is implied.
    """
    if len(count_cols) != 3:
        raise ValueError("Exactly three MAG count columns are required")
    if calibration["range"].duplicated().any() or calibration["range"].isna().any():
        raise ValueError("Calibration ranges must be unique and nonmissing")
    out = df.copy()
    lookup = calibration.set_index("range")
    if not out[range_col].isin(lookup.index).all():
        raise ValueError("Missing calibration for one or more MAG ranges")
    coefficients = lookup.reindex(out[range_col]).reset_index(drop=True)
    numeric = coefficients[[f"{kind}_{axis}" for kind in ("gain", "zero") for axis in "xyz"]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Calibration coefficients must be finite")
    changed = out[range_col].ne(out[range_col].shift()).to_numpy(copy=True)
    if len(changed):
        changed[0] = False
    out["range_changed"] = changed
    for column, axis in zip(count_cols, "xyz"):
        values = ((out[column].to_numpy(float) - coefficients[f"zero_{axis}"].to_numpy(float))
                  * coefficients[f"gain_{axis}"].to_numpy(float))
        if flag_range_changes:
            values[changed] = np.nan
        out[f"B{axis}_{frame}"] = values
    out.attrs["calibration"] = "caller-supplied MAG gains and zero levels"
    return out


def calibrate_er(df, channels, *, accumulation_col="accumulation_s"):
    """Convert unpacked ER counts to differential number flux using supplied factors

    channels requires count_col, flux_col, geometric_factor_cm2_sr,
    energy_width_ev, efficiency. Flux is counts divided by live accumulation
    time, geometric factor, energy width and efficiency. The caller must
    decode count compression and correct dead time/background as needed.
    """
    out = df.copy()
    duration = out[accumulation_col].to_numpy(float)
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Accumulation times must be positive and finite")
    if channels["flux_col"].duplicated().any() or channels.empty:
        raise ValueError("Output flux columns must be unique and nonempty")
    factors = channels[["geometric_factor_cm2_sr", "energy_width_ev", "efficiency"]].to_numpy(float)
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("ER calibration factors must be positive and finite")
    for (_, channel), factor in zip(channels.iterrows(), factors.prod(axis=1)):
        counts = out[channel["count_col"]].to_numpy(float)
        counts = np.where(counts >= 0, counts, np.nan)
        out[channel["flux_col"]] = counts / (duration * factor)
    out.attrs["calibration"] = "caller-supplied ER response and live accumulation time"
    return out
