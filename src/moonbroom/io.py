"""Label-driven archive readers and CSV helpers"""

from pathlib import Path

import numpy as np
import pandas as pd


def find_label(path):
    """Find the matching XML label without assuming filename case"""
    path = Path(path).expanduser()
    if path.suffix.lower() == ".xml":
        return path
    matches = sorted(p for p in path.parent.iterdir()
                     if p.stem.lower() == path.stem.lower() and p.suffix.lower() == ".xml")
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one XML label for {path}, found {len(matches)}")
    return matches[0]


def read_pds4(path, *, table=0, scale=True):
    """Read an ASCII or binary PDS4 table, retaining units and masking flags

    Repeated group fields become columns such as flux[0,1], with zero-based
    indices in label order. Scale and offset are applied once by pds4_tools.
    This reads labelled raw tables as well as calibrated products; it does
    not decode unlabelled spacecraft telemetry or perform instrument calibration.
    """
    from pds4_tools import pds4_read

    label = find_label(path)
    structures = pds4_read(str(label), quiet=True, lazy_load=True,
                          decode_strings=True, no_scale=not scale)
    tables = [structure for structure in structures if structure.is_table()]
    if isinstance(table, str):
        matches = [structure for structure in tables if structure.id == table]
        if len(matches) != 1:
            raise KeyError(f"Expected one table named {table!r}")
        structure = matches[0]
    else:
        structure = tables[table]
    data = structure.as_masked().data
    columns = {}
    units = {}
    for i in range(len(data.dtype.names)):
        field = data.field(i)
        name = field.meta_data["name"]
        for index in np.ndindex(field.shape[1:]):
            suffix = "[" + ",".join(map(str, index)) + "]" if index else ""
            column = name + suffix
            if column in columns:
                raise ValueError(f"Duplicate field name {column!r}; select a distinct table")
            values = field[(slice(None),) + index]
            mask = np.ma.getmaskarray(values)
            values = np.asarray(values)
            if values.dtype.kind in "iufc" and mask.any():
                values = values.astype(float)
                values[mask] = np.nan
            elif mask.any():
                values = values.astype(object)
                values[mask] = None
            columns[column] = values
            units[column] = field.meta_data.get("unit")
    df = pd.DataFrame(columns)
    df.attrs = {"label": str(label), "table": structure.id, "units": units,
                "scaled": scale}
    return df


def load_lp_csvs(input_dir, recursive=False, *, pattern="*_nightside.csv"):
    """Combine processed CSVs, recording the full source path for each row"""
    input_dir = Path(input_dir).expanduser()
    paths = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir}")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "utc" in df:
            df["utc"] = pd.to_datetime(df["utc"], utc=True, format="mixed", errors="raise")
        df["source_file"] = str(path.resolve())
        frames.append(df)
    if not frames:
        raise ValueError("All input tables are empty")
    return pd.concat(frames, ignore_index=True)


def write_csv(df, out_path):
    """Write UTC timestamps with an explicit offset, preserving the input frame"""
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "utc" in out:
        out["utc"] = pd.to_datetime(out["utc"], utc=True, format="mixed", errors="raise")
    out.to_csv(out_path, index=False)
    return out_path
