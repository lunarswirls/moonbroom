"""Readers for calibrated LP MAG lunar-coordinate and surface-track tables"""

from xml.etree import ElementTree as et

import numpy as np
import pandas as pd

from .coordinates import add_field_magnitude, position_from_lonlat
from .io import find_label, read_pds4


l1b_columns = ["utc", "et_days", "Bx_SEL", "By_SEL", "Bz_SEL",
               "Bx_SSE", "By_SSE", "Bz_SSE", "Brms",
               "X_SEL", "Y_SEL", "Z_SEL", "X_SSE", "Y_SSE", "Z_SSE", "ISUN"]


def read_lpmag_l1b_tab(path):
    """Read the documented 16-column spin-averaged lunar-coordinate ASCII format

    Use read_mag for a label-driven read with archive missing constants masked.
    et_days retains the legacy column name; the value is decimal day of year,
    not SPICE ephemeris time.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if len(df.columns) != len(l1b_columns):
        raise ValueError(f"Expected 16 lunar-coordinate columns in {path}")
    df.columns = l1b_columns
    df["utc"] = pd.to_datetime(df["utc"], utc=True, format="mixed", errors="raise")
    df[l1b_columns[1:]] = df[l1b_columns[1:]].apply(pd.to_numeric, errors="raise")
    for columns, flag in [(l1b_columns[2:9], 999.999), (l1b_columns[9:15], 999999.99),
                          (["et_days"], 999.999999)]:
        df[columns] = df[columns].replace(flag, np.nan)
    return add_field_magnitude(df)


def read_mag(path, *, table=0):
    """Read a labelled MAG product, preserving archive columns and RMS values"""
    df = read_pds4(path, table=table)
    time_col = next((column for column in ("utc", "Time", "time") if column in df), None)
    if time_col is None:
        raise ValueError("No absolute UTC field found; use read_mag_l2 for hour-of-day tables")
    df["utc"] = pd.to_datetime(df[time_col], utc=True, format="mixed", errors="raise")
    rms_col = next((column for column in df if column.lower() in ("b_rms", "brms")), None)
    if rms_col is not None and "Brms" not in df:
        df["Brms"] = df[rms_col]
    for frame in ("SEL", "SSE"):
        if all(f"B{axis}_{frame}" in df for axis in "xyz"):
            df = add_field_magnitude(df, frame=frame)
    return df


def read_mag_l2(path, *, radius_km=1737.4, table=0):
    """Read a labelled Hour/SG/Latitude/Longitude/Altitude table

    SG magnetic components are mapped to SEL. Hour is measured from midnight
    on the label start date; resets across midnight advance the day. No SSE
    position is invented and no field magnitude is substituted for RMS.
    """
    label = find_label(path)
    root = et.parse(label).getroot()
    start = next((element.text for element in root.iter()
                  if element.tag.split("}")[-1] == "start_date_time"), None)
    if start is None:
        raise ValueError(f"Missing start_date_time in {label}")
    df = read_pds4(label, table=table)
    required = ["Hour", "Bx_SG", "By_SG", "Bz_SG", "Latitude", "Longitude", "Altitude"]
    missing = set(required).difference(df.columns)
    if missing:
        raise KeyError(f"Missing L2 columns: {sorted(missing)}")
    hours = pd.to_numeric(df["Hour"], errors="raise")
    if not np.isfinite(hours).all() or (hours < 0).any():
        raise ValueError("Hour values must be finite and nonnegative")
    resets = hours.diff() < -12
    elapsed = hours + 24 * resets.cumsum()
    df["utc"] = pd.to_datetime(start, utc=True).normalize() + pd.to_timedelta(elapsed, unit="h")
    df = position_from_lonlat(df, radius_km=radius_km)
    for axis in "xyz":
        df[f"B{axis}_SEL"] = df[f"B{axis}_SG"]
    return add_field_magnitude(df)


def load_pds4_lp(tab_path, *, radius_km=1737.4):
    """Compatibility name for the labelled L2 reader"""
    return read_mag_l2(tab_path, radius_km=radius_km)
