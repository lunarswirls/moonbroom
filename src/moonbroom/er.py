"""LP Electron Reflectometer tables, without assuming an energy-bin assignment"""

import numpy as np
import pandas as pd

from .io import read_pds4


def read_er(path, *, table=0, time_col="Time"):
    """Read a labelled ER table, including repeated flux or pitch-angle fields

    Archive field names, units and flagged values come from the label. Energy
    bin definitions remain in the product's companion energy-bin table.
    """
    df = read_pds4(path, table=table)
    if time_col in df:
        df["utc"] = pd.to_datetime(df[time_col], utc=True, format="mixed", errors="raise")
    return df


def read_lper_txt(path):
    """Read the legacy three-column high-resolution ER flux ASCII format

    The f1_340ev/f2_140ev aliases are retained for existing callers. Consult
    E_BINS for actual channel energies; these columns contain electron flux.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if len(df.columns) != 3:
        raise ValueError("Expected Time and two ER flux columns; use read_er with an XML label")
    df.columns = ["Time", "f1_340ev", "f2_140ev"]
    df["utc"] = pd.to_datetime(df["Time"], utc=True, format="mixed", errors="raise")
    for column in ("f1_340ev", "f2_140ev"):
        df[column] = pd.to_numeric(df[column], errors="raise").replace(
            [9.999e-99, -9.99e-99, -9.999e-99], np.nan)
    return df
