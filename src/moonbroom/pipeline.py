"""MAG postprocessing pipelines shared by scripts and analysis packages"""

from pathlib import Path

import numpy as np
import pandas as pd

from .coordinates import compute_lonlat_from_sel
from .geometry import classify_shadow_alt_sza, select_wake_from_nightside, sel_to_sse
from .io import write_csv
from .mag import read_mag
from .processing import select_time_window


def postprocess_mag(df, *, start=None, stop=None, abcorr="LT+S", chunk_size=20000,
                    wake_model="cone", wake_options=None):
    """Add shadow, ellipsoidal altitude, SZA, lon/lat and explicit wake membership

    Requires caller-loaded SPICE kernels. Missing SSE positions are computed
    from SEL; supplied SSE positions are retained. wake_options is forwarded
    to select_wake_from_nightside. A None wake_model skips wake selection.
    """
    out = select_time_window(df, start, stop).reset_index(drop=True)
    out = compute_lonlat_from_sel(out)
    state, altitude, sza = classify_shadow_alt_sza(out, abcorr=abcorr, chunk_size=chunk_size)
    out["shadow_state"] = state
    out["altitude_km"] = altitude
    out["sza_deg"] = sza
    out["is_wake"] = False
    out["wake_t_km"] = np.nan
    out["wake_rperp_km"] = np.nan
    if wake_model is not None:
        if not all(column in out for column in ("X_SSE", "Y_SSE", "Z_SSE")):
            out = sel_to_sse(out)
        night = out.loc[out["shadow_state"].isin(["umbra", "penumbra", "antumbra"])].copy()
        wake = select_wake_from_nightside(night, model=wake_model, **(wake_options or {}))
        out.loc[wake.index, "is_wake"] = True
        out.loc[wake.index, ["wake_t_km", "wake_rperp_km"]] = wake[["wake_t_km", "wake_rperp_km"]]
    return out


def process_mag_directory(input_dir, out_dir, *, reader=read_mag, pattern="*.tab",
                          reader_options=None, **kwargs):
    """Write all, nightside and nightside_wake CSVs per input; return a manifest

    Filename matching is case-insensitive and errors propagate to the caller.
    Time bounds and geometry options are passed to postprocess_mag.
    """
    from fnmatch import fnmatchcase

    input_dir, out_dir = Path(input_dir).expanduser(), Path(out_dir).expanduser()
    paths = sorted(path for path in input_dir.iterdir()
                   if path.is_file() and fnmatchcase(path.name.lower(), pattern.lower()))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir}")
    records = []
    for path in paths:
        df = postprocess_mag(reader(path, **(reader_options or {})), **kwargs)
        night = df.loc[df["shadow_state"].isin(["umbra", "penumbra", "antumbra"])]
        subsets = {"all": df, "nightside": night, "nightside_wake": df.loc[df["is_wake"]]}
        for name, subset in subsets.items():
            output = write_csv(subset, out_dir / f"{path.stem}_{name}.csv")
            records.append((str(path), name, str(output), len(subset)))
    return pd.DataFrame(records, columns=["source", "subset", "output", "rows"])
