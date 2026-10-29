"""Reusable time selection, quality filtering and altitude-weighted detrending"""

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_datetime(df, utc_col="utc"):
    """Return a copy with UTC-aware timestamps, rejecting malformed values"""
    out = df.copy()
    out[utc_col] = pd.to_datetime(out[utc_col], utc=True, format="mixed", errors="raise")
    if out[utc_col].isna().any():
        raise ValueError("Missing UTC timestamps")
    return out


def select_time_window(df, start=None, stop=None, *, utc_col="utc"):
    """Select an inclusive UTC interval; omitted bounds retain all available data"""
    out = ensure_datetime(df, utc_col)
    start = pd.to_datetime(start, utc=True) if start is not None else None
    stop = pd.to_datetime(stop, utc=True) if stop is not None else None
    if start is not None and stop is not None and start > stop:
        raise ValueError("start must not be later than stop")
    mask = pd.Series(True, index=out.index)
    if start is not None:
        mask &= out[utc_col] >= start
    if stop is not None:
        mask &= out[utc_col] <= stop
    return out.loc[mask].copy()


def filter_altitude_brms(df, *, alt_col="altitude_km", brms_col="Brms",
                         alt_max_km=40.0, brms_quantile=0.9):
    """Retain altitude < limit, then RMS <= quantile within that altitude subset"""
    if not 0 <= brms_quantile <= 1:
        raise ValueError("brms_quantile must be between 0 and 1")
    if not np.isfinite(alt_max_km):
        raise ValueError("alt_max_km must be finite")
    out = df.copy()
    out[[alt_col, brms_col]] = out[[alt_col, brms_col]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(out[[alt_col, brms_col]].to_numpy(float)).all(axis=1)
    out = out.loc[finite & (out[alt_col] < alt_max_km) & (out[brms_col] >= 0)].copy()
    if out.empty:
        return out
    return out.loc[out[brms_col] <= out[brms_col].quantile(brms_quantile)].copy()


def filter_wake_file(path, **kwargs):
    return filter_altitude_brms(pd.read_csv(Path(path)), **kwargs)


def segment_tracks_by_time(df, *, utc_col="utc", gap_sec=1200.0):
    """Sort samples in UTC and split tracks at gaps larger than gap_sec"""
    if not np.isfinite(gap_sec) or gap_sec <= 0:
        raise ValueError("gap_sec must be positive and finite")
    out = ensure_datetime(df, utc_col).sort_values(utc_col, kind="stable").reset_index(drop=True)
    out["track_id"] = out[utc_col].diff().dt.total_seconds().gt(gap_sec).cumsum().astype(int)
    return out


def compute_along_track_distance(df):
    """Cumulative Cartesian distance in km; positions must be present and finite"""
    position = df[["X_SEL", "Y_SEL", "Z_SEL"]].to_numpy(float)
    if not np.isfinite(position).all():
        raise ValueError("Along-track distance requires finite SEL positions")
    if df.empty:
        return np.array([], dtype=float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(position, axis=0), axis=1))]


def build_design_matrix(s, degree):
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    return np.vander(np.asarray(s, dtype=float), degree + 1, increasing=True)


def bayesian_trend_fit(s, y, h, *, degree=2, sigma0=1.0, h0=20.0, tau=1e4):
    """Fit y = polynomial(s) with altitude-weighted noise and N(0, tau²) priors

    s should be normalized to [0, 1] for stable polynomial fits. Noise variance
    is sigma0² * (1 + (h/h0)²). Returns fitted values and MAP coefficients.
    """
    if not all(np.isfinite(value) and value > 0 for value in (sigma0, h0, tau)):
        raise ValueError("sigma0, h0 and tau must be positive and finite")
    s, y, h = (np.asarray(values, dtype=float) for values in (s, y, h))
    if s.ndim != 1 or s.shape != y.shape or s.shape != h.shape:
        raise ValueError("s, y and h must be matching one-dimensional arrays")
    if not all(np.isfinite(values).all() for values in (s, y, h)):
        raise ValueError("Fit inputs must be finite")
    design = build_design_matrix(s, degree)
    if len(s) < degree + 1:
        raise ValueError("Insufficient samples for the polynomial degree")
    weights = 1 / (sigma0**2 * (1 + (h / h0)**2))
    weighted = design * np.sqrt(weights[:, None])
    augmented = np.vstack([weighted, np.eye(degree + 1) / tau])
    target = np.r_[y * np.sqrt(weights), np.zeros(degree + 1)]
    coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
    return design @ coefficients, coefficients


def detrend_track(df_track, *, alt_col="altitude_km", degree=2,
                  sigma0=1.0, h0=20.0, tau=1e4):
    """Add spherical trend and residual columns for one continuous track

    Br_crust/Btheta_crust/Bphi_crust retain legacy names for compatibility;
    subtracting this empirical trend does not establish a crustal origin.
    """
    out = df_track.copy()
    if "utc" in out:
        out = ensure_datetime(out).sort_values("utc", kind="stable").reset_index(drop=True)
    distance = compute_along_track_distance(out)
    if len(distance) and distance[-1] > 0:
        distance = distance / distance[-1]
    height = out[alt_col].to_numpy(float)
    for component in ("Br", "Btheta", "Bphi"):
        values = out[component].to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(height)
        trend = np.full(len(out), np.nan)
        if valid.sum() >= degree + 1:
            trend[valid], _ = bayesian_trend_fit(distance[valid], values[valid], height[valid],
                                               degree=degree, sigma0=sigma0, h0=h0, tau=tau)
        out[f"{component}_trend"] = trend
        out[f"{component}_crust"] = values - trend
    return out


def detrend_tracks(df, *, gap_sec=1200.0, **kwargs):
    """Detrend each time-continuous track separately, including empty inputs"""
    out = segment_tracks_by_time(df, gap_sec=gap_sec)
    if out.empty:
        return detrend_track(out, **kwargs)
    frames = [detrend_track(track, **kwargs) for _, track in out.groupby("track_id", sort=False)]
    return pd.concat(frames, ignore_index=True)
