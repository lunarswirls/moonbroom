"""SEL field and position conversions, with explicit coordinate conventions"""

import numpy as np
import pandas as pd


def wrap_lon_180(lon_deg):
    return (np.asarray(lon_deg) + 180.0) % 360.0 - 180.0


def compute_lonlat_from_sel(df):
    """Add east-positive longitude and planetocentric latitude in degrees"""
    out = df.copy()
    position = out[["X_SEL", "Y_SEL", "Z_SEL"]].apply(pd.to_numeric, errors="coerce")
    x, y, z = position.to_numpy(float).T
    radius = np.linalg.norm(position, axis=1)
    out["lon_deg"] = np.where(radius > 0, wrap_lon_180(np.degrees(np.arctan2(y, x))), np.nan)
    out["lat_deg"] = np.where(radius > 0, np.degrees(np.arctan2(z, np.hypot(x, y))), np.nan)
    return out


def sel_to_spherical(df):
    """Add outward Br, southward Btheta and eastward Bphi without changing altitude"""
    out = df.copy()
    x, y, z = out[["X_SEL", "Y_SEL", "Z_SEL"]].to_numpy(float).T
    bx, by, bz = out[["Bx_SEL", "By_SEL", "Bz_SEL"]].to_numpy(float).T
    radius = np.sqrt(x*x + y*y + z*z)
    safe = np.where(radius > 0, radius, np.nan)
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z / safe, -1, 1))
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    out["r_km"] = radius
    out["theta_rad"] = theta
    out["phi_rad"] = phi
    out["Br"] = bx*st*cp + by*st*sp + bz*ct
    out["Btheta"] = bx*ct*cp + by*ct*sp - bz*st
    out["Bphi"] = np.where(radius > 0, -bx*sp + by*cp, np.nan)
    return out


def add_field_magnitude(df, *, frame="SEL"):
    """Add field magnitude in nT, requiring all three finite components"""
    out = df.copy()
    out[f"Bmag_{frame}"] = np.linalg.norm(
        out[[f"B{axis}_{frame}" for axis in "xyz"]].to_numpy(float), axis=1)
    return out


def position_from_lonlat(df, *, lat_col="Latitude", lon_col="Longitude",
                        alt_col="Altitude", radius_km=1737.4):
    """Add SEL positions using planetocentric angles and spherical altitude in km"""
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("radius_km must be positive and finite")
    out = df.copy()
    lat = np.deg2rad(out[lat_col].to_numpy(float))
    lon = np.deg2rad(out[lon_col].to_numpy(float))
    radius = radius_km + out[alt_col].to_numpy(float)
    out["X_SEL"] = radius * np.cos(lat) * np.cos(lon)
    out["Y_SEL"] = radius * np.cos(lat) * np.sin(lon)
    out["Z_SEL"] = radius * np.sin(lat)
    out["altitude_km"] = out[alt_col]
    return out
