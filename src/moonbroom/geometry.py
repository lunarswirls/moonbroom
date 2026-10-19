"""Optional SPICE geometry and explicit lunar wake selections"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


@contextmanager
def loaded_kernels(paths):
    """Temporarily load explicit kernel paths without clearing the caller's pool

    Prefer individual kernel paths. A metakernel's nested paths follow SPICE
    resolution rules, relative to the process working directory.
    """
    import spiceypy as spice

    if isinstance(paths, (str, Path)):
        paths = [paths]
    existing = {str(Path(spice.kdata(i, "ALL")[0]).resolve())
                for i in range(spice.ktotal("ALL"))}
    owned = []
    try:
        for path in paths:
            path = str(Path(path).expanduser().resolve())
            if path not in existing:
                owned.append(path)
                spice.furnsh(path)
                existing.add(path)
        yield
    finally:
        for path in reversed(owned):
            spice.unload(path)


def mean_radius_km(body="MOON"):
    import spiceypy as spice

    return float(np.mean(spice.bodvrd(body, "RADII", 3)[1]))


def _ephemeris_times(times):
    import spiceypy as spice

    times = pd.to_datetime(times, utc=True, format="mixed", errors="raise")
    if times.isna().any():
        raise ValueError("Geometry requires valid UTC timestamps")
    return np.asarray([spice.str2et(value.strftime("%Y-%m-%dT%H:%M:%S.%f")) for value in times])


def sel_to_sse(df, *, abcorr="NONE"):
    """Rotate SEL positions and fields using IAU_MOON and a Sun/ecliptic SSE basis

    SSE x points toward the Sun, y is ecliptic north cross x, and z completes
    the orthonormal basis. This uses the IAU_MOON approximation to archive SEL.
    """
    import spiceypy as spice

    out = df.copy()
    times = _ephemeris_times(out["utc"])
    position = out[["X_SEL", "Y_SEL", "Z_SEL"]].to_numpy(float)
    has_field = all(f"B{axis}_SEL" in out for axis in "xyz")
    field = out[[f"B{axis}_SEL" for axis in "xyz"]].to_numpy(float) if has_field else None
    positions = np.full_like(position, np.nan)
    fields = np.full_like(position, np.nan)
    for i, epoch in enumerate(times):
        sun = np.asarray(spice.spkpos("SUN", epoch, "ECLIPJ2000", abcorr, "MOON")[0])
        x = sun / np.linalg.norm(sun)
        y = np.cross([0.0, 0.0, 1.0], x)
        y /= np.linalg.norm(y)
        rotation = np.vstack([x, y, np.cross(x, y)]) @ spice.pxform("IAU_MOON", "ECLIPJ2000", epoch)
        positions[i] = rotation @ position[i]
        if has_field:
            fields[i] = rotation @ field[i]
    out[["X_SSE", "Y_SSE", "Z_SSE"]] = positions
    if has_field:
        out[["Bx_SSE", "By_SSE", "Bz_SSE"]] = fields
    return out


def classify_shadow_alt_sza(df, *, abcorr="LT+S", chunk_size=20000):
    """Return disc-shadow state, ellipsoidal altitude and surface-normal SZA

    SEL positions are treated as IAU_MOON. Shadow disc radii use the mean
    lunar radius; altitude and SZA use the SPICE triaxial ellipsoid. Invalid
    positions/times and points inside the Moon retain missing results.
    """
    import spiceypy as spice

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    state = pd.Series(pd.NA, index=df.index, dtype="string")
    altitude = np.full(len(df), np.nan)
    sza = np.full(len(df), np.nan)
    if df.empty:
        return state, altitude, sza
    a, b, c = map(float, spice.bodvrd("MOON", "RADII", 3)[1])
    sun_radius = mean_radius_km("SUN")
    moon_radius = (a + b + c) / 3
    positions = df[["X_SEL", "Y_SEL", "Z_SEL"]].to_numpy(float)
    utc = pd.to_datetime(df["utc"], utc=True, format="mixed", errors="coerce")
    valid = np.isfinite(positions).all(axis=1) & utc.notna().to_numpy()
    valid &= np.sum((positions / [a, b, c])**2, axis=1) > 1
    indices = np.flatnonzero(valid)
    times = _ephemeris_times(utc.iloc[indices])
    for start in range(0, len(indices), chunk_size):
        selected = indices[start:start + chunk_size]
        epochs = times[start:start + chunk_size]
        suns = np.asarray(spice.spkpos("SUN", epochs, "IAU_MOON", abcorr, "MOON")[0]).reshape(-1, 3)
        for index, sun in zip(selected, suns):
            position = positions[index]
            to_moon = -position
            to_sun = sun - position
            moon_distance = np.linalg.norm(to_moon)
            sun_distance = np.linalg.norm(to_sun)
            alpha_m = np.arcsin(np.clip(moon_radius / moon_distance, 0, 1))
            alpha_s = np.arcsin(np.clip(sun_radius / sun_distance, 0, 1))
            theta = np.arccos(np.clip(np.dot(to_moon, to_sun) / (moon_distance * sun_distance), -1, 1))
            value = "sunlit"
            if theta < abs(alpha_m - alpha_s):
                value = "umbra" if alpha_m > alpha_s else "antumbra"
            elif theta < alpha_m + alpha_s:
                value = "penumbra"
            state.iloc[index] = value
            surface, altitude[index] = spice.nearpt(np.ascontiguousarray(position), a, b, c)
            normal = spice.surfnm(a, b, c, surface)
            direction = sun - surface
            cosz = np.dot(normal, direction) / (np.linalg.norm(normal) * np.linalg.norm(direction))
            sza[index] = np.degrees(np.arccos(np.clip(cosz, -1, 1)))
    return state, altitude, sza


def select_wake_from_nightside(df_night, *, aberr_deg=4.0, aberr_sign=-1,
                              wake_radius_km=None, min_downstream_km=0.0,
                              model="cone", terminator_margin_deg=15.0,
                              moon_radius_km=1737.4):
    """Select exterior downstream SSE points using a cone or cylindrical wake

    The cone uses solar angle >= 90 + terminator_margin_deg and a downstream
    test on the aberrated antisun axis, matching the original L1B selection.
    The cylinder uses perpendicular distance <= wake_radius_km. Neither
    selection assumes a particular event or time interval.
    """
    if model not in ("cone", "cylinder"):
        raise ValueError("model must be cone or cylinder")
    if not 0 <= terminator_margin_deg < 90:
        raise ValueError("terminator_margin_deg must be in [0, 90)")
    if not np.isfinite(moon_radius_km) or moon_radius_km <= 0:
        raise ValueError("moon_radius_km must be positive and finite")
    radius = moon_radius_km if wake_radius_km is None else wake_radius_km
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("wake_radius_km must be positive and finite")
    position = df_night[["X_SSE", "Y_SSE", "Z_SSE"]].to_numpy(float)
    angle = np.deg2rad(aberr_sign * aberr_deg)
    axis = np.array([-np.cos(angle), -np.sin(angle), 0.0])
    downstream = position @ axis
    perpendicular = np.linalg.norm(position - downstream[:, None] * axis, axis=1)
    distance = np.linalg.norm(position, axis=1)
    mask = (downstream > min_downstream_km) & (distance > moon_radius_km)
    if model == "cone":
        cosine = np.divide(position[:, 0], distance, out=np.full(len(distance), np.nan), where=distance > 0)
        mask &= cosine <= np.cos(np.deg2rad(90 + terminator_margin_deg))
    else:
        mask &= perpendicular <= radius
    out = df_night.loc[mask].copy()
    out["wake_t_km"] = downstream[mask]
    out["wake_rperp_km"] = perpendicular[mask]
    return out
