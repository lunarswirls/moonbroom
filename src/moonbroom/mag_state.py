"""Observed MAG hardware context and explicit coefficient applicability.

Status snapshots constrain collection time, including for delayed burst data.
Changes are bracketed observations, not precisely timed hardware operations.
"""

import hashlib

import numpy as np
import pandas as pd


MAG_HARDWARE_FIELDS = ("electronics_side", "boom_deployed", "telemetry_version", "reset_count")
MAG_SCOPE_FIELDS = MAG_HARDWARE_FIELDS + ("mag_hardware_state_id",)


def mag_model_modes(coefficients):
    """Legacy coefficients apply to real-time MAG; missing mode never means both."""
    modes = coefficients.get("mode", pd.Series("realtime", index=coefficients.index)).fillna("realtime")
    if not modes.isin(["realtime", "burst"]).all():
        raise ValueError("MAG coefficient mode must be realtime or burst")
    return modes


def mag_model_mask(samples, model, *, mode=None, _times=None):
    """Match range, UTC, mode and every nonmissing hardware constraint."""
    times = _times if _times is not None else pd.to_datetime(samples["utc"], utc=True, format="mixed")
    model_mode = model.get("mode", "realtime")
    model_mode = "realtime" if pd.isna(model_mode) else model_mode
    sample_mode = samples.get("mode", pd.Series("realtime", index=samples.index))
    selected = (samples["range"].eq(model["range"]) & sample_mode.eq(mode or model_mode)
                & times.ge(pd.to_datetime(model["valid_from"], utc=True))
                & times.lt(pd.to_datetime(model["valid_to"], utc=True)))
    for field in MAG_SCOPE_FIELDS:
        if pd.notna(model.get(field)):
            selected &= samples[field].eq(model[field]).fillna(False) if field in samples else False
    return selected.fillna(False)


def validate_mag_model_scopes(coefficients):
    """Reject overlapping applicability, allowing disjoint modes/hardware states."""
    modes = mag_model_modes(coefficients)
    table = coefficients.copy()
    table["mode"] = modes
    for field in ("valid_from", "valid_to"):
        table[field] = pd.to_datetime(table[field], utc=True, format="mixed")
    if (table[["valid_from", "valid_to", "range"]].isna().any().any()
            or table["valid_from"].ge(table["valid_to"]).any()):
        raise ValueError("MAG model scopes require finite ordered UTC bounds and ranges")
    for _, group in table.groupby(["range", "mode"]):
        rows = list(group.sort_values("valid_from").iterrows())
        for i, (_, left) in enumerate(rows):
            for _, right in rows[i + 1:]:
                if right["valid_from"] >= left["valid_to"]:
                    break
                disjoint = any(pd.notna(left.get(f)) and pd.notna(right.get(f))
                               and left[f] != right[f] for f in MAG_SCOPE_FIELDS)
                if not disjoint:
                    raise ValueError("MAG calibration applicability overlaps for a range, mode and hardware state")


def mag_model_assignments(samples, coefficients, *, allow_realtime_transfer=False):
    """Return row masks, giving explicit burst models priority over opt-in transfer."""
    times = pd.to_datetime(samples["utc"], utc=True, format="mixed")
    masks = [mag_model_mask(samples, row, _times=times) for _, row in coefficients.iterrows()]
    assigned = pd.Series(False, index=samples.index)
    for mask in masks:
        assigned |= mask
    if allow_realtime_transfer:
        modes = mag_model_modes(coefficients)
        for i, (_, row) in enumerate(coefficients.iterrows()):
            if modes.iloc[i] == "realtime":
                masks[i] |= mag_model_mask(samples, row, mode="burst", _times=times) & ~assigned
    return masks


def annotate_mag_hardware_state(samples, subcom, *, burst_markers=None, max_age_s=32.0):
    """Attach bounded collection-time state and bracket observed changes.

    Burst context comes exclusively from its own marker sequence. Missing or
    stale snapshots remain unknown. The local hardware cohort prevents offset
    interpolation across A-to-B-to-A changes as well as single transitions.
    """
    if not np.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("Hardware snapshot maximum age must be positive and finite")
    out = samples.reset_index(drop=True).copy()
    for field in MAG_HARDWARE_FIELDS:
        out[field] = np.nan
    for field in ("mag_hardware_state_id", "mag_hardware_cohort"):
        out[field] = pd.Series(None, index=out.index, dtype=object)
    out["mag_hardware_state_source"] = "unavailable"
    out["mag_hardware_state_valid"] = False
    out["mag_hardware_transition"] = False

    def attach(selected, snapshots, source):
        if not selected.any() or snapshots.empty:
            return
        states = snapshots.copy()
        for field in MAG_HARDWARE_FIELDS:
            if field not in states:
                states[field] = np.nan
        states = states.loc[np.isfinite(states["major_clock_s"])].sort_values("major_clock_s")
        for _, group in states.groupby("major_clock_s"):
            if len(group[list(MAG_HARDWARE_FIELDS)].drop_duplicates()) > 1:
                raise ValueError("Conflicting hardware snapshots at one collection clock")
        states = states.drop_duplicates("major_clock_s").reset_index(drop=True)
        if states.empty:
            return
        clocks = states["major_clock_s"].to_numpy(float)
        positions = np.flatnonzero(selected.to_numpy())
        values = out.loc[selected, "clock_s"].to_numpy(float)
        index = np.searchsorted(clocks, values, side="right") - 1
        safe = np.clip(index, 0, len(states) - 1)
        supported = np.isfinite(values) & (index >= 0) & (values - clocks[safe] < max_age_s)
        status = states[list(MAG_HARDWARE_FIELDS)]
        complete = status.notna().all(axis=1).to_numpy()
        changed = np.zeros(len(states), dtype=bool)
        # Compare available observations per field, including across missing words.
        for field in MAG_HARDWARE_FIELDS:
            seen = np.flatnonzero(states[field].notna().to_numpy())
            for before, after in zip(seen[:-1], seen[1:]):
                if states[field].iloc[before] != states[field].iloc[after]:
                    changed[after] = True
                    bracket = selected & out["clock_s"].gt(clocks[before]) & out["clock_s"].lt(clocks[after])
                    out.loc[bracket, "mag_hardware_transition"] = True
        anchors = np.maximum.accumulate(np.where(changed, clocks, clocks[0]))
        for field in MAG_HARDWARE_FIELDS:
            out.loc[positions[supported], field] = states[field].to_numpy()[safe[supported]]
        valid = supported & complete[safe]
        out.loc[positions[supported], "mag_hardware_state_source"] = source
        out.loc[positions[valid], "mag_hardware_state_valid"] = True
        identities = []
        for row in status.itertuples(index=False, name=None):
            identities.append(hashlib.sha256(repr(tuple(int(v) for v in row)).encode()).hexdigest()[:20]
                              if all(pd.notna(v) for v in row) else None)
        identities = np.asarray(identities, dtype=object)
        out.loc[positions[valid], "mag_hardware_state_id"] = identities[safe[valid]]
        cohorts = np.asarray([f"{value}:{anchor:.9f}" if value is not None else None
                              for value, anchor in zip(identities, anchors)], dtype=object)
        out.loc[positions[valid], "mag_hardware_cohort"] = cohorts[safe[valid]]

    attach(out["mode"].eq("realtime"), subcom, "digital_subcom")
    if burst_markers is not None and not burst_markers.empty and "burst_id" in out:
        for burst_id, states in burst_markers.groupby("burst_id"):
            attach(out["mode"].eq("burst") & out["burst_id"].eq(burst_id), states, "burst_marker")
    out.index = samples.index
    out.attrs.update(samples.attrs)
    out.attrs["hardware_state_basis"] = "bounded collection-time snapshots; changes bracketed, execution times unknown"
    return out


def mag_hardware_calibration_checks(samples):
    """Summarize support by observed state, range and mode without claiming truth.

    These are applicability checks. Independent reference residuals are still
    required to validate a coefficient's accuracy in a new hardware state.
    """
    keys = ["mode", "range", "mag_hardware_state_id"]
    table = samples.copy()
    for field, default in (("mag_hardware_state_id", None), ("mag_hardware_state_valid", False),
                           ("mag_hardware_transition", False), ("calibration_valid", False),
                           ("calibration_hardware_scoped", False), ("calibration_mode_transfer", False)):
        if field not in table:
            table[field] = default
    table["calibration_hardware_scoped"] &= table["calibration_valid"]
    result = table.groupby(keys, dropna=False).agg(
        n_samples=("calibration_valid", "size"), n_calibrated=("calibration_valid", "sum"),
        n_state_supported=("mag_hardware_state_valid", "sum"),
        n_transition=("mag_hardware_transition", "sum"),
        n_hardware_scoped=("calibration_hardware_scoped", "sum"),
        n_mode_transferred=("calibration_mode_transfer", "sum")).reset_index()
    result["review_required"] = ((result["n_state_supported"] < result["n_samples"])
                                 | (result["n_calibrated"] < result["n_samples"])
                                 | result["n_transition"].gt(0)
                                 | (result["n_hardware_scoped"] < result["n_calibrated"])
                                 | result["n_mode_transferred"].gt(0))
    result.attrs["basis"] = "observed state and coefficient applicability; independent calibration accuracy not established"
    return result
