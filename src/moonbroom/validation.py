"""Byte-level MAG audits and leakage-checked archive calibration experiments."""

import numpy as np
import pandas as pd

from .archive_calibration import (
    fit_mag_archive_calibration, evaluate_mag_archive_calibration,
    fit_er_archive_response, apply_er_archive_response,
)


def audit_mag_decoding(frames, samples):
    """Independently extract scalar bit fields from retained source payloads.

    Tests exact codes, signed values, range/status and nominal collection
    clocks. This verifies extraction, not absolute UTC or analog calibration.
    Full-burst frames contain no real-time samples. Unexpected, duplicate and
    missing sample identities all fail the audit.
    """
    keys = ["source_file", "record_index", "sample_index"]
    raw = samples.loc[samples["mode"].eq("realtime")].copy()
    if raw[keys].duplicated().any():
        raise ValueError("Duplicate native sample identity")
    indexed = raw.set_index(keys)
    seen = set()
    rows = []
    for frame in frames.itertuples(index=False):
        if not frame.header_valid:
            continue
        payload = bytes(frame.mag_er)
        kind = payload[0] & 63
        if 36 <= kind <= 62 and payload[0] & 128:
            continue
        block = payload[3:85]
        if len(block) != 82:
            raise ValueError("Incomplete source MAG block")
        # Expand individual bits LSB first, independently of the vectorized decoder.
        bits = [(byte >> shift) & 1 for byte in block[1:] for shift in range(8)]
        for j in range(18):
            key = (frame.source_file, frame.record_index, j)
            seen.add(key)
            expected = [sum(bits[(j*3+a)*12+k] << k for k in range(12)) for a in range(3)]
            present = key in indexed.index
            exact = False
            clock_ok = False
            if present:
                row = indexed.loc[key]
                exact = all(row['b'+a] == q and row['b'+a+'_signed'] == q-2048
                            for a, q in zip('xyz', expected))
                exact = exact and row['range'] == (block[0] & 7)
                exact = exact and bool(row['calibration_mode']) == bool(block[0] & 8)
                exact = exact and row['mag_frame'] == block[0] >> 4
                clock_ok = np.isclose(row['clock_s'], frame.frame_clock_s+j/9+1/36-4,
                                      rtol=0, atol=1e-8)
            rows.append((*key, present, exact, clock_ok))
    for key in indexed.index:
        if key not in seen:
            rows.append((*key, True, False, False))
    out = pd.DataFrame(rows, columns=keys+["present", "codes_status_exact", "clock_exact"])
    out["audit_pass"] = out["present"] & out["codes_status_exact"] & out["clock_exact"]
    out.attrs["absolute_timing_validated"] = False
    out.attrs["high_frequency_validated"] = False
    return out


def check_validation_split(training, validation, *, min_gap_s=0):
    """Reject shared reference spectra and overlapping or insufficiently purged intervals.

    Intervals are half open. Supply a guard at least as long as any temporal
    preprocessing support. Different dates are recommended for transfer tests
    but not asserted to be statistically independent solely by this check.
    """
    if not np.isfinite(min_gap_s) or min_gap_s < 0:
        raise ValueError("min_gap_s must be finite and nonnegative")
    arrays = []
    for table in (training, validation):
        if table.empty:
            raise ValueError("Both splits need observations")
        start = pd.to_datetime(table["start_utc"], utc=True, format="mixed")
        stop = pd.to_datetime(table["stop_utc"], utc=True, format="mixed")
        if start.isna().any() or stop.isna().any() or (stop <= start).any():
            raise ValueError("Collection bounds must be complete and increasing")
        arrays.append((start, stop))
    if "reference_id" in training and "reference_id" in validation:
        if set(training["reference_id"].dropna()) & set(validation["reference_id"].dropna()):
            raise ValueError("Shared reference identities leak between splits")
    ts, te = arrays[0]
    vs, ve = arrays[1]
    # Sort training starts and use their running maximum stops to test all intervals.
    order = np.argsort(ts.astype("datetime64[ns, UTC]").astype('int64'))
    starts = ts.astype("datetime64[ns, UTC]").astype('int64').to_numpy()[order]
    ends = np.maximum.accumulate(te.astype("datetime64[ns, UTC]").astype('int64').to_numpy()[order])
    guard = int(min_gap_s*1e9)
    right = np.searchsorted(starts, ve.astype("datetime64[ns, UTC]").astype('int64').to_numpy()+guard, side='left')
    has_left = right > 0
    if (ends[np.maximum(right-1, 0)][has_left] > vs.astype("datetime64[ns, UTC]").astype('int64').to_numpy()[has_left]-guard).any():
        raise ValueError("Training and validation collection intervals overlap or violate guard")
    return pd.DataFrame({"role": ["training", "validation"],
                         "start_utc": [ts.min(), vs.min()], "stop_utc": [te.max(), ve.max()],
                         "n_rows": [len(training), len(validation)], "purge_gap_s": min_gap_s})


def validate_mag_pseudocalibration(training, validation, *, valid_from, valid_to,
                                    source, max_rmse_nt, max_bias_nt,
                                    min_validation_windows=20, min_gap_s=120,
                                    fit_xy_offsets=True, min_windows=20):
    """Fit training only and approve each range against explicit held-out limits.

    Alignment must already be frozen in supplied windows. Unsupported windows
    remain in predictions and the denominator. A validation decision does not
    validate fluctuations faster than the reference averaging interval.
    """
    if (not np.isfinite([max_rmse_nt, max_bias_nt]).all()
            or min(max_rmse_nt, max_bias_nt) < 0 or min_validation_windows < 1):
        raise ValueError("Supply finite nonnegative error limits and positive support")
    manifest = check_validation_split(training, validation, min_gap_s=min_gap_s)
    models = fit_mag_archive_calibration(training, valid_from=valid_from, valid_to=valid_to,
                                         source=source, fit_xy_offsets=fit_xy_offsets,
                                         min_windows=min_windows)
    prediction = evaluate_mag_archive_calibration(validation, models)
    metrics = []
    from .mag_state import MAG_SCOPE_FIELDS
    keys = ['range'] + [field for field in ('mode',) + MAG_SCOPE_FIELDS if field in models]
    for key, group in prediction.groupby(keys, dropna=False):
        supported = group['empirical_calibration_supported'].fillna(False)
        error = group.loc[supported, ['error_b'+a for a in 'xyz']].to_numpy(float)
        error = error[np.isfinite(error).all(axis=1)]
        row = dict(zip(keys, key))
        row.update(n_total=len(group), n_validation=len(error),
                   supported_fraction=len(error)/len(group), validation_rmse_nt=np.nan,
                   max_abs_bias_nt=np.nan, validation_pass=False)
        if len(error):
            row.update(validation_rmse_nt=float(np.sqrt(np.mean(error**2))),
                       max_abs_bias_nt=float(np.max(np.abs(error.mean(axis=0)))),
                       median_vector_error_nt=float(np.median(np.linalg.norm(error, axis=1))))
            for i, a in enumerate('xyz'):
                row['bias_'+a+'_nt'] = float(error[:, i].mean())
                row['rmse_'+a+'_nt'] = float(np.sqrt(np.mean(error[:, i]**2)))
            row['validation_pass'] = (len(error) >= min_validation_windows
                                      and row['validation_rmse_nt'] <= max_rmse_nt
                                      and row['max_abs_bias_nt'] <= max_bias_nt)
        metrics.append(row)
    metrics = pd.DataFrame(metrics)
    models = models.merge(metrics, on=keys, how='left', validate='one_to_one')
    models['validation_pass'] = models['validation_pass'].fillna(False).astype(bool)
    models['validation_start'] = manifest.iloc[1]['start_utc']
    models['validation_stop'] = manifest.iloc[1]['stop_utc']
    models['max_allowed_rmse_nt'] = max_rmse_nt
    models['max_allowed_bias_nt'] = max_bias_nt
    return dict(coefficients=models, predictions=prediction, metrics=metrics, split=manifest)


def validate_er_pseudocalibration(training, validation, *, valid_from, valid_to,
                                   source, max_median_relative_error,
                                   min_validation_pairs=4, min_gap_s=120):
    """Validate response by mode, energy, angle and version on withheld spectra."""
    if (not np.isfinite(max_median_relative_error) or max_median_relative_error < 0
            or min_validation_pairs < 1):
        raise ValueError("Supply finite nonnegative error limit and positive support")
    manifest = check_validation_split(training, validation, min_gap_s=min_gap_s)
    models = fit_er_archive_response(training, valid_from=valid_from, valid_to=valid_to, source=source)
    prediction = apply_er_archive_response(validation, models)
    keys = ['mode', 'kind', 'energy_step', 'angle_bin']
    if 'telemetry_version' in models:
        keys.append('telemetry_version')
    rows = []
    for key, group in prediction.groupby(keys):
        good = (group['match_valid'].fillna(False) & group['calibration_valid'].fillna(False)
                & group['reference_flux'].gt(0) & np.isfinite(group['flux_estimate']))
        residual = (group.loc[good, 'flux_estimate']/group.loc[good, 'reference_flux']-1).abs()
        row = dict(zip(keys, key))
        row.update(n_total=len(group), n_validation=len(residual),
                   median_relative_error=float(residual.median()),
                   validation_pass=len(residual) >= min_validation_pairs
                   and residual.median() <= max_median_relative_error)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    models = models.merge(metrics, on=keys, how='left', validate='one_to_one')
    models['validation_pass'] = models['validation_pass'].fillna(False).astype(bool)
    return dict(coefficients=models, predictions=prediction, metrics=metrics, split=manifest)


def compare_native_mag_reference(samples, reference_windows, *, min_coverage=0.9):
    """Aggregate actual pseudo-calibrated native rows at reference clock bounds.

    This independently checks native application rather than predicting via
    averaged design features. No reference field is interpolated to 9 Hz.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError('Coverage must lie in (0, 1]')
    raw = samples.loc[samples['mode'].eq('realtime')].sort_values('clock_s')
    clock = raw['clock_s'].to_numpy(float)
    if not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
        raise ValueError('Native clocks must be finite, unique and increasing')
    out = reference_windows.copy()
    duration = out['stop_clock_s']-out['start_clock_s']
    if not np.isfinite(duration).all() or duration.le(0).any():
        raise ValueError('Reference collection bounds must increase')
    field = raw[['B'+a+'_SCD' for a in 'xyz']].to_numpy(float)
    usable = raw['calibration_valid'].fillna(False).to_numpy(bool) & np.isfinite(field).all(axis=1)
    sums = np.vstack([np.zeros(3), np.cumsum(np.where(usable[:,None], field, 0), axis=0)])
    bad = np.r_[0, np.cumsum(~usable)]
    left = np.searchsorted(clock, out['start_clock_s'])
    right = np.searchsorted(clock, out['stop_clock_s'])
    count = right-left
    coverage = count/(duration.to_numpy()*9)
    good = (count >= 3) & (coverage >= min_coverage) & (coverage <= 1.05) & (bad[right] == bad[left])
    # No aggregation across an internal telemetry gap or a range change.
    for i, (lo, hi) in enumerate(zip(left, right)):
        if hi-lo > 1 and np.diff(clock[lo:hi]).max() > 1.5/9:
            good[i] = False
        if 'range' in raw and raw.iloc[lo:hi]['range'].nunique() > 1:
            good[i] = False
    predicted = (sums[right]-sums[left])/np.maximum(count[:,None], 1)
    good &= np.isfinite(out[['reference_b'+a for a in 'xyz']]).all(axis=1).to_numpy()
    predicted[~good] = np.nan
    out['comparison_valid'], out['n_samples'], out['coverage'] = good, count, coverage
    for i, a in enumerate('xyz'):
        out['native_mean_b'+a] = predicted[:,i]
        out['error_b'+a] = predicted[:,i]-out['reference_b'+a].to_numpy(float)
    out['high_frequency_validated'] = False
    return out
