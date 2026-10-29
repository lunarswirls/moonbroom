"""Conservative LP magnetotail candidate survey with explicit boundary assumptions.

Shue et al. (1998), doi:10.1029/98JA01103 is a magnetopause model.
Far-tail extrapolation and orbit-average attitude are approximations.
"""

import numpy as np
import pandas as pd

from .geometry import loaded_kernels


def shue_parameters(pressure_npa, bz_gsm_nt):
    """Return the published 1998 r0 (Earth radii) and flaring exponent."""
    pressure, bz = np.broadcast_arrays(np.asarray(pressure_npa, float), np.asarray(bz_gsm_nt, float))
    good = np.isfinite(pressure) & (pressure > 0) & np.isfinite(bz)
    p = np.where(good, pressure, 1)
    r0 = (10.22+1.29*np.tanh(0.184*(bz+8.14)))*p**(-1/6.6)
    alpha = (0.58-0.007*bz)*(1+0.024*np.log(p))
    good &= (alpha > 0) & (alpha < 1)
    return np.where(good, r0, np.nan), np.where(good, alpha, np.nan)


def shue_tail_radius(x_gse_re, pressure_npa, bz_gsm_nt):
    """Solve the model cross section at downstream x, including the tail axis.

    Returns NaN for x >= 0 or nonphysical parameters. This geometrical
    extrapolation does not establish model validity at lunar distance.
    """
    x, p, bz = np.broadcast_arrays(np.asarray(x_gse_re, float),
                                  np.asarray(pressure_npa, float), np.asarray(bz_gsm_nt, float))
    r0, alpha = shue_parameters(p, bz)
    valid = np.isfinite(x) & (x < 0) & np.isfinite(r0)
    low, high = np.full(x.shape, np.pi/2), np.full(x.shape, np.pi-1e-7)
    for _ in range(64):
        theta = (low+high)/2
        radius = r0*(2/(1+np.cos(theta)))**alpha
        toward_tail = radius*np.cos(theta) > x
        low = np.where(toward_tail, theta, low)
        high = np.where(toward_tail, high, theta)
    theta = (low+high)/2
    rho = r0*(2/(1+np.cos(theta)))**alpha*np.sin(theta)
    return np.where(valid, rho, np.nan)


def native_variability_windows(samples, *, field_frame, window_s=60, min_coverage=0.9,
                                max_gap_s=0.25, sample_rate_hz=9):
    """Compare native vector variability with variability of five-second means.

    UTC windows are half open and no data are filled. Mixed ranges or
    coefficient cohorts invalidate a window. No PSD or wave validation is
    inferred from these descriptive statistics. Output retains all windows
    that contain source rows, including rejected windows.
    """
    if field_frame not in ('GSE', 'SCD', 'SSE', 'J2000'):
        raise ValueError("Declare an explicit supported field frame")
    if (not np.isfinite([window_s, max_gap_s, sample_rate_hz]).all()
            or window_s < 5 or max_gap_s <= 0 or sample_rate_hz <= 0
            or not 0 < min_coverage <= 1):
        raise ValueError("Invalid window, cadence, gap or coverage settings")
    raw = samples.copy()
    raw['utc'] = pd.to_datetime(raw['utc'], utc=True, format='mixed')
    if raw['utc'].isna().any() or raw['utc'].duplicated().any():
        raise ValueError("Variability requires unique valid UTC")
    raw = raw.sort_values('utc')
    raw['_window'] = np.floor(raw['utc'].astype('datetime64[ns, UTC]').astype('int64')/1e9/window_s).astype('int64')
    fields = ['B'+a+'_'+field_frame for a in 'xyz']
    rows = []
    for number, group in raw.groupby('_window', sort=True):
        start = pd.Timestamp(number*window_s, unit='s', tz='UTC')
        field = group[fields].to_numpy(float)
        mask = group['calibration_valid'].fillna(False).to_numpy(bool) & np.isfinite(field).all(axis=1)
        t = group['utc'].astype('datetime64[ns, UTC]').astype('int64').to_numpy()/1e9
        coverage = mask.sum()/(window_s*sample_rate_hz)
        gaps = np.diff(t[mask])
        mixed = any(group[col].nunique(dropna=False) != 1 for col in (
            'range', 'cohort', 'calibration_source', 'calibration_model_id', 'mager_command_cohort') if col in group)
        valid = (mask.sum() >= 3 and min_coverage <= coverage <= 1.05
                 and not mixed and (not len(gaps) or gaps.max() <= max_gap_s)
                 and mask.all())
        row = dict(utc=start+pd.Timedelta(seconds=window_s/2), start_utc=start,
                   stop_utc=start+pd.Timedelta(seconds=window_s), n_total=len(group),
                   n_valid=int(mask.sum()), coverage=coverage, window_valid=bool(valid),
                   field_frame=field_frame, sigma_native_nt=np.nan, sigma_5s_nt=np.nan,
                   mean_b_nt=np.nan, normalized_variability=np.nan, n_5s=0,
                   max_observed_gap_s=float(gaps.max()) if len(gaps) else np.nan)
        for name in ('range', 'cohort', 'calibration_source', 'calibration_model_id', 'high_frequency_validated'):
            if name in group:
                row[name] = group[name].iloc[0] if group[name].nunique(dropna=False) == 1 else None
        if valid:
            f = field[mask]
            mean = f.mean(axis=0)
            sigma = float(np.sqrt(np.mean(np.sum((f-mean)**2, axis=1))))
            row.update(sigma_native_nt=sigma, mean_b_nt=float(np.linalg.norm(mean)),
                       magnitude_std_nt=float(np.std(np.linalg.norm(f, axis=1))))
            if row['mean_b_nt'] > 0.1:
                row['normalized_variability'] = sigma/row['mean_b_nt']
            # Use only complete five-second bins contained in this window.
            bins = np.floor(t/5).astype('int64')
            means = [field[bins == b].mean(axis=0) for b in np.unique(bins)
                     if b*5 >= number*window_s-1e-6
                     and (b+1)*5 <= (number+1)*window_s+1e-6
                     and np.sum(bins == b) >= 5*sample_rate_hz*min_coverage]
            row['n_5s'] = len(means)
            if len(means) >= 2:
                means = np.asarray(means)
                row['sigma_5s_nt'] = float(np.sqrt(np.mean(np.sum((means-means.mean(axis=0))**2, axis=1))))
        rows.append(row)
    columns = ['utc', 'start_utc', 'stop_utc', 'window_valid', 'sigma_native_nt', 'sigma_5s_nt']
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def spice_survey_geometry(utc, kernels, *, earth_radius_km=6371.2):
    """Geometric Earth-centered LP positions in a constructed epoch GSE basis.

    SPICE ECLIPJ2000 defines ecliptic north. LP is target -25. This explicit
    basis avoids assuming that a built-in GSE frame exists. No aberration is
    applied to instantaneous geometric positions. Kernels are caller supplied.
    """
    import spiceypy as spice
    times = pd.to_datetime(pd.Series(utc), utc=True, format='mixed')
    if times.isna().any() or not np.isfinite(earth_radius_km) or earth_radius_km <= 0:
        raise ValueError("Supply valid times and positive Earth radius")
    out = pd.DataFrame({'utc': times})
    values = []
    with loaded_kernels(kernels):
        for t in times:
            et = spice.str2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
            sun = np.asarray(spice.spkpos('SUN', et, 'ECLIPJ2000', 'NONE', 'EARTH')[0])
            x = sun/np.linalg.norm(sun)
            y = np.cross([0., 0., 1.], x)
            y /= np.linalg.norm(y)
            basis = np.vstack([x, y, np.cross(x, y)])
            position = np.asarray(spice.spkpos('-25', et, 'ECLIPJ2000', 'NONE', 'EARTH')[0])
            moon = np.asarray(spice.spkpos('-25', et, 'ECLIPJ2000', 'NONE', 'MOON')[0])
            solar = np.asarray(spice.spkpos('SUN', et, 'ECLIPJ2000', 'NONE', 'MOON')[0])
            sza = np.degrees(np.arccos(np.clip(moon@solar/np.linalg.norm(moon)/np.linalg.norm(solar), -1, 1)))
            values.append([*(basis@position/earth_radius_km), np.linalg.norm(moon), sza])
    out[['X_GSE_RE', 'Y_GSE_RE', 'Z_GSE_RE', 'lunar_radius_km', 'lunar_sza_deg']] = np.asarray(values).reshape(-1, 5)
    out.attrs.update(kernels=list(map(str, kernels)), earth_radius_km=earth_radius_km,
                     position_correction='NONE', frame='epoch Sun with ECLIPJ2000 north')
    return out


def survey_magnetotail(windows, geometry, upstream, *, max_upstream_age_s=120,
                       boundary_margin_re=2, pressure_fraction=0.2, bz_uncertainty_nt=1,
                       max_downstream_re=100):
    """Join bounded upstream context and classify conservative geometric candidates.

    upstream requires utc, pressure_npa, bz_gsm_nt and quality_valid. Its time
    tags must already represent the desired terrestrial boundary context. No
    extra OMNI propagation is silently applied. Sensitivity envelope is an
    input perturbation test, not a calibrated statistical confidence interval.
    """
    if (not np.isfinite([max_upstream_age_s, boundary_margin_re, pressure_fraction,
                         bz_uncertainty_nt, max_downstream_re]).all()
            or max_upstream_age_s < 0 or boundary_margin_re < 0
            or not 0 <= pressure_fraction < 1 or bz_uncertainty_nt < 0 or max_downstream_re <= 0):
        raise ValueError("Invalid upstream age, boundary sensitivity or tail scope")
    left, geo, wind = windows.copy(), geometry.copy(), upstream.copy()
    if not {'utc', 'pressure_npa', 'bz_gsm_nt', 'quality_valid'} <= set(wind):
        raise ValueError('Upstream requires utc, pressure_npa, bz_gsm_nt and quality_valid')
    # Object strings such as "False" must never become truthy quality flags.
    quality = wind['quality_valid'].astype('string').str.lower()
    if not quality.dropna().isin(['true', 'false', '1', '0']).all():
        raise ValueError('Upstream quality_valid must contain boolean flags')
    wind['quality_valid'] = quality.isin(['true', '1'])
    for table in (left, geo, wind):
        table['utc'] = pd.to_datetime(table['utc'], utc=True, format='mixed')
        if table['utc'].isna().any() or table['utc'].duplicated().any():
            raise ValueError("Context tables need unique finite UTC")
    out = left.merge(geo, on='utc', how='left', validate='one_to_one').sort_values('utc')
    wind = wind.rename(columns={'utc': 'upstream_utc'})
    out = pd.merge_asof(out, wind.sort_values('upstream_utc'), left_on='utc', right_on='upstream_utc',
                        direction='backward', tolerance=pd.Timedelta(seconds=max_upstream_age_s))
    out['upstream_age_s'] = (out['utc']-out['upstream_utc']).dt.total_seconds()
    supported_interval = np.ones(len(out), dtype=bool)
    if 'upstream_interval_stop_utc' in out:
        out['upstream_interval_stop_utc'] = pd.to_datetime(out['upstream_interval_stop_utc'], utc=True, format='mixed')
        supported_interval = (out['utc'].lt(out['upstream_interval_stop_utc'])
                              & out['upstream_interval_stop_utc'].gt(out['upstream_utc'])).to_numpy()
    x = out['X_GSE_RE'].to_numpy(float)
    rho = np.hypot(out['Y_GSE_RE'], out['Z_GSE_RE']).to_numpy()
    p, bz = out['pressure_npa'].to_numpy(float), out['bz_gsm_nt'].to_numpy(float)
    out['upstream_context_valid'] = (out['quality_valid'].fillna(False).to_numpy(bool)
                                     & supported_interval & np.isfinite(p) & (p > 0) & np.isfinite(bz))
    out['shue_r0_re'], out['shue_alpha'] = shue_parameters(p, bz)
    out['tail_radius_re'] = shue_tail_radius(x, p, bz)
    radii = np.array([shue_tail_radius(x, p*(1+dp), bz+db)
                      for dp in (-pressure_fraction, 0, pressure_fraction)
                      for db in (-bz_uncertainty_nt, 0, bz_uncertainty_nt)])
    model_good = np.isfinite(radii).all(axis=0)
    lower, upper = np.min(radii, axis=0), np.max(radii, axis=0)
    out['tail_radius_min_re'], out['tail_radius_max_re'] = lower, upper
    out['boundary_margin_re'] = lower-rho
    good = (out['window_valid'].fillna(False).to_numpy(bool)
            & out['upstream_context_valid'].to_numpy(bool) & model_good
            & np.isfinite(rho) & (x >= -max_downstream_re))
    state = np.full(len(out), 'insufficient_context', dtype=object)
    state[good] = 'boundary_ambiguous'
    state[good & (rho < lower-boundary_margin_re)] = 'tail_candidate'
    state[good & (rho > upper+boundary_margin_re)] = 'modeled_external'
    # Dayside locations cannot be tail encounters even though tail formula is undefined.
    dayside = (out['window_valid'].fillna(False).to_numpy(bool)
               & np.isfinite(x) & (x >= 0) & np.isfinite(rho))
    state[dayside] = 'not_downstream'
    out['region'] = state
    out['boundary_model'] = 'Shue 1998 magnetopause, far-tail extrapolation'
    out['high_frequency_validated'] = False
    return out


def magnetotail_catalogue(windows):
    """Merge only adjacent candidate windows. Gaps and all other states split."""
    selected = windows.sort_values('start_utc').copy()
    for name in ('start_utc', 'stop_utc'):
        selected[name] = pd.to_datetime(selected[name], utc=True, format='mixed')
    rows, active = [], None
    for row in selected.itertuples(index=False):
        candidate = row.region == 'tail_candidate'
        if active is not None and (not candidate or row.start_utc != active['stop_utc']):
            rows.append(active)
            active = None
        if candidate:
            if active is None:
                active = dict(start_utc=row.start_utc, stop_utc=row.stop_utc, n_windows=1,
                              label='tail_candidate', sigma_native_sum_nt=row.sigma_native_nt)
            else:
                active['stop_utc'] = row.stop_utc
                active['n_windows'] += 1
                active['sigma_native_sum_nt'] += row.sigma_native_nt
    if active is not None:
        rows.append(active)
    out = pd.DataFrame(rows, columns=['start_utc', 'stop_utc', 'n_windows', 'label', 'sigma_native_sum_nt'])
    out['duration_s'] = (out['stop_utc']-out['start_utc']).dt.total_seconds() if len(out) else pd.Series(dtype=float)
    out['mean_window_sigma_nt'] = out.pop('sigma_native_sum_nt')/out['n_windows']
    return out


def variability_region_summary(windows):
    """Descriptive association with modeled region, retaining unsupported exposure.

    Samples within windows are dependent. No significance or native-rate
    accuracy is inferred. Missing time between observed windows is not exposure.
    """
    table = windows.copy()
    table['duration_s'] = (pd.to_datetime(table['stop_utc'], utc=True)-pd.to_datetime(table['start_utc'], utc=True)).dt.total_seconds()
    return table.groupby('region', dropna=False).agg(
        n_windows=('region', 'size'), exposure_s=('duration_s', 'sum'),
        median_sigma_native_nt=('sigma_native_nt', 'median'),
        median_sigma_5s_nt=('sigma_5s_nt', 'median')).reset_index()


def rotate_scd_to_gse(samples, attitude, kernels, *, attitude_frame, max_attitude_age_s):
    """Rotate native SCD fields with a bounded, explicitly framed spin-axis table.

    Orbit-average attitude is held backward within the supplied age limit.
    This is not a full reconstructed attitude solution. Unsupported rows
    remain with missing fields and survey calibration_valid=False.
    """
    import spiceypy as spice
    if not attitude_frame or not np.isfinite(max_attitude_age_s) or max_attitude_age_s < 0:
        raise ValueError("Declare attitude frame and finite nonnegative age limit")
    raw, att = samples.copy(), attitude.copy()
    raw['utc'] = pd.to_datetime(raw['utc'], utc=True, format='mixed')
    att['utc'] = pd.to_datetime(att['utc'], utc=True, format='mixed')
    if (raw['utc'].isna().any() or att['utc'].isna().any() or att['utc'].duplicated().any()
            or not np.isfinite(att[['longitude_deg', 'latitude_deg']]).all().all()
            or att['latitude_deg'].abs().gt(90).any()):
        raise ValueError("Require valid times, unique attitude epochs and finite valid angles")
    raw['_order'] = np.arange(len(raw))
    out = pd.merge_asof(raw.sort_values('utc'), att.rename(columns={'utc': 'rotation_attitude_utc'}).sort_values('rotation_attitude_utc'),
                        left_on='utc', right_on='rotation_attitude_utc', direction='backward',
                        tolerance=pd.Timedelta(seconds=max_attitude_age_s))
    fields = np.full((len(out), 3), np.nan)
    with loaded_kernels(kernels):
        for i, row in enumerate(out.itertuples(index=False)):
            if pd.isna(row.rotation_attitude_utc) or not getattr(row, 'attitude_quality_valid', True):
                continue
            et = spice.str2et(row.utc.strftime("%Y-%m-%dT%H:%M:%S.%f"))
            lon, lat = np.deg2rad([row.longitude_deg, row.latitude_deg])
            z = np.array([np.cos(lon)*np.cos(lat), np.sin(lon)*np.cos(lat), np.sin(lat)])
            z = spice.pxform(attitude_frame, 'ECLIPJ2000', et)@z
            sun = np.asarray(spice.spkpos('SUN', et, 'ECLIPJ2000', 'NONE', 'MOON')[0])
            sun /= np.linalg.norm(sun)
            x = sun-(sun@z)*z
            if np.linalg.norm(x) < 1e-8:
                continue
            x /= np.linalg.norm(x)
            scd = np.column_stack([x, np.cross(z, x), z])
            sun_earth = np.asarray(spice.spkpos('SUN', et, 'ECLIPJ2000', 'NONE', 'EARTH')[0])
            gx = sun_earth/np.linalg.norm(sun_earth)
            gy = np.cross([0., 0., 1.], gx)
            gy /= np.linalg.norm(gy)
            gse = np.vstack([gx, gy, np.cross(gx, gy)])
            fields[i] = gse@scd@np.array([row.Bx_SCD, row.By_SCD, row.Bz_SCD])
    out[['B'+a+'_GSE' for a in 'xyz']] = fields
    out['survey_attitude_valid'] = np.isfinite(fields).all(axis=1)
    if 'attitude_utc' not in out:
        out['attitude_utc'] = out['rotation_attitude_utc']
    out['calibration_valid'] = out['calibration_valid'].fillna(False) & out['survey_attitude_valid']
    out = out.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    out.attrs.update(attitude_frame=attitude_frame, max_attitude_age_s=max_attitude_age_s,
                     attitude_basis='bounded orbit-average spin axis')
    return out


def variability_boundary_association(windows, *, block_s=86400, n_bootstrap=500, seed=0):
    """Spearman association with input-envelope margin and day-block uncertainty.

    Positive boundary margin is inside the most restrictive model. Bootstrap
    intervals require at least three represented time blocks. Descriptive
    correlations do not independently classify a plasma region.
    """
    if not np.isfinite(block_s) or block_s <= 0 or n_bootstrap < 1:
        raise ValueError("Require positive time blocks and bootstrap count")
    good = windows['region'].isin(['tail_candidate', 'boundary_ambiguous', 'modeled_external'])
    table = windows.loc[good].copy()
    times = pd.to_datetime(table['utc'], utc=True, format='mixed').astype('datetime64[ns, UTC]')
    table['_block'] = np.floor(times.astype('int64')/1e9/block_s).astype('int64')
    rows = []

    def correlation(x, y):
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return np.nan
        return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])

    for name in ('sigma_native_nt', 'sigma_5s_nt'):
        selected = table.loc[np.isfinite(table[name]) & np.isfinite(table['boundary_margin_re'])]
        blocks = [group for _, group in selected.groupby('_block')]
        estimate = correlation(selected['boundary_margin_re'].to_numpy(), selected[name].to_numpy())
        interval = [np.nan, np.nan]
        if len(blocks) >= 3:
            rng = np.random.default_rng(seed)
            values = []
            for _ in range(n_bootstrap):
                sample = pd.concat([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
                values.append(correlation(sample['boundary_margin_re'].to_numpy(), sample[name].to_numpy()))
            finite = np.asarray(values)[np.isfinite(values)]
            if len(finite) >= .9*n_bootstrap:
                interval = np.quantile(finite, [.025, .975])
        rows.append(dict(variable=name, spearman_r=estimate, n_windows=len(selected),
                         n_blocks=len(blocks), block_s=block_s, ci_low=interval[0], ci_high=interval[1],
                         confidence_basis='time-block bootstrap' if np.isfinite(interval).all() else 'descriptive only'))
    return pd.DataFrame(rows)
