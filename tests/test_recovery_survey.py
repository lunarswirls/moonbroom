from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moonbroom import (decode_level0, audit_mag_decoding, check_validation_split,
                       validate_mag_pseudocalibration, validate_er_pseudocalibration,
                       shue_parameters, shue_tail_radius, native_variability_windows,
                       survey_magnetotail, magnetotail_catalogue, variability_region_summary)
from moonbroom.archive_calibration import _mag_design

FIXTURES = Path(__file__).parent/'fixtures'


def test_native_audit_detects_corrupted_and_missing_samples():
    products = decode_level0(FIXTURES/'M9906003_excerpt.LBL')
    audit = audit_mag_decoding(products['frames'], products['mag'])
    assert len(audit) > 2000
    assert audit['audit_pass'].all()
    fast = decode_level0(FIXTURES/'M9906003_excerpt.LBL', mag_only=True)
    assert audit_mag_decoding(fast['frames'], fast['mag'])['audit_pass'].all()
    changed = products['mag'].copy()
    changed.loc[0, 'bx'] += 1
    assert not audit_mag_decoding(products['frames'], changed)['audit_pass'].all()
    assert not audit_mag_decoding(products['frames'], changed.iloc[1:])['audit_pass'].all()


def test_split_checks_all_intervals_and_shared_reference_spectra():
    def table(a, b, ref):
        return pd.DataFrame(dict(start_utc=pd.to_datetime(a, unit='s', utc=True),
                                 stop_utc=pd.to_datetime(b, unit='s', utc=True), reference_id=ref))
    train = table([0, 10], [3, 12], ['a', 'b'])
    test = table([5], [7], ['c'])
    check_validation_split(train, test, min_gap_s=2)
    with pytest.raises(ValueError, match='guard'):
        check_validation_split(train, test, min_gap_s=3)
    with pytest.raises(ValueError, match='Shared'):
        check_validation_split(train, test.assign(reference_id='a'))
    with pytest.raises(ValueError, match='overlap'):
        check_validation_split(train, table([11], [13], ['x']))


def test_blocked_mag_fit_measures_heldout_bias_and_rejects_bad_transfer():
    rng = np.random.default_rng(31)
    def windows(day):
        w = pd.DataFrame(rng.normal(size=(40, 5)), columns=['raw_u','raw_v','raw_z','phase_cos','phase_sin'])
        w['start_utc'] = pd.Timestamp(day, tz='UTC')+pd.to_timedelta(np.arange(40)*5, unit='s')
        w['stop_utc'] = w['start_utc']+pd.Timedelta(seconds=5)
        w['reference_id'] = day+'_'+w.index.astype(str)
        w['range'], w['window_valid'], w['window_s'] = 1, True, 5.
        w[['reference_b'+a for a in 'xyz']] = (_mag_design(w)@np.array([.2,.01,.3,1.,2.,3.])).reshape(-1,3)
        return w
    train, test = windows('1998-01-17'), windows('1998-01-19')
    args = dict(valid_from='1998-01-16', valid_to='1998-01-21', source='synthetic', max_rmse_nt=.01, max_bias_nt=.01)
    out = validate_mag_pseudocalibration(train, test, **args)
    assert out['coefficients']['validation_pass'].all()
    test['reference_bx'] += 1
    out = validate_mag_pseudocalibration(train, test, **args)
    assert not out['coefficients']['validation_pass'].any()
    assert out['metrics']['bias_x_nt'].iloc[0] == pytest.approx(-1)


@pytest.mark.parametrize('cohort', ['1998_01', '1998_01_format2', '1998_06', '1999_03', '1999_03_sweep34'])
def test_er_withheld_pds_dates(cohort):
    base = FIXTURES/'mission_fit'
    train = pd.read_csv(base/(cohort+'_training.csv'))
    test = pd.read_csv(base/(cohort+'_validation.csv'))
    out = validate_er_pseudocalibration(train, test, valid_from=train['start_utc'].min()[:10],
          valid_to=(pd.to_datetime(test['stop_utc'], utc=True).max()+pd.Timedelta(days=1)).isoformat(),
          source='PDS regression fixture', max_median_relative_error=.05)
    assert out['coefficients']['validation_pass'].all()


def test_shue_known_parameters_axis_and_invalid_pressure():
    r0, alpha = shue_parameters(2, 0)
    assert r0 == pytest.approx((10.22+1.29*np.tanh(.184*8.14))*2**(-1/6.6))
    theta = 2.5
    r = r0*(2/(1+np.cos(theta)))**alpha
    rho = shue_tail_radius(r*np.cos(theta), 2, 0)
    assert rho == pytest.approx(r*np.sin(theta))
    assert np.isnan(shue_tail_radius(-60, -1, 0))
    assert np.isnan(shue_tail_radius(1, 2, 0))
    assert np.isfinite(shue_tail_radius(-60, 2, 0))


def native_samples():
    time = np.arange(1080)/9
    return pd.DataFrame(dict(utc=pd.Timestamp('1998-01-19', tz='UTC')+pd.to_timedelta(time, unit='s'),
                             Bx_GSE=10+np.sin(2*np.pi*time), By_GSE=0., Bz_GSE=0.,
                             calibration_valid=True, range=1))


def test_variability_retains_native_signal_and_rejects_gap_range_and_bad_rows():
    raw = native_samples()
    w = native_variability_windows(raw, field_frame='GSE')
    assert w['window_valid'].all()
    np.testing.assert_allclose(w['sigma_native_nt'], np.sqrt(.5), atol=1e-8)
    assert w['sigma_5s_nt'].max() < 1e-8
    bad = raw.drop(index=range(100, 105))
    assert not native_variability_windows(bad, field_frame='GSE')['window_valid'].iloc[0]
    raw.loc[100, 'range'] = 2
    assert not native_variability_windows(raw, field_frame='GSE')['window_valid'].iloc[0]
    raw.loc[100, 'range'] = 1
    raw.loc[100, 'calibration_valid'] = False
    assert not native_variability_windows(raw, field_frame='GSE')['window_valid'].iloc[0]


def test_tail_axis_stale_upstream_boundary_and_catalogue_gaps():
    w = native_variability_windows(native_samples(), field_frame='GSE')
    geo = w[['utc']].assign(X_GSE_RE=-60., Y_GSE_RE=0., Z_GSE_RE=0.)
    wind = w[['utc']].assign(pressure_npa=2., bz_gsm_nt=0., quality_valid=True)
    result = survey_magnetotail(w, geo, wind)
    assert result['region'].eq('tail_candidate').all()
    assert magnetotail_catalogue(result)['duration_s'].tolist() == [120.]
    stale = survey_magnetotail(w, geo, wind.iloc[:1], max_upstream_age_s=10)
    assert stale['region'].tolist() == ['tail_candidate', 'insufficient_context']
    geo['Y_GSE_RE'] = shue_tail_radius(-60, 2, 0)
    assert survey_magnetotail(w, geo, wind)['region'].eq('boundary_ambiguous').all()
    geo['Y_GSE_RE'] = 100.
    assert survey_magnetotail(w, geo, wind)['region'].eq('modeled_external').all()
    shifted = result.copy()
    for col in ('start_utc', 'stop_utc'):
        shifted.loc[1, col] += pd.Timedelta(minutes=1)
    assert len(magnetotail_catalogue(shifted)) == 2
    assert variability_region_summary(result)['exposure_s'].sum() == 120
    assert magnetotail_catalogue(result.iloc[:0]).empty


def test_spice_geometry_and_scd_rotation_use_lp_position_and_explicit_basis(tmp_path):
    import spiceypy as spice
    from moonbroom import spice_survey_geometry, rotate_scd_to_gse, loaded_kernels
    lsk = FIXTURES/'naif0012.tls'
    with loaded_kernels([lsk]):
        begin, end = [spice.str2et(t) for t in ('1998-01-19', '1998-01-20')]
    spk = tmp_path/'survey.bsp'
    handle = spice.spkopn(str(spk), 'synthetic survey', 0)
    for target, position in [(399, [0,0,0]), (301, [-60*6371.2,0,0]),
                             (-25, [-60*6371.2+1800,0,0]), (10, [1.5e8,0,0])]:
        states = np.zeros((2,6))
        states[:, :3] = position
        spice.spkw09(handle, target, 0, 'ECLIPJ2000', begin, end, 'survey test', 1, 2,
                     states, np.array([begin,end]))
    spice.spkcls(handle)
    kernels = [lsk, spk]
    utc = ['1998-01-19T01:00:00Z', '1998-01-19T02:00:00Z']
    geo = spice_survey_geometry(utc, kernels)
    np.testing.assert_allclose(geo['X_GSE_RE'], -60+1800/6371.2)
    np.testing.assert_allclose(geo['lunar_radius_km'], 1800)
    raw = pd.DataFrame(dict(utc=utc, Bx_SCD=3., By_SCD=4., Bz_SCD=2., calibration_valid=True))
    attitude = pd.DataFrame(dict(utc=utc[:1], longitude_deg=0., latitude_deg=90.))
    rotated = rotate_scd_to_gse(raw, attitude, kernels, attitude_frame='ECLIPJ2000', max_attitude_age_s=60)
    np.testing.assert_allclose(rotated.loc[0, ['Bx_GSE','By_GSE','Bz_GSE']].astype(float), [3,4,2])
    assert not rotated['survey_attitude_valid'].iloc[1]
    assert not rotated['calibration_valid'].iloc[1]
    flagged = raw.copy()
    flagged['attitude_quality_valid'] = False
    flagged['attitude_utc'] = pd.to_datetime(flagged['utc'], utc=True)
    rejected = rotate_scd_to_gse(flagged, attitude, kernels,
                                  attitude_frame='ECLIPJ2000', max_attitude_age_s=60)
    assert not rejected['survey_attitude_valid'].any()
    assert rejected[['Bx_GSE', 'By_GSE', 'Bz_GSE']].isna().all().all()


def test_decode_cli_hashes_raw_and_writes_reproducible_outputs(tmp_path):
    from moonbroom.analysis_cli import run_analysis
    cfg = dict(task='decode', raw_paths=[str(FIXTURES/'M9906003_excerpt.LBL')],
               output_directory=str(tmp_path/'run'))
    result = run_analysis(cfg)
    assert result['audit']['audit_pass'].all()
    import json
    manifest = json.loads((tmp_path/'run'/'manifest.json').read_text())
    assert len(manifest['inputs']) == 2
    assert all(len(p['sha256']) == 64 for p in manifest['inputs'])
    with pytest.raises(ValueError, match='new output'):
        run_analysis(cfg)


def test_association_blocks_keep_single_day_descriptive_and_bootstrap_repeatable():
    from moonbroom import variability_boundary_association
    time = pd.date_range('1998-01-19', periods=12, freq='12h', tz='UTC')
    table = pd.DataFrame(dict(utc=time, region='tail_candidate', boundary_margin_re=np.arange(12),
                              sigma_native_nt=np.arange(12)*2., sigma_5s_nt=np.arange(12)*.5))
    out = variability_boundary_association(table, n_bootstrap=30)
    np.testing.assert_allclose(out['spearman_r'], 1)
    assert out['n_blocks'].eq(6).all()
    assert out['ci_low'].notna().all()
    pd.testing.assert_frame_equal(out, variability_boundary_association(table, n_bootstrap=30))
    assert variability_boundary_association(table.iloc[:2])['ci_low'].isna().all()


def test_native_application_reaggregation_against_separate_date_five_second_reference():
    from moonbroom import (fit_mag_spin_offsets, calibrate_mag_9hz, compare_native_mag_reference)
    base = FIXTURES/'mission_fit'
    raw = pd.read_csv(base/'mag_native_validation.csv')
    models = pd.read_csv(base/'mag_native_model.csv')
    reference = pd.read_csv(base/'mag_native_validation_windows.csv')
    native = calibrate_mag_9hz(raw, models, offsets=fit_mag_spin_offsets(raw))
    comparison = compare_native_mag_reference(native, reference)
    assert len(native) == len(raw) == 540
    pd.testing.assert_series_equal(native['clock_s'], raw['clock_s'])
    pd.testing.assert_series_equal(native['bx_signed'], raw['bx_signed'])
    assert comparison['comparison_valid'].sum() == 11
    error = comparison.loc[comparison['comparison_valid'], ['error_b'+a for a in 'xyz']].to_numpy()
    assert np.sqrt(np.mean(error**2)) < .04
    native.loc[100, 'calibration_valid'] = False
    assert compare_native_mag_reference(native, reference)['comparison_valid'].sum() == 10


def test_survey_cli_writes_catalogue_summary_and_association(tmp_path):
    from moonbroom.analysis_cli import run_analysis
    raw = native_samples()
    raw.to_csv(tmp_path/'native.csv', index=False)
    w = native_variability_windows(raw, field_frame='GSE')
    w[['utc']].assign(X_GSE_RE=-60., Y_GSE_RE=0., Z_GSE_RE=0.).to_csv(tmp_path/'geo.csv', index=False)
    w[['utc']].assign(pressure_npa=2., bz_gsm_nt=0., quality_valid=True).to_csv(tmp_path/'upstream.csv', index=False)
    out = run_analysis(dict(task='survey', output_directory=str(tmp_path/'result'),
                            samples_csv=str(tmp_path/'native.csv'), geometry_csv=str(tmp_path/'geo.csv'),
                            upstream_csv=str(tmp_path/'upstream.csv'), field_frame='GSE'))
    assert len(out['encounters']) == 1
    assert (tmp_path/'result'/'association.csv').exists()
