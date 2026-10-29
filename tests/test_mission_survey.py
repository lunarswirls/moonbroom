import json
import numpy as np
import pandas as pd
import pytest

from moonbroom.magnetotail import native_variability_windows
from moonbroom.mission_survey import run_mission_survey, stream_variability


def samples():
    seconds = np.arange(1080) / 9
    return pd.DataFrame(dict(utc=pd.Timestamp('1998-01-19', tz='UTC') + pd.to_timedelta(seconds, unit='s'),
                             Bx_GSE=10 + np.sin(2 * np.pi * seconds), By_GSE=0., Bz_GSE=0.,
                             calibration_valid=True, range=1, calibration_model_id='frozen'))


def write_parts(tmp_path, table):
    paths = []
    for number, part in enumerate([table.iloc[:280], table.iloc[280:810], table.iloc[810:]]):
        path = tmp_path / f'{number}.csv.gz'
        part.to_csv(path, index=False)
        paths.append(path)
    return paths


def test_survey_carries_partial_windows_between_shards(tmp_path):
    raw = samples()
    paths = write_parts(tmp_path, raw)
    windows, stats = stream_variability(paths, field_frame='GSE')
    expected = native_variability_windows(raw, field_frame='GSE')
    pd.testing.assert_frame_equal(windows, expected, check_exact=False, atol=1e-10)
    assert stats == dict(input_native_rows=1080, invalid_utc_rows=0)
    np.testing.assert_allclose(windows['sigma_native_nt'], np.sqrt(.5))
    assert windows['sigma_5s_nt'].max() < 1e-9
    with pytest.raises(ValueError, match='out of order'):
        stream_variability(paths[::-1], field_frame='GSE')


def test_survey_reports_invalid_utc_and_rejects_coefficient_transition(tmp_path):
    raw = samples()
    raw.loc[100, 'calibration_model_id'] = 'new model'
    raw.loc[800, 'utc'] = pd.NaT
    windows, stats = stream_variability(write_parts(tmp_path, raw), field_frame='GSE')
    assert stats['invalid_utc_rows'] == 1
    assert not windows['window_valid'].iloc[0]


def test_variability_window_rejects_crossing_a_recorded_mager_action():
    raw = samples()
    raw['mager_command_cohort'] = 'before'
    raw.loc[raw.index >= 270, 'mager_command_cohort'] = 'after'
    result = native_variability_windows(raw, field_frame='GSE')
    assert result['window_valid'].tolist() == [False, True]


def test_mission_survey_produces_catalogue_and_manifest(tmp_path):
    paths = write_parts(tmp_path, samples())
    centers = pd.date_range('1998-01-19T00:00:30Z', periods=2, freq='min')
    geometry = tmp_path / 'geometry.csv'
    upstream = tmp_path / 'upstream.csv'
    pd.DataFrame(dict(utc=centers, X_GSE_RE=-60., Y_GSE_RE=0., Z_GSE_RE=0.)).to_csv(geometry, index=False)
    pd.DataFrame(dict(utc=centers, pressure_npa=2., bz_gsm_nt=0., quality_valid=True)).to_csv(upstream, index=False)
    result = run_mission_survey(dict(native_paths=[str(p) for p in paths], field_frame='GSE',
                                     geometry_csv=str(geometry), upstream_csv=str(upstream),
                                     output_directory=str(tmp_path / 'survey')))
    assert result['windows']['region'].eq('tail_candidate').all()
    assert result['encounters']['duration_s'].tolist() == [120.]
    manifest = json.loads((tmp_path / 'survey/manifest.json').read_text())
    assert manifest['input_native_rows'] == 1080
    assert len(manifest['inputs']) == 5
    assert not manifest['high_frequency_validated']
