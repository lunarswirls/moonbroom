"""OMNI format, timing, provenance and Shue integration regression checks."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from moonbroom.magnetotail import survey_magnetotail, shue_parameters
from moonbroom.omni import read_omni_hro, prepare_omni_upstream, main
from moonbroom.mission_survey import run_mission_survey


def hro_record(*, year=1998, day=19, hour=0, minute=0, pressure=2., bz=-5., **changes):
    """Independent serialization of the published SPDF 46-word Fortran format."""
    values = [year, day, hour, minute, 71, 51, 4, 1, 0, 2400, 60, 1., 60,
              6., 1., 2., 25., 3., bz, .2, .3, 400., -399., 20., -10., 5.,
              100000., pressure, 2., .5, 8., 200., 1., 2., 14., 0., 0.,
              20, -10, 10, 0, -5, 1, 2, 1., 6.]
    formats = (['4d']*2 + ['3d']*4 + ['4d']*3 + ['7d']*2 + ['6.2f', '7d']
               + ['8.2f']*8 + ['8.1f']*4 + ['7.2f', '9.0f', '6.2f']
               + ['7.2f']*2 + ['6.1f'] + ['8.2f']*6 + ['6d']*7 + ['7.2f', '5.1f'])
    for key, value in changes.items():
        values[int(key.removeprefix('word'))-1] = value
    return ''.join(format(value, fmt) for value, fmt in zip(values, formats)) + '\n'


def config(tmp_path, **changes):
    cfg = dict(start_utc='1998-01-19T00:00:00Z', stop_utc='1998-01-19T00:04:00Z',
               cache_directory=str(tmp_path / 'cache'), output_csv=str(tmp_path / 'upstream.csv'),
               retries=1)
    cfg.update(changes)
    return cfg


class Response(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {'Content-Length': str(len(payload) if length is None else length),
                        'Last-Modified': 'Mon, 13 Feb 2023 12:00:00 GMT'}


def serve(monkeypatch, payload):
    calls = []

    def request(req, **kwargs):
        calls.append(req.full_url)
        return Response(payload.encode('ascii'))

    monkeypatch.setattr('moonbroom.omni.urlopen', request)
    return calls


def test_published_flow_pressure_and_gsm_column_are_used_without_rescaling(tmp_path):
    path = tmp_path / 'omni.asc'
    path.write_text(hro_record(pressure=3.45))
    row = read_omni_hro(path).iloc[0]
    assert row.utc == pd.Timestamp('1998-01-19T00:00:00Z')
    assert row.pressure_npa == 3.45  # Density/speed imply 1.60: never substitute.
    assert row.bz_gsm_nt == -5 and row.bz_gse_nt == 25
    assert row.quality_valid and row.quality_reason == 'valid'
    assert row.omni_timeshift_s == 2400  # Diagnostic only, not added to UTC.
    assert row.omni_plasma_spacecraft_id == 51


def test_fills_and_nonphysical_pressure_remain_invalid_minutes(tmp_path):
    path = tmp_path / 'omni.asc'
    path.write_text(hro_record(pressure=99.99) + hro_record(minute=1, bz=9999.99)
                    + hro_record(minute=2, pressure=0.) + hro_record(minute=3, pressure=-1.)
                    + hro_record(minute=4, word5=99, word8=999)
                    + hro_record(minute=5, pressure=140.20, word13=-10))
    table = read_omni_hro(path)
    assert table.quality_valid.tolist() == [False]*5 + [True]
    assert pd.isna(table.pressure_npa.iloc[0])
    assert pd.isna(table.bz_gsm_nt.iloc[1])
    assert 'invalid_imf_source' in table.quality_reason.iloc[4]
    assert 'invalid_plasma_points' in table.quality_reason.iloc[4]
    assert table.omni_out_of_sequence.iloc[5]


@pytest.mark.parametrize('payload', [hro_record()[:-4], '<html>not an archive</html>\n',
                                      hro_record(day=366), hro_record(hour=24),
                                      hro_record(minute=1) + hro_record(), hro_record()*2])
def test_bad_structure_and_invalid_or_duplicate_epochs_fail(tmp_path, payload):
    path = tmp_path / 'bad.asc'
    path.write_text(payload)
    with pytest.raises(ValueError):
        read_omni_hro(path)


def test_leap_day_and_year_boundary(tmp_path):
    path = tmp_path / 'omni.asc'
    path.write_text(hro_record(year=2000, day=366, hour=23, minute=59)
                    + hro_record(year=2001, day=1))
    table = read_omni_hro(path)
    assert table.utc.tolist() == list(pd.to_datetime(['2000-12-31T23:59Z', '2001-01-01T00:00Z']))


def test_preparation_retains_gaps_and_pins_sources_for_offline_reuse(tmp_path, monkeypatch):
    calls = serve(monkeypatch, hro_record() + hro_record(minute=1, pressure=99.99) + hro_record(minute=3))
    manifest = prepare_omni_upstream(config(tmp_path))
    assert calls == ['https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/omni_min199801.asc']
    table = pd.read_csv(tmp_path / 'upstream.csv')
    assert table.quality_valid.tolist() == [True, False, False, True]
    assert table.quality_reason.iloc[2] == 'missing_omni_minute'
    assert manifest['coverage'] == dict(expected_minutes=4, valid_minutes=2, missing_source_minutes=1, invalid_minutes=2)
    assert manifest['imf_frame'] == 'GSM (HRO word 19)'
    assert '2e-6' in manifest['pressure_convention']
    assert manifest['sources'][0]['origin'] == 'NASA_SPDF_HTTPS'
    monkeypatch.setattr('moonbroom.omni.urlopen', lambda *a, **k: pytest.fail('Unexpected network'))
    assert prepare_omni_upstream(config(tmp_path, download=False)) == manifest


@pytest.mark.parametrize('name', ['upstream.csv', 'cache/omni_min199801.asc', 'cache/omni_min199801.asc.json'])
def test_changed_outputs_or_cached_inputs_are_not_silently_reused(tmp_path, monkeypatch, name):
    serve(monkeypatch, hro_record())
    prepare_omni_upstream(config(tmp_path))
    with (tmp_path / name).open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='changed'):
        prepare_omni_upstream(config(tmp_path))


def test_truncated_transfer_never_promotes_source_or_output(tmp_path, monkeypatch):
    data = hro_record().encode('ascii')
    monkeypatch.setattr('moonbroom.omni.urlopen', lambda *a, **k: Response(data, length=len(data)+1))
    with pytest.raises(OSError, match='Truncated'):
        prepare_omni_upstream(config(tmp_path))
    assert not (tmp_path / 'cache/omni_min199801.asc').exists()
    assert not (tmp_path / 'upstream.csv').exists()


def test_cache_only_mode_and_config_relative_paths(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match='download disabled'):
        prepare_omni_upstream(config(tmp_path, download=False))
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'omni_min199801.asc').write_text(hro_record())
    cfg = config(tmp_path, cache_directory='cache', output_csv='upstream.csv', download=False)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path.parent)
    main([str(path)])
    manifest = json.loads((tmp_path / 'upstream.manifest.json').read_text())
    assert manifest['sources'][0]['origin'] == 'existing_local_file'


def test_preparation_crosses_month_and_year_without_including_exclusive_stop(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'omni_min199812.asc').write_text(hro_record(day=365, hour=23, minute=59))
    (cache / 'omni_min199901.asc').write_text(hro_record(year=1999, day=1) + hro_record(year=1999, day=1, minute=1))
    manifest = prepare_omni_upstream(config(tmp_path, start_utc='1998-12-31T23:59:00Z',
                    stop_utc='1999-01-01T00:01:00Z', download=False))
    assert manifest['coverage']['expected_minutes'] == manifest['coverage']['valid_minutes'] == 2
    table = pd.read_csv(tmp_path / 'upstream.csv')
    assert pd.to_datetime(table.utc, utc=True).iloc[-1] == pd.Timestamp('1999-01-01T00:00:00Z')
    assert len(manifest['sources']) == 2


def test_missing_cached_source_can_be_restored_only_with_the_pinned_bytes(tmp_path, monkeypatch):
    serve(monkeypatch, hro_record())
    original = prepare_omni_upstream(config(tmp_path))
    (tmp_path / 'cache/omni_min199801.asc').unlink()
    restored = prepare_omni_upstream(config(tmp_path, output_csv=str(tmp_path / 'restored.csv')))
    assert restored['sources'] == original['sources']
    (tmp_path / 'cache/omni_min199801.asc').unlink()
    serve(monkeypatch, hro_record(pressure=8.))
    with pytest.raises(OSError, match='pinned SHA-256'):
        prepare_omni_upstream(config(tmp_path, output_csv=str(tmp_path / 'changed.csv')))
    assert not (tmp_path / 'cache/omni_min199801.asc').exists()


def test_shue_receives_omni_pressure_and_does_not_bridge_invalid_minutes(tmp_path, monkeypatch):
    serve(monkeypatch, hro_record(pressure=3.45) + hro_record(minute=1, pressure=99.99)
          + hro_record(minute=3, pressure=20.))
    cfg = config(tmp_path)
    omni_config = tmp_path / 'omni.json'
    omni_config.write_text(json.dumps(cfg))
    seconds = np.arange(4*60*9)/9
    samples = pd.DataFrame(dict(utc=pd.Timestamp(cfg['start_utc']) + pd.to_timedelta(seconds, unit='s'),
                                Bx_GSE=10 + np.sin(2*np.pi*seconds), By_GSE=0., Bz_GSE=0.,
                                calibration_valid=True, range=1))
    samples.to_csv(tmp_path / 'native.csv', index=False)
    centers = pd.date_range('1998-01-19T00:00:30Z', periods=4, freq='min')
    pd.DataFrame(dict(utc=centers, X_GSE_RE=-60., Y_GSE_RE=0., Z_GSE_RE=0.)).to_csv(tmp_path / 'geo.csv', index=False)
    result = run_mission_survey(dict(native_paths=[str(tmp_path / 'native.csv')], field_frame='GSE',
                  geometry_csv=str(tmp_path / 'geo.csv'), upstream_csv=cfg['output_csv'],
                  omni_config=str(omni_config), output_directory=str(tmp_path / 'survey')))
    table = result['windows']
    assert table.region.tolist() == ['tail_candidate', 'insufficient_context', 'insufficient_context', 'tail_candidate']
    np.testing.assert_allclose(table.shue_r0_re.iloc[[0, 3]], shue_parameters([3.45, 20.], -5.)[0])
    assert table.upstream_age_s.eq(30).all()
    assert result['encounters'].duration_s.tolist() == [60., 60.]
    provenance = json.loads((tmp_path / 'survey/manifest.json').read_text())['upstream_provenance']
    assert provenance['settings']['product'] == 'OMNI_HRO_1MIN'


def test_shue_rejects_expired_omni_intervals_and_string_false_flags():
    times = pd.to_datetime(['1998-01-19T00:00:30Z', '1998-01-19T00:01:30Z'])
    windows = pd.DataFrame(dict(utc=times, window_valid=True))
    geometry = windows[['utc']].assign(X_GSE_RE=-60., Y_GSE_RE=0., Z_GSE_RE=0.)
    upstream = pd.DataFrame(dict(utc=['1998-01-19T00:00:00Z'], pressure_npa=2., bz_gsm_nt=0.,
                                 quality_valid=True, upstream_interval_stop_utc=['1998-01-19T00:01:00Z']))
    result = survey_magnetotail(windows, geometry, upstream)
    assert result.region.tolist() == ['tail_candidate', 'insufficient_context']
    upstream['quality_valid'] = 'False'
    assert survey_magnetotail(windows, geometry, upstream).region.eq('insufficient_context').all()
    upstream['quality_valid'] = 'not a flag'
    with pytest.raises(ValueError, match='boolean'):
        survey_magnetotail(windows, geometry, upstream)
