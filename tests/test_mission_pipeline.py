import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moonbroom import decode_level0, read_merged_telemetry
from moonbroom.mission_archive import (PDS_ROOT, _download_one, parse_volume_checksums,
                                       discover_mission, validate_plan, file_hash)
from moonbroom.mission_pipeline import (preflight, process_mission, process_shard,
                                        _assign_clock_cycles, _shard_inputs, _mission_offsets)


FIXTURES = Path(__file__).parent / 'fixtures'


def test_record_slicing_keeps_original_identity_and_validates_whole_label():
    path = FIXTURES / 'M9906003_excerpt.LBL'
    full = read_merged_telemetry(path)
    selected = read_merged_telemetry(path, record_slice=(20, 55))
    pd.testing.assert_frame_equal(selected, full.iloc[20:55].reset_index(drop=True))
    assert selected['byte_offset'].iloc[0] == 20 * 472
    with pytest.raises(ValueError, match='record_slice'):
        read_merged_telemetry(path, record_slice=(0, 129))


def test_mission_clock_epoch_survives_rollover_inside_source_and_new_shards(tmp_path):
    data = (FIXTURES / 'M9906003_excerpt.B').read_bytes()
    paths = []
    for number, counters in enumerate([[2**24-2, 2**24-1, 0, 1], [1, 2, 3, 4]]):
        path = tmp_path / f'M99060{number:02d}.B'
        records = []
        for index, counter in enumerate(counters):
            record = bytearray(data[index*472:(index+1)*472])
            record[6:9] = counter.to_bytes(3, 'big')
            records.append(record)
        path.write_bytes(b''.join(records))
        paths.append(dict(path=str(path), label=str(path), records=4))
    _assign_clock_cycles(paths)
    assert [row['clock_cycle'] for row in paths] == [0, 1]
    assert _shard_inputs(paths, 1, 1, 4, 1)[2] == 1
    initial = decode_level0(paths[0]['path'], mag_only=True)
    later = decode_level0(paths[1]['path'], mag_only=True, clock_cycle=1)
    assert initial['frames']['vcdu_unwrapped'].tolist() == [2**24-2, 2**24-1, 2**24, 2**24+1]
    assert later['frames']['vcdu_unwrapped'].tolist() == [2**24+1, 2**24+2, 2**24+3, 2**24+4]


def test_global_counter_epoch_preserves_existing_bank_offset_window_grid():
    from moonbroom.native_mag import fit_mag_spin_offsets
    raw = pd.read_csv(FIXTURES / 'mission_fit/mag_native_validation.csv')
    expected = fit_mag_spin_offsets(raw)
    shifted = raw.copy()
    shifted['clock_s'] += 2**25
    actual = _mission_offsets(shifted)
    np.testing.assert_allclose(actual[['zero_x', 'zero_y']], expected[['zero_x', 'zero_y']], atol=1e-6)
    np.testing.assert_allclose(actual['clock_center_s'] - 2**25, expected['clock_center_s'])
    np.testing.assert_allclose(actual['offset_window'], expected['offset_window'])


def checksums(entries):
    return '\n'.join(f'{hashlib.md5(data).hexdigest()}  {name}' for name, data in entries.items())


def minimal_files():
    return {'MERGED/D06X/M9906003.B': b'raw', 'MERGED/D06X/M9906003.LBL': b'label',
            'SUNPULSE/D06X/S9906003.B': b'pulse', 'SUNPULSE/D06X/S9906003.LBL': b'label'}


def test_md5_catalog_supports_volume_prefix_and_rejects_unpaired_or_unsafe_paths():
    files = minimal_files()
    text = checksums({'./LP_0015/' + name: data for name, data in files.items()})
    rows = parse_volume_checksums(text, 'lp_0015')
    assert len(rows) == 4 and rows[0]['volume_path'] == 'MERGED/D06X/M9906003.B'
    with pytest.raises(ValueError, match='unpaired'):
        parse_volume_checksums(checksums(dict(list(files.items())[1:])), 'lp_0015')
    with pytest.raises(ValueError, match='Unsafe'):
        parse_volume_checksums(text + '\n' + '0' * 32 + '  ../outside', 'lp_0015')
    with pytest.raises(ValueError, match='Duplicate'):
        parse_volume_checksums(text + '\n' + text.splitlines()[0].lower(), 'lp_0015')


def test_full_inventory_uses_all_19_official_catalogs_and_resolves_case(tmp_path, monkeypatch):
    from moonbroom.mission_archive import VOLUMES
    listing = {PDS_ROOT: []}
    text = checksums(minimal_files()).encode()
    for volume in VOLUMES:
        listing[PDS_ROOT].extend([PDS_ROOT + volume.upper() + '/', PDS_ROOT + volume + '_121210.md5'])
        root = PDS_ROOT + volume.upper() + '/'
        listing[root] = [root + 'MERGED/', root + 'SUNPULSE/']
        for folder, tag in [('MERGED', 'M'), ('SUNPULSE', 'S')]:
            listing[root + folder + '/'] = [root + folder + '/D06X/']
            listing[root + folder + '/D06X/'] = [root + folder + f'/D06X/{tag}9906003.{suffix}' for suffix in ['B', 'LBL']]
    monkeypatch.setattr('moonbroom.mission_archive.archive_links', lambda url: listing[url])
    monkeypatch.setattr('moonbroom.mission_archive._fetch', lambda url: text)
    plan = discover_mission(tmp_path)
    assert plan['official_inventory_complete'] and len(plan['catalogs']) == 19
    assert len(plan['products']) == 19 * 4
    assert '/LP_0001/MERGED/D06X/' in plan['products'][0]['url']
    plan['products'].pop()
    with pytest.raises(ValueError, match='differs'):
        validate_plan(plan)


class Response:
    def __init__(self, data, status=200, headers=None):
        import io
        self.stream = io.BytesIO(data)
        self.status = status
        self.headers = headers or {'Content-Length': str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def read(self, size):
        return self.stream.read(size)


@pytest.mark.parametrize('accept_range', [False, True])
def test_download_resumes_or_restarts_and_checks_official_checksum(tmp_path, monkeypatch, accept_range):
    data = b'original NASA PDS bytes'
    row = dict(relative_path='lp_0015/merged/file.B', url=PDS_ROOT + 'lp_0015/merged/file.B',
               md5=hashlib.md5(data).hexdigest())
    partial = tmp_path / 'lp_0015/merged/file.B.partial'
    partial.parent.mkdir(parents=True)
    partial.write_bytes(data[:7])
    requests = []

    def fetch(request, timeout):
        requests.append(request)
        assert request.headers['Range'] == 'bytes=7-'
        return (Response(data[7:], 206, {'Content-Range': f'bytes 7-{len(data)-1}/{len(data)}',
                                        'Content-Length': str(len(data)-7)}) if accept_range else Response(data))

    monkeypatch.setattr('moonbroom.mission_archive.urlopen', fetch)
    result = _download_one(row, tmp_path, timeout=1, retries=1)
    assert result['status'] == 'available'
    assert Path(result['path']).read_bytes() == data
    assert result['sha256'] == hashlib.sha256(data).hexdigest()
    assert _download_one(row, tmp_path, timeout=1, retries=1)['status'] == 'available'
    assert len(requests) == 1  # Verified completed files cause no new HTTP call.


def test_checksum_failure_never_commits_product(tmp_path, monkeypatch):
    row = dict(relative_path='lp_0015/raw.B', url=PDS_ROOT + 'raw.B', md5='0' * 32)
    monkeypatch.setattr('moonbroom.mission_archive.urlopen', lambda *a, **k: Response(b'bad'))
    result = _download_one(row, tmp_path, timeout=1, retries=1)
    assert result['status'] == 'failed' and 'checksum mismatch' in result['error']
    assert not (tmp_path / row['relative_path']).exists()


def local_archive(tmp_path):
    data = {}
    for tag, directory in [('M', 'merged'), ('S', 'sunpulse')]:
        for suffix in ['B', 'LBL']:
            name = f'{tag}9906003.{suffix}'
            original = (FIXTURES / f'{tag}9906003_excerpt.{suffix}').read_bytes()
            data[f'{directory}/d06x/{name}'] = original.replace(b'_excerpt', b'') if suffix == 'LBL' else original
    catalog = tmp_path / 'checksums.md5'
    catalog.write_text(checksums(data))
    products = parse_volume_checksums(catalog.read_text(), 'lp_0015')
    downloads = []
    for row in products:
        row['relative_path'] = 'lp_0015/' + row['volume_path']
        row['url'] = PDS_ROOT + row['relative_path']
        path = tmp_path / 'archive' / row['relative_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data[row['volume_path']])
        downloads.append(dict(row, path=str(path.resolve()), sha256=file_hash(path), status='available'))
    plan = dict(source=PDS_ROOT, volumes=['lp_0015'], official_inventory_complete=False,
                catalogs=[dict(volume='lp_0015', path=str(catalog), sha256=file_hash(catalog))], products=products)
    return plan, downloads


def test_mission_shards_preserve_mag_er_and_resume_with_source_and_product_checks(tmp_path):
    plan, downloads = local_archive(tmp_path)
    cfg = dict(archive_directory=str(tmp_path / 'archive'), output_directory=str(tmp_path / 'out'),
               records_per_shard=64, halo_records=128, clock_only=True, allow_uncalibrated=True)
    result = process_mission(plan, downloads, cfg)
    assert result['processing_complete'] and not result['full_mission_processed']
    assert result['native_mag_rows'] == 2304
    assert result['er_count_rows'] == 10128
    assert result['pseudo_mag_valid_rows'] == result['pseudo_er_valid_rows'] == 0
    paths = sorted((tmp_path / 'out/shards').rglob('native_mag.csv.gz'))
    native = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    assert native['source_sample_id'].is_unique and len(native) == 2304
    counts = pd.concat([pd.read_csv(path) for path in sorted((tmp_path / 'out/shards').rglob('er_counts.csv.gz'))], ignore_index=True)
    assert not counts.duplicated(['source_accumulation_id', 'kind', 'sample', 'energy_step', 'angle_bin']).any()
    full = decode_level0(FIXTURES / 'M9906003_excerpt.LBL')
    np.testing.assert_array_equal(native[['bx_signed', 'by_signed', 'bz_signed']], full['mag'][['bx_signed', 'by_signed', 'bz_signed']])
    order = ['accumulation_id', 'kind', 'sample', 'energy_step', 'angle_bin']
    np.testing.assert_allclose(counts.sort_values(order)['counts_lower'],
                               full['er_counts'].sort_values(order)['counts_lower'], equal_nan=True)
    assert process_mission(plan, downloads, cfg) == result
    paths[0].write_bytes(b'changed completed output')
    with pytest.raises(ValueError, match='Completed shard changed'):
        process_mission(plan, downloads, cfg)


def test_source_mutation_is_rejected_and_missing_calibration_is_explicit(tmp_path):
    plan, downloads = local_archive(tmp_path)
    cfg = dict(archive_directory=str(tmp_path / 'archive'), output_directory=str(tmp_path / 'out'), clock_only=True)
    with pytest.raises(ValueError, match='mag_bank'):
        preflight(cfg)
    cfg['allow_uncalibrated'] = True
    Path(downloads[0]['path']).write_bytes(b'changed source')
    with pytest.raises(ValueError, match='Source hash mismatch'):
        process_mission(plan, downloads, cfg)


def test_mission_automatically_ingests_ancillaries_and_records_quality_provenance(tmp_path, monkeypatch):
    import moonbroom.mission_pipeline as pipeline
    import csv
    import io
    plan, downloads = local_archive(tmp_path)
    frames = read_merged_telemetry(FIXTURES / 'M9906003_excerpt.LBL')
    begin = frames['ert'].min().floor('min') - pd.Timedelta(minutes=1)
    end = frames['ert'].max().ceil('min') + pd.Timedelta(minutes=1)
    tables = {
        'index/OUTAGES.TAB': [['LP_0015', 'example.MERGE', begin.isoformat(), end.isoformat(), 'NOISY DATA', 'NO CORRECTION']],
        'geometry/ATTITUDE.TAB': [[(begin - pd.Timedelta(hours=1)).isoformat(), 12., 90., 85.]],
        'command/C9906042.TAB': [[begin.isoformat(), 'A14REOR', 'CONFIRM', 42],
                                 [(begin + pd.Timedelta(seconds=32)).isoformat(), 'EXEC', 'CONFIRM', 42],
                                 [(begin + pd.Timedelta(minutes=2)).isoformat(), 'STOPFIRE', 'CONFIRM', 42],
                                 [(begin + pd.Timedelta(minutes=3)).isoformat(), 'MTLOADW, 0, 2, 350', 'CONFIRM', 42]],
    }
    extra = {}
    for name, rows in tables.items():
        text = io.StringIO()
        csv.writer(text).writerows(rows)
        extra[name] = text.getvalue().encode('ascii')
        extra[name[:-3] + 'LBL'] = f'FILE_RECORDS = {len(rows)}\n^TABLE = "{Path(name).name}"\n'.encode('ascii')
    for name, data in extra.items():
        relative = 'lp_0015/' + name
        path = tmp_path / 'archive' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        row = dict(volume='lp_0015', volume_path=name, relative_path=relative,
                   url=PDS_ROOT + relative, md5=hashlib.md5(data).hexdigest())
        plan['products'].append(row)
        downloads.append(dict(row, path=str(path), status='available', sha256=file_hash(path)))
    catalog = Path(plan['catalogs'][0]['path'])
    catalog.write_text('\n'.join(f"{row['md5']}  {row['volume_path']}" for row in plan['products']))
    plan['catalogs'][0]['sha256'] = file_hash(catalog)
    original = pipeline.decode_level0
    monkeypatch.setattr(pipeline, 'decode_level0', lambda paths, **kw: original(paths, light_time_s=3.28, **kw))
    cfg = dict(archive_directory=str(tmp_path / 'archive'), output_directory=str(tmp_path / 'out'),
               records_per_shard=64, halo_records=128, clock_only=True, allow_uncalibrated=True)
    summary = process_mission(plan, downloads, cfg)
    assert summary['processing_complete'] and summary['native_mag_rows'] == 2304
    native = pd.concat([pd.read_csv(path) for path in sorted((tmp_path / 'out/shards').rglob('native_mag.csv.gz'))])
    assert native['archive_noisy'].all() and not native['science_valid'].any()
    assert native['attitude_available'].any()
    assert native['post_maneuver_settling'].any()
    assert native['mager_command_transition'].any()
    assert not native['attitude_quality_valid'].any()
    manifest = json.loads((tmp_path / 'out/run.json').read_text())
    assert manifest['archive_quality']['outages_source'] == 'lp_0015/index/OUTAGES.LBL'
    assert manifest['archive_quality']['attitude_frame'] == 'ECLIPJ2000'
    assert manifest['sunpulse_selection'][0]['selected'].endswith('S9906003.LBL')
    assert len(pd.read_csv(tmp_path / 'out/archive_outages.csv')) == 1
    assert len(pd.read_csv(tmp_path / 'out/maneuver_intervals.csv')) == 1
    actions = pd.read_csv(tmp_path / 'out/mager_command_actions.csv')
    assert actions['command'].tolist() == ['MTLOADW, 0, 2, 350']
    assert not actions['arrival_scet_valid'].any()  # Clock-only, never invented.


def test_shard_boundary_inside_er_accumulation_preserves_assembly():
    path = str((FIXTURES / 'M9906003_excerpt.LBL').resolve())
    source = FIXTURES / 'M9906003_excerpt.B'
    frames = decode_level0(path)
    first = frames['er_packets'].drop_duplicates('accumulation_id')
    # Use one record beyond the first complete accumulation's first packet.
    edge = int(first['record_index'].iloc[1]) + 1
    results = [process_shard([path], {path: (0, 128)}, source, a, b, [], {'clock_only': True}, {}, None)[0]
               for a, b in [(0, edge), (edge, 128)]]
    total = pd.concat([part['er_counts'] for part in results], ignore_index=True)
    assert len(total) == len(frames['er_counts'])
    assert total.groupby('source_accumulation_id')['accumulation_id'].nunique().eq(1).all()
    order = ['accumulation_id', 'kind', 'sample', 'energy_step', 'angle_bin']
    np.testing.assert_allclose(total.sort_values(order)['counts_upper'],
                               frames['er_counts'].sort_values(order)['counts_upper'], equal_nan=True)


def test_mission_er_application_preserves_response_metadata():
    from moonbroom.validation import validate_er_pseudocalibration
    from moonbroom.mission_calibration import apply_er_mission_response
    train = pd.read_csv(FIXTURES / 'mission_fit/1998_01_training.csv')
    test = pd.read_csv(FIXTURES / 'mission_fit/1998_01_validation.csv')
    bank = validate_er_pseudocalibration(train, test, valid_from='1998-01-16', valid_to='1998-01-23',
                                         source='public fixture', max_median_relative_error=.05)['coefficients']
    bank['cohort'] = '1998_01'
    out = apply_er_mission_response(test, bank)
    assert out['calibration_valid'].all()
    assert out['response_cohort'].eq('1998_01').all()
    assert out['response_valid_from'].notna().all()
