import json
from pathlib import Path

import pytest

from moonbroom.auxiliary_data import prepare_auxiliary_data
from moonbroom.mission_archive import file_hash


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_calibration_snapshot_matches_its_provenance():
    manifest = json.loads((ROOT / 'calibration/manifest.json').read_text())
    for entry in manifest['files']:
        path = ROOT / 'calibration' / entry['path']
        assert path.stat().st_size == entry['bytes']
        assert file_hash(path) == entry['sha256']
    config = json.loads((ROOT / 'configs/pseudocalibrate_mission.json').read_text())
    for key in ('mag_bank', 'er_bank', 'pad_bank', 'energy_bins'):
        path = (ROOT / 'configs' / config[key]).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
    for key in ('archive_directory', 'output_directory'):
        assert (ROOT / 'configs' / config[key]).resolve().is_relative_to(ROOT)


def test_auxiliary_inputs_reuse_verified_local_files_without_network(tmp_path, monkeypatch):
    kernel = tmp_path / 'naif0012.tls'
    kernel.write_bytes(b'example kernel')
    manifest = tmp_path / 'auxiliary.json'
    manifest.write_text(json.dumps(dict(files=[dict(path=kernel.name, sha256=file_hash(kernel),
                bytes=kernel.stat().st_size, url='https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls')],
                output_manifest='results.json')))
    monkeypatch.setattr('moonbroom.mission_archive.urlopen', lambda *a, **k: pytest.fail('Unexpected network request'))
    result = prepare_auxiliary_data(manifest)
    assert result[0]['status'] == 'available'
    assert json.loads((tmp_path / 'results.json').read_text())[0]['sha256'] == file_hash(kernel)


def test_auxiliary_inputs_require_pinned_official_sources(tmp_path):
    manifest = tmp_path / 'auxiliary.json'
    manifest.write_text(json.dumps(dict(files=[dict(path='kernel.bsp', sha256='0'*64,
                                                   url='https://example.org/kernel.bsp')])))
    with pytest.raises(ValueError, match='official NAIF'):
        prepare_auxiliary_data(manifest)
