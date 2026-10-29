"""Fetch pinned official SPICE kernels into a standalone Moonbroom checkout."""

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from .mission_archive import _download_one, write_json


def prepare_auxiliary_data(manifest):
    path = Path(manifest).resolve()
    config = json.loads(path.read_text())
    results = []
    for entry in config['files']:
        target = (path.parent / Path(entry['path']).expanduser()).resolve()
        url = urlparse(entry['url'])
        if (url.scheme != 'https' or url.netloc != 'naif.jpl.nasa.gov'
                or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256'])):
            raise ValueError('Auxiliary inputs require an official NAIF URL and a pinned SHA-256')
        row = dict(relative_path=target.name, url=entry['url'], sha256_expected=entry['sha256'])
        result = _download_one(row, target.parent, timeout=config.get('timeout_s', 120),
                               retries=config.get('retries', 4))
        if result['status'] == 'available' and entry.get('bytes') is not None and result['bytes'] != entry['bytes']:
            result.update(status='failed', error='Pinned auxiliary file size mismatch')
        results.append(result)
        print(f"{target.name}: {result['status']}", flush=True)
    if config.get('output_manifest'):
        write_json(path.parent / config['output_manifest'], results)
    failed = [row for row in results if row['status'] != 'available']
    if failed:
        raise OSError(f"Could not prepare {len(failed)} auxiliary input(s). See auxiliary download manifest")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args(argv)
    prepare_auxiliary_data(args.manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
