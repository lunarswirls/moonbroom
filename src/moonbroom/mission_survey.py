"""Run the manuscript magnetotail analyses over native mission output shards."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .magnetotail import (native_variability_windows, rotate_scd_to_gse, spice_survey_geometry,
                         survey_magnetotail, magnetotail_catalogue, variability_region_summary,
                         variability_boundary_association)
from .mission_archive import file_hash, write_json
from .omni import load_omni_config, prepare_omni_upstream


def stream_variability(paths, *, field_frame, variability_settings=None, attitude=None,
                       kernels=(), attitude_settings=None):
    """Carry partial UTC windows between files without averaging native vectors.

    Files must be chronological, nonoverlapping native output shards. Rows
    lacking UTC are counted separately because they cannot enter UTC windows.
    Invalid calibrated rows with valid UTC still enter the coverage denominator.
    """
    settings = variability_settings or {}
    window_s = settings.get('window_s', 60)
    if window_s < 5:
        raise ValueError('Variability windows must be at least five seconds')
    pending, windows = pd.DataFrame(), []
    previous = None
    input_rows, invalid_utc = 0, 0
    for path in paths:
        raw = pd.read_csv(path)
        input_rows += len(raw)
        raw['utc'] = pd.to_datetime(raw['utc'], utc=True, format='mixed', errors='raise')
        invalid_utc += int(raw['utc'].isna().sum())
        raw = raw.loc[raw['utc'].notna()].copy()
        if raw.empty:
            continue
        if not raw['utc'].is_monotonic_increasing or raw['utc'].duplicated().any():
            raise ValueError(f'Native UTC must be ordered and unique: {path}')
        if previous is not None and raw['utc'].iloc[0] <= previous:
            raise ValueError(f'Native files overlap or are out of order: {path}')
        previous = raw['utc'].iloc[-1]
        frame = field_frame
        if attitude is not None:
            if frame != 'SCD':
                raise ValueError('Attitude rotation requires field_frame SCD')
            raw = rotate_scd_to_gse(raw, attitude, kernels, **(attitude_settings or {}))
            frame = 'GSE'
        combined = pd.concat([pending, raw], ignore_index=True) if not pending.empty else raw
        numbers = (combined['utc'].astype('datetime64[ns, UTC]').astype('int64') / 1e9 // window_s)
        last = numbers.max()
        finished = combined.loc[numbers.lt(last)]
        pending = combined.loc[numbers.eq(last)].copy()
        if not finished.empty:
            windows.append(native_variability_windows(finished, field_frame=frame, **settings))
    if not pending.empty:
        frame = 'GSE' if attitude is not None else field_frame
        windows.append(native_variability_windows(pending, field_frame=frame, **settings))
    if not windows:
        raise ValueError('No UTC-supported native samples were available for the survey')
    return pd.concat(windows, ignore_index=True), dict(input_native_rows=input_rows, invalid_utc_rows=invalid_utc)


def run_mission_survey(config):
    cfg = dict(config)
    target = Path(cfg['output_directory'])
    if target.exists():
        raise ValueError('Use a new survey output directory')
    files = []
    upstream_provenance = None
    mission_summary = None
    if cfg.get('mission_directory'):
        root = Path(cfg['mission_directory'])
        manifest = root / 'processing_status.json'
        statuses = json.loads(manifest.read_text())
        files.append(manifest)
        paths = []
        for status in statuses:
            if status['status'] != 'processed':
                continue
            for product in status['products']:
                if Path(product['path']).name == 'native_mag.csv.gz':
                    path = root / product['path']
                    if file_hash(path) != product['sha256']:
                        raise ValueError(f'Native mission product changed: {path}')
                    paths.append(path)
        if (root / 'mission_status.json').exists():
            mission_summary = json.loads((root / 'mission_status.json').read_text())
            files.append(root / 'mission_status.json')
    else:
        paths = [Path(path) for path in cfg['native_paths']]
    if not paths:
        raise ValueError('No processed native MAG shards are available for the survey')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    files.extend(paths)
    attitude = pd.read_csv(cfg['attitude_csv']) if cfg.get('attitude_csv') else None
    if attitude is not None:
        files.append(Path(cfg['attitude_csv']))
    kernels = cfg.get('kernels', [])
    if cfg.get('geometry_csv'):
        geometry = pd.read_csv(cfg['geometry_csv'])
        files.append(Path(cfg['geometry_csv']))
    else:
        for path in kernels:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    if cfg.get('omni_config'):
        omni_cfg = load_omni_config(cfg['omni_config'])
        if Path(omni_cfg['output_csv']).resolve() != Path(cfg['upstream_csv']).resolve():
            raise ValueError('omni_config output_csv must match survey upstream_csv')
        upstream_provenance = prepare_omni_upstream(omni_cfg)
        files.extend([Path(cfg['omni_config']),
                      Path(omni_cfg.get('output_manifest', Path(omni_cfg['output_csv']).with_suffix('.manifest.json')))])
    upstream = pd.read_csv(cfg['upstream_csv'])
    if upstream.empty or not {'utc', 'pressure_npa', 'bz_gsm_nt', 'quality_valid'} <= set(upstream):
        raise ValueError('Provide nonempty upstream boundary context with utc, pressure_npa, bz_gsm_nt, quality_valid')
    files.append(Path(cfg['upstream_csv']))
    windows, counts = stream_variability(paths, field_frame=cfg['field_frame'],
                                         variability_settings=cfg.get('variability_settings'),
                                         attitude=attitude, kernels=kernels,
                                         attitude_settings=cfg.get('attitude_settings'))
    if not cfg.get('geometry_csv'):
        geometry = spice_survey_geometry(windows['utc'], kernels)
    files.extend(Path(path) for path in kernels)
    surveyed = survey_magnetotail(windows, geometry, upstream, **cfg.get('boundary_settings', {}))
    results = dict(windows=surveyed, encounters=magnetotail_catalogue(surveyed),
                   region_summary=variability_region_summary(surveyed),
                   association=variability_boundary_association(surveyed, **cfg.get('association_settings', {})))
    target.mkdir(parents=True)
    for name, table in results.items():
        table.to_csv(target / (name + '.csv'), index=False)
    write_json(target / 'manifest.json', dict(config=cfg, **counts, source_mission_status=mission_summary,
                upstream_provenance=upstream_provenance,
                inputs=[dict(path=str(path.resolve()), sha256=file_hash(path)) for path in dict.fromkeys(files)],
                code_sha256={path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
                high_frequency_validated=False, boundary_model='Shue 1998, far-tail extrapolation'))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent
    for key in ('output_directory', 'mission_directory', 'upstream_csv', 'omni_config', 'attitude_csv', 'geometry_csv'):
        if cfg.get(key):
            cfg[key] = str((base / Path(cfg[key]).expanduser()).resolve())
    for key in ('native_paths', 'kernels'):
        if key in cfg:
            cfg[key] = [str((base / Path(path).expanduser()).resolve()) for path in cfg[key]]
    result = run_mission_survey(cfg)
    print(f"Wrote {len(result['windows'])} windows and {len(result['encounters'])} candidate intervals")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
