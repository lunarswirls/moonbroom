"""Run reproducible recovery validation and magnetotail surveys from JSON configs."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import pandas as pd

from .level0 import decode_level0
from .mission_calibration import prepare_mag_interval, prepare_archive_interval
from .native_mag import calibrate_mag_9hz, fit_mag_spin_offsets
from .archive_calibration import mag_sse_reference_to_scd, mag_fixed_reference_windows
from .mag import read_mag
from .validation import (audit_mag_decoding, validate_mag_pseudocalibration,
                         validate_er_pseudocalibration, compare_native_mag_reference)
from .magnetotail import (native_variability_windows, spice_survey_geometry,
                         survey_magnetotail, magnetotail_catalogue,
                         variability_region_summary, rotate_scd_to_gse, variability_boundary_association)


def run_analysis(config):
    """No downloads or implicit file discovery. Paths are relative to caller cwd.

    Existing Moonbroom preparation tools produce MAG windows or ER pairs.
    A survey accepts calibrated GSE fields, or SCD fields plus an explicit
    attitude table. A fresh output directory prevents mixing previous runs.
    """
    cfg = dict(config)
    directory = Path(cfg['output_directory'])
    if directory.exists():
        raise ValueError("Use a new output directory for each reproducible run")
    files = []

    def read(path):
        files.append(Path(path))
        return pd.read_csv(path)

    task = cfg['task']
    settings = cfg.get('settings', {})
    if task == 'decode':
        options = cfg.get('decode_options', {})
        products = decode_level0(cfg['raw_paths'], **options)
        audit = audit_mag_decoding(products['frames'], products['mag'])
        result = dict(audit=audit, mag=products['mag'])
        files.extend(Path(p) for p in cfg['raw_paths'])
        # Include binary partners even when the user passes detached labels.
        files.extend(Path(p) for p in products['frames']['source_file'].unique())
        files.extend(Path(p) for p in options.get('timing_kernels', []))
    elif task in ('prepare_mag', 'native'):
        kernels = cfg['kernels']
        prepared = prepare_mag_interval(cfg['raw_paths'], cfg['sunpulse_paths'],
                                       kernels=kernels, receiver=cfg['receiver'],
                                       **cfg.get('preparation_settings', {}))
        result = dict(prepared)
        files.extend(Path(p) for p in cfg['raw_paths']+cfg['sunpulse_paths']+kernels)
        files.extend(Path(p) for p in prepared['mag_raw']['source_file'].unique())
        # Hash both members of detached SUNPULSE inputs.
        for path in cfg['sunpulse_paths']:
            path = Path(path)
            if path.suffix.lower() == '.lbl':
                files.extend(p for p in path.parent.iterdir()
                             if p.stem.lower() == path.stem.lower() and p.suffix.lower() == '.b')
        if task == 'prepare_mag':
            reference_path = Path(cfg['reference_path'])
            files.append(reference_path)
            if reference_path.suffix.lower() == '.csv':
                reference = pd.read_csv(reference_path)
            else:
                reference = read_mag(reference_path)
                files.extend(p for p in reference_path.parent.iterdir()
                             if p.stem.lower() == reference_path.stem.lower() and p.suffix.lower() == '.tab')
            reference = mag_sse_reference_to_scd(reference, read(cfg['attitude_csv']), kernels,
                                                 attitude_frame=cfg['attitude_frame'])
            result['windows'] = mag_fixed_reference_windows(prepared['mag'], reference,
                                                             **cfg.get('window_settings', {}))
        else:
            result['native_mag'] = calibrate_mag_9hz(prepared['mag_raw'], read(cfg['coefficients_csv']),
                                                     offsets=prepared['offsets'])
    elif task == 'prepare_er':
        result = prepare_archive_interval(cfg['raw_paths'], cfg['sunpulse_paths'], cfg['reference_path'],
                                          kernels=cfg['kernels'], receiver=cfg['receiver'],
                                          reference_source=cfg['reference_source'], **settings)
        files.extend(Path(p) for p in cfg['raw_paths'] + cfg['sunpulse_paths'] + cfg['kernels'])
        files.extend(Path(p) for p in result['mag_raw']['source_file'].unique())
        files.append(Path(cfg['reference_path']))
        for path in cfg['sunpulse_paths'] + [cfg['reference_path']]:
            path = Path(path)
            files.extend(p for p in path.parent.iterdir() if p.stem.lower() == path.stem.lower()
                         and p.suffix.lower() in ('.b', '.tab', '.dat'))
    elif task == 'apply_native':
        samples = read(cfg['samples_csv'])
        coefficients = read(cfg['coefficients_csv'])
        offsets = fit_mag_spin_offsets(samples, **cfg.get('offset_settings', {})) if cfg.get('local_spin_offsets', True) else None
        result = dict(native_mag=calibrate_mag_9hz(samples, coefficients, offsets=offsets))
        if offsets is not None:
            result['offsets'] = offsets
    elif task == 'native_compare':
        comparison = compare_native_mag_reference(read(cfg['samples_csv']), read(cfg['reference_windows_csv']), **settings)
        errors = comparison.loc[comparison['comparison_valid'], ['error_b' + a for a in 'xyz']]
        metrics = dict(n_total=len(comparison), n_supported=len(errors),
                       component_rmse_nt=float((errors ** 2).to_numpy().mean() ** .5) if len(errors) else None)
        for axis in 'xyz':
            metrics['bias_' + axis + '_nt'] = float(errors['error_b' + axis].mean()) if len(errors) else None
            metrics['rmse_' + axis + '_nt'] = float((errors['error_b' + axis] ** 2).mean() ** .5) if len(errors) else None
        result = dict(comparison=comparison, metrics=pd.DataFrame([metrics]))
    elif task in ('mag', 'er'):
        train, test = read(cfg['training_csv']), read(cfg['validation_csv'])
        function = validate_mag_pseudocalibration if task == 'mag' else validate_er_pseudocalibration
        result = function(train, test, **settings)
    elif task == 'survey':
        samples = read(cfg['samples_csv'])
        kernels = cfg.get('kernels', [])
        frame = cfg['field_frame']
        if frame == 'SCD' and 'attitude_csv' in cfg:
            samples = rotate_scd_to_gse(samples, read(cfg['attitude_csv']), kernels,
                                        **cfg['attitude_settings'])
            frame = 'GSE'
        windows = native_variability_windows(samples, field_frame=frame, **cfg.get('variability_settings', {}))
        geometry = read(cfg['geometry_csv']) if 'geometry_csv' in cfg else spice_survey_geometry(windows['utc'], kernels)
        windows = survey_magnetotail(windows, geometry, read(cfg['upstream_csv']), **settings)
        result = dict(windows=windows, encounters=magnetotail_catalogue(windows),
                      region_summary=variability_region_summary(windows),
                      association=variability_boundary_association(windows, **cfg.get('association_settings', {})))
        files.extend(Path(p) for p in kernels)
    else:
        raise ValueError("task must be decode, prepare_mag, prepare_er, native, apply_native, native_compare, mag, er or survey")
    # Hash inputs before writing, including explicitly named kernels and references.
    inventory = []
    for path in dict.fromkeys(files):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        inventory.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    directory.mkdir(parents=True)
    for name, table in result.items():
        table.to_csv(directory/(name+'.csv'), index=False)
    manifest = dict(config=cfg, inputs=inventory, python=platform.python_version(),
                    pandas=pd.__version__, high_frequency_validated=False,
                    created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')})
    manifest['outputs'] = [dict(path=str((directory / (name + '.csv')).resolve()), rows=len(table),
                                sha256=hashlib.sha256((directory / (name + '.csv')).read_bytes()).hexdigest())
                           for name, table in result.items()]
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    result = run_analysis(json.loads(args.config.read_text()))
    if 'audit' in result and (result['audit'].empty or not result['audit']['audit_pass'].all()):
        raise SystemExit('Native MAG audit failed. See audit.csv')


if __name__ == '__main__':
    main()
