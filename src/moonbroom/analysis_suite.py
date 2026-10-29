"""Run an ordered set of manuscript analysis JSON configurations."""

import argparse
import json
from pathlib import Path

from .analysis_cli import run_analysis
from .mission_archive import file_hash, write_json


def resolve_analysis_config(config, directory):
    cfg = dict(config)
    for key, value in cfg.items():
        if isinstance(value, str) and (key.endswith('_csv') or key.endswith('_path') or key == 'output_directory'):
            cfg[key] = str((directory / Path(value).expanduser()).resolve())
        elif key in ('raw_paths', 'sunpulse_paths', 'kernels'):
            cfg[key] = [str((directory / Path(path).expanduser()).resolve()) for path in value]
    if 'decode_options' in cfg:
        cfg['decode_options'] = dict(cfg['decode_options'])
        if 'timing_kernels' in cfg['decode_options']:
            cfg['decode_options']['timing_kernels'] = [str((directory / Path(path).expanduser()).resolve())
                                                       for path in cfg['decode_options']['timing_kernels']]
    return cfg


def run_suite(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    statuses = []
    output = (path.parent / cfg['output_directory']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for relative in cfg['jobs']:
        job = (path.parent / relative).resolve()
        settings = resolve_analysis_config(json.loads(job.read_text()), job.parent)
        manifest = Path(settings['output_directory']) / 'manifest.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            code = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
            if (previous['config'] != settings or previous['code_sha256'] != code
                    or not previous.get('outputs')
                    or any(file_hash(row['path']) != row['sha256'] for row in previous['inputs'] + previous['outputs'])):
                raise ValueError(f'Completed analysis changed. Use a new output directory: {job}')
            state = 'reused'
        else:
            print(f"Running {job.name}: {settings['task']}", flush=True)
            result = run_analysis(settings)
            if 'audit' in result and (result['audit'].empty or not result['audit']['audit_pass'].all()):
                raise ValueError(f'Native bit audit failed: {job}')
            state = 'completed'
        statuses.append(dict(config=str(job), config_sha256=file_hash(job), task=settings['task'],
                              output_directory=settings['output_directory'], status=state))
        write_json(output / 'pipeline_status.json', statuses)
    write_json(output / 'pipeline_manifest.json', dict(config=str(path), config_sha256=file_hash(path),
               jobs=statuses, high_frequency_validated=False))
    return statuses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args(argv)
    statuses = run_suite(args.config)
    print(f"Completed {len(statuses)} configured analyses")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
