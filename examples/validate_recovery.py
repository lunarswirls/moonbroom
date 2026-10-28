"""Reproduce small public-fixture checks. These are not full-mission estimates.

Run from a source checkout:
    python examples/validate_recovery.py --output /path/to/new/results
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from moonbroom import (decode_level0, audit_mag_decoding, calibrate_mag_9hz,
                       fit_mag_spin_offsets, compare_native_mag_reference)
from moonbroom.analysis_cli import run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]/'tests'/'fixtures'
    root = args.output
    if root.exists():
        raise ValueError('Choose a new results directory')
    root.mkdir(parents=True)
    products = decode_level0(base/'M9906003_excerpt.LBL')
    audit = audit_mag_decoding(products['frames'], products['mag'])
    audit.to_csv(root/'decode_audit.csv', index=False)
    summary = dict(raw_records=len(products['frames']), native_samples=len(audit),
                   exact_samples=int(audit['audit_pass'].sum()),
                   high_frequency_validated=False)
    cfg = dict(task='mag', output_directory=str(root/'mag'),
               training_csv=str(base/'archive_fit'/'mag_training_windows.csv'),
               validation_csv=str(base/'archive_fit'/'mag_validation_windows.csv'),
               settings=dict(valid_from='1999-03-01', valid_to='1999-03-02',
                             source='PDS public regression windows', max_rmse_nt=.1, max_bias_nt=.1,
                             min_gap_s=120))
    mag = run_analysis(cfg)
    summary['mag_metrics'] = mag['metrics'].to_dict('records')
    er_metrics = []
    for cohort in ['1998_01','1998_01_format2','1998_06','1999_03','1999_03_sweep34']:
        train = base/'mission_fit'/(cohort+'_training.csv')
        test = base/'mission_fit'/(cohort+'_validation.csv')
        tr, va = pd.read_csv(train), pd.read_csv(test)
        config = dict(task='er', output_directory=str(root/cohort), training_csv=str(train), validation_csv=str(test),
                      settings=dict(valid_from=tr['start_utc'].min()[:10],
                                    valid_to=(pd.to_datetime(va['stop_utc'], utc=True).max()+pd.Timedelta(days=1)).isoformat(),
                                    source='PDS public regression pairs', max_median_relative_error=.05,
                                    min_gap_s=120))
        out = run_analysis(config)
        er_metrics.append(out['metrics'].assign(cohort=cohort))
    er = pd.concat(er_metrics, ignore_index=True)
    er.to_csv(root/'er_metrics.csv', index=False)
    native_raw = pd.read_csv(base/'mission_fit'/'mag_native_validation.csv')
    model = pd.read_csv(base/'mission_fit'/'mag_native_model.csv')
    native = calibrate_mag_9hz(native_raw, model, offsets=fit_mag_spin_offsets(native_raw))
    reference = pd.read_csv(base/'mission_fit'/'mag_native_validation_windows.csv')
    comparison = compare_native_mag_reference(native, reference)
    comparison.to_csv(root/'native_five_second_comparison.csv', index=False)
    error = comparison.loc[comparison['comparison_valid'], ['error_b'+a for a in 'xyz']]
    summary['native_input_sha256'] = {name: hashlib.sha256((base/'mission_fit'/name).read_bytes()).hexdigest()
                                      for name in ['mag_native_validation.csv', 'mag_native_model.csv', 'mag_native_validation_windows.csv']}
    summary['native_five_second'] = dict(n_samples=len(native),
          original_clock_preserved=bool(native['clock_s'].equals(native_raw['clock_s'])),
          n_windows=int(comparison['comparison_valid'].sum()), component_rmse_nt=float((error**2).to_numpy().mean()**.5))
    summary['er_groups'] = len(er)
    summary['er_max_group_median_relative_error'] = float(er['median_relative_error'].max())
    summary['er_withheld_pairs'] = int(er['n_validation'].sum())
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    (root/'summary.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False)+'\n')
    if audit.empty or not audit['audit_pass'].all():
        raise SystemExit('Decode audit failed')


if __name__ == '__main__':
    main()
