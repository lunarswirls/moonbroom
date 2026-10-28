# Recovered LP pseudo-calibration banks

These CSV snapshots are included in Moonbroom so mission processing needs no
other analysis repository. The original values, accepted/rejected flags,
training/validation dates, response keys, proposed validity scopes and PDS
source identifiers are retained. `manifest.json` records exact SHA-256 hashes.

| File | Purpose |
| --- | --- |
| `mag_models.csv` | Native MAG gain/alignment/offset models and held-out validation |
| `er_3d_models.csv` | Effective ER response by mode, version, energy and angular bin |
| `pad_models.csv` | Approximate PAD aggregate normalization with held-out checks |
| `cohorts.csv` | Training and validation date assignments |
| `validation_summary.csv` | Original cohort validation summaries |
| `er_energy_bins.tab` | Original PDS 15-step energy table with both archived time conventions |

These are empirical calibration-transfer models, not recovered flight-team
coefficient tables. Applying them outside their acceptance flags, keys or
time scopes is unsupported. Monthly validity intervals do not establish
continuous independent validation. High-frequency MAG truth and resolved
pitch-bin ER response remain unvalidated. The demonstration gates used by the
small manuscript fixtures are separate from these banks' original gates.

The CSVs retain rejected fits so unsupported ranges and groups are visible.
`mission_pipeline` selects only explicitly accepted rows. Use a new output
directory when changing a coefficient bank, and retain new training/holdout
evidence with any replacement.
