# Native MAG recovery, pseudo-calibration and magnetotail survey

Run `python -m moonbroom.analysis_cli CONFIG.json` from a source checkout with
`PYTHONPATH=src`, or install Moonbroom and its `geometry` extra. All input
paths and scientific choices are explicit. Each run writes CSV tables and a
JSON manifest with input hashes, source-code hashes, settings and versions.
Output directories must be new. The analysis CLI functions do not download
data automatically. The mission survey can explicitly prepare OMNI via its
`omni_config` setting; see [OMNI upstream inputs](omni_upstream.md).

## Recovery and calibration sequence

1. `task: decode` reads raw merged files and audits every real-time MAG sample
   against an independent scalar bit extraction from the retained payload.
   Missing, extra, duplicate and modified identities fail. Unsigned codes,
   excess-2048 values, status, range and nominal collection clocks are checked.
   This does not validate absolute UTC or high-frequency analog accuracy.
2. `task: prepare_mag` uses raw and SUNPULSE paths, kernels and a named receiver.
   It prepares local transverse offsets, rotates the separate five-second SSE
   reference to SCD with explicitly framed attitude, and creates matched windows.
   A chosen timing offset must be frozen from training before validation.
3. `task: mag` fits only the training-window CSV and predicts withheld windows.
   Use `fit_xy_offsets: false` for locally offset-corrected preparation output.
   It records per-range support, component biases, RMSE and acceptance based on
   explicit limits. Unsupported validation ranges remain in metrics even when
   they have no fitted coefficient. An unrepresented validation range cannot
   silently become accepted.
4. `task: native` prepares raw telemetry and applies a selected coefficient
   cohort to the original 9 Hz rows using local offsets. Pass a nonoverlapping
   selected cohort, not a bank containing repeated ranges. Rejected or absent
   validation support leaves calibrated fields missing.
5. `task: er` fits and validates matched ER response pairs using the existing
   preparation and reference matching functions. Units and count bounds are
   unchanged. The response is effective, not a unique detector efficiency.

Example MAG fit config:

```json
{
  "task": "mag",
  "output_directory": "results/independent_mag_fit",
  "training_csv": "results/train/windows.csv",
  "validation_csv": "results/validation/windows.csv",
  "settings": {
    "valid_from": "1998-01-16",
    "valid_to": "1998-01-23",
    "source": "exact PDS product identifiers",
    "fit_xy_offsets": false,
    "min_gap_s": 120,
    "max_rmse_nt": 0.1,
    "max_bias_nt": 0.1,
    "min_validation_windows": 20
  }
}
```

The 0.1 nT limits are demonstration gates, not established science requirements.
Predeclare science limits and examine axis-specific errors. Never enlarge a
limit solely to accept a failed holdout. The public March range-2 example
fails this RMSE gate and remains rejected.

A preparation config includes `raw_paths`, `sunpulse_paths`, `kernels`,
`receiver`, `reference_path`, `attitude_csv`, `attitude_frame`, and
`window_settings` such as `{"time_offset_s": 0, "window_s": 5}`.
`preparation_settings` are passed to `prepare_mag_interval`. A native config
uses the same raw inputs plus `coefficients_csv` and omits the reference.
A decode config needs only `raw_paths` and optional `decode_options`.

## Why different windows and dates

The 60-second offset interval resolves multiple approximately five-second
spins while approximating slowly varying offsets. It is not an averaging of
the returned native fields. Five-second reference windows test the archived
product's resolution. Longer training intervals supply field/phase diversity
for identifiable gains and offsets. Counts alone do not guarantee design rank.

Validation uses withheld contiguous intervals and preferably another date
within a stable instrument cohort. This tests transfer through changing
field/thermal conditions without extrapolating across firmware, range, sweep
or known configuration changes. Randomly splitting native samples, adjacent
reference rows or cells of one spectrum would share temporal information.
`check_validation_split` rejects all shared reference IDs and physical
interval overlaps, including interleaved blocks. A 120-second default purge
is a baseline for the local one-minute offset support. Use a larger gap when
preprocessing or residual correlations require one. It is not proof that
observations 120 seconds apart are independent. Keep a final untouched date
for reporting after model or alignment selection.

Existing fixtures span versions 1 through 5 on the date pairs 17/19 January
1998, 25/27 January, 10/12 June, 10/12 March 1999 and 30/31 March. These dates
exercise observed layouts and sweep cohorts. They are not optimal windows
selected by an exhaustive mission study. The March 1 MAG fixture is a weaker
same-day transfer check. Multiple independent dates and operating ranges are
required before broad application.

## Magnetotail survey

Survey configs have `task: survey`, `samples_csv`, `upstream_csv`, and
`field_frame`. Supply either `geometry_csv` or `kernels`. The CSV geometry
requires `utc`, `X_GSE_RE`, `Y_GSE_RE`, `Z_GSE_RE`, evaluated at the variability
window centers. With kernels, geometric LP positions are computed relative
to Earth, with the actual LP displacement from the Moon included. The GSE
basis uses the epoch Earth-Sun direction and J2000 ecliptic north, an explicit
approximation to a date-dependent ecliptic convention.

Native fields may be supplied in GSE or SCD. For SCD, provide `attitude_csv`,
`attitude_settings` with `attitude_frame` and `max_attitude_age_s`, and kernels
to rotate to GSE. Attitude has `utc`, `longitude_deg` and `latitude_deg`, with
angles in the declared inertial frame. Unavailable or stale attitude stays
unsupported. Without this rotation SCD statistics are explicitly labeled SCD.

Upstream inputs require `utc`, `pressure_npa`, `bz_gsm_nt` and `quality_valid`.
Their time tags must already represent the desired terrestrial boundary
context. No propagation is applied silently. Stale inputs are rejected with
backward matching and a bounded age. Published pressure must have a recorded
composition convention. No bow-shock classification is inferred.
The default mission survey prepares this table from published one-minute
OMNI flow pressure and GSM Bz. Invalid and absent minutes are retained,
and OMNI records expire at their one-minute interval end.

`variability_settings` defaults are 60-second windows, 90 percent nominal
coverage and a maximum 0.25-second sample gap. Windows crossing ranges,
cohorts or calibration sources are rejected. All observed windows retain a
quality result. Native vector variability is compared with variability of
complete five-second vector means. Input fields must be despun and flagged
with `calibration_valid`.

`settings` control upstream maximum age, pressure/Bz perturbations, boundary
margin and downstream scope. The Shue 1998 cross section is solved at the
spacecraft x position, including the exact tail axis. Its far-tail use is an
extrapolation. A sensitivity envelope and radial margin define conservative
`tail_candidate`, `boundary_ambiguous`, `modeled_external`, `not_downstream`
and `insufficient_context` labels. The default envelope perturbs pressure by
20 percent and Bz by 1 nT, and uses a 2 Earth-radius margin. These are explicit
sensitivity choices, not fitted probabilities or established model errors.

Outputs include per-window metrics, contiguous candidate intervals, exposure
by region and Spearman association of variability with model margin. Day-block
bootstrap intervals require at least three represented blocks. Synthetic
geometry tests verify transformations and region logic. A full-mission survey
has not been executed with these additions. Classification remains contextual
evidence. Local wake/crustal signatures, spacecraft charging, model limitations
and offset processing can confound a physical interpretation.

## Reproduce public checks

`python examples/validate_recovery.py --output NEW_DIRECTORY` runs the payload
audit and held-out MAG/ER public fixtures, writing hashes and numerical results.
The native audit contains 2304 samples from 128 source records. ER validation
has 120 selected cells in 20 energy/angular/version groups. These small
fixtures do not establish full-mission coverage or native-frequency accuracy.

Scientific basis: Lin et al. (1998), doi:10.1126/science.281.5382.1480.
Acuña et al. (1992), doi:10.1029/92JE00344, describes instrument heritage rather
than an LP flight coefficient table. Plaschke et al. (2019),
doi:10.5194/gi-8-63-2019, discusses spin-based calibration identifiability.
Shue et al. (1998), doi:10.1029/98JA01103, defines the magnetopause model.
PDS MAG and ER processing descriptions remain the authoritative sources for
LP archive-specific conventions.
