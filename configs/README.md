# Run configurations

Run these commands from this standalone `moonbroom` checkout. Configuration
paths resolve relative to their own files. Create a dedicated environment and
install the dependencies declared by this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[geometry,test]'
```

## Download the full mission

`download_mission.json` specifies the local archive directory and download
settings. Original downloads go to `data/pds_level0/` inside this repository.

```bash
python -m moonbroom.mission_pipeline configs/download_mission.json --stage inventory
python -m moonbroom.mission_pipeline configs/download_mission.json --stage download
```

Or run the download script:

```bash
python scripts/download_lp_mission.py
```

Edit `CONFIG_FILE` and `STAGE` at the top of the download script to choose the
configuration and `inventory` or `download` stage. Its default is `download`.
This script does not decode or calibrate telemetry.

## Pseudo-calibrate the downloaded mission data

```bash
python scripts/pseudocalibrate_lp_mission.py
```

The new script's hardcoded `CONFIG_FILE` is
`configs/pseudocalibrate_mission.json`. It reads the downloaded archive's
`inventory.json`, `downloads.json` and original telemetry, then decodes and
pseudo-calibrates MAG, ER 3D and supported PAD data over the mission. It runs
the processing stage only, without downloading the raw archive or performing
magnetotail encounter analysis.

The configuration uses the banks and energy table included in `calibration/`
and kernels in `spice_kernels/`. Missing kernels are fetched automatically from
official NAIF URLs and checked against `configs/auxiliary_data.json`.
Processed MAG/ER shards go to `output/mission_processed/`, which is the
mission input directory in `configs/survey_mission.json`.

The equivalent module command is:

```bash
python -m moonbroom.mission_pipeline configs/pseudocalibrate_mission.json --stage process
```

All scripts resolve their hardcoded configuration paths relative to the
repository, independently of shell cwd.

Rerunning checks existing files and resumes verified work. Processing records
unsupported calibration and missing ephemeris support explicitly. It never
extrapolates a coefficient bank to make all mission rows calibrated. The
configured trajectories end on 30 July 1999, and any later unsupported raw
data remain clock-only under `retain_clock_only: true`.

## Reproduce the manuscript analyses

All nine analysis jobs use included public fixtures. No downloads, missing
input paths or placeholder coefficients are required.

```bash
python -m moonbroom.analysis_suite configs/analysis/pipeline.json
```

This runs the bit audit, MAG holdout, all five ER cohorts, native MAG model
application and independent five-second reaggregation. Each job has its own
editable JSON file under `analysis/`. Results go to
`moonbroom/output/manuscript-config-analysis/`. Completed jobs can be reused
only when their inputs, configuration, code and products still match their
hashes. Use new output directories after changing scientific settings or code.

The March MAG model is expected to fail the configured 0.1 nT acceptance
gate. A completed analysis reports that scientific outcome and does not
promote the rejected coefficients.

## Analyze the mission magnetotail encounters

After pseudo-calibration, run the magnetotail encounter analysis only:

```bash
python scripts/analyze_lp_mission.py
```

Edit `CONFIG_FILE` at the top of the analysis script to choose its configuration.
This script consumes the pseudo-calibrated native MAG shards. It does not run
telemetry decoding or pseudo-calibration.
The default survey prepares its upstream input using `omni_mission.json`.
It downloads the 19 monthly NASA/SPDF standard one-minute OMNI HRO files
covering January 1998 through July 1999, then writes
`data/upstream_boundary_context.csv` and its provenance manifest. To prepare
these inputs separately before running the survey:

```bash
python scripts/prepare_omni_upstream.py
# Equivalent module command:
python -m moonbroom.omni configs/omni_mission.json
```

The upstream CSV contains `utc,pressure_npa,bz_gsm_nt,quality_valid` and
diagnostics. Shue uses published flow pressure (HRO word 28) directly and
GSM Bz (word 19). The documented HRO pressure convention is
`P[nPa] = 2e-6 * Np[cm^-3] * Vp[km/s]^2`; no extra helium correction or
proton-only replacement is applied. OMNI UTC is already arrival at Earth's
bow-shock nose and marks the start of each one-minute average. No Moon shift
or extra L1 propagation is applied. Missing records and fill values remain
invalid minutes, and each record expires at the end of its minute.

The interval in `omni_mission.json` is half open: `start_utc` is included and
`stop_utc` is excluded. Raw ASCII sources are cached in `data/omni_hro_1min/`.
Set `download: false` to require existing cache files. Local source hashes,
timing/pressure conventions and valid/invalid minute counts are recorded;
the survey embeds that provenance in its manifest. Existing outputs are
reused only when settings, code, raw sources, sidecars and output hashes
match. Choose new output paths after a scientific or source change.

For a separately prepared boundary-context CSV, remove `omni_config` from
the survey JSON and supply its frame, timing and composition conventions.
A Moon-shifted SEL event table cannot be substituted for terrestrial GSM
boundary context without explicit conversion. See
[the OMNI guide](../docs/omni_upstream.md) for the format and quality rules.

The provided survey explicitly measures variability in SCD. Geometry still
uses the LP trajectory relative to Earth and the Sun. Optional terrestrial
field-component analysis requires an explicitly framed attitude table. Add
`attitude_csv` and `attitude_settings` with `attitude_frame` and
`max_attitude_age_s` to rotate SCD fields into the constructed GSE basis.

Results go to `output/magnetotail_survey/`. See
[the full guide](../docs/full_mission.md) for data products, scientific limits
and configuration options. This environment's Python network access could
not resolve the PDS host, so the full mission download has not been executed.
