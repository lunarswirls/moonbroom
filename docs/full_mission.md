# Manuscript analyses and complete LP MAG/ER recovery

This checkout implements the analyses in `frontiers.tex` from
the LP MAG recovery project. It includes the earlier recovery/validation
additions and a new mission downloader, bounded MAG/ER processing driver and
mission survey runner. No mission-wide numerical results are assumed.

## Run the archive, calibration and encounter steps

Install into an environment with Python 3.11 or newer:

```bash
python -m pip install -e '.[geometry,test]'
python scripts/download_lp_mission.py
python scripts/pseudocalibrate_lp_mission.py
```

The download script uses `configs/download_mission.json` and defaults to the
`download` stage. The pseudo-calibration script uses
`configs/pseudocalibrate_mission.json` and runs the `process` stage on the
downloaded archive. Edit `CONFIG_FILE` in each script to select a different
configuration without command-line arguments. `analyze_lp_mission.py` is
reserved for the magnetotail encounter analysis described below.

Both mission examples use the MAG, ER and PAD bank snapshots included in
`calibration/`, and repository-local timing kernels. Configuration paths resolve
relative to their JSON files. Missing kernels are fetched from official NAIF
URLs with pinned SHA-256 checksums using `configs/auxiliary_data.json`. Large
kernels are kept in ignored `spice_kernels/`, so a fresh clone obtains them
without a dependency on another repository. Downloads and products also stay
inside the checkout, in ignored `data/` and `output/`. See
[configs/README.md](../configs/README.md) for the ready-to-run command sequence.

The included model banks retain their own training dates, held-out tests,
acceptance flags and proposed validity intervals. Their acceptance limits are
not the manuscript's 0.1 nT/5% demonstration limits. The small public checks
below reproduce those manuscript examples separately. Applying a monthly
bank does not establish continuous independent validation of that month.

Together, the download and pseudo-calibration commands perform these stages:

1. Discover all 19 `LP_0001`–`LP_0019` volumes from the official
   [PDS archive](https://pds-geosciences.wustl.edu/lunar/prospectorcd/).
   Download each published MD5 catalog and resolve the actual HTTP filename
   case from directory listings. Inventory every original cataloged volume
   file, including all merged telemetry, SUNPULSE, geometry, commands,
   documentation, errata and indexes. No calibrated-product start/stop date
   truncates the raw archive. Merged files necessarily contain the other LP
   instruments as well as MAG/ER.
2. Stream and retry downloads with four workers. Resume partial HTTP transfers
   when supported, restart when the server ignores Range, check published MD5
   hashes, and record independent SHA-256 hashes. A failed download prevents
   the processing stage from declaring complete input coverage. Previously
   downloaded files are checked before reuse.
3. Decode every merged binary in chronological order in shards of 1800
   records, approximately one hour. Include at least 128 surrounding records
   on either side for digital subcom, ER accumulation, timing, pulse and offset
   context. Neighboring source files provide context at file boundaries.
   Track the mission-wide 24-bit counter rollover epoch across files and shards.
   Keep local offset windows on the original counter grid used by the supplied
   banks, then express their centers in the unwrapped mission clock.
   Memory use depends on a shard and its context rather than mission length.
4. Audit real-time MAG against an independent expansion of the packed bits.
   Reconstruct collection UTC with the supplied kernels and explicit receiver.
   Reconstruct measured SUNPULSE phase and local transverse offsets. Apply
   independently accepted MAG models to the original nine-hertz samples.
   Preserve original 18 Hz burst MAG separately in `native_burst_mag` shards.
   Burst collection markers provide phase and hardware context. Explicit burst
   models are used by default; `allow_realtime_burst_transfer: true` permits
   labelled fallback transfer of accepted real-time coefficients. The default
   is false, so existing real-time banks leave burst fields unsupported.
   Offset fits stay within their burst and hardware cohort. Each shard writes
   `mag_hardware_checks` with support and review counts by mode/range/state.
5. Decode real-time and burst ER, retain compressed codes and count bounds,
   assign the official 15-step energy table, and apply accepted ER response
   models only within their mode/version/energy/angular/time support. Optional
   `pad_bank` applies the existing approximate PAD geometry and independently
   validated aggregate normalization in a separate product. Burst and cut
   bins receive no unsupported calibration by inheritance.
6. Write compressed CSV shards, source/model/code provenance, bit-audit results,
   product hashes, per-shard failures and mission coverage totals. Successful
   shards resume only when configuration, source, bank, kernel, code and output
   hashes agree. Failed shards are retried on the next run.

You can run stages separately:

```bash
python -m moonbroom.mission_pipeline configs/download_mission.json --stage inventory
python -m moonbroom.mission_pipeline configs/download_mission.json --stage download
python -m moonbroom.mission_pipeline configs/pseudocalibrate_mission.json --stage process
```

Inventory fetches small catalogs/listings and no telemetry binaries. Download
mirrors all original volume files. Processing requires the completed verified
inventory and download manifest, and does not fetch raw archive files.
It may fetch missing auxiliary SPICE kernels. The archive is large, and long-form ER plus raw-byte CSV products
need substantially more disk than the downloaded binaries. Start with enough
storage for both. The driver does not delete original downloads.

The checksum-verified archive contains a truncated final record in
`lp_0012/merged/D35X/m9835916.b` (1998-12-25). Its label declares 5,683
472-byte records, but the published binary contains 5,682 complete records
and 44 bytes of the final record. The file matches the official MD5
`142b121e682fe310a26af914d05b2cfc`. Downloading it again does not restore the
missing 428 bytes. This discrepancy was found by checking the binary against
its label and checksum catalog, rather than reported in the archive errata.

Mission processing defaults to `allow_incomplete_final_record: true` after
verifying each source against the official inventory and local SHA-256.
It decodes the complete records and preserves the final fragment as hex in
`archive_record_anomalies.csv`, `run.json` and shard `input_files` tables.
The report includes the original record count, complete count, byte offset,
received and missing byte counts, and source hashes. It never pads or decodes
the fragment. Strict complete-record header validation remains enabled.
Recovery requires a matching label that counts exactly one partial final
record with a valid frame header. Other label-length mismatches still fail.
Set this option to `false` to reject incomplete final records as well.
`mission_status.json` distinguishes successful processing of available complete
records from source completeness through `source_records_complete`,
`incomplete_source_records`, `source_trailing_bytes` and `missing_source_bytes`.

## Archive quality and ancillary precedence

Mission processing automatically reads the latest available cumulative
`INDEX/OUTAGES.LBL` and `GEOMETRY/ATTITUDE.LBL`, and all downloaded command
tables. It writes their normalized contents to `archive_outages.csv`,
`archive_attitude.csv`, `archive_commands.csv` and `maneuver_intervals.csv`.
`mager_command_actions.csv` selects MAG/ER requests with their action category,
original parameters, transmission status, station and timing evidence.
`run.json` records the selected sources, settings and SUNPULSE selection
decisions, alongside the existing input and code hashes. After upgrading,
choose a new output directory: previous processed shards have different code
and are not silently reused.

SUNPULSE selection occurs before decoding or phase interpolation. As directed
by the [archive errata](https://pds-ppi.igpp.ucla.edu/data/LP-L-ENG_GRS_NS_APS_MAG_ER-1-MDR-V1.0/ERRATA.TXT),
the `LP_0019/SUNPULSE/DXXX/S9812114.B` replacement supersedes its `LP_0005`
copy, regardless of input order. Identical copies of other products are used
once; conflicting copies without an explicit replacement rule raise an error.
The same selection helper is used by interval calibration preparation.

Outages are matched against merged **ERT**, including in clock-only runs.
Each two-second received frame is tested for overlap. The official table has
minute precision; the default conservatively includes the entire stated stop
minute. Descriptions containing NOISE/NOISY set `archive_noisy`; other
unresolved outages set `archive_gap`. Original descriptions, statuses and
interval identities remain available. Only explicit RECOVERED, CORRECTED or
FILLED statuses clear the exclusion. Any affected packet flags its entire ER
accumulation, including across shard boundaries and for burst playback; the
outage is not incorrectly compared to burst collection UTC.

Command-based maneuver candidates require a transmitted (`CONFIRM`) thruster
selection followed by `EXEC`. Heater executions, ABORT and SKIP entries do
not start candidates. Identical duplicate command records are collapsed.
`STOPFIRE` closes an active candidate; `SETCMDREG` only resets parameter
registers and does not prove firing stopped. `STOPFIRE` without a
known start still supplies a settling interval. An unclosed candidate remains
open until `STOPFIRE`, with no invented end time. These are conservative
quality flags, not a reconstructed firing history: CONFIRM means transmitted,
and errata report examples of commands that did not execute. Reconstructed
command arrival SCET is used where supported, otherwise DSN transmission UTC
is retained as an explicit proxy. Times are padded by five seconds as a policy. Collection intervals
that overlap a candidate or the following eight-hour settling interval are
excluded from science calibration. The eight-hour choice follows the upper
end of VOLSIS's six-to-eight-hour nutation description; it is not a measured
settling time for each maneuver.

MAG/ER requests are classified using LVL0SIS Appendix B: electronics-side and
ER power, heaters, processor/circuitry resets, cover/boom actions and memory or
table operations. Readouts, buffer staging and selecting the boot image for a
future reset are recorded without being treated as immediate science changes.
Unknown MAG/ER-like commands remain unmapped. The documented failed
`MERHTROFF` request in C9802342 at 1998-01-23T00:49:53.200 is explicitly marked
as not executed; its confirmed transmission does not create a transition.

`mager_command_transition` flags candidate science-changing requests with
configurable padding. If a hardware request is immediately followed by a
confirmed `EXEC`, the candidate interval extends through that EXEC. This
adjacency is recorded as evidence, not a demonstrated execution association.
MAG-only and ER-only commands affect their respective science rows. These
flags gate calibration and local MAG offset fitting. `mager_command_cohort`
identifies the preceding request; local offset fits reject mixed request
cohorts and offset interpolation stays within a cohort. Variability windows that cross request
cohorts are rejected. These are command-history cohorts, not proof of stable
hardware configuration or independently valid calibration cohorts. Memory and
table parameters are retained verbatim: the separate MAG/ER Commanding
document referenced by MAGERFMT is still needed to interpret their internal
tables, anode masks, test-pulser settings and other detailed control values.

Orbit-average attitude is held backward for at most 48 hours by default.
The table is in `ECLIPJ2000` (LVL0SIS section 2.1.2). Output includes support,
age, staleness, the approximately one-degree uncertainty stated in VOLSIS,
and `attitude_sun_sensor_only` for unambiguous periods documented in ERRATA.
That flag is not an exhaustive sensor-quality reconstruction: the ambiguous
May 1998 date range in ERRATA is not guessed. `attitude_quality_valid` requires
fresh support outside candidate burns and settling periods. Inertial survey
rotation honors this flag. Missing attitude alone does not invalidate SCD
counts or independently measured spin phase. Known orbit-average spin rates
at or above 15 rpm invalidate SUNPULSE-based phase support, as required by
ERRATA.

The default settings are explicit in `configs/pseudocalibrate_mission.json`:

```json
"archive_quality": {
  "outage_stop_padding_s": 60,
  "maneuver_padding_s": 5,
  "mager_command_padding_s": 5,
  "maneuver_settling_s": 28800,
  "max_attitude_age_s": 172800
}
```

`archive_quality_assessed`, `maneuver_quality_assessed` and
`science_quality_assessed` distinguish checked evidence from unavailable
ancillaries or UTC. A true `science_quality_valid` means no identified archive
or maneuver/command-transition failure; use the assessed flags when requiring verified coverage.
Known failures gate MAG science validity **before local offset fitting** and
gate ER count rates and flux. Digital codes, count bounds and source identities
remain unchanged. Attitude support gates inertial rotation separately from
calibration in the despun spacecraft frame. Per-shard statistics count noisy,
maneuver/settling and attitude-supported MAG rows.

## Scientific inputs and completeness

The main inputs are `kernels`, `receiver`, `mag_bank`, `er_bank`, `energy_bins`
and `energy_time_basis`. The energy table in `calibration/er_energy_bins.tab`
is the unmodified official low-resolution companion table containing all 15
energy steps. The archive's UTC and Unix energy-change columns disagree in
some records. Choose the convention explicitly, then hold it fixed.

Kernels must support Earth–LP reception geometry over the intended raw data
span. `EARTH` is an explicit geocentric receiver approximation. The configured
trajectory filenames end on 30 July 1999. Any raw data outside actual
kernel coverage are still recovered as clock-only digital products when
`retain_clock_only` is true. Their UTC/calibration remains unsupported, and
the exception is saved in the shard status. Other errors fail the shard.
Set `retain_clock_only` to false to require timing support for every shard.

`clock_only: true` and `allow_uncalibrated: true` explicitly allow digital
recovery without trajectory kernels or coefficient banks. These options do
not create calibrated measurements. Missing, rejected and out-of-scope models
retain original rows with invalid/missing calibrated fields. Native MAG rows
carry coefficient hashes, cohort, source identity and failure reasons.
Observed hardware state is attached at collection time, with transition brackets
and missing/stale snapshots excluded from calibrated fields. Optional coefficient
state constraints are enforced in both modes. Unscoped legacy coefficients remain
explicitly unscoped; an applicability check is not reference validation.
`native_mag` remains the nine-hertz input to existing mission surveys. Burst rows
are counted separately in `burst_mag_rows` and `calibrated_burst_mag_rows`.

`identical_mager` is an explicit overlap policy for a MAG/ER-only science
recovery: adjacent equal-counter records must have identical frame headers,
all MAG/ER bytes and tail bytes, and receive times within 0.1 s. Differences
in other-instrument bytes are recorded with both original record hashes.
Original complete binaries remain in the mirror. Differing MAG/ER payloads
are rejected. Choose `exact` for whole-record equality or `identical_payload`
to also require every physical instrument payload byte to agree. The older
`identical_mag` option remains restricted to the MAG-only decoder.

An ER accumulation belongs to the shard containing its first retained packet.
Its persistent ID combines mode, original source filename and record index.
All count cells for that accumulation stay together, including cells collected
before the packet was downlinked. Shard-local accumulation/frame IDs are
accompanied by the original source identities. Missing packets stay missing.
Long interruptions beyond the context halo retain incomplete-accumulation
flags. Increase the halo for such intervals instead of filling missing data.

Inspect `mission_status.json` and `processing_status.json`. Complete archive
download, successful digital processing, UTC support and valid
pseudo-calibration counts are separate quantities. `full_mission_processed`
requires all 19 volume inventories and every processing shard. It never means
all rows are calibrated. `high_frequency_validated` remains false.

## Analyses required by the manuscript

| Manuscript analysis | Command/API | Main outputs |
| --- | --- | --- |
| Independent bit extraction and sample identity | `analysis_cli` task `decode` | Per-sample exact audit, native MAG |
| SUNPULSE, offsets, independent five-second MAG reference | task `prepare_mag` | Offset fits and matched reference windows |
| Independent ER distribution preparation | task `prepare_er` | Matched pairs, native counts, timing diagnostics |
| Blocked MAG transfer with fixed acceptance limits | task `mag` | Coefficients, biases, RMSE, support, split checks |
| ER transfer by version/energy/angular response | task `er` | Count/flux bounds, relative errors, acceptance |
| Actual native-output reaggregation | task `native_compare` | Five-second vector residuals and component metrics |
| Native/five-second variability and region context | `mission_survey` | Window metrics, candidate intervals, exposure, association |

Run the fixture analyses used for the manuscript with:

```bash
python examples/validate_recovery.py --output output/manuscript-regression
python -m pytest tests -q
```

These checks exercise the 2304-sample bit audit, the failed March MAG
demonstration gate, the five ER cohorts and the January native-output
reaggregation. They do not substitute for a mission-wide science evaluation.
See [recovery_validation_survey.md](recovery_validation_survey.md) for the
fit and split settings and the meanings of the distinct temporal windows.

`prepare_er` uses `raw_paths`, `sunpulse_paths`, `reference_path`,
`reference_source`, `kernels`, `receiver`, `output_directory` and optional
`settings` passed to `prepare_archive_interval`. Use separate configurations
for the training and validation dates. Task `er` consumes their `pairs.csv`
files. `native_compare` uses `samples_csv`, `reference_windows_csv`,
`output_directory` and optional `settings` for `compare_native_mag_reference`.
`analysis_cli` paths retain their existing caller-working-directory convention.

## Mission magnetotail survey

```bash
python scripts/analyze_lp_mission.py
```

The default configuration prepares its upstream context from one-minute
NASA/SPDF OMNI using `configs/omni_mission.json`: published flow pressure
and GSM Bz, with bow-shock-nose arrival timestamps and no additional shift.
Run `python scripts/prepare_omni_upstream.py` to prepare it separately.
See [OMNI upstream inputs](omni_upstream.md) for pressure composition,
quality handling, cache verification and offline preparation. A custom
upstream CSV can still supply `utc`, `pressure_npa`, `bz_gsm_nt`,
`quality_valid` when `omni_config` is omitted; document its timing and
composition conventions explicitly.
Provide trajectory kernels spanning the survey, or a `geometry_csv` at window
centers with `utc`, `X_GSE_RE`, `Y_GSE_RE`, `Z_GSE_RE`.

The runner streams native output shards, carrying unfinished windows across
shard boundaries. It preserves invalid calibrated rows in the coverage
denominator, separately counts rows lacking UTC and rejects coefficient
transitions. The example explicitly computes SCD variability. For terrestrial
field components, add `attitude_csv` and `attitude_settings` containing
`attitude_frame` and `max_attitude_age_s`. The runner then rotates SCD vectors
to the explicitly constructed GSE approximation before calculating variability.

The outputs are `windows.csv`, `encounters.csv`, `region_summary.csv`,
`association.csv` and `manifest.json`. They implement the draft's conservative
Shue envelope, gap-separated candidate catalogues and day-block uncertainty
intervals. Invalid attitude, stale upstream input and rejected windows remain
visible. Modeled inclusion is contextual evidence, with far-tail model
extrapolation and unresolved lunar wake/crustal/charging effects. It is not a
physical plasma-region identification by itself.

## Source and execution status

The [PDS volume guide](https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0019/document/volsis.htm)
documents the monthly volumes, merged streams, indexes and SUNPULSE products.
No decoder or calibration implementation is downloaded or executed from PDS.

The implementation has been exercised with public excerpts, synthetic geometry
and simulated HTTP transfers, including interrupted downloads and checksum
failures. This task environment cannot resolve the PDS host from the Python
runtime, so a live full-mission transfer and processing run was not performed.
