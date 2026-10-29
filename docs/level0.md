# LP Level-0 MAG/ER decoding

`decode_level0` reads official LP merged telemetry files and returns a mapping
of named pandas DataFrames. Inputs are chronological `.LBL` or `.B` paths.
Decode a continuous sequence together so that digital subcom, ER
accumulations and burst markers can cross file boundaries.

```python
from moonbroom import decode_level0, write_level0_tables

products = decode_level0([
    "data/level0/M9906003.LBL",
    "data/level0/M9906008.LBL",
])
mag = products["mag"]
er = products["er_counts"]
manifest = write_level0_tables(products, "output/level0")
```

The implementation follows the official [Science Data Interface
Specification, section 2.5](https://pds-ppi.igpp.ucla.edu/archive1/LP_1201/DOCUMENT/PROJECT/LVL0SIS.PDF)
for the merged envelope and the revised [MAG/ER Telemetry Format, sections
2.1–2.6](https://pds-ppi.igpp.ucla.edu/data/LP-L-MAG-5-SURFACE-FIELD-MAP-V1.0/DOCUMENT/INST/MAGERFMT.PDF)
for instrument packets. File overlap and spacecraft-counter rollover handling
also account for the [Level-0 archive errata](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/errata.txt).

## Outputs

| Product | Contents |
| --- | --- |
| `frames` | Every merged record, source location, unwrapped VCDU, ERT, reception quality, header flags, original bytes and optional UTC correlation |
| `input_files` | Source labels, record counts and any trailing bytes retained as a quality diagnostic |
| `mag` | Real-time and burst 12-bit component codes, signed counts, range, diagnostic mode, sample clocks, optional UTC, spin phase and quality flags |
| `er_counts` | Long-form energy/sample/angle channel codes, decompressed count bounds, prescale, saturation possibility, accumulation bounds and completeness |
| `er_accumulations` | Version, mode, packet coverage, header counters, pitch-map version, timing validity and links to source frames |
| `er_angles` | Format-3 (software 3–5) and burst magnetic directions, or version-1/2 sample-parity bits, indexed by quarter spin |
| `er_bits` | All 72 real-time ancillary bits in explicit bit order |
| `digital_subcom` | Decoded status, spin phase/period, sweep settings and nominal digital housekeeping conversions per major frame |
| `er_packets` | Original real-time/burst ER payloads linked to accumulations and source records |
| `burst_packets` | Original playback records, slot, type and counter-gap flags |
| `burst_markers` | Collection and transmission major-frame counters, reconstructed collection clocks, burst status and snapshot subcom |
| `control_tables` | Original PLL, MAG, burst and ER control-table bytes |
| `memory_dumps` | Dump address and original memory bytes |
| `issues` | Missing timing, partial accumulations, unknown versions, missing packets and frame inconsistencies |

Raw engineering/spectrometer envelopes remain in `frames`; their non-MAG/ER
instrument contents are not interpreted. Control-table contents whose field
definitions are only referenced through a separate commanding document are
preserved as bytes rather than assigned invented parameter meanings.
Reserved burst types and fill packets are retained in `burst_packets`.

`write_level0_tables` writes one CSV per product and returns a manifest. Raw
byte columns are hexadecimal strings in those CSVs. In-memory byte columns
remain bytes. UTC columns include their timezone offset. Large sequences can
produce large long-form ER tables, select contiguous groups of input files
that fit the available memory.

## Timing

The default is **clock-relative timing**, with missing UTC. `frames.ert` is
the archive's Earth receive time. The decoder never relabels it as spacecraft
event time and never applies the delayed-playback correction a second time.
`ert_iso` retains an ERT leap-second label exactly. Its pandas `ert` value is
missing at that instant because pandas cannot represent second 60.

For absolute UTC, supply either a clock-correlation DataFrame or explicitly
computed one-way light times:

```python
import pandas as pd
from moonbroom import decode_level0

correlation = pd.read_csv("data/spacecraft_clock_correlation.csv")
products = decode_level0("data/level0/M9906003.LBL", clock_correlation=correlation)
```

The correlation columns are `vcdu_unwrapped` and `utc`, representing UTC of
the spacecraft frame header. At least two increasing anchors are required.
Frame UTC is interpolated only within the supplied anchor interval, allowing
clock drift. Instrument sample offsets use the specified nominal seconds per
frame and the measured instrument spin period.

Alternatively, `light_time_s` accepts a nonnegative scalar or one value per
deduplicated merged record. It sets `frame_utc = ert - light_time_s`.
Compute that propagation correction from appropriate station/spacecraft
geometry, the decoder has no hardcoded Earth–Moon delay or mission epoch.
This lower-level option subtracts only the supplied correction, callers must
include any data-system delay in their correlation. Prefer the new
`timing_kernels` and `receiver` options for a physically reconstructed frame
pulse, including the documented two-second data-system delay. See
[timing and calibration](calibration.md). With this option all sample UTC
values are recomputed by bounded interpolation in SPICE ephemeris seconds,
including leap seconds. Burst collection times outside the available clock
correlations remain missing. The `timing_kernels` output records input paths.

Real-time MAG clocks include the published four-second pipeline delay and
the mid-time of each pair averaged to 9 Hz. Burst MAG uses 18 Hz sampling,
the marker's collection time, the MAG status frame and the half-frame bit.
Burst collection time is reconstructed from the modulo-65536 difference of
collection/transmission major counters. Playback time is never used as burst
collection time. A missing marker or unmarked major-frame wrap masks timing
until a new marker is available.

ER bounds use decoded phase, spin period and header half-spin index. The
previous major-frame context is used when required by the timing rule.
Missing phase/period/header data gives missing bounds. High-rate PAD, slow
PAD, cut and 3D bounds are handled separately. The `interval_s` column is a
collection interval, **not detector live time**. `nominal_integration_s` is
provided for 3D bins, including polar-bin summation. PAD effective exposure
requires instrument response and onboard binning information.

`read_sunpulse` reads the separate eight-byte sun-pulse records. It handles
the preceding-four-second window, major-frame wrap, no-pulse code and
measured versus estimated source flag. Pass the decoded `frames` table to
align rollover and interpolate pulse UTC within its correlated interval.
Pulse uncertainty is retained independently of source flag.

## Counts, layouts and calibration

MAG outputs preserve unsigned excess-2048 codes in `bx/by/bz` and provide
`bx_signed/by_signed/bz_signed`. No nominal instrument gain is silently
applied. Calibration-mode samples, ADC endpoints, range changes and frame
number mismatches are flagged. `science_valid` summarizes these instrument
checks, it does not certify calibration, timing coverage or scientific use.
Use `calibrate_mag` with the actual gain/offset table, followed by the
appropriate alignment, despinning and attitude conversion.

`decompress_er_counts` returns quantization intervals, not exact original
counts. It accounts for truncation in both compression and prescaling.
The largest code is flagged as potentially saturated with an infinite upper
bound. Long PAD samples use the header's per-band factor-of-eight prescale,
high-rate PAD, cuts, burst PAD and 3D data do not. Total/rate header counters
use their separate factor of sixteen. When a needed prescale header is
missing, codes remain available and restored count bounds are missing.

Telemetry versions 1, 2 and 3 have independent documented layouts. Software
versions 4/5 use an empirical format-3 compatibility layout, with their actual
version and layout provenance preserved in accumulation metadata. The version comes
from the digital subcom, not the pitch-map version in the ER header. A caller
may supply `telemetry_version=...` when the version is independently known.
Unsupported or unavailable versions leave packets intact and produce an
issue. Pitch-bin remapping, count-to-flux response, detector corrections and
calibrated energies remain separate from decoding.
Digital subcom retains the raw ER-dependent fields when ER is off, flags
`er_parameters_valid=False`, and prevents those stale phase/period settings
from producing collection times.

`er_angular_bins()` returns the 88-bin phase-space mapping.
`er_energy_bins(sweep_max, sweep_rate)` evaluates the published continuous
sweep recurrence and labels the results as **nominal** energy bounds. It
does not assume undocumented DAC rounding or an instrument response curve.

## Gaps and printed-table inconsistencies

Exact duplicate boundary records are removed. Conflicting counter overlaps
raise an error. Missing merged records conservatively split real-time ER
accumulations, missing burst playback counters split burst accumulations.
Partial ER groups retain their received codes with missing bytes masked.
They are never padded with zero electron counts. Structural corruption raises
by default, `strict=False` retains invalid headers with flags and excludes
them from science decoding. Truncated trailing bytes are reported in the
standalone reader's attributes, label length mismatches still raise.

For a labelled partial final frame, `allow_incomplete_final_record=True` on
`read_merged_telemetry` or `decode_level0` explicitly recovers only complete
records while preserving strict header checks. The label must count exactly
one additional record, and the tail must contain its valid ten-byte frame
header. No bytes are padded or interpreted as science data. Attributes and
the decoder's `input_files` table retain `complete_records`, `label_records`,
`trailing_byte_offset`, `trailing_hex`, `incomplete_final_record` and
`missing_bytes`. Standalone decoding defaults to rejecting such incomplete
records. The mission driver enables this recovery after checksum verification
and writes a separate [source anomaly report](full_mission.md).

The implementation makes the following explicit resolutions of printed
inconsistencies, verified by byte coverage, dimensions and archive data:

- ER sample indices follow each energy's occurrence order. Some printed
  sample labels repeat or retain numbering from an older layout
- Burst packet 50 continues at 3D byte 1144, consistent with the preceding
  packet and the total byte count, the printed 1044 is inconsistent
- A dump's two-byte address occupies bytes 85–86, followed by the 81 memory
  bytes at 87–167, the printed data range overlaps the address
- ER phase delay uses `(phase % 2048) / 4096 * period_s`, using the declared
  phase and period units. The printed decimal coefficient in seconds differs
  by a factor of 1000
- Sun-pulse uncertainty uses 1/1800 second, consistent with the SIS and the
  label's decimal resolution, the label's isolated `1/8000` fraction disagrees

Ancillary bits are exposed with the lower frame-code bit first, then the
higher bit, for each frame in order. The original high code bits remain in
the raw frame. The test-pulser command itself is not inferred from an
unspecified external commanding-table definition.

## Validation

Tests cover independent packed-bit examples, compression boundaries,
prescale rules, all ER layouts, burst packet ordering, timing, rollover,
file boundaries, overlapping records, missing markers and CSV export.
The public regression excerpts and their provenance are in
`tests/fixtures/README.md`. The independently archived sun-pulse times also
check subcom phase alignment. Tests require no network or private data.

The full March 1999 archive file was also decoded locally. Validation of
telemetry decoding does not replace gain/response calibration or science
validation of an analysis built on the decoded measurements.
