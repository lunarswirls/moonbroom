# Mission-spanning recovery and native 9 Hz MAG

The reusable tools fit and apply empirical corrections recovered from official
PDS raw and calibrated products. Study dates, cohort boundaries, downloaded
inputs, recovered coefficient banks and event interpretation belong in the
calling analysis repository. These models reproduce aspects of the archive
pipeline; they are not the missing flight calibration tables.

## Prepare a study

`discover_lp_products(date)` inventories all merged telemetry files beginning
on a date, matching SUNPULSE files and calibrated MAG/3D/high/low references.
It uses the archive directory listings and preserves filename case.
`max_raw_files` limits a calibration experiment; leave it `None` to inventory
a complete day. Include adjacent dates for boundary accumulations. Network
access happens only on an explicit call. `cache_pds_products` returns byte
counts, SHA-256 hashes and per-file download status. 

`prepare_archive_interval` reconstructs collection UTC using caller-supplied
trajectory kernels and a named receiver, reconstructs spin phase, estimates
local transverse offsets and matches 3D reference distributions. Its output
is a dictionary of DataFrames, including original MAG samples, offset fits,
ER pairs, PAD counts, direction packets and timing diagnostics. Decode
chronological files together. The optional `telemetry_version` override is
explicit and must be justified from packet structure and reference content.

PDS reference spectra with incomplete or invalid energy rows can be skipped
explicitly with `er_3d_reference(..., drop_invalid=True)`. The excluded count
is retained in attributes and preparation diagnostics, strict reading remains
the default. 

## Fit on training dates, validate on other dates

Choose independent dates for each fitted range/response before fitting. Different
MAG ranges may require different held-out dates if their operating coverage
differs. Never use a range's validation samples to fit that range. Freeze gains, response, timing
alignment and quality thresholds before evaluating the held-out date. Keep
energy sweeps and operating versions in separate cohorts. Proposed monthly
validity bounds do not prove continuous accuracy between sampled dates.

The MAG model can use a separate five-second MAG reference after explicit
SSE-to-SCD rotation. `mag_fixed_reference_windows` builds matched windows.
Fit their coefficients with `fit_xy_offsets=False` if transverse offsets
were already removed locally. Inspect residuals by range on another date,
store `validation_pass` and the number/error of those comparisons in the
coefficient table. Unsupported ranges retain their raw samples with missing
calibrated fields. Never transfer a nominal range gain into an unobserved
range merely because its gain ratio looks plausible.

`fit_mag_spin_offsets` fits each range in local windows to a rotating field
plus a linear trend. It separates constant x/y sensor offsets from spin
modulation. Default windows are 60 seconds and need at least 120 valid samples,
3.5 spins of coverage and an identified, conditioned fit. It does not recover
z offsets. `correct_mag_spin_offsets` only interpolates near supported offset
centers, distant gaps remain unsupported. The inferred offsets can absorb
real spin-synchronous signals, so this processing needs separate scrutiny for
wave analysis.

## Preserve the original MAG samples

```python
from moonbroom import (
    decode_level0, read_sunpulse, reconstruct_spin_phase,
    fit_mag_spin_offsets, calibrate_mag_9hz,
)
import pandas as pd

products = decode_level0(raw_paths, timing_kernels=kernels, receiver=receiver)
pulses = pd.concat([read_sunpulse(path, frames=products["frames"])
                    for path in sunpulse_paths], ignore_index=True)
samples = reconstruct_spin_phase(products["mag"], pulses)
offsets = fit_mag_spin_offsets(samples)
mag = calibrate_mag_9hz(samples, model_bank, offsets=offsets)
mag.to_csv(output_path, index=False)
```

`calibrate_mag_9hz` keeps every original real-time sample, its clock, UTC,
raw signed counts, source file/record/sample identity and quality flags.
It applies calibrated gains/rotation and measured spin phase sample by sample.
The telemetry has 18 real-time vectors per two-second frame (nominally 9 Hz)
SPICE-reconstructed UTC can have small clock corrections. It performs no
resampling, interpolation of reference fields, averaging or gap filling.
Burst MAG at 18 Hz is a distinct product and is excluded by this helper.
Unsupported samples remain rows with missing SCD fields.

If `validation_pass` is present, rejected coefficient rows cannot calibrate
samples. Models labelled `offset_model="local spin fit"` require an offset
table. `select_mag_models` can select accepted, nonoverlapping models for a
date. `high_frequency_validated=False` remains explicit: agreement with
five-second PDS averages cannot establish the accuracy of 9 Hz fluctuations.

## ER versions and PAD

Software versions 1–3 use the documented MAGERFMT layouts. Software versions
4 and 5 use an explicitly labelled empirical format-3 compatibility layout,
checked against raw/calibrated distributions on separate mission dates.
Actual software version stays in `telemetry_version`; `telemetry_layout`
records the layout and `layout_basis` records its evidence. An explicit
version override also retains `reported_software_version` and
`telemetry_version_overridden` in accumulation metadata. Unknown versions
remain unsupported. Do not assume a firmware version and an energy sweep
identify each other.

`apply_er_mission_response(counts, bank)` applies independently accepted 3D
models across nonoverlapping time scopes, checking mode, kind, energy,
angular bin and version. It preserves unsupported rows and rejects overlapping
model keys. A spectrum crossing a model or energy-change boundary is not
partially calibrated.

PAD has additional geometry assumptions. `pad_field_directions` uses the
explicit downlinked polar angle for later layouts. For versions 1/2 it uses
the parity-selected raw MAG vector as an approximation to the on-board field.
Packets with unavailable clocks are excluded, when supplied, `mode` excludes
burst directions from real-time PAD processing.

`calibrate_er_pad` pools the recovered 3D anode response, uses the default
16 equal 11.25-degree pitch bins and a phase-averaged circular-anode geometry,
and assumes 1/16 energy duty. It requires at least eight represented anodes,
85% direction coverage and slowly changing polar direction (default standard
deviation at most 10 degrees). Unknown pitch maps and unsupported energies,
versions or time scopes stay missing. This is an approximate angular model.

`pad_omnidirectional` integrates all 16 supported bins with spherical
solid-angle weights. `er_omni_reference` normalizes the official 5-second high-
resolution and 80-second low-resolution total-flux products.
`fit_pad_normalization` estimates one aggregate flux scale and chooses a
reference-window offset using training data only. Freeze these choices and
use `compare_pad_omni` on another date. Reference-window alignment never
changes reconstructed sample UTC. Reject poorly predicted energies.

`apply_pad_mission_response` applies accepted aggregate normalizations with
the matching 3D response cohort. The banks must retain cohort identities.
`omnidirectional_model_validated` concerns total flux only,
`pitch_bin_response_validated=False` always remains explicit. The archive's
high/low products do not independently identify individual pitch-bin
sensitivities. Cut bins and burst response have no independently recovered
mission bank and must not inherit a PAD or real-time 3D calibration.

## Primary documentation

- [Official MAGERFMT instrument format](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0019/document/magerfmt.pdf)
- [PDS 3D flux processing](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/document/data-3deleflux-desc.txt)
- [PDS MAG processing](https://pds-ppi.igpp.ucla.edu/data/lp-mag-calibrated/document/data-lunarcrds-desc.txt)
- [Official LP trajectory kernels](https://naif.jpl.nasa.gov/pub/naif/LPM/kernels/spk/)

## Diagnose missing native MAG calibration

`mag_calibration_diagnostics` adds `calibration_model_available` and a
mutually exclusive `calibration_failure_reason`. `calibrate_mag_9hz` now
includes these fields automatically. Failure precedence is diagnostic mode,
ADC limit, frame mismatch, range transition, other telemetry quality, missing
raw values, UTC, spin phase, local offsets, then range-model support. Original
flags remain available for overlapping-cause counts. A missing model is split
into an untrained range, insufficient/unidentified training, insufficient or
failed independent validation, and out-of-scope time. Keep the original sample
set fixed when comparing before/after recovery, extra training data must not
change the denominator of the reported improvement.

`prepare_mag_interval` needs raw merged files, matching SUNPULSE, timing kernels
and a receiver, but no ER reference product. Its `mag_only=True` decoder path
unpacks real-time MAG in batches. It omits ER/burst products and obtains spin
phase from SUNPULSE. Tests compare its original counts, sample clocks and
quality flags with the complete decoder.

Strict overlap handling remains the default. Explicit `overlap_policy=
"identical_mag"` is restricted to MAG-only output: adjacent records must share
the counter, frame header and all MAG/ER payload bytes, and their receive times
must agree within 0.1 second. The first record is retained and every removed
record/time difference is saved in `overlap_records`. Other-instrument or
engineering differences are not silently accepted for full telemetry decoding.

With `sunpulse_strict=False`, odd-counter SUNPULSE records remain in the
returned `sunpulse` table with `counter_valid=False` and `timing_valid=False`.
Their pulse clocks stay missing. Counter-order violations still raise. These
options allow valid portions of imperfect archive files to be processed
without inventing pulses or discarding their source-quality evidence.
