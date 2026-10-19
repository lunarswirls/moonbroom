# Derive effective calibration from PDS reference products

`archive_calibration.py` fits empirical corrections from paired raw telemetry
and calibrated PDS measurements. Coefficients preserve their source, proposed
validity interval, training coverage and unsupported groups. They are labelled
**archive-derived**, rather than official flight calibration coefficients.
Keep study file choices, time intervals, derived model artifacts and scientific
interpretation in the calling analysis repository.

## Match raw and calibrated ER distributions

```python
from moonbroom import (
    decode_level0, er_3d_reference, match_er_archive, read_pds4,
    fit_er_archive_response, apply_er_archive_response,
)

reference = er_3d_reference(read_pds4(reference_path), source=reference_product_url)
training = match_er_archive(training_counts, reference)
validation = match_er_archive(validation_counts, reference)
coefficients = fit_er_archive_response(
    training, valid_from=scope_start, valid_to=scope_stop,
    source=reference_product_url,
)
predicted = apply_er_archive_response(validation, coefficients)
```

`training_counts` and `validation_counts` come from independent `decode_level0`
sequences with absolute UTC reconstructed. Use one telemetry mode per matching
call. Accumulation identities must be globally unique within each input table.
Optionally join `telemetry_version` from `er_accumulations` before fitting,
this makes format version an additional application constraint.

`er_3d_reference` uses native Unix `Spectrum Time`. The PDS `Time` field advances
artificially through energy rows and must not identify individual spectra.
Zero native time is treated as unavailable. Every spectrum requires 15 energy
rows and 88 flux bins. The normalizer also preserves its accompanying SCD MAG
vector for the MAG fit.

`match_er_archive` proposes a unique nearest spectrum within `tolerance_s`
(default ten seconds), then independently checks the zero-count pattern and
rank correlation of count-derived intensities against reference flux. At least
88 comparable cells, 90% zero-pattern agreement and 0.95 rank correlation are
required by default. A match never changes sample UTC. Reused reference spectra
raise an error. Raw rows remain present when matching fails, with
`match_valid=False`. Exclude such rows from both fitting and error statistics.

The ER model estimates effective geometric response separately by mode, kind,
energy step and angular bin, plus format version when supplied. With exposure
`t`, center energy `e`, decompressed count `c` and PDS flux `f`:

`g_effective = c / (t - c * dead_time_s) / (f * e)`

Defaults follow the documented PDS deadtime convention: 0.3 microseconds and
masking above a correction factor of 1.25. Only matched positive, unsaturated
counts and positive finite flux constrain the fit. Log-compression bounds are
propagated into response bounds, the estimate uses the median across paired
observations. The table retains median absolute deviation, counts and training
times. A group needs at least eight observations by default. Energy-sweep
changes must be split into separate cohorts.

Application requires matching energy to 0.01 eV, matching keys and collection
bounds within the proposed validity interval. Sparse groups, unobserved bins,
other modes/versions and out-of-scope times remain missing. The recovered
response incorporates the assumed exposure and reference pipeline; these
comparisons cannot identify absolute detector efficiency and live exposure
independently. The current reference workflow constrains **3D** bins. It does
not infer individual PAD responses from an omnidirectional reference product.

## Derive MAG corrections in a common frame and averaging interval

```python
from moonbroom import (
    mag_archive_windows, fit_mag_archive_calibration,
    evaluate_mag_archive_calibration, apply_mag_archive_calibration,
)

training_windows = mag_archive_windows(training_mag, training)
validation_windows = mag_archive_windows(validation_mag, validation)
mag_coefficients = fit_mag_archive_calibration(
    training_windows, valid_from=scope_start, valid_to=scope_stop,
    source=reference_product_url,
)
comparison = evaluate_mag_archive_calibration(validation_windows, mag_coefficients)
corrected_samples = apply_mag_archive_calibration(validation_mag, mag_coefficients)
```

MAG samples require signed counts, range, collection clock, science flags and
SUNPULSE-derived `spin_phase_deg`/`spin_phase_valid`. Reference vectors are the
calibrated SCD field accompanying each ER 3D distribution. Matched raw collection
clocks define the averaging interval, avoiding a second assumption about the
native reference timestamp. Windows with invalid phase/science samples, mixed
ranges or less than 90% nominal 9 Hz coverage are excluded.

The model applies a common planar gain and rotation, a separate z gain, and
three sensor-frame offsets. For raw signed vector `c`, it fits the six
parameters `a,b,gz,ox,oy,oz` in:

```text
sensor_x = a*cx - b*cy + ox
sensor_y = b*cx + a*cy + oy
sensor_z = gz*cz + oz
```

The model then rotates x/y by the measured spin phase into SCD and averages
over the reference interval. Fitting uses robust reweighting, column scaling,
full-rank checks and a condition-number limit. A range requires at least 20
valid windows by default. Coefficients include an equivalent gain/zero/rotation
table, model parameters and fit diagnostics. Sparse or unidentified ranges
remain explicitly unsupported.

These measurements do not uniquely recover the team's general alignment
matrix, minute-dependent offsets or any removed high-frequency signal.
`apply_mag_archive_calibration` applies the effective correction to original
raw samples and sets `high_frequency_validated=False`. It does not interpolate
the calibrated reference onto the raw sampling grid. Validate comparisons at
the reference resolution before interpreting native-rate output.

For a separate check against the five-second MAG product,
`mag_sse_reference_to_scd` rotates the archived SSE field using supplied attitude
angles and kernels. The input `attitude_frame` must be explicit: original labels
contain inconsistent prose about ecliptic versus equatorial angles. The orbit-
average attitude is an approximation. `mag_fixed_reference_windows` builds
the same averaging windows with an explicit candidate `time_offset_s`. Choose
any empirical alignment using training data only, freeze it, then test on a
later interval. This alignment does not alter reconstructed UTC or establish
an official clock solution.

## Validation and application

Split by contiguous telemetry interval before fitting. Randomly dividing cells
of the same spectrum would leak information between training and validation.
Report unsupported ranges/bins, positive-flux error statistics and common-frame
MAG residuals on separate reference distributions. Keep model scope narrow
until separate dates, gain ranges, energy sweeps and operating modes have been
validated. Coefficient validity fields are proposed scope constraints, does not
guarantee mission-wide accuracy.

Public PDS-derived regression fixtures cover independent M9906003 and M9906008
intervals and have provenance in `tests/fixtures/archive_fit/README.md`.
The calling study script stores full inputs, input hashes, coefficients,
alignment checks and validation tables outside the public package.

The real-file comparison also exposed duplicate ERT anchors in the source
telemetry. UTC reconstruction now flags both records in a non-increasing
anchor pair with `ert_anchor_rejected`, excludes their ephemeris anchors and
interpolates product times only between remaining valid anchors. Original ERT
and records are retained.

Source descriptions: [PDS ER 3D processing](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/document/data-3deleflux-desc.txt),
[PDS MAG processing](https://pds-ppi.igpp.ucla.edu/data/lp-mag-calibrated/document/data-lunarcrds-desc.txt).

For local transverse-offset recovery, model banks across independently tested
dates, firmware 4/5 compatibility, approximate PAD total-flux validation and
original 9 Hz sample output, see [mission calibration](mission_calibration.md).
Mission-date ER regression fixtures are in `tests/fixtures/mission_fit/`.
