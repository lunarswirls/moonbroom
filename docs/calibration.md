# LP timing and calibration

The shared functions live in moonbroom. File locations, receiving-station
choices, event windows and interpretation belong in the calling analysis
repository. Functions do not download data or load kernels on import.

## UTC reconstruction

```python
from pathlib import Path
from moonbroom import decode_level0, read_sunpulse, reconstruct_spin_phase

kernel_dir = Path("spice_kernels")
kernels = [kernel_dir / name for name in (
    "naif0012.tls", "de440s.bsp", "lp_ask_981101-990331.bsp",
)]
products = decode_level0(
    "data/level0/M9906003.LBL", timing_kernels=kernels, receiver="EARTH",
)
pulses = read_sunpulse("data/level0/S9906003.LBL", frames=products["frames"])
mag = reconstruct_spin_phase(products["mag"], pulses)
```

NASA NAIF provides [four LP spacecraft trajectory SPKs](https://naif.jpl.nasa.gov/pub/naif/LPM/kernels/spk/).
Select the SPK covering the input interval, along with a leap-second kernel
and planetary ephemeris. The archive's `LP_EPHEM.TSP` is a transfer-format
planetary ephemeris; it is not an LP spacecraft-clock kernel or a replacement
for the spacecraft trajectory. No LP SCLK kernel was found in that collection.

`reconstruct_frame_times` calculates converged reception light time (`CN`)
from the explicitly named receiver to spacecraft `-25`, then computes:

`frame_et = ert_et - light_time_s - data_system_delay_s`

The default data-system delay is two seconds, as specified in
[MAGERFMT section 1.5](https://pds-ppi.igpp.ucla.edu/archive1/LP_1201/DOCUMENT/INST/MAGERFMT.PDF).
The merged archive has already subtracted 3200 seconds from delayed-playback
ERT and corrected its counter; that correction is not applied again.
The separate four-second MAG instrument pipeline offset remains part of
sample decoding. An alternative delay must be supplied explicitly.

The instrument-specific nominal clock offset of real-time MAG sample `i`
(zero-based, 0 through 17) is `-4 + i/9 + 1/36` seconds relative to its packet's
frame clock. The `1/36` term places each averaged 18-Hz pair at its midpoint.
These collection clocks are correlated to SCET using the bounded ephemeris
interpolation below. ER uses its header, spin phase/period and the 2.4-ms
instrument timing term; a blanket 32-second science delay is not substituted.

## Command transmission and spacecraft arrival

[Command labels](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0011/command/c9831942.lbl)
define their UTC as the time a DSN station **sent** the command. It is neither
Earth receive time nor a measured spacecraft execution time.
`annotate_command_actions` preserves that value as `transmit_utc` (and the
legacy `utc` column). `reconstruct_command_times` calculates converged
transmission light time with SPICE `XCN` and adds it in ephemeris seconds:

`arrival_scet_et = transmit_et + uplink_light_time_s`

No two-second telemetry buffering, four-second MAG pipeline delay or
3200-second merged-playback correction applies to this uplink calculation.
Output retains `arrival_scet_utc`, its leap-second-preserving ISO companion,
`arrival_scet_valid`, `uplink_light_time_s`, the endpoint and timing basis.
ABORT/SKIP entries and missing SPK coverage have no claimed arrival SCET.
An arrival estimate does not establish that the command executed, or when.

The mission driver uses its configured kernels and `command_transmitter`
(default `EARTH`, explicitly a geocentric approximation). Each command's DSN
station number remains in the log. For station-specific work, group commands
by their actual station and call the helper with that endpoint and its
station/frame kernels; a single global endpoint does not reconstruct each
station's geometry. Clock-only processing keeps command arrival unavailable
and flags padded DSN-time proxies explicitly. It never silently adds the
historical mean 1.28-second Earth–Moon delay.

## Telemetry timing support

`receiver="EARTH"` is an explicit geocentric approximation. Station timing
requires the actual DSN station identity and matching station/frame kernels.
The merged record does not identify that station. Receiver uncertainty,
archive ERT accuracy and the absence of an independent oscillator drift
solution limit absolute timing accuracy, microsecond formatting does not
imply microsecond accuracy.

`reconstruct_product_times` maps collection clocks to frame anchors by
bounded interpolation in ephemeris seconds. It applies to MAG, ER bounds,
accumulations, angles, markers and any supplied sun-pulse table. Earlier
burst collection requires earlier frame correlations; later playback is
not used to extrapolate those collection times. Edge samples outside the
input coverage remain missing, so include neighboring input files.

Every reconstructed UTC has an ISO-string companion and a validity flag.
UTC second 60 remains in `*_iso`, with `*_leap_second=True`; the pandas
timestamp is missing at that instant. Interpolation never treats that second
as a discontinuity. Timing-kernel paths are exported in `timing_kernels`.

Archived SUNPULSE is preferable to extrapolating the ER phase-locked loop
through eclipse. `reconstruct_spin_phase` interpolates only between adjacent
valid pulses separated by at most `max_gap_s` (default six seconds). It
deduplicates repeated reports, prefers measured pulses and retains estimated
pulse flags and uncertainties. Choose the gap threshold for the applicable
spin rate. A missing pulse never becomes an artificial slower rotation.

## Official energy tables

```python
from moonbroom import assign_er_energies, read_er_energy_bins

energy_bins = read_er_energy_bins("data/er/e_bins.xml", time_basis="unix")
counts = assign_er_energies(products["er_counts"], energy_bins)
```

The [low-resolution companion table](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-lowresflux-anc/e_bins/)
contains all 15 center energies in telemetry step order. The
[high-resolution table](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-highresflux-anc/e_bins/)
contains two selected channels. For that table supply `telemetry_version`:
format 1 selects steps 6 and 9, formats 2–3 select steps 9 and 11.

`time_basis` is required because the archive contains inconsistent time tags:
the first high-resolution row has 16 January 1998 in Unix seconds and
1 January in its UTC text, the last low-resolution row differs by six seconds.
Both original times and `time_disagreement_s` are retained. Choose the
appropriate convention explicitly and record it with analysis settings.
Energies apply over half-open intervals. Accumulations crossing a change,
preceding the first definition, or lacking UTC remain unassigned.

## MAG flight coefficients

```python
import pandas as pd
from moonbroom import calibrate_mag_telemetry, despin_mag

coefficients = pd.read_csv("data/calibration/mag_coefficients.csv")
calibrated = calibrate_mag_telemetry(mag, coefficients, source="flight coefficient product identifier")
despun = despin_mag(calibrated, phase_offset_deg=mounting_phase_deg)
```

The coefficient DataFrame requires `range`, `valid_from`, `valid_to`,
`zero_x`, `zero_y`, `zero_z`, `gain_x`, `gain_y`, `gain_z`. Zero values use
signed ADC counts, matching the decoder's `bx_signed/by_signed/bz_signed`
columns, and gains use nT/count. Correction is `(signed_counts - zero)*gain`.
Intervals with overlapping range/mode/hardware applicability may touch but must
not overlap. Optional `mode` selects `realtime` or `burst`; a missing mode means
real-time only. Nonmissing electronics-side, boom, software-version and reset-count
constraints must match observed sample state. See
[native MAG and hardware checks](mission_calibration.md#calibrate-original-18-hz-burst-mag).

Optional `align_xx` through `align_zz` define a right-handed approximately
orthonormal sensor-to-spacecraft matrix, applied after the gain correction.
All nine values are required together. Uncovered samples, diagnostic mode,
ADC limits and flagged range transitions remain missing. `calibration_source`,
`calibration_available` and `calibration_valid` identify the result.
Without alignment output is `Bx_SEN/By_SEN/Bz_SEN`; aligned output uses `_SC`.
Despinning requires the rotation phase of spacecraft x from sunward SCD x,
including an explicitly supplied mounting/sun-sensor phase correction.

The [MAG processing description](https://pds-ppi.igpp.ucla.edu/data/lp-mag-calibrated/document/data-lunarcrds-desc.txt)
specifies minute-dependent x/y offsets, a constant z offset, per-range flight
gains and misalignment correction. It does not supply those coefficient
tables or enough detail to reproduce the team's offset-fitting procedure.
MAGERFMT calls its gain table nominal and leaves flight values TBD. The code
therefore requires supplied coefficients rather than manufacturing them.

## ER count rates and response calibration

```python
from moonbroom import calibrate_er_telemetry

rates = calibrate_er_telemetry(counts)
response = pd.read_csv("data/calibration/er_effective_response.csv")
response.attrs["source"] = "effective response product identifier"
flux = calibrate_er_telemetry(counts, response=response)
```

`response` contains one row per `accumulation_id`, `kind`, `energy_step`,
`sample`, `angle_bin`, with `exposure_s` and `geometric_factor_cm2_sr`.
These must describe the actual angular bin and energy integration, including
pitch-angle occupancy and applicable detector efficiency. Duplicate keys and
nonpositive factors are rejected; missing response rows produce missing flux.

The [PDS ER processing convention](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/document/data-3deleflux-desc.txt)
is `r = counts/exposure`, `r_corrected = r/(1-r*dead_time_s)`, then
`flux = r_corrected/(geometric_factor_cm2_sr*energy_ev)`. Center energy is
used rather than energy-bin width. Defaults are 0.3 microseconds deadtime and
a maximum correction factor of 1.25. Background remains unsubtracted.

`counts_lower/counts_upper` are propagated into rate and flux bounds;
the reported estimate is their midpoint propagated through the correction.
If the upper bound exceeds the allowed deadtime correction, or the code
could indicate saturation, the calibrated result is missing. This avoids
presenting lossy telemetry as exact counts. Output units are electrons
cm^-2 s^-1 sr^-1 eV^-1.

Without a response table, 3D/cut count rates use the documented basic
integration time of period/256, multiplied by the 3D polar-bin summation.
PAD rates remain missing because their effective exposure depends on
pitch-angle occupancy. **All flux stays missing without an effective response.**
The approximate full-instrument 0.02 cm2 sr value in the collection
description is not a per-angular-bin response. Applying it to individual
bins failed comparison against the published 1 March 1999 3D product, so it
is deliberately not used as a default. MAGERFMT section 2.3.6 leaves the
anode-efficiency table and the detailed pitch-angle mapping arithmetic TBS.

`calibrate_level0` coordinates these functions and adds `calibration_info`,
`mag_calibrated` when coefficients are supplied, and `er_flux` when energies
are supplied. The manifest explicitly reports missing inputs, the full flight 
calibration has not been reproduced as the coefficients were not archived in
the PDS alongside the data.
