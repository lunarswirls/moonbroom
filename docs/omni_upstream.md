# OMNI inputs for the Shue magnetopause investigation

The mission survey uses NASA/SPDF's **standard one-minute high-resolution
OMNI** (`OMNI_HRO_1MIN`) monthly ASCII product. The default interval covers
1998-01-01 00:00 UTC through, but excluding, 1999-08-01 00:00 UTC. This
includes the entire LP science mission, with padding before orbital data.
The product is selected explicitly; hourly OMNI2 and the modified/definitive
HRO product are different inputs and are not substituted automatically.

After preparing the native MAG mission products, run:

```bash
python scripts/analyze_lp_mission.py
```

`configs/survey_mission.json` points to `omni_mission.json`. The survey
validates the existence of its native shards and geometry inputs, prepares
or verifies the OMNI table, then runs the Shue analysis. OMNI can also be
prepared independently:

```bash
python scripts/prepare_omni_upstream.py
python -m moonbroom.omni configs/omni_mission.json
# Installed entry point:
moonbroom-omni configs/omni_mission.json
```

These are alternative commands for the same preparation step. JSON paths
resolve relative to their configuration file. `start_utc` and `stop_utc`
must be ordered and minute aligned; the stop is exclusive. No user-facing
configuration is needed beyond the provided mission defaults.

## Physical inputs and time convention

The reader follows the published **46-word, fixed-width** one-minute HRO
format. It rejects malformed records, invalid calendar fields, duplicate
epochs, unordered epochs and data outside a cached file's named month.

| Survey input | HRO word | Convention |
| --- | --- | --- |
| `utc` | 1–4 | Year, day of year, hour, minute; start of one-minute average |
| `pressure_npa` | 28 | Published flow pressure, nPa |
| `bz_gsm_nt` | 19 | IMF Bz in GSM, nT |
| `bz_gse_nt` | 17 | Diagnostic only; does not drive Shue |
| `upstream_interval_stop_utc` | Derived | UTC plus 60 seconds, exclusive |

SPDF documents HRO flow pressure as

\[
P_{\rm dyn}[\mathrm{nPa}]=2\times10^{-6}
N_p[\mathrm{cm}^{-3}]V_p^2[\mathrm{km}^2\,\mathrm{s}^{-2}].
\]

The supplied word 28 is used directly. The coefficient includes SPDF's
composition convention. The reader does not recalculate pressure, add a
helium multiplier or substitute a fallback when word 28 is missing. Speed,
velocity components and proton density are retained for inspection.

OMNI has already been propagated to **Earth's bow-shock nose**. Its UTC is
target arrival time, not the original spacecraft observation time. The
stored timeshift and its RMS are diagnostic columns and are never added to
UTC again. No Moon propagation or boundary response delay is introduced.
Using bow-shock-nose arrival for contemporaneous terrestrial magnetopause
forcing is an explicit approximation; no far-tail propagation/response
model has been validated here.

## Missing data and quality

The preparation builds every minute in the requested interval. Absent
source records become `quality_reason=missing_omni_minute`, with missing
physical inputs and `quality_valid=false`. Documented fills, including
`99.99` nPa pressure and `9999.99` nT Bz, become missing values. A pressure
above 99.99 nPa is not rejected merely for exceeding the fill value.

Valid inputs require finite positive pressure, finite GSM Bz, documented
IMF/plasma source IDs and positive non-fill IMF/plasma point counts. Reasons
for rejection remain in the table. This is an **input availability gate**,
not certification of propagation accuracy. Percent-interpolated PFNs,
timeshift/RMS and observation-time separation are retained. Negative
observation-time separation is flagged as `omni_out_of_sequence` without
silently imposing another selection threshold.

The survey matches backward within `max_upstream_age_s` and also enforces
the exclusive interval end. Thus a minute cannot be extended into a missing
next minute or beyond the configured stop even when the maximum age would
permit it. Invalid minutes remain in `windows.csv` as unsupported boundary
context and split candidate encounters. No gap interpolation is performed.

## Provenance and reuse

Sources are downloaded from NASA/SPDF over HTTPS into
`data/omni_hro_1min/omni_minYYYYMM.asc`. Downloads use temporary files;
truncated HTTP responses and malformed data are rejected before promotion.
Each source gets a sidecar with its SHA-256, byte count, origin, URL and,
for downloads, retrieval time and HTTP Last-Modified. These local hashes
pin the first acquired bytes; they are not independent official checksums.
Existing ASCII files without sidecars can be adopted after validation and
are explicitly marked `existing_local_file` with unknown retrieval time.

`data/upstream_boundary_context.manifest.json` records source and sidecar
hashes, code hash, product identity, pressure/time conventions, interval,
coverage counts and the output hash. Reuse verifies all of these inputs
without accessing the network. Modified sources, code, settings or output
require a new output path. With `download: false`, missing cache files fail
explicitly. The survey embeds the preparation manifest and hashes both
the upstream CSV and manifest as analysis inputs.

`windows.csv` records the matched upstream inputs and time, context validity,
nominal `shue_r0_re`, `shue_alpha`, `tail_radius_re` and the existing
pressure/Bz sensitivity envelope. The geometric model still uses
Shue et al. (1998), the constructed terrestrial solar basis and an explicit
far-tail extrapolation. An OMNI-supported candidate is contextual evidence;
it does not independently identify a magnetotail plasma region.

## References and execution status

- [NASA/SPDF HRO documentation, format and derived-parameter equations](https://omniweb.gsfc.nasa.gov/html/omni_min_data.html)
- [Standard HRO ASCII format](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/hroformat.txt)
- [Monthly one-minute ASCII archive](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/)
- [Shue et al. (1998), magnetopause location under extreme solar wind conditions](https://doi.org/10.1029/98JA01103)

Regression checks cover column/frame selection, published-pressure use,
fills, UTC/leap handling, missing minutes, interval expiration, source/output
integrity, truncated downloads and the survey connection. The Python runtime
could not resolve NASA's host during development. Live mission downloads and
mission-wide encounter statistics have not been executed in this checkout.
