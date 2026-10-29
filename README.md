# moonbroom

Python tools for postprocessing Lunar Prospector Magnetometer and Electron
Reflectometer data.

## Install

Create an isolated environment in this checkout, then install:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[geometry,plots,test]'
```

The core dependencies are NumPy, pandas and pds4_tools. SPICE geometry and
plotting dependencies are optional. Python 3.11 or newer is required.

## Read and select data

```python
from moonbroom import read_mag, read_er, select_time_window, write_csv

mag = read_mag("data/mag/ma990301.xml")
er = read_er("data/er/EH990301.xml")
mag = select_time_window(mag, "1999-03-01T00:00:00Z", "1999-03-02T23:59:59Z")
write_csv(mag, "output/mag.csv")
```

The sample window is 1–2 March 1999 as a representative example. Naive 
timestamps are interpreted as UTC, written UTC timestamps include an 
offset. Selection includes both endpoints.

| Function | Input and result |
| --- | --- |
| `decode_level0(paths, ...)` | Official merged Level-0 MAG/ER records, including documented real-time formats 1–3, empirical firmware 4/5 compatibility and burst playback, returned as named DataFrames |
| `read_merged_telemetry(path)` | Labelled 472-byte merged records with raw payloads, ERT and quality flags |
| `read_sunpulse(path, frames=None)` | Official eight-byte measured/estimated sun-pulse records and clock-relative pulse times |
| `read_pds4(path, table=0, scale=True)` | Label-driven ASCII or binary tables, including raw tables with PDS4 labels |
| `read_mag(path)` | Calibrated MAG with absolute timestamps, archive columns, `utc`, `Brms` alias and `Bmag_SEL`/`Bmag_SSE` where available |
| `read_mag_l2(path, radius_km=1737.4)` | Labelled `Hour`, SG field, latitude, longitude and altitude tables, with derived UTC and SEL positions |
| `read_lpmag_l1b_tab(path)` | Legacy 16-column lunar-coordinate ASCII table without requiring XML |
| `read_er(path, table=0, time_col="Time")` | Labelled ER flux or pitch-angle tables with archive field names |
| `read_lper_txt(path)` | Legacy three-column high-resolution ER flux table |
| `read_raw_records(path, fields, record_bytes=...)` | Fixed-size unpacked binary records with a caller-supplied layout |
| `load_lp_csvs(directory, recursive=False, pattern="*_nightside.csv")` | Combined CSVs with UTC timestamps and source paths |

PDS4 readers apply label scaling/offsets and mask numeric special constants.
Set `scale=False` to inspect stored values. Repeated group fields are expanded
into columns such as `flux[0,1]`, using zero-based indices in label order.
The original label path, table identifier and per-column units are in
`df.attrs`. CSV output stores tabular values, it does not retain `attrs`.
Use `table="table name"` or a zero-based table index when a label has multiple
tables. Unlabelled binary files and PDS3 labels are not accepted by `read_pds4`.

The legacy MAG column `et_days` contains decimal day of year, not SPICE ET.
`Brms` is the archive RMS variability measurement. `Bmag_SEL` is vector
magnitude and is never substituted for RMS. L2 products without RMS cannot
use the RMS filter until a suitable variability measure is provided.

ER channel energies and pitch-angle definitions come from each product's
companion tables. `read_er` does not assign energies from channel number.
The legacy `f1_340ev`/`f2_140ev` names are compatibility aliases for flux,
check the energy-bin table for their actual energies.

## Geometry and directory processing

```python
from pathlib import Path
from moonbroom import loaded_kernels, process_mag_directory

kernel_dir = Path("spice_kernels")
kernels = [kernel_dir / name for name in ("naif0012.tls", "pck00010.tpc", "de440s.bsp")]

with loaded_kernels(kernels):
    manifest = process_mag_directory(
        "data/mag", "output/mag",
        start="1999-03-01T00:00:00Z", stop="1999-03-02T23:59:59Z",
    )
```

Provide your own kernels covering the data interval. Kernels are not bundled.
The context loads explicit paths, unloads kernels it loaded, and does not call
`kclear`. Individual paths are preferable to metakernels with relative paths.

`process_mag_directory` writes `_all.csv`, `_nightside.csv` and
`_nightside_wake.csv` per input and returns an output manifest. It matches
filename case insensitively and propagates processing errors. Set
`reader=read_mag_l2` for labelled L2 tables, or
`reader=read_lpmag_l1b_tab` for legacy ASCII. `postprocess_mag(df, ...)` returns
the full annotated frame directly. Neither function loads kernels implicitly.

Shadow classification uses apparent lunar/solar discs with mean lunar radius.
Altitude uses the SPICE triaxial ellipsoid and SZA uses its surface normal.
Selenographic (SEL) frame of reference is approximated by `IAU_MOON`. Invalid 
geometry retains missing results. Existing SSE coordinates are used, missing SSE 
coordinates are computed with `sel_to_sse`, never copied from SEL. L2 position 
reconstruction uses the supplied spherical reference radius.

The default cone selection retains downstream exterior points at least 15°
past the terminator, with a 4° aberrated antisun axis for the downstream test.
Use `wake_model="cylinder"` and `wake_options={"wake_radius_km": ...}` for a
cylindrical cut, or `wake_model=None` to skip wake classification. **These are
geometric selections, not a physical model of plasma depletion.**

## Filtering, detrending and plotting

```python
from moonbroom import filter_altitude_brms, sel_to_spherical, detrend_tracks, plot_timeseries

clean = filter_altitude_brms(mag, alt_max_km=40, brms_quantile=0.9)
spherical = sel_to_spherical(clean)
residuals = detrend_tracks(spherical, gap_sec=1200, degree=2)
ax = plot_timeseries(residuals, ["Br", "Br_trend"], ylabel="Magnetic field [nT]")
ax.figure.savefig("output/trends.png", dpi=200, bbox_inches="tight")
```

Filtering computes the RMS quantile after the altitude cut. Spherical fields
use outward `Br`, southward `Btheta`, and eastward `Bphi`. Radius is stored in
`r_km` without changing `altitude_km`. Detrending fits each time-continuous
track separately against normalized along-track distance, with
altitude-weighted noise and a Gaussian coefficient prior. The legacy
`*_crust` columns contain empirical residuals. 
Plot functions return axes and do not call `show()`.

## Raw records and calibration

Use the official Level-0 decoder directly on archive files:

```python
from moonbroom import decode_level0, write_level0_tables

products = decode_level0("data/level0/M9906003.LBL")
mag_counts = products["mag"]
er_counts = products["er_counts"]
manifest = write_level0_tables(products, "output/level0")
```

This decodes packed MAG vectors, the three documented ER layouts and
explicitly labelled firmware 4/5 compatibility,
burst MAG/ER, digital subcom, prescaled/log-compressed count bounds and
collection clocks. It retains raw packets and reports gaps, partial groups,
unknown versions and unavailable timing. Decode adjacent files together to
reassemble accumulations across their boundaries.

UTC can use official LP trajectory kernels with an explicit receiver, or
supplied frame-clock correlations. The default retains clock-relative times
and keeps Earth receive time separate. See [timing and calibration](docs/calibration.md)
for SPICE reconstruction, archived sun pulses, official ER energy tables,
flight-coefficient MAG calibration and ER effective-response calibration.
The required MAG flight coefficients and ER angular response tables were
not found in the public archive directories checked, missing coefficients
never receive nominal substitutes. See [Level-0 decoding](docs/level0.md) for the output tables,
timing rules, source specifications, printed-table corrections and validation.

You can now [derive effective corrections from paired raw and calibrated PDS
products](docs/archive_calibration.md). The shared functions match ER 3D
spectra, estimate per-bin responses, fit an identifiable MAG gain/phase/offset
model and validate against separate intervals. Derived coefficients retain
their scope and provenance, unsupported bins/ranges remain missing, and MAG
native-rate accuracy is not inferred from averaged references.

For custom records that have already been unpacked, the lower-level reader
also accepts an explicit layout:

```python
import pandas as pd
from moonbroom import read_raw_records, calibrate_mag

fields = pd.DataFrame(
    [("range", "u1", 0), ("bx", ">u2", 1), ("by", ">u2", 3), ("bz", ">u2", 5)],
    columns=["name", "dtype", "offset_bytes"],
)
raw = read_raw_records("data/unpacked_mag.bin", fields, record_bytes=7)
calibration = pd.read_csv("data/mag_calibration.csv")
calibrated = calibrate_mag(raw, calibration, frame="SEN")
```

This example layout is illustrative. Use `decode_level0` for official merged
telemetry and `read_raw_records` for explicitly described custom records.
Neither applies a guessed epoch or instrument calibration.

`calibrate_mag` requires one calibration row per `range`, with
`gain_x/gain_y/gain_z` in nT/count and `zero_x/zero_y/zero_z` in counts.
It computes `(counts - zero) * gain` in the caller-named frame and flags the
first sample after each range change by default. Time-varying offsets,
misalignment corrections, despinning and attitude rotation remain explicit
upstream operations.

The generic, custom-record helper `calibrate_er` requires an `accumulation_s` column and a channel DataFrame
containing `count_col`, `flux_col`, `geometric_factor_cm2_sr`,
`energy_width_ev` and `efficiency`. It returns differential number flux as
counts divided by live accumulation time and the three response factors.
Supply decoded counts, corrected live times and appropriate instrument
response, background/dead-time corrections are not inferred.
For decoded LP ER telemetry, use `calibrate_er_telemetry` instead: it uses
the PDS center-energy convention, explicit angular response and deadtime
masking, and propagates the bounds on log-compressed counts.

For mission-spanning model banks, independent-date validation, approximate PAD
response and **original 9 Hz MAG output**, see
[mission calibration](docs/mission_calibration.md).
The native calibration API also provides `calibrate_mag_18hz` for original burst
MAG, explicit coefficient-transfer provenance, and checks across observed hardware
states. Mission processing writes burst samples and hardware checks separately.

## Use from analysis repositories

Install moonbroom in the same environment as the calling analysis package,
then import functions directly:

```python
from moonbroom import read_mag, read_er, postprocess_mag, sel_to_spherical
```

Reusable processing belongs here. Project data, event windows, event markers,
interpretation, exploratory notebooks and science figures belong in the
calling analysis repository. The editable examples in `examples/` contain
only sample paths and sample dates, they have no command-line parser.

## Validation

```sh
python -m pytest
```

Tests use synthetic archive labels/records and small public PDS telemetry
excerpts with provenance, not private science data. Geometry tests use
temporary synthetic kernels. Tests require the optional
geometry and plotting dependencies as well as pytest.

## Archive references

- [LP MAG lunar-coordinate format and processing](https://pds-ppi.igpp.ucla.edu/data/lp-mag-calibrated/document/data-lunarcrds-desc.txt)
- [LP Level-0 archive and instrument format documentation](https://pds-ppi.igpp.ucla.edu/volume/LP_0010/AAREADME.HTM)
- [PDS4 Python Tools API](https://pdssbn.astro.umd.edu/tools/pds4_tools_docs/current/api_quick_start.html)
# Manuscript analyses and full-mission recovery

See [the full-mission guide](docs/full_mission.md) for the manuscript analysis
map, a resumable downloader for all 19 PDS Level-0 volumes, original MAG/ER
decoding, accepted-bank pseudo-calibration and the mission magnetotail survey.
Run `scripts/download_lp_mission.py` to download the archive,
`scripts/pseudocalibrate_lp_mission.py` to decode and pseudo-calibrate the
downloaded mission data, and `scripts/analyze_lp_mission.py` for magnetotail
encounter analysis. The default survey prepares NASA/SPDF one-minute OMNI
flow pressure and GSM Bz for the Shue model; use
`scripts/prepare_omni_upstream.py` to prepare those inputs separately.
See [OMNI upstream inputs](docs/omni_upstream.md) for conventions and quality
handling. Each script has a hardcoded `CONFIG_FILE` pointing
to its repository-local JSON configuration. Coefficient banks and
provenance are included in `calibration/`. Missing official SPICE kernels are
fetched and hash-checked automatically. [Run configurations](configs/README.md)
cover download, processing, manuscript regression and mission survey commands.
