# Public Level-0 regression excerpt

`M9906003_excerpt.B` contains the first 128 consecutive 472-byte records
(60416 bytes) of the public PDS file M9906003.B, without modifying any bytes.
The small companion label records the excerpt length and original ERT start.
It is a test label, not the archive's original detached label.

Original file SHA-256:
`6ed53e431cce544c8b134adca280b5cdd402dfb2fbb01d1915232d4987d85fd1`

Sources:

- [Original binary](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/merged/d06x/M9906003.B)
- [Original label](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/merged/d06x/M9906003.LBL)
- [Archive errata](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/errata.txt)

This is mission telemetry for 1 March 1999. It contains no private analysis
or event annotations. The reference values in test_level0.py are literal
record and packed-channel values checked against the official field layout.

`S9906003_excerpt.B` contains the first 64 unmodified eight-byte records of
the [matching sun-pulse file](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/sunpulse/D06X/S9906003.B).
Its [official label](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/sunpulse/D06X/S9906003.LBL)
defines pulse timing and source flags. The small test label records only the
excerpt filename, eight-byte record size and count. These independent pulse
times check the subcom's major-frame alignment and spin-phase reconstruction.

`er_energy_low/e_bins.tab` and `er_energy_high/e_bins.tab`, with their XML
labels, are unmodified official PDS companion tables from
[low-resolution ER](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-lowresflux-anc/e_bins/)
and [high-resolution ER](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-highresflux-anc/e_bins/).
The original directory separation is retained because both tables have the
same basename. Tests verify their energy values and inconsistent time tags.

`naif0012.tls` is the official
[NAIF leap-second kernel](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls).
Timing tests create a temporary synthetic spacecraft/receiver SPK with a
known 1.5-second light time, while this kernel supplies real UTC leap seconds.
No large trajectory files or local event settings are included in fixtures.
