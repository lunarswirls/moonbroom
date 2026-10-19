# Public PDS calibration-transfer regression data

The MAG tables contain processed raw-spin window features paired with the
SCD magnetic vectors accompanying the ER 3D product. The ER tables contain
selected raw count bounds and unmodified calibrated flux values for energy
steps 9 and 11 and angular bins 36 and 37. These are derived test tables,
not original PDS products or official coefficient tables.

Training uses M9906003, with validation on the separate M9906008 interval
of 1 March 1999. The reference distributions do not overlap between splits.
Raw decoding, reconstructed SUNPULSE phase and matched collection bounds
produce the MAG features. The tests check recovery against data that were
not used to estimate coefficients.

Sources:

- [Merged telemetry directory](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/merged/d06x/)
- [Sun-pulse directory](https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_0015/sunpulse/D06X/)
- [Calibrated ER 3D reference](https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux/1999/060_090MAR/3D990301.xml)
- [LP trajectory kernels](https://naif.jpl.nasa.gov/pub/naif/LPM/kernels/spk/)

Timing used the LP trajectory, naif0012 leap seconds, de440s planetary
ephemeris and explicit EARTH receiver approximation. Duplicate ERT anchor
pairs were excluded. The fields record collection time and reference identity;
they contain no private event annotations or interpretation.
