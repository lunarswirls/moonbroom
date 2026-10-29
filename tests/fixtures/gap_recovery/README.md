# Public additional-date MAG recovery fixtures

These tables contain raw-spin averaging features and official calibrated
five-second PDS field vectors rotated from SSE to SCD. Training and held-out
validation dates are distinct for every range. They are processed regression
inputs, not flight calibration tables.

- range 0: training 14 May 1998, validation 12 May 1998
- range 1: training 14 February 1998, validation 12 February 1998
- range 2: training 25 September 1998, validation 26 September 1998

Features use locally estimated transverse offsets, archived SUNPULSE phase,
LP SPICE reception timing with the EARTH receiver approximation and explicit
ECLIPJ2000 orbit-average attitude. Each table samples up to 200 training or
100 validation reference windows. `provenance.csv` records PDS URLs and
original-product SHA-256 hashes. The regression fits only training targets
and predicts the separate-date references with transverse offsets fixed to
zero after local correction. Native-frequency accuracy remains unvalidated.
