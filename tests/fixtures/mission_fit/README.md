# Public PDS regression pairs on independent mission dates

These small derived tables retain raw count bounds, nominal integration time,
center energy and unmodified calibrated PDS flux for selected 3D angular bins.
They are regression fixtures, not flight calibration coefficients.
`inventory.csv` records the distinct training/validation dates and software
versions 1–5. `provenance.csv` records source URLs and SHA-256 hashes of the
original downloaded products.

The extraction decoded the official merged telemetry, reconstructed UTC using
LP trajectory kernels with EARTH as a geocentric receiver approximation, then
matched complete distributions by time, zero pattern and count/flux rank
correlation. Each split retains at most twelve training or six validation
observations per selected energy/angular key. The tests fit only training
observations and predict counts on the other date. They exercise the
empirical format-3 compatibility of software versions 4/5 without changing
the reported version.

These fixtures test response recovery and application. They do not validate
individual PAD pitch bins, burst calibration or native-frequency MAG accuracy.

The MAG minute excerpts preserve signed counts, reconstructed clock/phase and
source identities from 17/19 January 1998. The model was fit on 17 January
and its five-second SCD reference windows are from 19 January. Native-output
regression checks the original sample grid and compares averages against the
separate-date reference; it makes no claim about high-frequency accuracy.

`firmware4_packets.bin` and `firmware5_packets.bin` each contain one complete
set of 36 original 83-byte real-time ER payloads, ordered by packet type.
`packet_inventory.csv` identifies the source product and inclusive record
indices; product hashes are in `provenance.csv`. Companion reference tables
retain all 1320 paired 3D cells. The regression decodes the original bytes
and checks agreement with the matched calibrated distribution.
