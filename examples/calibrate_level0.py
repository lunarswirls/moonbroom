"""Edit generic local input paths to reconstruct LP UTC and inspect calibration availability"""

from pathlib import Path

from moonbroom import (
    calibrate_level0, decode_level0, read_er_energy_bins, read_sunpulse,
    write_level0_tables,
)


kernel_dir = Path("spice_kernels")
kernels = [kernel_dir / name for name in (
    "naif0012.tls", "de440s.bsp", "lp_ask_981101-990331.bsp",
)]
receiver = "EARTH"
merged_paths = [Path("data/level0/M9906003.LBL")]
sunpulse_path = Path("data/level0/S9906003.LBL")
energy_bins_path = Path("data/er_low_energy/e_bins.xml")
energy_time_basis = "unix"
output_dir = Path("output/calibrated")


if __name__ == "__main__":
    products = decode_level0(merged_paths, timing_kernels=kernels, receiver=receiver)
    pulses = read_sunpulse(sunpulse_path, frames=products["frames"])
    energies = read_er_energy_bins(energy_bins_path, time_basis=energy_time_basis)
    calibrated = calibrate_level0(products, er_energy_bins=energies, sunpulse=pulses)
    manifest = write_level0_tables(calibrated, output_dir)
    print(calibrated["calibration_info"].to_string(index=False))
    print(manifest.to_string(index=False))
