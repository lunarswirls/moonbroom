"""Edit paths and the sample interval before running this script"""

from pathlib import Path

from moonbroom import loaded_kernels, process_mag_directory


input_dir = Path("data/mag")
out_dir = Path("output/mag")
kernel_dir = Path("spice_kernels")
kernels = [kernel_dir / name for name in ("naif0012.tls", "pck00010.tpc", "de440s.bsp")]
start = "1999-03-01T00:00:00Z"
stop = "1999-03-02T23:59:59Z"

if __name__ == "__main__":
    with loaded_kernels(kernels):
        manifest = process_mag_directory(input_dir, out_dir, start=start, stop=stop)
    print(manifest.to_string(index=False))
