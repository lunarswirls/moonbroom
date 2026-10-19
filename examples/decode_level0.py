"""Edit archive paths; no event interval or unverified absolute clock is assumed"""

from pathlib import Path

from moonbroom import decode_level0, write_level0_tables


paths = [Path("data/level0/M9906003.LBL"), Path("data/level0/M9906008.LBL")]
out_dir = Path("output/level0")

if __name__ == "__main__":
    products = decode_level0(paths)
    manifest = write_level0_tables(products, out_dir)
    print(manifest.to_string(index=False))
