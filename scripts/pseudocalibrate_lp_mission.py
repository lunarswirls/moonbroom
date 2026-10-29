"""
Decode and pseudo-calibrate the full-mission LP MAG/ER archive.

Reads the local inventory, download manifest and original telemetry produced by
download_lp_mission.py. Applies the accepted MAG/ER/PAD coefficient banks and
preserves unsupported rows.
"""

from pathlib import Path

from moonbroom.mission_pipeline import main


CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "pseudocalibrate_mission.json"


if __name__ == '__main__':
    raise SystemExit(main([str(CONFIG_FILE), "--stage", "process"]))
