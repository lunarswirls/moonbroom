"""
Download all LP MAG/ER Level-0 archive volumes for pseudo-calibration.
"""

from pathlib import Path

from moonbroom.mission_pipeline import main


CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "download_mission.json"
STAGE = "download"  # inventory or download


if __name__ == '__main__':
    raise SystemExit(main([str(CONFIG_FILE), "--stage", STAGE]))
