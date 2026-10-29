"""
Analyze magnetotail encounters in pseudo-calibrated 9 Hz LP MAG data.

Run pseudocalibrate_lp_mission.py first to prepare the 9 Hz products.
"""

from pathlib import Path

from moonbroom.mission_survey import main


CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "survey_mission.json"


if __name__ == '__main__':
    raise SystemExit(main([str(CONFIG_FILE)]))
