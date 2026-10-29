"""Prepare one-minute OMNI flow pressure and GSM Bz for the Shue survey."""

from pathlib import Path

from moonbroom.omni import main


CONFIG_FILE = Path(__file__).resolve().parents[1] / 'configs' / 'omni_mission.json'


if __name__ == '__main__':
    raise SystemExit(main([str(CONFIG_FILE)]))
