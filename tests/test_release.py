from pathlib import Path
import pandas as pd
from moonbroom import decode_level0
from moonbroom.release import audit_native_payloads, inventory_local_mag


def test_release_scale_audit_is_exact_and_detects_negative_count_corruption():
    path=Path(__file__).parent/'fixtures'/'M9906003_excerpt.LBL'
    p=decode_level0(path,mag_only=True)
    assert audit_native_payloads(p['frames'],p['mag'])['audit_pass']
    bad=p['mag'].copy()
    bad.loc[0,'bx_signed']+=1
    assert not audit_native_payloads(p['frames'],bad)['audit_pass']
