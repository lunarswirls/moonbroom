"""Prepare NASA/SPDF one-minute OMNI inputs for the terrestrial Shue model.

Reads the standard 46-word HRO ASCII product, not hourly OMNI2 or a
Moon-shifted event table. Published flow pressure is used without rescaling.
https://omniweb.gsfc.nasa.gov/html/omni_min_data.html
"""

import argparse
import calendar
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .mission_archive import file_hash, write_json


OMNI_ROOT = 'https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/'
OMNI_PRODUCT = 'OMNI_HRO_1MIN'
# (2I4,4I3,3I4,2I7,F6.2,I7,8F8.2,4F8.1,F7.2,F9.0,F6.2,
#  2F7.2,F6.1,6F8.2,7I6,F7.2,F5.1)
HRO_WIDTHS = [4]*2 + [3]*4 + [4]*3 + [7]*2 + [6, 7] + [8]*8 + [8]*4 + [7, 9, 6] + [7]*2 + [6] + [8]*6 + [6]*7 + [7, 5]
# Zero-based word offsets, names and documented fill values.
HRO_FIELDS = {
    4: ('omni_imf_spacecraft_id', 99), 5: ('omni_plasma_spacecraft_id', 99),
    6: ('omni_imf_points', 999), 7: ('omni_plasma_points', 999),
    8: ('omni_percent_interpolated', 999), 9: ('omni_timeshift_s', 999999),
    10: ('omni_timeshift_rms_s', 999999), 12: ('omni_observation_delta_s', 999999),
    16: ('bz_gse_nt', 9999.99), 18: ('bz_gsm_nt', 9999.99),
    21: ('speed_km_s', 99999.9), 22: ('vx_gse_km_s', 99999.9),
    23: ('vy_gse_km_s', 99999.9), 24: ('vz_gse_km_s', 99999.9),
    25: ('proton_density_cm3', 999.99), 27: ('pressure_npa', 99.99),
}


def read_omni_hro(path):
    """Read standard fixed-width HRO ASCII, retaining invalid input minutes.

    quality_valid is an availability gate for pressure, GSM Bz, documented
    source IDs and positive non-fill point counts; it is not a propagation
    accuracy assessment. Timing and interpolation diagnostics are retained.
    No pressure fallback, interpolation or additional time shift is applied.
    """
    path = Path(path)
    with path.open(encoding='ascii') as stream:
        for number, line in enumerate(stream, 1):
            if len(line.rstrip('\r\n')) != sum(HRO_WIDTHS):
                raise ValueError(f'Expected 46-word fixed-width HRO record: {path}:{number}')
    raw = pd.read_fwf(path, widths=HRO_WIDTHS, names=range(46), dtype=float)
    if raw.empty:
        raise ValueError(f'Empty OMNI source: {path}')
    stamp = raw.iloc[:, :4].to_numpy()
    if not np.isfinite(stamp).all() or not np.equal(stamp, np.floor(stamp)).all():
        raise ValueError('OMNI year/day/hour/minute must be finite integers')
    year, day, hour, minute = stamp.astype(int).T
    leap_days = np.array([365 + calendar.isleap(int(y)) for y in year])
    if ((day < 1) | (day > leap_days) | (hour < 0) | (hour > 23)
            | (minute < 0) | (minute > 59)).any():
        raise ValueError('Invalid OMNI day/hour/minute')
    utc = (pd.to_datetime(pd.Series(year).astype(str) + '-01-01', utc=True)
           + pd.to_timedelta(day-1, unit='D') + pd.to_timedelta(hour, unit='h')
           + pd.to_timedelta(minute, unit='min'))
    if utc.duplicated().any() or not utc.is_monotonic_increasing:
        raise ValueError('OMNI UTC must be unique and ordered')
    out = pd.DataFrame({'utc': utc})
    for offset, (name, fill) in HRO_FIELDS.items():
        out[name] = raw[offset].mask(raw[offset].eq(fill))
    flags = {
        'invalid_pressure': ~np.isfinite(out['pressure_npa']) | out['pressure_npa'].le(0),
        'missing_bz_gsm': ~np.isfinite(out['bz_gsm_nt']),
        'invalid_imf_source': ~out['omni_imf_spacecraft_id'].isin([50, 51, 52, 60, 71]),
        'invalid_plasma_source': ~out['omni_plasma_spacecraft_id'].isin([50, 51, 52, 60, 71]),
        'invalid_imf_points': ~out['omni_imf_points'].between(1, 998),
        'invalid_plasma_points': ~out['omni_plasma_points'].between(1, 998),
    }
    reasons = pd.Series('', index=out.index)
    for reason, mask in flags.items():
        reasons.loc[mask] += reason + ';'
    out['quality_reason'] = reasons.str.rstrip(';').replace('', 'valid')
    out['quality_valid'] = out['quality_reason'].eq('valid')
    out['omni_out_of_sequence'] = out['omni_observation_delta_s'].lt(0)
    out['omni_source_file'] = path.name
    return out


def _month_source(month, directory, *, download, timeout_s, retries):
    """Cache a validated monthly source and pin its local SHA-256 on first use."""
    name = f'omni_min{month:%Y%m}.asc'
    target = directory / name
    metadata = target.with_suffix('.asc.json')
    url = OMNI_ROOT + name
    origin = 'existing_local_file'
    pinned = json.loads(metadata.read_text()) if metadata.exists() else None
    if pinned is not None and pinned['url'] != url:
        raise ValueError(f'Cached OMNI source URL changed: {target}')
    details, table = {}, None
    if target.exists():
        if pinned is not None and pinned['sha256'] != file_hash(target):
            raise ValueError(f'Cached OMNI source changed: {target}')
    else:
        if not download:
            raise FileNotFoundError(f'Missing cached OMNI source with download disabled: {target}')
        directory.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.asc.partial')
        for attempt in range(retries):
            try:
                with urlopen(Request(url, headers={'User-Agent': 'moonbroom/0.1 OMNI inputs'}), timeout=timeout_s) as response:
                    length = response.headers.get('Content-Length')
                    with partial.open('wb') as stream:
                        while block := response.read(1024*1024):
                            stream.write(block)
                    if length is not None and partial.stat().st_size != int(length):
                        raise OSError('Truncated OMNI HTTP response')
                    details = dict(retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                                   http_last_modified=response.headers.get('Last-Modified'))
                # Validate before promoting the downloaded bytes into the cache.
                table = read_omni_hro(partial)
                _check_month(table, month)
                if pinned is not None and file_hash(partial) != pinned['sha256']:
                    raise ValueError('Restored OMNI source differs from its pinned SHA-256')
                partial.replace(target)
                origin = 'NASA_SPDF_HTTPS'
                break
            except (OSError, ValueError) as error:
                partial.unlink(missing_ok=True)
                if attempt + 1 == retries:
                    raise OSError(f'Could not fetch OMNI source {url}: {error}') from error
                time.sleep(min(2**attempt, 8))
    if table is None:
        table = read_omni_hro(target)
    table['omni_source_file'] = name
    _check_month(table, month)
    if pinned is None:
        details.update(path=str(target.resolve()), url=url, origin=origin,
                       sha256=file_hash(target), bytes=target.stat().st_size)
        write_json(metadata, details)
    else:
        # A copied cache retains its original provenance but uses local paths.
        details = dict(pinned, path=str(target.resolve()))
    return table, details, metadata


def _check_month(table, month):
    stop = month + pd.offsets.MonthBegin()
    if not table['utc'].between(month, stop, inclusive='left').all():
        raise ValueError('OMNI source contains UTC outside its named month')


def load_omni_config(path):
    """Resolve OMNI config paths relative to the JSON file."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    for key in ('cache_directory', 'output_csv', 'output_manifest'):
        if cfg.get(key):
            cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    return cfg


def prepare_omni_upstream(config):
    """Prepare a dense minute grid over [start_utc, stop_utc), plus provenance.

    Missing records remain explicit invalid minutes. Source/metadata hashes,
    code, and output hashes allow verified offline reuse. Use a new output
    path after changing scientific settings or source bytes.
    """
    cfg = dict(config)
    start, stop = [pd.to_datetime(cfg[k], utc=True) for k in ('start_utc', 'stop_utc')]
    if pd.isna(start) or pd.isna(stop) or start >= stop or start != start.floor('min') or stop != stop.floor('min'):
        raise ValueError('Use ordered, minute-aligned start_utc and exclusive stop_utc')
    timeout, retries = cfg.get('timeout_s', 120), cfg.get('retries', 4)
    if not np.isfinite(timeout) or timeout <= 0 or not isinstance(retries, int) or retries < 1:
        raise ValueError('Use positive timeout_s and integer retries')
    if not isinstance(cfg.get('download', True), bool):
        raise ValueError('download must be a JSON boolean')
    cache = Path(cfg['cache_directory']).resolve()
    target = Path(cfg['output_csv']).resolve()
    metadata = Path(cfg.get('output_manifest', target.with_suffix('.manifest.json'))).resolve()
    if metadata == target:
        raise ValueError('OMNI CSV and manifest must use different paths')
    settings = dict(start_utc=start.isoformat(), stop_utc=stop.isoformat(),
                    cache_directory=str(cache), product=OMNI_PRODUCT, source=OMNI_ROOT)
    code_hash = file_hash(__file__)
    if target.exists() or metadata.exists():
        if not target.is_file() or not metadata.is_file():
            raise ValueError('Incomplete existing OMNI output; use a new output path')
        saved = json.loads(metadata.read_text())
        unchanged = (saved['settings'] == settings and saved['code_sha256'] == code_hash
                     and saved['output']['path'] == str(target)
                     and saved['output']['sha256'] == file_hash(target)
                     and all(Path(p['path']).is_file() and file_hash(p['path']) == p['sha256'] for p in saved['inputs']))
        if not unchanged:
            raise ValueError('Existing OMNI inputs, settings, code or output changed; use a new output path')
        return saved
    months = pd.date_range(start.replace(day=1).normalize(), stop-pd.Timedelta(minutes=1), freq='MS')
    tables, inputs, sources = [], [], []
    for month in months:
        table, source, sidecar = _month_source(month, cache, download=cfg.get('download', True),
                                               timeout_s=timeout, retries=retries)
        tables.append(table.loc[table['utc'].ge(start) & table['utc'].lt(stop)])
        sources.append(source)
        inputs.extend([dict(path=source['path'], sha256=source['sha256']),
                       dict(path=str(sidecar.resolve()), sha256=file_hash(sidecar))])
        print(f'OMNI {month:%Y-%m}: {len(table)} source minutes', flush=True)
    grid = pd.date_range(start, stop, freq='min', inclusive='left', name='utc')
    out = pd.concat(tables, ignore_index=True).set_index('utc').reindex(grid).reset_index()
    missing = out['quality_valid'].isna()
    out.loc[missing, 'quality_reason'] = 'missing_omni_minute'
    out['quality_valid'] = out['quality_valid'].eq(True)
    out['upstream_interval_stop_utc'] = out['utc'] + pd.Timedelta(minutes=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    out.to_csv(temporary, index=False)
    temporary.replace(target)
    manifest = dict(settings=settings, code_sha256=code_hash, inputs=inputs, sources=sources,
                    created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    time_convention='Earth bow-shock-nose arrival, start of one-minute average; no extra shift',
                    pressure_source='Published HRO word 28 flow pressure; no fallback or rescaling',
                    pressure_convention='P[nPa] = 2e-6 * Np[cm^-3] * Vp[km/s]^2 (SPDF HRO convention)',
                    imf_frame='GSM (HRO word 19)', interval='[start_utc, stop_utc)',
                    quality_basis='Finite positive pressure, finite GSM Bz, documented spacecraft IDs and positive non-fill point counts',
                    documentation='https://omniweb.gsfc.nasa.gov/html/omni_min_data.html',
                    units=dict(pressure_npa='nPa', bz_gsm_nt='nT'),
                    coverage=dict(expected_minutes=len(out), valid_minutes=int(out['quality_valid'].sum()),
                                  missing_source_minutes=int(missing.sum()),
                                  invalid_minutes=int((~out['quality_valid']).sum())),
                    output=dict(path=str(target), sha256=file_hash(target), rows=len(out)))
    write_json(metadata, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args(argv)
    result = prepare_omni_upstream(load_omni_config(args.config))
    print(f"Prepared {result['coverage']['valid_minutes']}/{result['coverage']['expected_minutes']} valid OMNI minutes")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
