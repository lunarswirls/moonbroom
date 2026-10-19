"""Discover and cache official Lunar Prospector archive products on explicit request"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen
import hashlib
import re
import time

import pandas as pd


def _fetch(url, *, timeout=120, retries=3):
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 4))


def archive_links(url, *, cache_dir=None):
    """Resolve case-preserving links from an official PDS directory listing"""
    target = None
    if cache_dir is not None:
        target = Path(cache_dir) / (hashlib.sha256(url.encode()).hexdigest() + ".html")
    if target is not None and target.exists():
        data = target.read_bytes()
    else:
        data = _fetch(url)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    links = re.findall(r'href=["\']([^"\']+)', data.decode(errors="replace"), re.I)
    return [urljoin(url, value) for value in links if not value.startswith("?")]


def discover_lp_products(date, *, cache_dir=None, max_raw_files=None,
                          collections=("3d", "mag", "high", "low"), min_raw_hour=0):
    """Inventory raw/sun-pulse records and calibrated references for one mission date

    max_raw_files limits a study inventory; None includes all files beginning
    on that date. Include neighboring dates when continuous coverage is needed.
    Directory listings supply actual filename case and existence.
    """
    date = pd.Timestamp(date).normalize().tz_localize(None)
    if not pd.Timestamp("1998-01-16") <= date <= pd.Timestamp("1999-07-29"):
        raise ValueError("Choose a date in the archived LP calibrated mission")
    if max_raw_files is not None and (not isinstance(max_raw_files, int) or max_raw_files < 1):
        raise ValueError("max_raw_files must be positive or None")
    volume = (date.year - 1998)*12 + date.month
    root = f"https://pds-geosciences.wustl.edu/lunar/prospectorcd/lp_{volume:04d}/"
    block = f"d{date.dayofyear//10:02d}x/"
    prefix = date.strftime("%y%j")
    links = archive_links(root + "merged/" + block, cache_dir=cache_dir)
    labels = sorted(url for url in links if re.fullmatch("m" + prefix + r"\d{2}\.lbl", Path(urlparse(url).path).name, re.I))
    labels = [url for url in labels if int(Path(urlparse(url).path).stem[-2:]) >= min_raw_hour]
    if max_raw_files is not None:
        labels = labels[:max_raw_files]
    if not labels:
        raise FileNotFoundError(f"No merged LP products beginning on {date.date()}")
    sun = archive_links(root + "sunpulse/" + block, cache_dir=cache_dir)
    rows = []
    for label in labels:
        stem = Path(urlparse(label).path).stem
        chosen = [url for url in links if Path(urlparse(url).path).stem.lower() == stem.lower()]
        chosen += [url for url in sun if Path(urlparse(url).path).stem.lower() == "s" + stem[1:].lower()]
        for url in chosen:
            name = Path(urlparse(url).path).name
            rows.append((str(date.date()), "raw" if name.lower().startswith("m") else "sunpulse", name, url))
    start = date.replace(day=1).dayofyear
    stop = (date + pd.offsets.MonthEnd(0)).dayofyear
    month = f"{start:03d}_{stop:03d}" + ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][date.month - 1]
    names = {"3d": ("lp-er-calibrated/data-3deleflux", "3d"),
             "mag": ("lp-mag-calibrated/data-lunarcrds", "ma"),
             "high": ("lp-er-calibrated/data-highresflux", "eh"),
             "low": ("lp-er-calibrated/data-lowresflux", "el")}
    for collection in collections:
        directory, tag = names[collection]
        url = f"https://pds-ppi.igpp.ucla.edu/data/{directory}/{date.year}/{month}/"
        desired = tag + date.strftime("%y%m%d")
        selected = [link for link in archive_links(url, cache_dir=cache_dir)
                    if Path(urlparse(link).path).stem.lower() == desired]
        if not selected:
            raise FileNotFoundError(f"No {collection} reference on {date.date()}")
        rows.extend((str(date.date()), collection, Path(urlparse(link).path).name, link) for link in selected)
    return pd.DataFrame(rows, columns=["date", "kind", "filename", "url"]).drop_duplicates("url")


def cache_pds_products(manifest, directory, *, workers=4):
    """Download an explicit product inventory, keeping hashes and failures per file

    Existing files are reused. Partial downloads never replace a completed
    product. This function has no network side effects until called.
    """
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        target = directory / row.filename
        if target.name != row.filename or row.filename in (".", ".."):
            raise ValueError("Product filenames must be basenames")
        result = row._asdict()
        try:
            if not target.exists():
                data = _fetch(row.url)
                partial = target.with_name(target.name + ".partial")
                partial.write_bytes(data)
                partial.replace(target)
            data = target.read_bytes()
            result.update({"path": str(target.resolve()), "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest(), "status": "available", "error": None})
        except (OSError, TimeoutError) as error:
            result.update({"path": str(target.resolve()), "bytes": None, "sha256": None,
                           "status": "failed", "error": str(error)})
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fetch, manifest.itertuples(index=False)))
    return pd.DataFrame(rows)
