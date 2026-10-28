"""Inventory and mirror all 19 official LP Level-0 volumes with checksum checks.

No date inferred from the calibrated archive truncates this raw inventory.
The volume MD5 catalogs enumerate the original files, including merged MAG/ER
telemetry, SUNPULSE, geometry, commands, indexes, documentation and errata.
Network activity occurs only when these functions are explicitly called.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .pds_archive import archive_links, _fetch


PDS_ROOT = "https://pds-geosciences.wustl.edu/lunar/prospectorcd/"
VOLUMES = tuple(f"lp_{number:04d}" for number in range(1, 20))


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_relative(value):
    """Reject remote paths which could escape a destination or alias a file."""
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or ":" in value
            or "\x00" in value or "?" in value or "#" in value):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return path.as_posix()


def parse_volume_checksums(text, volume):
    """Parse GNU/BSD MD5 listings and check case-insensitive path uniqueness."""
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{32})\s+\*?(.+)", line)
        if match:
            checksum, name = match.groups()
        else:
            match = re.fullmatch(r"MD5\s*\((.+)\)\s*=\s*([0-9a-fA-F]{32})", line)
            if not match:
                raise ValueError(f"Unrecognized MD5 catalog row: {line!r}")
            name, checksum = match.groups()
        name = name.strip().replace("\\", "/")
        if name.startswith("./"):
            name = name[2:]
        parts = name.split("/")
        if parts[0].lower() == volume.lower():
            name = "/".join(parts[1:])
        name = safe_relative(name)
        key = name.lower()
        if key in rows:
            raise ValueError(f"Duplicate path in {volume} checksum catalog: {name}")
        rows[key] = dict(volume=volume, volume_path=name, md5=checksum.lower())
    for directory, prefix in (("merged", "m"), ("sunpulse", "s")):
        labels = {key[:-4] for key in rows if key.startswith(directory + "/")
                  and re.fullmatch(prefix + r"\d{7}\.lbl", PurePosixPath(key).name)}
        binaries = {key[:-2] for key in rows if key.startswith(directory + "/")
                    and re.fullmatch(prefix + r"\d{7}\.b", PurePosixPath(key).name)}
        if not labels or labels != binaries:
            raise ValueError(f"Missing or unpaired {directory} products in {volume}")
    return list(rows.values())


def _resolve_url(root, relative, listings):
    """Use actual listing case rather than assume index/MD5 case is an HTTP path."""
    current = root
    for component in PurePosixPath(relative).parts:
        if current not in listings:
            listings[current] = archive_links(current)
        matches = [link for link in listings[current]
                   if urlparse(link).netloc == urlparse(root).netloc
                   and unquote(urlparse(link).path.rstrip("/").split("/")[-1]).lower() == component.lower()
                   and urlparse(link).path.startswith(urlparse(current).path)]
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise ValueError(f"Expected one official listing link for {current}{component}")
        current = matches[0]
    return current


def discover_mission(directory, *, volumes=VOLUMES):
    """Build a fresh, complete inventory from official volume checksum catalogs.

    Inventory-only operation fetches directory listings and small MD5 catalogs,
    never telemetry binaries. Failed discovery never writes a complete plan.
    """
    directory = Path(directory)
    volumes = tuple(volumes)
    if not volumes or len(set(volumes)) != len(volumes) or not set(volumes) <= set(VOLUMES):
        raise ValueError("Choose unique official LP_0001 through LP_0019 volumes")
    links = archive_links(PDS_ROOT)
    listings = {PDS_ROOT: links}
    products, catalogs = [], []
    for volume in volumes:
        choices = sorted(link for link in links if re.fullmatch(
            re.escape(volume) + r"_\d{6}\.md5", PurePosixPath(urlparse(link).path).name, re.I))
        if not choices:
            raise ValueError(f"No official MD5 catalog found for {volume}")
        url = choices[-1]
        data = _fetch(url)
        name = PurePosixPath(urlparse(url).path).name
        target = directory / "catalogs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        catalogs.append(dict(volume=volume, url=url, path=str(target.resolve()),
                             sha256=hashlib.sha256(data).hexdigest()))
        for row in parse_volume_checksums(data.decode("ascii"), volume):
            relative = volume + "/" + row["volume_path"]
            row.update(relative_path=relative, url=_resolve_url(PDS_ROOT, relative, listings))
            products.append(row)
        print(f"Inventoried {volume}", flush=True)
    plan = dict(schema_version=1, source=PDS_ROOT, volumes=list(volumes),
                official_inventory_complete=set(volumes) == set(VOLUMES),
                catalogs=catalogs, products=products)
    validate_plan(plan)
    write_json(directory / "inventory.json", plan)
    return plan


def validate_plan(plan):
    if plan.get("source") != PDS_ROOT or not plan.get("products"):
        raise ValueError("Expected an official, nonempty LP volume inventory")
    seen = set()
    for row in plan["products"]:
        relative = safe_relative(row["relative_path"])
        url = urlparse(row["url"])
        if (relative.lower() in seen or relative.split("/")[0] not in VOLUMES
                or not row["url"].startswith(PDS_ROOT) or url.query or url.fragment
                or not re.fullmatch(r"[0-9a-f]{32}", row["md5"])):
            raise ValueError("Invalid or duplicate official inventory entry")
        seen.add(relative.lower())
    expected = set(plan["volumes"])
    if ({row["volume"] for row in plan["products"]} != expected
            or {row['volume'] for row in plan['catalogs']} != expected
            or len(plan['catalogs']) != len(expected)):
        raise ValueError("Volume inventory is inconsistent")
    for catalog in plan["catalogs"]:
        if file_hash(catalog["path"]) != catalog["sha256"]:
            raise ValueError("Official checksum catalog changed locally")
        actual = {row["volume_path"].lower(): row["md5"] for row in plan["products"]
                  if row["volume"] == catalog["volume"]}
        original = {row["volume_path"].lower(): row["md5"] for row in
                    parse_volume_checksums(Path(catalog["path"]).read_text(), catalog["volume"])}
        if actual != original:
            raise ValueError("Inventory differs from official checksum catalog")
    if any(row['relative_path'] != row['volume'] + '/' + row['volume_path'] for row in plan['products']):
        raise ValueError('Inventory path differs from its official volume path')


def _download_one(row, root, *, timeout, retries):
    target = root / safe_relative(row["relative_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    result = dict(row, path=str(target.resolve()))
    algorithm = 'sha256' if 'sha256_expected' in row else 'md5'
    expected = row['sha256_expected'] if algorithm == 'sha256' else row['md5']
    if target.is_file() and file_hash(target, algorithm) == expected:
        return dict(result, status="available", bytes=target.stat().st_size, sha256=file_hash(target))
    partial = target.with_name(target.name + ".partial")
    for attempt in range(retries):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "moonbroom/0.1 LP archive recovery"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(row["url"], headers=headers), timeout=timeout) as response:
                resume = offset > 0 and response.status == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise OSError("Server returned an inconsistent download range")
                # A server ignoring Range sends the whole file: restart safely.
                received = 0
                with partial.open("ab" if resume else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                length = response.headers.get("Content-Length")
                if length is not None and received != int(length):
                    raise OSError("Truncated HTTP response")
            if file_hash(partial, algorithm) != expected:
                partial.unlink()
                raise OSError(f"{algorithm.upper()} checksum mismatch")
            partial.replace(target)
            return dict(result, status="available", bytes=target.stat().st_size, sha256=file_hash(target))
        except (OSError, TimeoutError, ValueError) as error:
            # 416 means the partial file cannot be resumed. Restart on retry.
            if getattr(error, "code", None) == 416 and partial.exists():
                partial.unlink()
            if attempt + 1 == retries:
                return dict(result, status="failed", error=f"{type(error).__name__}: {error}")
            time.sleep(min(2 ** attempt, 8))


def download_mission(plan, directory, *, workers=4, timeout=120, retries=4):
    """Stream downloads, resume partials, verify official MD5 and record SHA-256."""
    validate_plan(plan)
    if not 1 <= workers <= 16 or retries < 1 or timeout <= 0:
        raise ValueError("Use 1..16 workers and positive timeout/retries")
    directory = Path(directory)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, row, directory / "archive",
                               timeout=timeout, retries=retries) for row in plan["products"]]
        for future in as_completed(futures):
            row = future.result()
            results[row["relative_path"]] = row
            if len(results) % 25 == 0 or row["status"] == "failed":
                write_json(directory / "downloads.json", list(results.values()))
                print(f"Checked {len(results)}/{len(futures)} archive files", flush=True)
    ordered = [results[row["relative_path"]] for row in plan["products"]]
    write_json(directory / "downloads.json", ordered)
    return ordered
