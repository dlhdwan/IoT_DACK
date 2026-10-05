"""Acquire official synchronized PPG-DaLiA pickles and README, recording hashes."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppg_iot.config import SUBJECTS, load_config, project_path
from ppg_iot.provenance import RunRecorder, fingerprint, now_utc, write_json
from ppg_iot.data.remote_zip import HTTPRangeFile

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip"
README_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf"
AUTHOR_PAGE = "https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html"
AUTHOR_URL = "https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP/download"

PAGE_URL = "https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia"


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Using existing source: {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "PPG-DaLiA-final-project/1.0"})
    total, last = 0, 0
    with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as stream:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status: {response.status}")
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            total += len(chunk)
            if total - last >= 256 * 1024 * 1024:
                print(f"{path.name}: {total / 1024**2:.0f} MiB received", flush=True)
                last = total
        expected = response.headers.get("Content-Length")
        if expected and total != int(expected):
            raise RuntimeError(f"Incomplete download: {total}/{expected} bytes")
    if not total:
        raise RuntimeError("Empty download")
    temporary.replace(path)
    print(f"Downloaded {path.name}: {total} bytes", flush=True)


def synchronized_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    found = {}
    for item in archive.infolist():
        match = re.fullmatch(r"(?:[^/]+/)*(S(?:[1-9]|1[0-5]))/\1\.pkl", item.filename)
        if match:
            subject = match[1]
            if subject in found:
                raise ValueError(f"Duplicate pickle member for {subject}")
            found[subject] = item
    if set(found) != set(SUBJECTS):
        raise ValueError(f"Expected 15 synchronized records, found {sorted(found)}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", default="results/data_sources.json")
    parser.add_argument("--source", choices=("author-range", "uci"), default="author-range")
    args = parser.parse_args()
    config = load_config(args.config)
    destination = project_path(args.root)
    out = project_path(args.out)
    run = RunRecorder(ROOT, out.parent / "data_acquisition", config,
                      ecode="DATA", kind="final", inputs={"config": project_path(args.config)})
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination).free < 8 * 1024**3:
            raise RuntimeError("Need at least 8 GiB free for download and synchronized pickles")
        archive_path = destination / "downloads/data.zip"
        readme = destination / "readme.pdf"
        download(README_URL, readme)
        remote = None
        if args.source == "uci":
            download(DATA_URL, archive_path)
            archive_source = archive_path
        else:
            remote = HTTPRangeFile(AUTHOR_URL)
            archive_source = remote
        sources = {}
        with zipfile.ZipFile(archive_source) as archive:
            members = synchronized_members(archive)
            required = sum(item.file_size for item in members.values())
            if shutil.disk_usage(destination).free < required + 256 * 1024**2:
                raise RuntimeError(f"Need {required} bytes free for selected pickles")
            for subject in SUBJECTS:
                member = members[subject]
                target = destination / subject / f"{subject}.pkl"
                target.parent.mkdir(parents=True, exist_ok=True)
                # Stream through ZipExtFile: extraction verifies member CRC at EOF.
                temporary = target.with_name(target.name + ".part")
                with archive.open(member) as stream, temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                if temporary.stat().st_size != member.file_size:
                    raise RuntimeError(f"Unexpected extracted size for {subject}")
                temporary.replace(target)
                sources[subject] = {**fingerprint(target), "archive_member": member.filename,
                                    "zip_crc32": f"{member.CRC:08x}"}
                print(f"Extracted {subject}: {member.file_size} bytes, CRC verified", flush=True)
        archive_info = ({**fingerprint(archive_path), "url": DATA_URL} if remote is None else
                        {"url": AUTHOR_URL, "author_page": AUTHOR_PAGE,
                         "bytes": remote.size, "etag": remote.etag,
                         "provided_checksums": remote.expected_checksums,
                         "downloaded_bytes": remote.downloaded_bytes,
                         "selection": "15 synchronized pickles; member CRC verified",
                         "whole_archive_sha256": None})
        if remote is not None:
            remote.close()
        report = {"schema_version": 1, "acquired_at": now_utc(),
                  "dataset": "PPG-DaLiA", "dataset_page": PAGE_URL,
                  "license": ("CC BY 4.0 (UCI)" if remote is None else
                              "Scientific/non-commercial use with attribution (author host)"),
                  "dataset_doi": "10.24432/C53890",
                  "archive": archive_info,
                  "readme": {**fingerprint(readme), "url": README_URL},
                  "subjects": sources,
                  "scope": "official synchronized pickles; no evaluation performed"}
        write_json(out, report)
        run.finish(evidence={"sources": out, "readme": readme})
        print(f"PASS acquisition: {out}", flush=True)
        return 0
    except Exception as exc:
        run.finish(error=f"{type(exc).__name__}: {exc}")
        print(f"FAIL: {exc}; evidence: {run.directory}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
