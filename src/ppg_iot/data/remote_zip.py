"""Seekable, bounded HTTP ranges using verified TLS and native curl."""
from __future__ import annotations
import io
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


class HTTPRangeFile(io.RawIOBase):
    """Read selected ZIP members without downloading unrelated raw-device files.

    Enforce 206/Content-Range/ETag and exact byte counts for every block.
    Native curl uses Windows trust on Windows; TLS verification stays enabled.
    """
    def __init__(self, url: str, *, block_size: int = 8 * 1024**2) -> None:
        super().__init__()
        self.url, self.block_size, self.position = url, block_size, 0
        self.curl = shutil.which("curl.exe") or shutil.which("curl")
        if not self.curl:
            raise RuntimeError("HTTP range acquisition requires curl with verified TLS")
        self.cache_index, self.cache = None, b""
        self.downloaded_bytes = 0
        self.size = None
        self.etag = None
        # Nextcloud treats end=0 as an open-ended range; use a 16-byte probe.
        _, headers = self._range(0, 15)
        self.expected_checksums = headers.get("oc-checksum")

    def _range(self, start: int, end: int) -> tuple[bytes, dict]:
        with tempfile.TemporaryDirectory(prefix="ppg-range-") as folder:
            directory = Path(folder)
            body, head = directory / "body.bin", directory / "headers.txt"
            command = [self.curl, "--silent", "--show-error", "--fail", "--location",
                       "--retry", "3", "--retry-all-errors", "--retry-delay", "1",
                       "--connect-timeout", "15", "--max-time", "60",
                       "--max-filesize", str(end - start + 1),
                       "--range", f"{start}-{end}", "--dump-header", str(head),
                       "--output", str(body), self.url]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise OSError(f"Range download failed: {completed.stderr.strip()}")
            raw = head.read_text(encoding="latin1").replace("\r\n", "\n")
            blocks = [block for block in raw.split("\n\n") if block.startswith("HTTP/")]
            final = blocks[-1] if blocks else ""
            lines = final.splitlines()
            if not lines or lines[0].split()[1] != "206":
                raise ValueError("Server did not honor HTTP Range; refusing corrupt concatenation")
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.lower()] = value.strip()
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("content-range", ""))
            if not match or tuple(map(int, match.groups()[:2])) != (start, end):
                raise ValueError("Wrong Content-Range")
            size = int(match[3])
            etag = headers.get("etag")
            if not etag:
                raise ValueError("Missing ETag for source consistency")
            if self.size is not None and (size != self.size or etag != self.etag):
                raise ValueError("Remote archive changed during acquisition")
            self.size, self.etag = size, etag
            data = body.read_bytes()
            if len(data) != end - start + 1:
                raise ValueError("Incomplete HTTP range body")
            self.downloaded_bytes += len(data)
            return data, headers

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        target = (offset if whence == io.SEEK_SET else
                  self.position + offset if whence == io.SEEK_CUR else
                  self.size + offset if whence == io.SEEK_END else None)
        if target is None or target < 0:
            raise ValueError("Invalid remote ZIP seek")
        self.position = target
        return target

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("Read from closed range stream")
        size = max(0, self.size - self.position) if size < 0 else min(
            size, max(0, self.size - self.position))
        pieces = []
        while size:
            block = self.position // self.block_size
            if block != self.cache_index:
                start = block * self.block_size
                self.cache, _ = self._range(start, min(self.size - 1, start + self.block_size - 1))
                self.cache_index = block
            offset = self.position % self.block_size
            count = min(size, len(self.cache) - offset)
            pieces.append(self.cache[offset:offset + count])
            self.position += count
            size -= count
        return b"".join(pieces)
