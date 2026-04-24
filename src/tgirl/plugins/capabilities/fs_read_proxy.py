"""Read-only filesystem proxy (FILESYSTEM_READ capability).

PRP Task 6, Y3. Implemented internally using ``pathlib`` — tgirl's own trusted
use of it. Plugin authors see only the curated read surface below; no write,
symlink, chmod, or chown methods are exposed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def read_text(path: str, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def exists(path: str) -> bool:
    return Path(path).exists()


def is_file(path: str) -> bool:
    return Path(path).is_file()


def is_dir(path: str) -> bool:
    return Path(path).is_dir()


def iterdir(path: str) -> Iterator[str]:
    """Yield child names as strings (not Path objects)."""
    for p in Path(path).iterdir():
        yield str(p)


def glob(root: str, pattern: str) -> Iterator[str]:
    for p in Path(root).glob(pattern):
        yield str(p)


def stat(path: str) -> dict[str, int | float]:
    """Return selected stat fields as a plain dict (no os.stat_result exposed)."""
    s = Path(path).stat()
    return {
        "size": s.st_size,
        "mtime": s.st_mtime,
        "ctime": s.st_ctime,
        "mode": s.st_mode,
    }
