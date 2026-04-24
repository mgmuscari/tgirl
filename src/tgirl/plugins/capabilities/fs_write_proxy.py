"""Read + write filesystem proxy (FILESYSTEM_WRITE capability).

PRP Task 6, Y3. Supersets the read proxy with write operations. Does NOT
expose ``symlink_to``, ``hardlink_to``, ``chmod``, or ``chown`` — those are
filesystem-layout-confusion escalation vectors deferred to a future v1.1
decision.
"""

from __future__ import annotations

from pathlib import Path

# Re-export every read method so plugin authors with FS_WRITE don't need a
# second grant to read. Source of truth is the read proxy.
from tgirl.plugins.capabilities.fs_read_proxy import (
    exists,
    glob,
    is_dir,
    is_file,
    iterdir,
    read_bytes,
    read_text,
    stat,
)


def write_text(path: str, content: str, encoding: str = "utf-8") -> int:
    return Path(path).write_text(content, encoding=encoding)


def write_bytes(path: str, data: bytes) -> int:
    return Path(path).write_bytes(data)


def mkdir(path: str, parents: bool = False, exist_ok: bool = False) -> None:
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def unlink(path: str) -> None:
    Path(path).unlink()


def rmdir(path: str) -> None:
    Path(path).rmdir()


def rename(src: str, dst: str) -> None:
    Path(src).rename(dst)


def touch(path: str, exist_ok: bool = True) -> None:
    Path(path).touch(exist_ok=exist_ok)


# Explicit __all__ — pin the public surface so the PRP's absent-vector
# requirements are not accidentally widened.
__all__ = [
    "exists",
    "glob",
    "is_dir",
    "is_file",
    "iterdir",
    "mkdir",
    "read_bytes",
    "read_text",
    "rename",
    "rmdir",
    "stat",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
]
