"""Core plugin data types: Capability, PluginManifest, CapabilityGrant.

PRP: plugin-architecture Task 1.

Pure data module — does NOT import from sandbox/registry internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class Capability(str, Enum):
    """The seven named capabilities a plugin may request.

    String values are hyphenated lowercase (user-facing TOML config uses
    hyphens). The hyphenated form is the canonical wire format — logs,
    error messages, and TOML all use it.
    """

    FILESYSTEM_READ = "filesystem-read"
    FILESYSTEM_WRITE = "filesystem-write"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    ENV = "env"
    CLOCK = "clock"
    RANDOM = "random"


@dataclass(frozen=True)
class PluginManifest:
    """Immutable declaration of a plugin's identity and capability claims.

    ``kind`` resolves cross-platform ambiguity:
      - ``"module"`` forces ``importlib.import_module`` (dotted path semantics).
      - ``"file"`` forces ``spec_from_file_location`` (filesystem path semantics).
      - ``"auto"`` applies the detection heuristic in the loader (Task 4).
    """

    name: str
    module: str
    allow: frozenset[Capability]
    kind: Literal["module", "file", "auto"] = "auto"
    source_path: Path | None = None


@dataclass(frozen=True)
class CapabilityGrant:
    """Runtime-effective capability set for a single plugin load.

    The grant is computed in ``load_plugin`` from ``manifest.allow`` and the
    server's ``--allow-capabilities`` flag (Task 9). ``CLOCK`` and ``RANDOM``
    are always present.
    """

    capabilities: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.CLOCK, Capability.RANDOM})
    )

    @classmethod
    def zero(cls) -> CapabilityGrant:
        """The default grant: CLOCK + RANDOM, nothing else."""
        return cls(
            capabilities=frozenset({Capability.CLOCK, Capability.RANDOM})
        )
