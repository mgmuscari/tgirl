"""Public plugin API for tgirl.

Three public types are exported:

* :class:`Capability` — the seven named capabilities a plugin may request.
* :class:`PluginManifest` — the TOML-loaded declaration of a single plugin.
* :class:`CapabilityGrant` — the runtime-effective capability set for one load.

Deferred to Task 13: re-export from ``tgirl.__init__`` (public top-level API).
"""

from __future__ import annotations

from tgirl.plugins.types import Capability, CapabilityGrant, PluginManifest

__all__ = ["Capability", "CapabilityGrant", "PluginManifest"]
