"""Plugin error types.

PRP: plugin-architecture Task 4.
"""

from __future__ import annotations

from tgirl.plugins.types import Capability


class PluginLoadError(RuntimeError):
    """Any failure during plugin module loading (discovery, import, register)."""


class PluginASTRejectedError(PluginLoadError):
    """Gate 1 rejection: plugin AST contains a forbidden construct.

    Attributes:
        plugin_name: name of the plugin whose AST was rejected.
        offense: short machine-readable reason code.
        detail: human-readable detail including line number when available.
    """

    def __init__(
        self, plugin_name: str, offense: str, detail: str
    ) -> None:
        self.plugin_name = plugin_name
        self.offense = offense
        self.detail = detail
        super().__init__(
            f"plugin {plugin_name!r} rejected at load ({offense}): {detail}"
        )


class CapabilityDeniedError(PluginLoadError):
    """Gate 3 rejection: plugin tried to invoke a capability it lacks.

    Attributes:
        capability: the missing capability.
        caller: dotted reference to the call site (e.g. "socket.create_connection").
        remediation_hint: one-line guidance to the plugin author or operator.
    """

    def __init__(
        self,
        capability: Capability | str,
        caller: str,
        remediation_hint: str,
    ) -> None:
        self.capability = capability
        self.caller = caller
        self.remediation_hint = remediation_hint
        cap_str = (
            capability.value if isinstance(capability, Capability) else str(capability)
        )
        super().__init__(
            f"capability {cap_str!r} required for {caller!r}: {remediation_hint}"
        )
