"""Native LinGO GSU -- TDL-to-Token Constraint Compiler.

Standalone module that reads HPSG grammars in TDL format (such as the
English Resource Grammar) and produces per-token valid masks through
the GrammarState protocol. Zero imports from other tgirl modules.
"""

from __future__ import annotations
