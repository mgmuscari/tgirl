"""Type hierarchy with O(1) subsumption for HPSG grammars.

Builds a type lattice from parsed TDL definitions with precomputed
ancestor and descendant sets for efficient subsumption checking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import structlog

from tgirl.lingo.tdl_parser import TdlDefinition

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TypeNode:
    """A single type in the hierarchy."""
    name: str
    supertypes: frozenset[str]


class TypeHierarchy:
    """Type lattice with multiple inheritance and O(1) subsumption.

    Three-pass build:
    1. Collect all type names and direct supertypes from := definitions
    2. Record :+ addenda (don't alter supertype graph in v1)
    3. Compute transitive ancestor sets via topological sort
    """

    def __init__(self, definitions: list[TdlDefinition]) -> None:
        # Pass 1: collect direct supertypes
        direct_supers: dict[str, set[str]] = {}
        all_names: set[str] = set()

        for defn in definitions:
            if defn.is_addendum:
                continue
            name = defn.name
            all_names.add(name)
            if name not in direct_supers:
                direct_supers[name] = set()
            for st in defn.supertypes:
                direct_supers[name].add(st)
                all_names.add(st)

        # Pass 2: record addenda
        self._addenda: dict[str, list[TdlDefinition]] = {}
        for defn in definitions:
            if not defn.is_addendum:
                continue
            all_names.add(defn.name)
            if defn.name not in direct_supers:
                direct_supers[defn.name] = set()
                logger.warning(
                    "Addendum for unknown type %r (no := def)",
                    defn.name,
                )
            self._addenda.setdefault(defn.name, []).append(defn)

        # Ensure all referenced supertypes have entries
        for name in list(all_names):
            if name not in direct_supers:
                direct_supers[name] = set()

        # Pass 3: compute transitive ancestors via BFS/topological order
        # ancestors[x] includes x itself (reflexive)
        self._ancestors: dict[str, frozenset[str]] = {}
        self._direct_supers = {k: frozenset(v) for k, v in direct_supers.items()}

        # Topological sort: process types whose supertypes are all resolved
        in_degree: dict[str, int] = {}
        children_of: dict[str, list[str]] = {n: [] for n in all_names}

        for name in all_names:
            # Count how many of this type's supertypes are in our type set
            count = 0
            for st in direct_supers.get(name, set()):
                if st in all_names:
                    count += 1
                    children_of[st].append(name)
            in_degree[name] = count

        queue: deque[str] = deque()
        for name in all_names:
            if in_degree[name] == 0:
                queue.append(name)

        while queue:
            name = queue.popleft()
            # Ancestors = union of all supertypes' ancestors + self
            anc: set[str] = {name}
            for st in direct_supers.get(name, set()):
                if st in self._ancestors:
                    anc.update(self._ancestors[st])
                else:
                    # Supertype not in our type set (external reference)
                    anc.add(st)
            self._ancestors[name] = frozenset(anc)

            for child in children_of[name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Handle any remaining types (cycles — shouldn't happen in valid TDL)
        for name in all_names:
            if name not in self._ancestors:
                logger.warning(
                    "Type %r in cycle, using partial ancestors",
                    name,
                )
                anc = {name}
                for st in direct_supers.get(name, set()):
                    if st in self._ancestors:
                        anc.update(self._ancestors[st])
                    else:
                        anc.add(st)
                self._ancestors[name] = frozenset(anc)

        # Build descendant sets by inverting ancestors
        self._descendants: dict[str, set[str]] = {n: set() for n in all_names}
        for name, ancs in self._ancestors.items():
            for anc in ancs:
                if anc != name and anc in self._descendants:
                    self._descendants[anc].add(name)

        # Direct children (for leaf computation)
        self._direct_children: dict[str, set[str]] = {n: set() for n in all_names}
        for name, supers in direct_supers.items():
            for st in supers:
                if st in self._direct_children:
                    self._direct_children[st].add(name)

        self._all_types = frozenset(all_names)

    def is_subtype(self, child: str, parent: str) -> bool:
        """True if child is the same as or a subtype of parent. O(1)."""
        if child not in self._ancestors:
            return False
        return parent in self._ancestors[child]

    def common_supertypes(self, a: str, b: str) -> frozenset[str]:
        """Return all types that are supertypes of both a and b."""
        anc_a = self._ancestors.get(a, frozenset())
        anc_b = self._ancestors.get(b, frozenset())
        # Exclude a and b themselves from intersection unless one is ancestor of other
        return frozenset(anc_a & anc_b - {a, b})

    def greatest_lower_bound(self, a: str, b: str) -> str | None:
        """Return the GLB of two types, or None if incompatible.

        The GLB is the most specific type that is a subtype of both a and b.
        """
        if a == b:
            return a
        # If one is subtype of the other, that's the GLB
        if self.is_subtype(a, b):
            return a
        if self.is_subtype(b, a):
            return b
        # Find types that are subtypes of both
        desc_a = self._descendants.get(a, set())
        desc_b = self._descendants.get(b, set())
        common = desc_a & desc_b
        if not common:
            return None
        # Find the most specific (deepest) — the one with fewest descendants
        best = min(common, key=lambda t: len(self._descendants.get(t, set())))
        return best

    def subtypes_of(self, type_name: str) -> frozenset[str]:
        """Return all (transitive) subtypes of a type."""
        return frozenset(self._descendants.get(type_name, set()))

    @property
    def all_types(self) -> frozenset[str]:
        """All type names in the hierarchy."""
        return self._all_types

    @property
    def leaf_types(self) -> frozenset[str]:
        """Types with no subtypes."""
        return frozenset(
            name for name in self._all_types
            if not self._direct_children.get(name)
        )
