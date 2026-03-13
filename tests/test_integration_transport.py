"""Integration tests for tgirl.transport — larger vocabs and zero coupling."""

from __future__ import annotations

import inspect

import pytest
import torch


class TestImportPaths:
    """Verify public exports are importable from tgirl and tgirl.transport."""

    def test_import_from_transport_module(self) -> None:
        from tgirl.transport import (
            TransportConfig,
            TransportResult,
            redistribute_logits,
        )

        assert TransportConfig is not None
        assert TransportResult is not None
        assert redistribute_logits is not None

    def test_import_from_tgirl_package(self) -> None:
        from tgirl import (
            TransportConfig,
            TransportResult,
            redistribute_logits,
        )

        assert TransportConfig is not None
        assert TransportResult is not None
        assert redistribute_logits is not None


class TestModuleDocstring:
    """Module-level docstring exists."""

    def test_has_docstring(self) -> None:
        import tgirl.transport

        assert tgirl.transport.__doc__ is not None
        assert len(tgirl.transport.__doc__) > 0


class TestZeroCouplingIntegration:
    """transport.py has zero imports from tgirl — verified via AST."""

    def test_no_tgirl_imports_ast(self) -> None:
        """Parse transport.py with ast to verify no tgirl imports."""
        import ast
        import pathlib

        transport_path = pathlib.Path(
            inspect.getfile(
                __import__("tgirl.transport", fromlist=["transport"])
            )
        )
        tree = ast.parse(transport_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("tgirl"), (
                        f"Found tgirl import: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("tgirl"), (
                    f"Found tgirl import: from {node.module}"
                )

    def test_transport_usable_standalone(self) -> None:
        """transport module works without importing any other tgirl module."""
        # Import only transport — no registry, grammar, compile
        import importlib
        import sys

        # Temporarily remove tgirl.registry etc from sys.modules to verify
        # transport doesn't depend on them at import time
        saved = {}
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("tgirl.") and mod_name != "tgirl.transport":
                saved[mod_name] = sys.modules.pop(mod_name)
        # Also remove the parent tgirl package temporarily
        if "tgirl" in sys.modules:
            saved["tgirl"] = sys.modules.pop("tgirl")

        try:
            # Force reimport
            if "tgirl.transport" in sys.modules:
                del sys.modules["tgirl.transport"]
            transport = importlib.import_module("tgirl.transport")
            # Verify it works
            logits = torch.randn(5)
            mask = torch.tensor([True, True, False, False, True])
            emb = torch.randn(5, 8)
            result = transport.redistribute_logits(logits, mask, emb)
            assert isinstance(result, transport.TransportResult)
        finally:
            # Restore modules
            sys.modules.update(saved)


class TestLargerVocabStress:
    """End-to-end tests with realistic vocabulary sizes."""

    @pytest.mark.parametrize("vocab_size", [1000, 2000, 5000])
    def test_large_vocab_e2e(self, vocab_size: int) -> None:
        from tgirl.transport import redistribute_logits

        torch.manual_seed(42)
        logits = torch.randn(vocab_size)
        # 30% valid tokens
        mask = torch.rand(vocab_size) < 0.3
        # Ensure at least 2 valid
        mask[:2] = True
        embeddings = torch.randn(vocab_size, 64)

        result = redistribute_logits(logits, mask, embeddings)

        # Basic invariants
        assert result.logits.shape == (vocab_size,)
        assert (result.logits[~mask] == float("-inf")).all()
        assert result.wasserstein_distance >= 0.0

        # Valid logits should be finite
        valid_logits = result.logits[mask]
        assert torch.isfinite(valid_logits).all()

    def test_random_inputs_no_crash(self) -> None:
        """Randomized inputs should not crash."""
        from tgirl.transport import redistribute_logits

        for seed in range(10):
            torch.manual_seed(seed)
            vocab_size = torch.randint(10, 100, (1,)).item()
            logits = torch.randn(vocab_size)
            mask = torch.rand(vocab_size) < 0.4
            # Ensure at least 2 valid
            mask[:2] = True
            embeddings = torch.randn(vocab_size, 32)
            result = redistribute_logits(logits, mask, embeddings)
            assert result.logits.shape == (vocab_size,)


class TestTelemetryCompatible:
    """TransportResult fields are suitable for telemetry logging."""

    def test_result_fields_serializable(self) -> None:
        from tgirl.transport import redistribute_logits

        torch.manual_seed(42)
        logits = torch.randn(20)
        mask = torch.tensor(
            [True, True, False, False, True] * 4
        )
        embeddings = torch.randn(20, 16)
        result = redistribute_logits(logits, mask, embeddings)

        # All non-tensor fields should be JSON-serializable types
        assert isinstance(result.wasserstein_distance, float)
        assert isinstance(result.bypassed, bool)
        assert result.bypass_reason is None or isinstance(
            result.bypass_reason, str
        )
        assert isinstance(result.iterations, int)
