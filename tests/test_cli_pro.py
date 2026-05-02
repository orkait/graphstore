"""Tests for `graphstore pro` CLI subcommand.

PR#3 ships read-only commands (check / status) end-to-end against a
mocked calibration cache. setup / probe are stubs that print the manual
fallback path; tested here to lock in the exit code (2) and the message
shape so users see actionable hints instead of a stack trace.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


GS = sys.executable  # use the venv python directly to dodge entry-point lookup


def _run(args: list[str], cwd=None) -> subprocess.CompletedProcess:
    """Run `python -m graphstore.cli pro <args>` capturing stdout+stderr."""
    return subprocess.run(
        [GS, "-m", "graphstore.cli", "pro", *args],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )


class TestProHelp:
    def test_top_level_help(self):
        r = _run(["--help"])
        assert r.returncode == 0
        assert "check" in r.stdout
        assert "setup" in r.stdout
        assert "probe" in r.stdout
        assert "status" in r.stdout

    def test_check_help_lists_slot_overrides(self):
        r = _run(["check", "--help"])
        assert r.returncode == 0
        for slot in ("--embedder", "--reranker", "--ingest-mode",
                     "--bonsai-quant", "--bonsai-skill",
                     "--vision", "--audio", "--ner",
                     "--json", "--cache-dir"):
            assert slot in r.stdout, f"missing flag {slot}"


# Slot args that produce a spec whose required_dists() are guaranteed
# present in the project's CI install (`pip install -e .[dev,playground,
# ingest]`). Default ProSpec uses ingest_mode="bonsai" which pulls
# llama-cpp-python (only in [vision] extra), so we explicitly pick
# deterministic ingest + none for vision/audio/embedder paths that need
# extras CI doesn't install. This isolates the tests to the calibration
# phase (the actual subject) instead of the extras-check phase.
_MINIMAL_DEPS_SPEC = [
    "--ingest-mode", "deterministic",
    "--embedder", "model2vec-256d",
    "--vision", "none",
    "--audio", "none",
]


class TestProCheckEmptyCache:
    """`pro check` against an empty cache: fits=False, exit 3."""

    def test_text_output_says_calibration_missing(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), *_MINIMAL_DEPS_SPEC])
        assert r.returncode == 3, f"stderr={r.stderr!r}"
        assert "fits" in r.stdout.lower()
        assert "NO" in r.stdout
        assert "calibration missing" in r.stdout.lower() or \
               "calibration: missing" in r.stdout.lower()

    def test_json_output_is_parseable(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), "--json",
                  *_MINIMAL_DEPS_SPEC])
        assert r.returncode == 3, f"stderr={r.stderr!r}"
        data = json.loads(r.stdout)
        assert data["fits"] is False
        assert data["calibration_source"] == "missing"
        assert isinstance(data["shortfalls"], list)
        assert any("calibration" in s.lower() for s in data["shortfalls"])
        assert data["spec"]["embedder"] == "model2vec-256d"

    def test_slot_override_reflected_in_json(self, tmp_path):
        r = _run([
            "check", "--cache-dir", str(tmp_path), "--json",
            "--embedder", "model2vec-256d",
            "--reranker", "none",
            "--ingest-mode", "deterministic",
            "--vision", "none", "--audio", "none",
        ])
        assert r.returncode == 3, f"stderr={r.stderr!r}"
        data = json.loads(r.stdout)
        assert data["spec"]["embedder"] == "model2vec-256d"
        assert data["spec"]["reranker"] == "none"
        assert data["spec"]["ingest_mode"] == "deterministic"


class TestProCheckHostFields:
    """JSON output must surface live host snapshot so callers can audit
    what the resolver decided against."""

    def test_host_block_present(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), "--json",
                  *_MINIMAL_DEPS_SPEC])
        assert r.returncode == 3
        data = json.loads(r.stdout)
        assert "host" in data
        h = data["host"]
        for f in ("ram_total_mb", "ram_available_mb", "disk_free_mb",
                  "cpu_cores_physical", "cpu_cores_logical",
                  "gpu_ready"):
            assert f in h, f"missing host field {f}"
        assert h["ram_total_mb"] > 0
        assert h["cpu_cores_logical"] >= 1


class TestProCheckExtrasGate:
    """Default spec needs llama-cpp-python (Bonsai ingest); CI's install
    matrix may or may not have it. Either way we lock in the contract:
    when extras are missing, exit 2 with structured error; when present,
    proceed to calibration."""

    def test_extras_missing_exits_2_with_structured_error(self, tmp_path):
        # Force a spec whose required_dists includes a definitely-absent
        # package. embedder=fastembed-bge-small needs `fastembed` which
        # is in [embedders-extra] only.
        r = _run([
            "check", "--cache-dir", str(tmp_path), "--json",
            "--embedder", "fastembed-bge-small",
            "--ingest-mode", "deterministic",
            "--vision", "none", "--audio", "none", "--reranker", "none",
            "--ner", "none",
        ])
        if r.returncode == 2:
            data = json.loads(r.stdout)
            assert data["error"] == "extra_not_installed"
            assert "fastembed" in data["missing_dists"]
        else:
            # If [embedders-extra] happens to be installed, we'd reach
            # calibration-missing instead. Both are valid path coverage.
            assert r.returncode == 3


class TestProSetupProbeStubs:
    """PR#3 ships these as stubs; lock the exit code + message shape."""

    @pytest.mark.parametrize("subcmd", ["setup", "probe"])
    def test_text_explains_pr_3_5(self, subcmd):
        r = _run([subcmd])
        assert r.returncode == 2  # not_implemented exit code
        # The message points users at the manual fallback so they can
        # still get value out of pro check today.
        msg = r.stderr if r.stderr else r.stdout
        assert "not yet implemented" in msg
        assert "PR#3.5" in msg
        assert "graphstore install-embedder" in msg

    @pytest.mark.parametrize("subcmd", ["setup", "probe"])
    def test_json_returns_structured_error(self, subcmd):
        r = _run([subcmd, "--json"])
        assert r.returncode == 2
        data = json.loads(r.stdout)
        assert data["error"] == "not_implemented"
        assert data["command"] == subcmd
        assert "PR#3.5" in data["message"]
