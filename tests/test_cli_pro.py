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


class TestProCheckEmptyCache:
    """`pro check` against an empty cache: fits=False, exit 3."""

    def test_text_output_says_calibration_missing(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path)])
        # Exit 3 = calibration missing per the design contract.
        assert r.returncode == 3
        assert "fits" in r.stdout.lower()
        assert "NO" in r.stdout
        assert "calibration missing" in r.stdout.lower() or \
               "calibration: missing" in r.stdout.lower()

    def test_json_output_is_parseable(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), "--json"])
        assert r.returncode == 3
        data = json.loads(r.stdout)
        assert data["fits"] is False
        assert data["calibration_source"] == "missing"
        assert isinstance(data["shortfalls"], list)
        assert any("calibration" in s.lower() for s in data["shortfalls"])
        # Spec echoed back so callers can verify what was checked.
        assert data["spec"]["embedder"] == "jina-v5-small"  # default

    def test_slot_override_reflected_in_json(self, tmp_path):
        r = _run([
            "check", "--cache-dir", str(tmp_path), "--json",
            "--embedder", "model2vec-256d",
            "--reranker", "none",
            "--vision", "smolvlm2-2.2b",
        ])
        data = json.loads(r.stdout)
        assert data["spec"]["embedder"] == "model2vec-256d"
        assert data["spec"]["reranker"] == "none"
        assert data["spec"]["vision"] == "smolvlm2-2.2b"


class TestProCheckHostFields:
    """JSON output must surface live host snapshot so callers can audit
    what the resolver decided against."""

    def test_host_block_present(self, tmp_path):
        r = _run(["check", "--cache-dir", str(tmp_path), "--json"])
        data = json.loads(r.stdout)
        assert "host" in data
        h = data["host"]
        for f in ("ram_total_mb", "ram_available_mb", "disk_free_mb",
                  "cpu_cores_physical", "cpu_cores_logical",
                  "gpu_ready"):
            assert f in h, f"missing host field {f}"
        assert h["ram_total_mb"] > 0
        assert h["cpu_cores_logical"] >= 1


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
