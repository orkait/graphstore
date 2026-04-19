"""Vision sidecar: spawn a local llama.cpp OpenAI-compatible HTTP server for
image captioning. Default model is SmolVLM-500M-Instruct (~400 MB GGUF).

Architecture: a separate process owns the model weights. graphstore talks to
it over loopback HTTP via the same VisionHandler used for Ollama/vLLM/cloud.
Crash isolation, weight dedup across GraphStore instances, and server-side
batching all fall out of keeping the seam at HTTP. We never load the model
in-process.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8418
DEFAULT_MODEL_REPO = "ggml-org/SmolVLM-500M-Instruct-GGUF"
DEFAULT_MODEL_FILE = "SmolVLM-500M-Instruct-Q8_0.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-SmolVLM-500M-Instruct-f16.gguf"
DEFAULT_CHAT_FORMAT = "llava-1-5"

_CACHE_DIR_ENV = "GRAPHSTORE_VLM_CACHE_DIR"


def _cache_dir() -> Path:
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(os.path.expanduser("~/.cache/graphstore/vlm"))


def _pid_file() -> Path:
    return _cache_dir() / "sidecar.pid"


def _log_file() -> Path:
    return _cache_dir() / "sidecar.log"


@dataclass(frozen=True, slots=True)
class SidecarStatus:
    running: bool
    pid: int | None
    port: int | None
    model: str | None
    base_url: str | None


def _read_pid() -> tuple[int, int, str] | None:
    """Return (pid, port, model) tuple if a live sidecar is recorded, else None."""
    pf = _pid_file()
    if not pf.exists():
        return None
    try:
        rec = json.loads(pf.read_text())
        pid = int(rec["pid"])
        port = int(rec["port"])
        model = str(rec.get("model", DEFAULT_MODEL_FILE))
    except (ValueError, KeyError, json.JSONDecodeError):
        logger.debug("vision sidecar pid file corrupt, ignoring", exc_info=True)
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return (pid, port, model)
    return (pid, port, model)


def _write_pid(pid: int, port: int, model: str) -> None:
    _cache_dir().mkdir(parents=True, exist_ok=True)
    _pid_file().write_text(json.dumps({"pid": pid, "port": port, "model": model}))


def _clear_pid() -> None:
    pf = _pid_file()
    if pf.exists():
        pf.unlink()


def _probe(host: str, port: int, timeout: float = 1.0) -> bool:
    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError):
        return False


def download_weights(
    repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
    mmproj_file: str = DEFAULT_MMPROJ_FILE,
) -> tuple[Path, Path]:
    """Download the GGUF model + mmproj. Returns (model_path, mmproj_path).

    Uses huggingface_hub.hf_hub_download so the files land in the standard
    HF snapshot layout under the graphstore cache dir. Idempotent - re-calling
    after a successful download returns the existing paths without re-fetching.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Vision sidecar requires the `vision` extra. "
            "Install with: pip install 'graphstore[vision]'"
        ) from e

    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(repo, filename=model_file, cache_dir=str(cache))
    mmproj_path = hf_hub_download(repo, filename=mmproj_file, cache_dir=str(cache))
    return Path(model_path), Path(mmproj_path)


def start(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
    mmproj_file: str = DEFAULT_MMPROJ_FILE,
    chat_format: str = DEFAULT_CHAT_FORMAT,
    n_threads: int = 8,
    n_ctx: int = 4096,
    wait_ready: bool = True,
    ready_timeout: float = 90.0,
) -> SidecarStatus:
    """Spawn the sidecar. If already running at ``host:port``, returns its status."""
    existing = _read_pid()
    if existing is not None:
        pid, existing_port, existing_model = existing
        if _probe(host, existing_port):
            return SidecarStatus(
                running=True,
                pid=pid,
                port=existing_port,
                model=existing_model,
                base_url=f"http://{host}:{existing_port}/v1",
            )
        _clear_pid()

    if _probe(host, port):
        return SidecarStatus(
            running=True,
            pid=None,
            port=port,
            model="external",
            base_url=f"http://{host}:{port}/v1",
        )

    model_path, mmproj_path = download_weights(model_repo, model_file, mmproj_file)

    log = _log_file().open("ab")
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--clip_model_path", str(mmproj_path),
        "--chat_format", chat_format,
        "--host", host,
        "--port", str(port),
        "--n_threads", str(n_threads),
        "--n_ctx", str(n_ctx),
    ]
    logger.info("vision sidecar: spawning %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    _write_pid(proc.pid, port, model_file)

    if wait_ready:
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                _clear_pid()
                raise RuntimeError(
                    f"vision sidecar died during startup (exit={proc.returncode}); "
                    f"see {_log_file()}"
                )
            if _probe(host, port, timeout=1.0):
                break
            time.sleep(0.5)
        else:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            _clear_pid()
            raise TimeoutError(
                f"vision sidecar did not become ready within {ready_timeout:.0f}s; "
                f"see {_log_file()}"
            )

    return SidecarStatus(
        running=True,
        pid=proc.pid,
        port=port,
        model=model_file,
        base_url=f"http://{host}:{port}/v1",
    )


def stop() -> bool:
    """Stop the sidecar. Returns True iff a process was signalled."""
    existing = _read_pid()
    if existing is None:
        return False
    pid, _, _ = existing
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_pid()
        return False
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _clear_pid()
            return True
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _clear_pid()
    return True


def status(host: str = DEFAULT_HOST) -> SidecarStatus:
    existing = _read_pid()
    if existing is None:
        return SidecarStatus(running=False, pid=None, port=None, model=None, base_url=None)
    pid, port, model = existing
    alive = _probe(host, port)
    return SidecarStatus(
        running=alive,
        pid=pid if alive else None,
        port=port if alive else None,
        model=model if alive else None,
        base_url=f"http://{host}:{port}/v1" if alive else None,
    )


def resolve_base_url(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    auto_start: bool = True,
) -> str | None:
    """Return the URL to use for VisionHandler, or None if no endpoint available.

    Precedence:
      1. ``GRAPHSTORE_VISION_URL`` env var (user-configured remote endpoint)
      2. Live sidecar from recorded PID file
      3. Probe the default port (maybe the user ran ``gs vision serve`` manually)
      4. ``auto_start`` -> spawn sidecar (requires ``[vision]`` extra) and return URL
    """
    env = os.environ.get("GRAPHSTORE_VISION_URL")
    if env:
        return env.rstrip("/")
    st = status(host)
    if st.running:
        return st.base_url
    if _probe(host, port):
        return f"http://{host}:{port}/v1"
    if auto_start:
        try:
            st = start(host=host, port=port)
            return st.base_url
        except ImportError:
            return None
        except (RuntimeError, TimeoutError) as e:
            logger.warning("vision sidecar auto-start failed: %s", e)
            return None
    return None
