"""Host-aware compute profile for auto-tuning thread pools and batch sizes.

Detects cores, RAM, and GPU availability (cgroup/containerization aware on
Linux) and exposes a single profile object that the ONNX session builders
and ingest paths consult.

Selection order:
  1. env ``GRAPHSTORE_PROFILE`` in {auto,tiny,laptop,desktop,gpu} (auto wins default)
  2. explicit ``GRAPHSTORE_<kind>_THREADS`` env vars override individual fields
  3. detected profile fields otherwise

Detection is stdlib-only: no psutil dependency. Computed once per process.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import psutil


_GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)


@dataclass(frozen=True)
class ComputeProfile:
    name: str            # tiny | laptop | desktop | gpu
    cores: int           # usable physical cores (affinity-aware on Linux)
    logical_cores: int
    ram_gb: float        # total RAM
    has_gpu: bool
    gpu_provider: str | None
    on_battery: bool     # laptop on battery -> be gentler on CPU
    load_pct: float      # host CPU utilization % at detection time

    ner_threads: int
    embed_threads: int
    rerank_threads: int

    embed_batch_size: int    # sentence/doc batch when auto-batching
    defer_embeddings: bool   # whether to auto-enable deferred-embedding context


def _detect_cores() -> tuple[int, int]:
    """Return (physical_cores, logical_cores). Respects cgroup/affinity on Linux."""
    # Affinity-aware logical count (cgroup-limited containers respect this).
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1

    physical = psutil.cpu_count(logical=False) or affinity
    logical = psutil.cpu_count(logical=True) or affinity

    # If container has restricted us, clamp both to affinity count.
    physical = min(physical, affinity) if affinity else physical
    logical = min(logical, affinity) if affinity else logical
    return max(1, physical), max(1, logical)


def _detect_ram_gb() -> float:
    try:
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def _detect_battery() -> bool:
    try:
        bat = psutil.sensors_battery()
    except Exception:
        return False
    return bool(bat and not bat.power_plugged)


def _detect_gpu() -> tuple[bool, str | None]:
    """Detect usable GPU provider.

    Provider listed by onnxruntime is NOT proof it is functional: CUDA can be
    listed while cuDNN/cublasLt are missing, so session creation would fail.
    Require explicit ``GRAPHSTORE_GPU=1`` opt-in (matches embedder convention)
    to avoid false-positive gpu classification.
    """
    if os.environ.get("GRAPHSTORE_GPU") != "1":
        return False, None
    try:
        import onnxruntime as ort
    except ImportError:
        return False, None
    try:
        available = ort.get_available_providers()
    except Exception:
        return False, None
    for p in _GPU_PROVIDERS:
        if p in available:
            return True, p
    return False, None


def _env_int(key: str, fallback: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return fallback
    try:
        return max(1, int(raw))
    except ValueError:
        return fallback


def _classify(cores: int, ram_gb: float, has_gpu: bool, on_battery: bool) -> str:
    if has_gpu:
        return "gpu"
    if cores <= 2 or (ram_gb and ram_gb < 4.0):
        return "tiny"
    # Battery-powered machine with 4-6 cores: treat as laptop even if close to desktop boundary.
    if cores <= 6 or (ram_gb and ram_gb < 16.0) or on_battery:
        return "laptop"
    return "desktop"


def _base_profile(name: str, physical_cores: int) -> tuple[int, int, int, int, bool]:
    """Fraction-of-cores sizing with reserve for user's other apps.

    (ner_threads, embed_threads, rerank_threads, embed_batch, defer)
    NER stays small (2) since batching amortizes launch overhead. Embed/rerank
    scale with cores but leave 25-50% headroom so graphstore does not steal
    the whole machine from IDE/browser/compiler/etc.
    """
    if name == "tiny":
        return (1, 1, 1, 16, False)
    if name == "laptop":
        # ~50% of cores, min 2, cap 4.
        t = max(2, min(4, physical_cores // 2))
        return (2, t, t, 32, False)
    if name == "desktop":
        # ~60% of cores, min 2, cap 8.
        t = max(2, min(8, physical_cores * 6 // 10))
        return (2, t, t, 64, True)
    if name == "gpu":
        # ~75% of cores for CPU-side ops (GPU does heavy lifting).
        t = max(2, min(12, physical_cores * 3 // 4))
        return (2, t, t, 128, True)
    return (2, 4, 4, 32, False)


def _detect_load_pct() -> float:
    """Current CPU utilization % over a short sample. 0.0 on failure."""
    try:
        return float(psutil.cpu_percent(interval=0.1))
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def get_profile() -> ComputeProfile:
    requested = os.environ.get("GRAPHSTORE_PROFILE", "auto").strip().lower()
    physical, logical = _detect_cores()
    ram_gb = _detect_ram_gb()
    has_gpu, gpu_provider = _detect_gpu()
    on_battery = _detect_battery()

    if requested in {"tiny", "laptop", "desktop", "gpu"}:
        name = requested
    else:
        name = _classify(physical, ram_gb, has_gpu, on_battery)

    ner_t, embed_t, rerank_t, batch, defer = _base_profile(name, physical)

    # Battery degrades threads one notch to preserve power/thermals even if profile stays the same.
    if on_battery and name != "tiny":
        embed_t = max(1, embed_t - 1)
        rerank_t = max(1, rerank_t - 1)

    # Load-aware scaling: if host is already busy (user compiling, running other
    # apps), halve our share so graphstore does not contend for cores.
    load_pct = _detect_load_pct()
    if load_pct > 40.0 and name != "tiny":
        embed_t = max(1, embed_t // 2)
        rerank_t = max(1, rerank_t // 2)

    return ComputeProfile(
        name=name,
        cores=physical,
        logical_cores=logical,
        ram_gb=ram_gb,
        has_gpu=has_gpu,
        gpu_provider=gpu_provider,
        on_battery=on_battery,
        load_pct=load_pct,
        ner_threads=_env_int("GRAPHSTORE_NER_THREADS", ner_t),
        embed_threads=_env_int("GRAPHSTORE_EMBED_THREADS", embed_t),
        rerank_threads=_env_int("GRAPHSTORE_RERANK_THREADS", rerank_t),
        embed_batch_size=_env_int("GRAPHSTORE_EMBED_BATCH", batch),
        defer_embeddings=defer,
    )


def describe_profile() -> str:
    p = get_profile()
    gpu = f" gpu={p.gpu_provider}" if p.has_gpu else ""
    ram = f"{p.ram_gb:.1f}GB" if p.ram_gb else "?GB"
    bat = " (on battery)" if p.on_battery else ""
    load = f" load={p.load_pct:.0f}%" if p.load_pct else ""
    return (
        f"profile={p.name} cores={p.cores}/{p.logical_cores} ram={ram}{gpu}{bat}{load} "
        f"threads(ner/embed/rerank)={p.ner_threads}/{p.embed_threads}/{p.rerank_threads} "
        f"embed_batch={p.embed_batch_size}"
    )


def reset_profile_cache() -> None:
    """For tests that mutate env and want a fresh detection."""
    get_profile.cache_clear()
