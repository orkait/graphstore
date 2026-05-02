# graphstore pro: design

Status: draft for review (not implemented)
Author: design
Targets: v0.6.0
Last updated: 2026-05-02

## Why

graphstore today ships a minimal core + many opt-in extras (`[ingest]`, `[vision]`, `[audio]`, `[gpu]`, ...). To wire it up as a complete agentic-memory store the user must:

- pick the right extras
- install models (embedder, NER, reranker, Bonsai GGUF) with separate CLI calls
- decide whether to enable GPU
- set `LD_LIBRARY_PATH`, `GRAPHSTORE_GPU=1`, `n_gpu_layers=-1` correctly per component
- size `n_ctx`, `n_batch`, batch sizes against their own RAM

This pushes too much onto the user. `pro` is a single profile that owns those decisions, runs the components together, refuses to start when the host genuinely cannot run them, and downgrades knobs (not silently, but visibly) when the host is tight.

`pro` is **opt-in**. `pip install graphstore` and `GraphStore()` keep today's behaviour. None of this proposal changes default-install code paths.

## Non-goals

- Apple Metal / AMD ROCm / Intel oneAPI support. Linux x86_64 + NVIDIA CUDA 12 only for v1.
- Auto-installing pip extras (`graphstore pro setup` will not run `pip install`).
- Hard-coded "min RAM" / "min cores" thresholds. The floor is computed from live calibration of the components the user picked, against the host's current free RAM/disk/VRAM/cores.
- Replacing the current core install. Today's `pip install graphstore` stays minimal.

## Shape of the API

### `ProSpec` — slotted choice, not a flat list

Components within graphstore are mostly mutually exclusive within a slot (one embedder at a time, one reranker, one ingest mode). A flat `components=["a", "b"]` list invites user errors like "two embedders." Slots make every choice explicit.

```python
@dataclass(frozen=True)
class ProSpec:
    embedder:     Literal[
        "jina-v5-small", "jina-v5-nano",
        "model2vec-256d", "embeddinggemma-300m",
        "fastembed-bge-small", "none",
    ] = "jina-v5-small"

    reranker:     Literal["jina-v3", "none"]                    = "jina-v3"

    ingest_mode:  Literal["bonsai", "deterministic"]            = "bonsai"
    bonsai_quant: Literal["tq1_0", "tq2_0"]                     = "tq1_0"
    bonsai_skill: Literal["lite", "full"]                       = "lite"

    vision:       Literal["smolvlm2-2.2b", "qwen-vl-3b", "none"] = "none"
    audio:        Literal["whisper-tiny", "whisper-base", "whisper-small", "none"] = "none"

    ner:          Literal["tinybert", "none"]                   = "tinybert"
```

Defaults match today's measured-best LoCoMo configuration: jina-v5-small embedder, jina-v3 reranker, Bonsai TQ1_0 with lite skill, TinyBERT NER, no vision/audio. Vision and audio default off because they're sporadic-use modalities; user opts in when they have image/audio sources.

### `HostSnapshot` — live, no assumptions

```python
@dataclass(frozen=True)
class HostSnapshot:
    ram_total_mb:      int  # psutil.virtual_memory().total
    ram_available_mb:  int  # psutil.virtual_memory().available
    disk_free_mb:      int  # shutil.disk_usage(cache_dir).free
    cpu_cores_physical: int
    cpu_cores_logical:  int
    gpu_ready:         bool                    # graphstore.gpu.setup().ready
    gpu_name:          str | None              # nvidia-smi --query-gpu=name
    gpu_vram_total_mb: int                     # 0 if no GPU
    gpu_vram_free_mb:  int                     # 0 if no GPU
    extras_installed:  set[str]                # via importlib.metadata
```

Captured at `resolve()` time via `psutil`, `shutil.disk_usage`, `os.cpu_count`, `nvidia-smi --query-gpu=memory.free,name --format=csv,noheader,nounits`, and `importlib.metadata.distributions()`. No persistent state; refreshed every call.

### `ResolvedConfig` — fit + knobs in one shot

```python
@dataclass(frozen=True)
class ResolvedConfig:
    spec:                  ProSpec
    host:                  HostSnapshot
    fits:                  bool             # False = ProUnsupportedHostError raised
    n_ctx:                 int
    bonsai_n_batch:        int
    bonsai_n_gpu_layers:   int              # -1 / N / 0
    reranker_max_len:      int
    reranker_gpu_layers:   int
    embed_batch:           int
    vision_offload:        bool             # vision sidecar on GPU?
    projected_tps:         dict[str, float] # bonsai, embedder, reranker, vision
    ram_budget_mb:         dict[str, int]   # per component
    vram_budget_mb:        dict[str, int]   # per component
    shortfalls:            list[str]        # only when fits=False
    warnings:              list[str]        # tight RAM, no GPU, partial offload, ...
    suggestions:           list[str]        # what to drop or upgrade
    calibration_source:    Literal["measured", "extrapolated", "missing"]
    calibration_age_s:     int | None       # None when calibration_source="missing"
```

`fits=False` means the host cannot run even the **minimal viable knobs** for the selected spec. That's the hard-stop. `fits=True` may still come with `warnings` (e.g. "RAM tight, n_ctx clamped to 2048; expect 30% slower throughput").

### Resolution algorithm

```
resolve(spec, host) -> ResolvedConfig:
  1. Look up calibration entry for each component in spec, keyed by
     (component_id, host_signature). If missing → calibration_source="missing".
  2. Compute MIN knobs (smallest n_ctx / batch each component still works at).
  3. Sum MIN-knob RAM/disk/VRAM for selected components.
  4. If sum > host.{ram_available, disk_free, vram_free} → fits=False, populate
     shortfalls with the components that don't fit.
  5. If fits, find MAX knobs that still fit by binary-searching n_ctx /
     n_batch / embed_batch / reranker_max_len upward, capped by sane upper
     bounds (n_ctx 32k, etc.).
  6. GPU allocation policy: layered, bonsai-first. Try to offload bonsai
     (-1). If VRAM remains, try reranker. If VRAM remains, try vision.
     Each transition logged in warnings.
  7. Compute projected_tps from calibration table interpolated at chosen
     knobs.
  8. Return ResolvedConfig.
```

This is one function, not "fit check + tier knobs separately." Knobs are the largest values that fit. If even the smallest values don't fit, `fits=False`.

### Calibration

Per-host calibration cache, populated by `graphstore pro setup` (which downloads + immediately probes each component) and refreshable via `graphstore pro probe`.

**Cache file**: `$XDG_CACHE_HOME/graphstore/calibration.json` (default `~/.cache/graphstore/calibration.json`).

**Schema**:

```json
{
  "graphstore_version": "0.6.0",
  "host_signature": "linux-x86_64-cpu_4c8t-ram_16384mb-gpu_RTX3060_12288mb",
  "measured_at": "2026-05-02T14:30:00Z",
  "components": {
    "embedder:jina-v5-small": {
      "model_id":  "jina-v5-small-retrieval",
      "disk_mb":   148,
      "ram_mb_idle":            230,
      "ram_mb_at_batch_64":     310,
      "vram_mb_full_offload":   180,
      "tps_cpu_threads": {"1": 32.0, "2": 58.0, "4": 95.0, "8": 130.0},
      "tps_gpu_full_offload":   480.0
    },
    "ingest:bonsai-tq1_0-lite": {
      "model_id":  "Ternary-Bonsai-4B-TQ1_0",
      "disk_mb":   1080,
      "ram_mb_n_ctx_2048_cpu":  1480,
      "ram_mb_n_ctx_4096_cpu":  1620,
      "vram_mb_full_offload":   1380,
      "tps_cpu_threads": {"2": 11.0, "4": 17.5, "8": 22.0, "16": 22.5},
      "tps_gpu_full_offload":   148.0
    },
    "...": "..."
  }
}
```

**Host signature** components: kernel, arch, physical/logical core count, total RAM bucket (rounded to 1 GB), GPU name + total VRAM. Different signature → cache invalidated.

**Bootstrap (cold cache)**:
- `graphstore pro check` on cold cache emits `calibration_source="missing"` warning + offers to run probe.
- `graphstore pro setup` always runs probe at the end of each component install, populating the cache as a side effect.
- `graphstore pro probe --refresh` re-runs all probes and overwrites cache.

**No hard-coded numbers anywhere.** If calibration is missing, `resolve()` returns `calibration_source="missing"` and refuses to make claims; user is told to run `pro setup` or `pro probe`. This is the "correct design" path the user asked for.

### Probe procedure (per component)

Sequential, not parallel (each component grabs RAM that may collide). Per probe:

1. Snapshot RSS + GPU VRAM free before load.
2. Load component (Bonsai → init Llama, embedder → load model, reranker → load, NER → load TinyBERT, vision → spawn sidecar, audio → load whisper).
3. Snapshot RSS + VRAM after load → idle delta.
4. Run a short workload:
   - Bonsai: 5 ingests of varied length (10/30/50 token outputs), report median TPS.
   - Embedder: embed 100 short texts batched, report TPS.
   - Reranker: rerank 50 query-doc pairs, report TPS.
   - NER: extract from 100 sentences, report TPS.
   - Vision: caption 1 small image (160×160), report seconds.
   - Audio: transcribe 5s WAV, report seconds.
5. Snapshot peak RSS during workload → resident-during-use number.
6. Tear down, free model, snapshot to confirm release.

CPU mode + GPU mode (when `gpu.is_ready()`) measured separately; cache stores both. CPU TPS measured at multiple thread counts so resolver can interpolate.

Wall time per component: 10-90s. Full pro suite cold probe: 3-8 min. Acceptable as one-time. Probe progress streams to stdout with a TTY-aware spinner.

## CLI surface

```
graphstore pro check
graphstore pro check --reranker none
graphstore pro check --vision smolvlm2-2.2b --audio whisper-base
graphstore pro check --spec ./my-spec.json
graphstore pro check --json

graphstore pro setup
graphstore pro setup --vision smolvlm2-2.2b
graphstore pro setup --skip-probe        # download models, defer probe to first use

graphstore pro probe
graphstore pro probe --refresh           # invalidate cache, re-measure
graphstore pro probe --component ingest:bonsai-tq1_0-lite

graphstore pro status
graphstore pro status --json
```

**Exit codes**:

- 0: spec fits the host (warnings allowed).
- 1: spec does not fit (`fits=False`); stderr lists shortfalls + suggestions.
- 2: `[pro]` extra not installed; stderr suggests `pip install 'graphstore[pro]'`.
- 3: calibration cache missing for selected spec; stderr suggests `pro setup` or `pro probe`.
- 4: GPU was requested explicitly via spec but driver/runtime probe failed.

`check` never installs anything. `setup` downloads models and probes. `probe` is a re-measurement that touches no model files. `status` is read-only.

### `graphstore pro check` example output

```
$ graphstore pro check

Host
  CPU         8 physical / 16 logical cores
  RAM         16384 MB total, 11200 MB available
  Disk        78400 MB free at ~/.cache/graphstore
  GPU         NVIDIA GeForce RTX 3060 (12288 MB total, 11800 MB free)

Spec
  embedder    jina-v5-small
  reranker    jina-v3
  ingest      bonsai-tq1_0 (lite)
  ner         tinybert
  vision      none
  audio       none

Resolved
  fits        YES
  n_ctx       4096          (clamped from 8192 by RAM headroom policy)
  bonsai      n_gpu_layers=-1, n_batch=512, ~148 tps projected
  embedder    embed_batch=128, ~480 tps projected (GPU)
  reranker    max_len=2048, gpu_layers=-1
  RAM budget  3120 MB / 11200 MB available
  VRAM budget 1560 MB / 11800 MB available

Warnings
  none

Calibration
  measured 2 days ago on this host (2026-04-30 09:14)

Run `graphstore pro probe --refresh` to recalibrate.
```

### `pro check` failure example

```
$ graphstore pro check

Host
  CPU         2 physical / 2 logical cores
  RAM         2048 MB total, 1100 MB available
  Disk        4200 MB free
  GPU         not detected

Spec  (default pro)
  ingest      bonsai-tq1_0 (lite)
  embedder    jina-v5-small
  reranker    jina-v3
  ner         tinybert

Resolved
  fits        NO

Shortfalls
  ingest:bonsai-tq1_0   needs 1480 MB RAM at smallest n_ctx (2048),
                        only 1100 MB available
  reranker:jina-v3      needs 800 MB RAM, would not fit alongside
                        bonsai even if RAM grew

Suggestions
  - drop reranker  (--reranker none)         saves 800 MB RAM
  - drop bonsai    (--ingest deterministic)  saves 1480 MB RAM, falls
                                             back to TinyBERT NER ingest
  - upgrade host   2048 MB total RAM is below the smallest combination
                   of pro components currently shippable

Calibration
  measured on this host 2 hours ago (cold setup)

Exit 1: spec does not fit this host.
```

The error never hand-waves about "pro needs X MB." It states what the picked spec's components actually measured at on this host and where the gap is.

## `GraphStore` integration

```python
GraphStore(profile="pro")                                    # ProSpec defaults
GraphStore(profile="pro", reranker="none", vision="smolvlm2-2.2b")
GraphStore(profile="pro", spec=ProSpec(...))                 # explicit
GraphStore()                                                 # untouched
```

Behaviour when `profile="pro"`:

1. Build `ProSpec` from kwargs + defaults.
2. `gpu.setup()` (idempotent, cached).
3. `pro.resolve(spec, HostSnapshot.capture())`.
4. If `fits=False` → raise `ProUnsupportedHostError` with the same structured message `pro check` prints. Don't fall back silently.
5. If `fits=True`:
   - Apply resolved knobs into `GraphStoreConfig` (`vector.gpu_layers`, `dsl.reranker_gpu_layers`, `vector.embedder`, `vector.embed_batch_size`, etc.).
   - Lazy-instantiate `BonsaiIngestor` on first `INGEST` (or first `RECALL_NL`-style call). Don't pay model load on `GraphStore()` construction.
   - Log `pro: ready spec=... knobs=...` at INFO; warnings at WARNING.

User-passed kwargs (`reranker_gpu_layers=-1`, `gpu_layers=N`, etc.) still override. Today's precedence (`config-file < env-var < kwargs`) extended with `pro-resolved < kwargs`. The resolver populates the config layer; user kwargs sit above it.

## Error model

```python
class ProUnsupportedHostError(GraphStoreError):
    """Selected pro spec does not fit current host. Read shortfalls /
    suggestions on the .resolved attribute for structured output."""
    resolved: ResolvedConfig

class ProCalibrationMissing(GraphStoreError):
    """No calibration data for the selected components on this host.
    Run `graphstore pro setup` or `graphstore pro probe`."""

class ProExtraNotInstalled(GraphStoreError):
    """[pro] extra not installed. pip install 'graphstore[pro]'."""
    missing: list[str]
```

All three carry a structured payload so callers can format their own UX (CLI prints a table, library users can swallow + remediate).

## `[pro]` extra in `pyproject.toml`

```toml
[project.optional-dependencies]
pro = [
  "graphstore[ingest,vision,audio,embedders-extra,gpu]",
]
```

Wraps the existing extras. No new pip dependencies introduced. `pip install 'graphstore[pro]'` unrolls to today's install with everything attached.

The Bonsai GGUF model is NOT a pip dep (it's 1.1 GB of weights, not a wheel). Downloaded by `graphstore pro setup` from `superkaiii/Ternary-Bonsai-4B-TQ1_0-GGUF` on Hugging Face Hub via `huggingface_hub.snapshot_download`. Cached under `~/.cache/graphstore/models/`.

Same for jina-v5-small, tinybert, jina-reranker-v3, smolvlm2-2.2b, whisper-base. All routed through the existing `graphstore install-embedder` / `graphstore vision serve --download` paths plus a thin orchestrator in `pro setup`.

## Failure modes + warning policy

| condition | behavior |
|---|---|
| `[pro]` extra not installed | `ProExtraNotInstalled`, exit 2, hint to `pip install 'graphstore[pro]'`. |
| `[pro]` installed, calibration missing | `ProCalibrationMissing`, exit 3, hint to run `pro setup`. |
| Spec fits, GPU not detected | `fits=True`, `warnings=["GPU not detected; bonsai on CPU at ~22 tps. CUDA setup: ..."]`. Run. |
| Spec fits, GPU detected, partial offload | `fits=True`, `warnings=["VRAM tight; offloaded 22/32 bonsai layers, reranker on CPU. ~80 tps projected vs 148 tps full offload."]`. Run. |
| Spec doesn't fit | `fits=False`, exit 1, structured shortfalls + suggestions. Refuse. |
| GPU spec explicit (`force_gpu=True`) but probe fails | exit 4, full diagnostic from `gpu.setup().error`. Refuse. |
| Calibration > 30 days old | `warnings=["calibration 35 days old; run pro probe --refresh"]`. Run anyway. |
| Calibration host signature mismatch (host changed) | treated as missing. |

## Test plan

| layer | tests |
|---|---|
| `ProSpec` | construction, slot validation, frozen, JSON round-trip via msgspec |
| `HostSnapshot.capture()` | mock `psutil` / `shutil` / `nvidia-smi` / `gpu.is_ready`; verify each field captured; verify graceful `gpu_*=0` when no GPU |
| `host_signature()` | stable across runs, changes when CPU count / RAM bucket / GPU model changes |
| calibration cache | read/write/atomic-replace, schema versioning, `graphstore_version` mismatch invalidation, host_signature mismatch invalidation |
| `resolve()` fit math | parametrized: (spec, host) → expected fits/knobs/shortfalls. Mock calibration. Cover: tight RAM clamps n_ctx, no GPU forces n_gpu_layers=0, layered VRAM allocation, vision-only spec on RAM-tight host |
| `resolve()` calibration_source | "measured" / "extrapolated" (cross-host) / "missing" paths |
| probe runner | each component-specific prober mockable; verify RAM delta / TPS recorded sanely; teardown frees memory |
| CLI | `pro check` / `setup` / `probe` / `status` exit codes + stdout/stderr formats; `--json` output stable |
| `GraphStore(profile="pro")` | applies resolved knobs to config; raises `ProUnsupportedHostError` when fits=False; user kwargs win over resolver |

Property-based tests (hypothesis) for `resolve()` knob monotonicity: more RAM → bigger n_ctx; more cores → bigger embed_batch; never below `min_*`.

## Migration

- v0.6.0 ships `pro` as opt-in. Today's `GraphStore()` semantics unchanged.
- `[pro]` extra is additive in `pyproject.toml`; no existing extras renamed.
- `graphstore pro` is a new top-level subcommand; doesn't conflict with existing `graphstore install-embedder` / `graphstore vision`.
- README gets a `## Pro mode` section and a new `website/docs/guides/pro-mode.md`.
- Bonsai installation docs reference `graphstore pro setup` as the recommended path; manual install steps stay documented for users who don't want pro.

## Open questions for review

1. **Slotted spec correctness**: are there mutually-exclusive choices I missed? (Multiple embedders for hybrid retrieval? Probably overkill for v1.)
2. **Calibration sequencing under tight RAM**: probe order matters when host can run components individually but not concurrently. Probably probe sequentially (load → measure → tear down → next), so peak RAM during calibration ≈ single-largest-component peak.
3. **Where the resolver lives**: pure module under `src/graphstore/pro.py` (no GraphStore dependency) so it's importable without instantiating a store. CLI handler in `src/graphstore/cli.py` adds the `pro` subcommand.
4. **`profile="pro"` precedence vs `config-file`**: a config file specifying `vector.gpu_layers=0` should still win over pro-resolved `-1`. User intent in config beats automation. Confirmed?
5. **What counts as "ingest_mode=deterministic"**: today's NER + sentence shadow path. No model download required. Is this still valuable as a fallback when bonsai doesn't fit?
6. **`ProSpec` versioning**: msgspec-encode the spec into the calibration cache so re-running with a changed spec invalidates only the affected component entries, not the whole cache.

## Out of scope (will revisit post v1)

- Apple Metal / AMD ROCm probing.
- Multi-GPU host (`CUDA_VISIBLE_DEVICES` enumeration). v1 picks GPU 0.
- Live in-process re-tier (knob downgrade when other processes start consuming RAM mid-session). v1 captures snapshot at construction.
- Calibration sharing across hosts (would need a server-side metric repo).
- Container / Kubernetes resource limit detection (`/sys/fs/cgroup/...`). v1 uses `psutil` numbers, which already respect cgroup quotas via `MemAvailable` on modern kernels but not always cleanly.
