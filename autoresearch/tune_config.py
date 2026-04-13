"""Optuna config tuner for GraphStore retrieval.

Optimizes retrieval config knobs on LongMemEval using model2vec (CPU, fast).
Finds the best config that maximizes recall_any@5.

Usage:
    python -m autoresearch.tune_config --trials 100
    python -m autoresearch.tune_config --trials 50 --per-category 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import optuna


_DSL_OUTPUT_KEYS = (
    "retrieval_depth",
    "recall_depth",
    "max_query_entities",
    "recency_boost_k",
    "recall_decay",
    "recency_half_life_days",
    "similar_to_oversample",
    "lexical_search_oversample",
    "retrieval_strategy",
    "fusion_method",
    "rrf_k",
    "recency_mode",
    "nucleus_expansion",
    "nucleus_hops",
    "nucleus_max_neighbors",
)

_VECTOR_OUTPUT_KEYS = ("search_oversample",)


def build_output_config(params: dict) -> dict:
    """Convert tuned params into graphstore.json override shape."""
    config_out = {
        "dsl": {k: v for k, v in params.items() if k in _DSL_OUTPUT_KEYS},
        "vector": {k: v for k, v in params.items() if k in _VECTOR_OUTPUT_KEYS},
    }
    return {k: v for k, v in config_out.items() if v}


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    retrieval_depth = trial.suggest_int("retrieval_depth", 2, 12)
    search_oversample = trial.suggest_int("search_oversample", 5, 20)
    recall_depth = trial.suggest_int("recall_depth", 1, 4)
    max_query_entities = trial.suggest_int("max_query_entities", 1, 6)
    recency_boost_k = trial.suggest_int("recency_boost_k", 1, 4)
    recall_decay = trial.suggest_float("recall_decay", 0.3, 0.95)
    recency_half_life_days = trial.suggest_float("recency_half_life_days", 7.0, 90.0, log=True)
    similar_to_oversample = trial.suggest_int("similar_to_oversample", 2, 8)
    lexical_search_oversample = trial.suggest_int("lexical_search_oversample", 2, 8)
    retrieval_strategy = trial.suggest_categorical(
        "retrieval_strategy",
        ["remember", "remember_graph", "remember_recency", "remember_lexical", "full"],
    )
    fusion_method = trial.suggest_categorical("fusion_method", ["rrf", "weighted"])
    recency_mode = trial.suggest_categorical("recency_mode", ["multiplicative", "additive"])
    nucleus_expansion = trial.suggest_categorical("nucleus_expansion", [False, True])
    if fusion_method == "rrf":
        rrf_k = trial.suggest_float("rrf_k", 20.0, 100.0)
    else:
        rrf_k = 60.0
    if nucleus_expansion:
        nucleus_hops = trial.suggest_int("nucleus_hops", 1, 2)
        nucleus_max_neighbors = trial.suggest_int("nucleus_max_neighbors", 1, 5)
    else:
        nucleus_hops = 1
        nucleus_max_neighbors = 3

    from benchmarks.framework.adapters.graphstore_ import GraphStoreAdapter
    from benchmarks.framework.datasets import load_longmemeval
    from benchmarks.framework.runner import run_benchmark

    dataset = load_longmemeval(
        args.data_path,
        variant="s",
        per_category=args.per_category,
    )

    config = {
        "embedder": "model2vec",
        "ceiling_mb": 256,
        "retrieval_depth": retrieval_depth,
        "search_oversample": search_oversample,
        "recall_depth": recall_depth,
        "max_query_entities": max_query_entities,
        "recency_boost_k": recency_boost_k,
        "recall_decay": recall_decay,
        "recency_half_life_days": recency_half_life_days,
        "similar_to_oversample": similar_to_oversample,
        "lexical_search_oversample": lexical_search_oversample,
        "retrieval_strategy": retrieval_strategy,
        "fusion_method": fusion_method,
        "rrf_k": rrf_k,
        "recency_mode": recency_mode,
        "nucleus_expansion": nucleus_expansion,
        "nucleus_hops": nucleus_hops,
        "nucleus_max_neighbors": nucleus_max_neighbors,
    }

    adapter = GraphStoreAdapter(config=config)
    result = run_benchmark(adapter, dataset, k=5, config=config, progress_every=999)
    accuracy = result.quality.accuracy

    # Log per-category
    cats = {cat: b.accuracy for cat, b in result.quality._categories.items()}
    trial.set_user_attr("per_category", cats)
    trial.set_user_attr("elapsed", round(result.total_elapsed_s, 1))

    return accuracy


def main():
    parser = argparse.ArgumentParser(prog="tune_config")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--per-category", type=int, default=5, help="records per category per trial")
    parser.add_argument("--data-path", default="/home/kai/longmemeval-data")
    parser.add_argument("--output", default="autoresearch/tuned_config.json")
    args = parser.parse_args()

    print(f"Optuna config tuner: {args.trials} trials, {args.per_category}/category")
    print(f"Dataset: {args.data_path}")

    study = optuna.create_study(direction="maximize", study_name="graphstore-config")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST: accuracy={best.value:.4f} (trial {best.number})")
    print(f"Params: {json.dumps(best.params, indent=2)}")
    print(f"Per-category: {best.user_attrs.get('per_category', {})}")
    print(f"{'='*60}")

    # Save best config as graphstore.json format (diffs only)
    config_out = build_output_config(best.params)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config_out, indent=2) + "\n")
    print(f"Saved to {out_path}")

    # Print top 5 trials
    print("\nTop 5 trials:")
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]:
        cats = t.user_attrs.get("per_category", {})
        print(f"  #{t.number}: acc={t.value:.4f} elapsed={t.user_attrs.get('elapsed', '?')}s {cats}")


if __name__ == "__main__":
    main()
