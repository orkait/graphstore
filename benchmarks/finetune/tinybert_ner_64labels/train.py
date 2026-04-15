from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from benchmarks.finetune.tinybert_ner_64labels.labels import NUM_LABELS, build_label_maps


def _normalize_bio_label(label: str) -> str:
    label = label.strip()
    if label == "O":
        return "O"
    if "-" not in label:
        return f"B-{label}"
    prefix, entity = label.split("-", 1)
    prefix = prefix.upper()
    if prefix not in {"B", "I"}:
        prefix = "B"
    return f"{prefix}-{entity}"


def _continuation_label(label: str) -> str:
    if label == "O":
        return "O"
    normalized = _normalize_bio_label(label)
    _, entity = normalized.split("-", 1)
    return f"I-{entity}"


def align_wordpiece_labels(
    word_ids: Sequence[int | None],
    word_labels: Sequence[str],
    label2id: Mapping[str, int],
) -> list[int]:
    aligned: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None or word_id >= len(word_labels):
            aligned.append(-100)
            previous_word_id = word_id
            continue
        label = _normalize_bio_label(word_labels[word_id])
        if word_id != previous_word_id:
            aligned.append(label2id[label])
        else:
            aligned.append(label2id[_continuation_label(label)])
        previous_word_id = word_id
    return aligned


def tokenize_and_align_labels(batch, tokenizer, label2id: Mapping[str, int]):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    aligned_labels = []
    for i, word_labels in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned_labels.append(align_wordpiece_labels(word_ids, word_labels, label2id))
    tokenized["labels"] = aligned_labels
    tokenized.pop("offset_mapping", None)
    return tokenized


@dataclass(frozen=True)
class TokenMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float


def compute_token_metrics(predictions: np.ndarray, labels: np.ndarray) -> TokenMetrics:
    pred_ids = predictions.argmax(axis=-1)
    mask = labels != -100
    if not np.any(mask):
        return TokenMetrics(0.0, 0.0, 0.0, 0.0)

    y_true = labels[mask]
    y_pred = pred_ids[mask]

    accuracy = float(np.mean(y_true == y_pred))

    positive_true = y_true != 0
    positive_pred = y_pred != 0

    tp = int(np.sum((y_true == y_pred) & positive_true))
    fp = int(np.sum(positive_pred & (y_true != y_pred)))
    fn = int(np.sum(positive_true & (y_true != y_pred)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return TokenMetrics(precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def _load_dataset(path: str | Path):
    from datasets import load_dataset

    path = Path(path)
    if path.is_dir():
        data_file = path / "train.jsonl"
    else:
        data_file = path
    if not data_file.exists():
        raise FileNotFoundError(f"dataset not found: {data_file}")
    return load_dataset("json", data_files=str(data_file))["train"]


def _build_model(base_model: str):
    from transformers import AutoModelForTokenClassification

    maps = build_label_maps()
    return AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=dict(maps.id2label),
        label2id=dict(maps.label2id),
        ignore_mismatched_sizes=True,
    )


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    base_model: str = "huawei-noah/TinyBERT_General_4L_312D",
    epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    batch_size: int = 16,
    seed: int = 42,
):
    from datasets import DatasetDict
    from transformers import (
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    dataset = _load_dataset(dataset_path)
    if "train" in dataset.features:
        raise ValueError("expected a flat JSONL dataset, not a nested train split")

    maps = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    split = dataset.train_test_split(test_size=0.05, seed=seed)
    ds = DatasetDict(train=split["train"], test=split["test"])

    def _tokenize(batch):
        return tokenize_and_align_labels(batch, tokenizer, maps.label2id)

    tokenized = ds.map(_tokenize, batched=True, remove_columns=ds["train"].column_names)

    model = _build_model(base_model)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def _metrics(eval_pred):
        logits, labels = eval_pred
        metrics = compute_token_metrics(np.asarray(logits), np.asarray(labels))
        return {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "accuracy": metrics.accuracy,
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        report_to=[],
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    return trainer, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinybert_ner_train")
    parser.add_argument("--dataset", required=True, help="Path to JSONL training data")
    parser.add_argument("--output-dir", required=True, help="Directory for the trained model")
    parser.add_argument(
        "--base-model",
        default="huawei-noah/TinyBERT_General_4L_312D",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
