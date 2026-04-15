from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from benchmarks.finetune.tinybert_ner_64labels.labels import build_label_maps


def load_onnx_classifier(model_dir: str | Path, onnx_file: str = "model_int8.onnx"):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(model_dir / onnx_file))
    return tokenizer, session


def predict(text: str, model_dir: str | Path, onnx_file: str = "model_int8.onnx") -> dict:
    import numpy as np

    tokenizer, session = load_onnx_classifier(model_dir, onnx_file=onnx_file)
    enc = tokenizer.encode(text)
    feed = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
    }
    input_names = {item.name for item in session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
    logits = session.run(None, feed)[0][0]
    pred_ids = logits.argmax(axis=-1)
    maps = build_label_maps()
    return {
        "tokens": tokenizer.encode(text).tokens,
        "labels": [maps.id2label[int(i)] for i in pred_ids],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinybert_ner_eval")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--onnx-file", default="model_int8.onnx")
    args = parser.parse_args(argv)
    result = predict(args.text, args.model_dir, onnx_file=args.onnx_file)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
