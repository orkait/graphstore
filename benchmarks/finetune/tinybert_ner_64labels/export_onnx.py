from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from benchmarks.finetune.tinybert_ner_64labels.labels import build_label_maps


def export_with_optimum(model_dir: str | Path, output_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(model_dir),
        "--task",
        "token-classification",
        "--opset",
        "17",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    return output_dir


@dataclass
class CalibrationTextReader:
    tokenizer_dir: Path
    texts: list[str]
    max_length: int = 256
    input_names: set[str] | None = None

    def __post_init__(self) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_dir / "tokenizer.json"))
        self._index = 0
        self._input_names = self.input_names or set()

    def get_next(self):
        if self._index >= len(self.texts):
            return None
        text = self.texts[self._index]
        self._index += 1
        enc = self._tokenizer.encode(text)
        import numpy as np

        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        return feed


def quantize_static_int8(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    calibration_texts: Sequence[str],
    model_file: str = "model.onnx",
    quantized_file: str = "model_int8.onnx",
    max_length: int = 256,
) -> Path:
    import numpy as np  # noqa: F401
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_model = model_dir / model_file
    output_model = output_dir / quantized_file
    import onnxruntime as ort

    input_names = {item.name for item in ort.InferenceSession(str(input_model)).get_inputs()}
    reader = CalibrationTextReader(
        model_dir,
        list(calibration_texts),
        max_length=max_length,
        input_names=input_names,
    )

    quantize_static(
        model_input=str(input_model),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return output_model


def _collect_calibration_texts(dataset_path: str | Path, limit: int = 128) -> list[str]:
    path = Path(dataset_path)
    texts: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if len(texts) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tokens = obj.get("tokens")
            if isinstance(tokens, list):
                texts.append(" ".join(str(token) for token in tokens))
    return texts[:limit]


def export_and_quantize(
    trained_model_dir: str | Path,
    export_dir: str | Path,
    *,
    calibration_dataset: str | Path,
) -> Path:
    export_with_optimum(trained_model_dir, export_dir)
    texts = _collect_calibration_texts(calibration_dataset)
    if not texts:
        raise ValueError("no calibration texts found for static quantization")
    quantized_dir = Path(export_dir) / "quantized"
    quantized_dir.mkdir(parents=True, exist_ok=True)
    quantize_static_int8(export_dir, quantized_dir, calibration_texts=texts)
    maps = build_label_maps()
    (quantized_dir / "config.json").write_text(
        json.dumps(
            {
                "id2label": {str(k): v for k, v in maps.id2label.items()},
                "label2id": dict(maps.label2id),
                "num_labels": len(maps.bio_labels),
            },
            indent=2,
        )
    )
    return quantized_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinybert_ner_export")
    parser.add_argument("--trained-model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--calibration-dataset", required=True)
    args = parser.parse_args(argv)
    export_and_quantize(
        args.trained_model_dir,
        args.export_dir,
        calibration_dataset=args.calibration_dataset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
