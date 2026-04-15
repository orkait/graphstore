# TinyBERT NER 64 Labels

This workspace contains the benchmark-side finetuning pipeline for the custom 64-label TinyBERT NER model.

## Layout

- `benchmarks/finetune/tinybert_64labels_teacher.py` generates the balanced synthetic teacher dataset via LiteLLM/Ollama.
- `benchmarks/finetune/tinybert_ner_64labels/train.py` fine-tunes TinyBERT with Hugging Face `Trainer`.
- `benchmarks/finetune/tinybert_ner_64labels/export_onnx.py` exports and quantizes the trained model.
- `benchmarks/finetune/tinybert_ner_64labels/eval.py` runs a small ONNX inference smoke test.

## Venv

Create an isolated venv inside this workspace and install the finetune stack from this directory.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## LLM config

Copy the local example and fill in the provider keys:

```bash
cp benchmarks/finetune/tinybert-ner-64labels/config.example.json \
   benchmarks/finetune/tinybert-ner-64labels/config.json
```

The config uses the same two-level provider/model shape as `autoresearch/config.example.json`.
Primary teacher routing is `local_ollama` with `gemma4:31b-cloud` over LiteLLM.
Keep the `local_ollama.api_key` literal exactly the same as the reference config: `REPLACE_WITH_OLLAMA_KEY_OR_LEAVE_EMPTY`.

## Teacher generation

```bash
python -m benchmarks.finetune.tinybert_64labels_teacher \
  --out benchmarks/finetune/tinybert-ner-64labels/data/teacher.jsonl
```

The teacher reads `benchmarks/finetune/tinybert-ner-64labels/config.json` first, then falls back to `autoresearch/config.json`. You can also override with `GRAPHSTORE_LITELLM_CONFIG`.

## Training

```bash
python -m benchmarks.finetune.tinybert_ner_64labels.train \
  --dataset benchmarks/finetune/tinybert-ner-64labels/data/teacher.jsonl \
  --output-dir benchmarks/finetune/tinybert-ner-64labels/artifacts/tinybert-ner
```

## Export

```bash
python -m benchmarks.finetune.tinybert_ner_64labels.export_onnx \
  --trained-model-dir benchmarks/finetune/tinybert-ner-64labels/artifacts/tinybert-ner \
  --export-dir benchmarks/finetune/tinybert-ner-64labels/artifacts/onnx \
  --calibration-dataset benchmarks/finetune/tinybert-ner-64labels/data/teacher.jsonl
```
