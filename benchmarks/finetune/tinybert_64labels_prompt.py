from __future__ import annotations

from benchmarks.finetune.tinybert_ner_64labels.labels import SEMANTIC_LABELS

_HARD_NEGATIVES = (
    "VARIABLE_NAME vs FUNCTION_NAME",
    "CLASS_NAME vs VARIABLE_NAME",
    "PROG_LANG vs LIBRARY",
    "DOMAIN_NAME vs URL",
    "CLOUD_SERVICE vs CLOUD_PROVIDER",
    "DATE vs TIME vs DURATION",
)


def build_teacher_system_prompt() -> str:
    return (
        "You are a senior data teacher generating high-quality synthetic NER "
        "training data for a code-biased schema. "
        "Output must be strict JSONL. Each line must contain keys "
        '`tokens` and `ner_tags`. '
        "Use BIO tags only. Do not add commentary."
    )


def build_teacher_prompt(label: str, examples_per_label: int = 100) -> str:
    labels = ", ".join(SEMANTIC_LABELS)
    hard_negatives = "; ".join(_HARD_NEGATIVES)
    return (
        f"Generate exactly {examples_per_label} one-sentence examples for the label "
        f"`{label}`. "
        "Each example must be a JSON object on its own line with the fields "
        "`tokens` (list of whitespace-tokenized strings) and `ner_tags` (list of BIO tags "
        "aligned to `tokens`). "
        "The output must be valid JSONL and nothing else. "
        "The label ontology is fixed and must not be merged or renamed. "
        "Do not merge labels that only look similar. "
        "Do not emit prose explanations, markdown, or code fences. "
        "Keep all entity spans short and realistic, and make the examples code-biased when "
        "the label naturally appears in technical contexts. "
        f"Hard negatives to emphasize: {hard_negatives}. "
        f"Available labels: {labels}. "
        "If the target label does not appear in a sentence, every tag must be `O`. "
        "If the target label does appear, use the exact BIO tag for that label. "
        "Ensure at least some examples include contrasting distractor entities."
    )


def build_teacher_messages(label: str, examples_per_label: int = 100) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_teacher_system_prompt()},
        {"role": "user", "content": build_teacher_prompt(label, examples_per_label)},
    ]
