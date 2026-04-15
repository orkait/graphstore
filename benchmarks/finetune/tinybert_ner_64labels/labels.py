from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SEMANTIC_LABELS: tuple[str, ...] = (
    "PROG_LANG",
    "FRAMEWORK",
    "LIBRARY",
    "VARIABLE_NAME",
    "FUNCTION_NAME",
    "CLASS_NAME",
    "DATA_TYPE",
    "API_ENDPOINT",
    "HTTP_METHOD",
    "HTTP_STATUS",
    "DATABASE_TYPE",
    "DB_QUERY_KEYWORD",
    "CONTAINER_TECH",
    "ORCHESTRATOR",
    "CLOUD_PROVIDER",
    "CLOUD_SERVICE",
    "CLI_COMMAND",
    "VERSION_CONTROL",
    "CI_CD_TOOL",
    "LOG_LEVEL",
    "EXCEPTION_TYPE",
    "MEMORY_ADDRESS",
    "PORT_NUMBER",
    "FILE_PATH",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "DOMAIN_NAME",
    "AUTH_MECHANISM",
    "ENCRYPTION_ALGO",
    "SECURITY_PROTOCOL",
    "IAM_ROLE",
    "SSH_KEY",
    "HARDWARE_CHIPSET",
    "OS_DISTRO",
    "PROJECT_CODENAME",
    "TICKET_ID",
    "ORGANIZATION",
    "VENDOR",
    "PRODUCT_NAME",
    "SERVICE_NAME",
    "ROLE_TITLE",
    "DEPARTMENT",
    "LEGAL_STATUTE",
    "CONTRACT_TYPE",
    "PERSON",
    "CITY",
    "COUNTRY",
    "REGION",
    "BUILDING",
    "DATE",
    "TIME",
    "DURATION",
    "FREQUENCY",
    "CURRENCY",
    "AMOUNT_NUM",
    "UNIT_MEASURE",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "URL",
    "EVENT_NAME",
    "LANGUAGE_HUMAN",
    "NATIONALITY",
    "COLOR",
    "MISC_ENTITY",
)


def _build_bio_labels() -> tuple[str, ...]:
    labels = ["O"]
    for semantic in SEMANTIC_LABELS:
        labels.append(f"B-{semantic}")
        labels.append(f"I-{semantic}")
    return tuple(labels)


BIO_LABELS: tuple[str, ...] = _build_bio_labels()
NUM_LABELS = len(BIO_LABELS)


@dataclass(frozen=True)
class LabelMaps:
    semantic_labels: tuple[str, ...]
    bio_labels: tuple[str, ...]
    label2id: Mapping[str, int]
    id2label: Mapping[int, str]


def build_label_maps() -> LabelMaps:
    bio_labels = BIO_LABELS
    label2id = {label: idx for idx, label in enumerate(bio_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return LabelMaps(
        semantic_labels=SEMANTIC_LABELS,
        bio_labels=bio_labels,
        label2id=label2id,
        id2label=id2label,
    )
