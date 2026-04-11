from __future__ import annotations

from collections.abc import Iterable, Sequence


CANONICAL_CLASS_COLUMNS: list[str] = [
    "class_healthy",
    "class_front_heart_attack",
    "class_first_degree_av_block",
    "class_bottom_heart_attack",
    "class_atrial_fibrillation",
    "class_complete_right_conduction_disorder",
    "class_incomplete_right_conduction_disorder",
    "class_complete_left_conduction_disorder",
]

FILE_COLUMN_KEYS = ("base", "dat", "hea")


def infer_label_columns(columns: Sequence[str] | Iterable[str]) -> list[str]:
    cols = list(columns)
    available = set(cols)

    canonical = [col for col in CANONICAL_CLASS_COLUMNS if col in available]
    if canonical:
        return canonical

    return [col for col in cols if col.startswith("class_")]


def infer_file_columns(columns: Sequence[str] | Iterable[str]) -> dict[str, str | None]:
    cols = list(columns)
    available = set(cols)

    return {
        "base": "local_record_base" if "local_record_base" in available else None,
        "dat": next((c for c in cols if c.endswith("_dat_file")), None),
        "hea": next((c for c in cols if c.endswith("_hea_file")), None),
    }


def missing_canonical_columns(columns: Sequence[str] | Iterable[str]) -> list[str]:
    available = set(columns)
    return [col for col in CANONICAL_CLASS_COLUMNS if col not in available]

