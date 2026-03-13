"""
Utilities for defining and working with multiple annotation rating scales.
"""

import re
from typing import List, Tuple, TypedDict

import pandas as pd


class RatingScale(TypedDict):
    name: str
    labels: List[str]
    column_name: str


def _slugify_scale_name(scale_name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", scale_name.strip()).strip("_").lower()
    return slug or "rating"


def _parse_labels(labels_text: str) -> List[str]:
    labels: List[str] = []
    for raw_label in labels_text.replace("\n", ",").split(","):
        label = raw_label.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def parse_rating_scales(
    scales_text: str, annotator_name: str
) -> Tuple[List[RatingScale], List[str]]:
    """
    Parse the rating scale definition entered by the user.

    Supported formats:
    - Legacy single-scale mode: ``0, 1, 2``
    - Multi-scale mode: one ``name: labels`` definition per line
    """
    errors: List[str] = []
    normalized_annotator = annotator_name.strip()
    normalized_text = scales_text.strip()

    if not normalized_annotator:
        return [], ["Please confirm an annotator name before defining rating scales."]

    if not normalized_text:
        return [], ["Please define at least one rating scale."]

    non_empty_lines = [
        line.strip() for line in normalized_text.splitlines() if line.strip()
    ]
    uses_named_scales = any(":" in line for line in non_empty_lines)
    scales: List[RatingScale] = []
    seen_columns = set()

    if uses_named_scales:
        if any(":" not in line for line in non_empty_lines):
            return [], [
                "Use one `dimension: label1, label2` definition per line when defining multiple scales."
            ]

        for line in non_empty_lines:
            scale_name_raw, labels_text = line.split(":", 1)
            scale_name = scale_name_raw.strip()
            labels = _parse_labels(labels_text)

            if not scale_name:
                errors.append(f"Missing scale name in line: `{line}`")
                continue

            if not labels:
                errors.append(
                    f"Scale `{scale_name}` needs at least one label after the colon."
                )
                continue

            column_name = (
                f"Rater_{normalized_annotator}_{_slugify_scale_name(scale_name)}"
            )
            if column_name in seen_columns:
                errors.append(
                    f"Scale `{scale_name}` creates a duplicate column name. Please rename that scale."
                )
                continue

            seen_columns.add(column_name)
            scales.append(
                {
                    "name": scale_name,
                    "labels": labels,
                    "column_name": column_name,
                }
            )
    else:
        labels = _parse_labels(normalized_text)
        if not labels:
            return [], ["Please provide at least one label."]

        scales.append(
            {
                "name": "Overall",
                "labels": labels,
                "column_name": f"Rater_{normalized_annotator}",
            }
        )

    return scales, errors


def ensure_rating_columns(
    df: pd.DataFrame, rating_scales: List[RatingScale]
) -> pd.DataFrame:
    """
    Ensure the dataframe contains all columns required by the active rating scales.
    """
    for scale in rating_scales:
        if scale["column_name"] not in df.columns:
            df[scale["column_name"]] = pd.NA
    return df


def get_rating_column_names(rating_scales: List[RatingScale]) -> List[str]:
    return [scale["column_name"] for scale in rating_scales]


def is_row_fully_annotated(
    df: pd.DataFrame, idx: int, rating_scales: List[RatingScale]
) -> bool:
    if not rating_scales:
        return False

    return all(pd.notna(df.at[idx, scale["column_name"]]) for scale in rating_scales)
