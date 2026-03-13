"""
Utilities for configuring and preparing multi-dimension evaluation mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TypedDict

import pandas as pd


LABEL_TYPE_OPTIONS = ["Integer", "Float", "Text"]
MISSING_TEXT_VALUES = {"", "none", "nan", "<na>", "n/a", "null"}


class EvaluationMapping(TypedDict):
    id: str
    name: str
    llm_field: str
    human_columns: List[str]
    label_type: str


@dataclass
class PreparedEvaluationData:
    mapping: EvaluationMapping
    analysis_data: pd.DataFrame
    labels: List[Any]
    warnings: List[str]


def clear_evaluation_result_cache(session_state: Any) -> None:
    """
    Remove cached evaluation results from Streamlit session state.
    """
    for key in list(session_state.keys()):
        if key.startswith("evaluation_results_"):
            del session_state[key]


def create_mapping_id() -> str:
    return uuid.uuid4().hex[:8]


def _default_mapping_name(index: int, llm_field: str) -> str:
    return llm_field or f"Mapping {index + 1}"


def _select_default_llm_field(
    selected_fields: Sequence[str],
    index: int,
    legacy_label_column: Optional[str] = None,
) -> str:
    if legacy_label_column is not None and legacy_label_column in selected_fields:
        return legacy_label_column

    fallback_index = min(index, len(selected_fields) - 1)
    return selected_fields[fallback_index]


def create_default_mapping(
    selected_fields: Sequence[str],
    annotation_columns: Sequence[str],
    index: int = 0,
    legacy_label_column: Optional[str] = None,
    legacy_label_type: Optional[str] = None,
) -> EvaluationMapping:
    llm_field: str = _select_default_llm_field(
        selected_fields=selected_fields,
        index=index,
        legacy_label_column=legacy_label_column,
    )
    label_type = (
        legacy_label_type if legacy_label_type in LABEL_TYPE_OPTIONS else "Text"
    )
    return {
        "id": create_mapping_id(),
        "name": _default_mapping_name(index, llm_field),
        "llm_field": llm_field,
        "human_columns": list(annotation_columns),
        "label_type": label_type,
    }


def sanitize_evaluation_mappings(
    raw_mappings: Optional[Sequence[Mapping[str, Any]]],
    selected_fields: Sequence[str],
    annotation_columns: Sequence[str],
    legacy_label_column: Optional[str] = None,
    legacy_label_type: Optional[str] = None,
    create_default_if_empty: bool = True,
) -> List[EvaluationMapping]:
    """
    Normalize mappings against the currently selected LLM and human columns.
    """
    if not selected_fields or not annotation_columns:
        return []

    sanitized: List[EvaluationMapping] = []
    seen_ids = set()
    raw_mappings = raw_mappings or []

    for index, raw_mapping in enumerate(raw_mappings):
        mapping_id = str(raw_mapping.get("id", "")).strip() or create_mapping_id()
        while mapping_id in seen_ids:
            mapping_id = create_mapping_id()
        seen_ids.add(mapping_id)

        raw_llm_field = raw_mapping.get("llm_field")
        llm_field = (
            raw_llm_field
            if isinstance(raw_llm_field, str) and raw_llm_field in selected_fields
            else _select_default_llm_field(
                selected_fields=selected_fields,
                index=index,
                legacy_label_column=legacy_label_column,
            )
        )

        raw_human_columns = (
            raw_mapping.get("human_columns") if "human_columns" in raw_mapping else None
        )
        if raw_human_columns is None:
            human_columns = list(annotation_columns)
        else:
            human_columns = [
                column
                for column in raw_human_columns
                if isinstance(column, str) and column in annotation_columns
            ]

        label_type = str(raw_mapping.get("label_type", legacy_label_type or "Text"))
        if label_type not in LABEL_TYPE_OPTIONS:
            label_type = (
                legacy_label_type if legacy_label_type in LABEL_TYPE_OPTIONS else "Text"
            )

        name = str(raw_mapping.get("name", "")).strip() or _default_mapping_name(
            index, llm_field
        )

        sanitized.append(
            {
                "id": mapping_id,
                "name": name,
                "llm_field": llm_field,
                "human_columns": human_columns,
                "label_type": label_type,
            }
        )

    if not sanitized and create_default_if_empty:
        sanitized.append(
            create_default_mapping(
                selected_fields=selected_fields,
                annotation_columns=annotation_columns,
                legacy_label_column=legacy_label_column,
                legacy_label_type=legacy_label_type,
            )
        )

    return sanitized


def mapping_label_type_to_alt_test(label_type: str) -> tuple[str, str]:
    normalized = label_type.lower()
    if normalized == "float":
        return "rmse", "float"
    if normalized == "integer":
        return "accuracy", "int"
    return "accuracy", "str"


def validate_krippendorff_mapping(
    mapping: EvaluationMapping, level_of_measurement: str
) -> Optional[str]:
    if len(mapping["human_columns"]) < 3:
        return "Krippendorff's alpha requires at least 3 human annotation columns."

    if mapping["label_type"] == "Text" and level_of_measurement != "nominal":
        return "Text mappings can only be used with nominal Krippendorff measurement."

    return None


def prepare_evaluation_data(
    results_df: pd.DataFrame,
    mapping: EvaluationMapping,
    row_filter: str = "complete",
    minimum_human_annotations: int = 1,
    encode_text_for_krippendorff: bool = False,
) -> PreparedEvaluationData:
    """
    Prepare evaluation data for one mapping and one metric family.
    """
    analysis_data = results_df.copy()
    warnings: List[str] = []

    for column_name, default_value in {
        "prompt_name": "streamlit_analysis",
        "iteration": 1,
        "split": "train",
        "run": 1,
    }.items():
        if column_name not in analysis_data.columns:
            analysis_data[column_name] = default_value

    if mapping["llm_field"] not in analysis_data.columns:
        warnings.append(
            f"LLM field `{mapping['llm_field']}` is missing from the current results."
        )
        return PreparedEvaluationData(
            mapping, analysis_data.iloc[0:0].copy(), [], warnings
        )

    missing_human_columns = [
        column
        for column in mapping["human_columns"]
        if column not in analysis_data.columns
    ]
    if missing_human_columns:
        warnings.append(
            f"Human columns missing from results: {', '.join(missing_human_columns)}."
        )
        return PreparedEvaluationData(
            mapping, analysis_data.iloc[0:0].copy(), [], warnings
        )

    if mapping["llm_field"] != "ModelPrediction":
        analysis_data = analysis_data.rename(
            columns={mapping["llm_field"]: "ModelPrediction"}
        )

    essential_columns = ["ModelPrediction"] + mapping["human_columns"]
    for column in essential_columns:
        coerced_series, column_warnings = _coerce_series(
            analysis_data[column],
            mapping["label_type"],
            column,
        )
        analysis_data[column] = coerced_series
        warnings.extend(column_warnings)

    if encode_text_for_krippendorff and mapping["label_type"] == "Text":
        analysis_data[essential_columns] = _encode_text_columns_for_krippendorff(
            analysis_data[essential_columns]
        )

    if row_filter == "complete":
        analysis_data = analysis_data.dropna(subset=essential_columns)
    elif row_filter == "minimum_humans":
        valid_human_count = analysis_data[mapping["human_columns"]].notna().sum(axis=1)
        analysis_data = analysis_data[
            analysis_data["ModelPrediction"].notna()
            & (valid_human_count >= minimum_human_annotations)
        ]

    labels = _extract_sorted_labels(analysis_data, essential_columns)

    return PreparedEvaluationData(mapping, analysis_data, labels, warnings)


def _normalize_missing_value(value: Any) -> Any:
    if value is None:
        return pd.NA

    try:
        if pd.isna(value):
            return pd.NA
    except TypeError:
        pass

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in MISSING_TEXT_VALUES:
            return pd.NA
        return stripped

    return value


def _extract_numeric_value(value: Any, target_type: str) -> Any:
    normalized = _normalize_missing_value(value)
    if normalized is pd.NA:
        return pd.NA

    if isinstance(normalized, bool):
        normalized = int(normalized)

    if isinstance(normalized, (int, float)):
        if target_type == "Integer":
            return int(normalized)
        return float(normalized)

    text_value = str(normalized).replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text_value)
    if not match:
        return pd.NA

    numeric_value = float(match.group(0))
    if target_type == "Integer":
        return int(numeric_value)
    return numeric_value


def _coerce_series(
    series: pd.Series,
    label_type: str,
    column_name: str,
) -> tuple[pd.Series, List[str]]:
    normalized = series.map(_normalize_missing_value)

    if label_type == "Text":
        coerced = normalized.map(
            lambda value: pd.NA if value is pd.NA else str(value)
        ).astype("string")
    elif label_type == "Integer":
        coerced = normalized.map(lambda value: _extract_numeric_value(value, "Integer"))
        coerced = pd.Series(coerced, index=series.index, dtype="Int64")
    else:
        coerced = normalized.map(lambda value: _extract_numeric_value(value, "Float"))
        coerced = pd.to_numeric(
            pd.Series(coerced, index=series.index),
            errors="coerce",
        )

    failed_mask = normalized.notna() & coerced.isna()
    warnings: List[str] = []
    if failed_mask.any():
        warnings.append(
            f"Column `{column_name}` had {int(failed_mask.sum())} value(s) that could not be coerced to {label_type}."
        )

    return coerced, warnings


def _extract_sorted_labels(
    analysis_data: pd.DataFrame, columns: Iterable[str]
) -> List[Any]:
    label_values = []
    for column in columns:
        if column in analysis_data.columns:
            label_values.extend(analysis_data[column].dropna().tolist())

    unique_values = list(dict.fromkeys(label_values))
    try:
        return sorted(unique_values)
    except TypeError:
        return sorted(unique_values, key=str)


def _encode_text_columns_for_krippendorff(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    unique_labels: List[str] = []
    for column in encoded.columns:
        values = [
            value
            for value in encoded[column].dropna().tolist()
            if value not in unique_labels
        ]
        unique_labels.extend(values)

    label_to_code: Dict[str, int] = {
        label: code for code, label in enumerate(unique_labels)
    }

    for column in encoded.columns:
        encoded[column] = encoded[column].map(
            lambda value: label_to_code.get(value, pd.NA) if pd.notna(value) else pd.NA
        )
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")

    return encoded
