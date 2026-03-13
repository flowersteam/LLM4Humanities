"""
Module for handling row annotation functionality in the Manual Annotation Tool.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from transformers import pipeline

from manual_annotator.rating_scales import RatingScale, is_row_fully_annotated


EMPTY_SELECTION = "__not_rated__"


@st.cache_resource(show_spinner=False)
def get_translator():
    """
    Translator pipeline for FR -> EN.

    Returns:
        A translation pipeline
    """
    return pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")


def _format_display_value(value: object) -> str:
    if pd.isna(value):
        return "Not rated"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _rating_widget_key(column_name: str) -> str:
    return f"rating_widget_{column_name}"


def _load_row_ratings_into_state(
    df: pd.DataFrame, idx: int, rating_scales: List[RatingScale]
) -> None:
    if st.session_state.get("loaded_rating_row") == idx:
        return

    selected_scale_labels: Dict[str, str] = {}
    for scale in rating_scales:
        raw_value = df.at[idx, scale["column_name"]]
        widget_value = (
            EMPTY_SELECTION if pd.isna(raw_value) else _format_display_value(raw_value)
        )
        st.session_state[_rating_widget_key(scale["column_name"])] = widget_value
        selected_scale_labels[scale["column_name"]] = (
            "" if widget_value == EMPTY_SELECTION else widget_value
        )

    st.session_state.selected_scale_labels = selected_scale_labels
    st.session_state.loaded_rating_row = idx


def _persist_row_ratings(
    df: pd.DataFrame, idx: int, rating_scales: List[RatingScale]
) -> Dict[str, str]:
    selected_scale_labels: Dict[str, str] = {}

    for scale in rating_scales:
        widget_key = _rating_widget_key(scale["column_name"])
        selected_value = st.session_state.get(widget_key, EMPTY_SELECTION)

        if selected_value == EMPTY_SELECTION:
            df.at[idx, scale["column_name"]] = pd.NA
            selected_scale_labels[scale["column_name"]] = ""
        else:
            df.at[idx, scale["column_name"]] = selected_value
            selected_scale_labels[scale["column_name"]] = str(selected_value)

    st.session_state.selected_scale_labels = selected_scale_labels
    return selected_scale_labels


def _count_remaining_unannotated(
    df: pd.DataFrame, indices: List[int], rating_scales: List[RatingScale]
) -> int:
    return sum(not is_row_fully_annotated(df, idx, rating_scales) for idx in indices)


def annotate_rows(
    df: pd.DataFrame,
    current_index: int,
    selected_columns: List[str],
    rating_scales: List[RatingScale],
    annotator_name: str,
    selected_scale_labels: Dict[str, str],
    translated_rows: Dict[int, Dict[str, str]],
    sort_column: Optional[str] = None,
    enable_sorting: bool = False,
) -> Tuple[pd.DataFrame, int, Dict[str, str], Dict[int, Dict[str, str]]]:
    """
    Step 7: Annotate rows one by one.
    """
    st.header("Step 7: Annotate Rows")

    if not rating_scales:
        st.info("Define and apply at least one rating scale to continue.")
        st.stop()

    if "selected_scale_labels" not in st.session_state:
        st.session_state.selected_scale_labels = selected_scale_labels

    filtered_indices = list(st.session_state.get("annotated_indices", []))
    navigation_indices = filtered_indices if filtered_indices else df.index.tolist()

    if enable_sorting and sort_column and sort_column in df.columns:
        navigation_indices = (
            df.loc[navigation_indices].sort_values(by=sort_column).index.tolist()
        )
        st.session_state.sorted_indices = navigation_indices
    else:
        st.session_state.pop("sorted_indices", None)

    if not navigation_indices:
        st.warning("No rows are available with the current filters.")
        st.stop()

    current_index = max(0, min(current_index, len(navigation_indices) - 1))
    idx = navigation_indices[current_index]
    st.session_state.current_index = current_index

    _load_row_ratings_into_state(df, idx, rating_scales)

    status_parts: List[str] = []
    if filtered_indices:
        status_parts.append("filtered subset")
    if enable_sorting and sort_column:
        status_parts.append(f"sorted by {sort_column}")

    if status_parts:
        st.info(
            f"Showing row {current_index + 1} of {len(navigation_indices)} ({', '.join(status_parts)})"
        )
    else:
        st.info(f"Showing row {current_index + 1} of {len(navigation_indices)}")

    flag_col = f"Invalid_{annotator_name}"
    flagged_val = bool(df.at[idx, flag_col]) if flag_col in df.columns else False
    remaining_unannotated = _count_remaining_unannotated(
        df, navigation_indices, rating_scales
    )

    st.markdown(f"**Row Index:** {idx}")
    st.markdown(f"**Is Invalid:** {flagged_val}")
    st.markdown(f"**Remaining incomplete rows:** {remaining_unannotated}")

    st.markdown("**Current ratings:**")
    for scale in rating_scales:
        current_value = _format_display_value(df.at[idx, scale["column_name"]])
        st.write(f"`{scale['name']}`: {current_value}")

    for col in selected_columns:
        val = df.at[idx, col]
        if pd.notna(val) and isinstance(val, float) and val.is_integer():
            val = int(val)
        st.write(f"**{col}:** {val}")

    translate_row = st.checkbox("Translate this row to English", key="translate_row")
    if translate_row:
        if idx not in translated_rows:
            translator = get_translator()
            translation_dict = {}
            for col in selected_columns:
                text = str(df.at[idx, col])
                try:
                    result = translator(text)
                    translation_dict[col] = result[0]["translation_text"]
                except Exception as e:
                    st.error(f"Error translating '{col}': {e}")
                    translation_dict[col] = "[Error]"
            translated_rows[idx] = translation_dict
            st.session_state.translated_rows = translated_rows

        translations = translated_rows[idx]
        st.markdown("### Translated Content:")
        for col, tval in translations.items():
            st.write(f"**{col}:** {tval}")

    st.subheader("Ratings")
    for scale in rating_scales:
        selected_value = st.radio(
            f"{scale['name']} ({', '.join(scale['labels'])})",
            options=[EMPTY_SELECTION] + scale["labels"],
            format_func=lambda option: (
                "Not rated" if option == EMPTY_SELECTION else option
            ),
            key=_rating_widget_key(scale["column_name"]),
            horizontal=True,
        )
        selected_scale_labels[scale["column_name"]] = (
            "" if selected_value == EMPTY_SELECTION else str(selected_value)
        )

    st.session_state.selected_scale_labels = selected_scale_labels

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Previous"):
            _persist_row_ratings(df, idx, rating_scales)
            current_index = max(0, current_index - 1)
            st.session_state.current_index = current_index
            st.rerun()

    with c2:
        if st.button("Next"):
            _persist_row_ratings(df, idx, rating_scales)
            current_index = min(len(navigation_indices) - 1, current_index + 1)
            st.session_state.current_index = current_index
            st.rerun()

    with c3:
        if st.button("Next incomplete"):
            _persist_row_ratings(df, idx, rating_scales)
            found = False

            for offset in range(current_index + 1, len(navigation_indices)):
                candidate_idx = navigation_indices[offset]
                if not is_row_fully_annotated(df, candidate_idx, rating_scales):
                    current_index = offset
                    found = True
                    break

            if found:
                st.session_state.current_index = current_index
                st.rerun()
            else:
                st.warning("No incomplete rows found.")

    with c4:
        if st.button("Invalid data"):
            df.at[idx, flag_col] = True
            _persist_row_ratings(df, idx, rating_scales)
            current_index = min(len(navigation_indices) - 1, current_index + 1)
            st.session_state.current_index = current_index
            st.rerun()

    return df, current_index, selected_scale_labels, translated_rows
