"""
Module for handling rating scale definition functionality in the Manual Annotation Tool.
"""

from typing import List, Tuple

import pandas as pd
import streamlit as st

from manual_annotator.rating_scales import (
    RatingScale,
    ensure_rating_columns,
    parse_rating_scales,
)


def _clear_rating_widget_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("rating_widget_"):
            del st.session_state[key]


def _render_scale_preview(rating_scales: List[RatingScale]) -> None:
    st.markdown("**Scale preview**")
    for scale in rating_scales:
        labels = ", ".join(scale["labels"])
        st.write(f"`{scale['name']}` -> {labels}  |  column: `{scale['column_name']}`")


def define_labels(
    df: pd.DataFrame,
    annotator_name: str,
    rating_scales_text: str,
    rating_scales: List[RatingScale],
) -> Tuple[pd.DataFrame, str, List[RatingScale]]:
    """
    Step 5: Define rating scales.

    Args:
        df: The dataframe being annotated
        annotator_name: Confirmed annotator name
        rating_scales_text: Raw rating scale definition text
        rating_scales: Currently applied rating scales

    Returns:
        A tuple containing:
        - df: The updated dataframe
        - rating_scales_text: The updated raw definition
        - rating_scales: The active rating scales
    """
    st.header("Step 5: Define Rating Scales")

    st.markdown(
        """
        Define the scales you want to apply to every row.

        Use either:
        - A single comma-separated label list for one overall scale, like `0, 1`
        - One scale per line in the form `name: label1, label2`, like:
          `clarity: 0, 1`
          `creativity: 0, 1, 2`
        """
    )

    updated_text = st.text_area(
        "Rating scales:",
        value=rating_scales_text,
        key="rating_scales_text",
        height=160,
        placeholder="clarity: 0, 1\ncreativity: 0, 1, 2",
    )

    parsed_scales: List[RatingScale] = []
    errors: List[str] = []
    if updated_text.strip():
        parsed_scales, errors = parse_rating_scales(updated_text, annotator_name)

    if parsed_scales:
        _render_scale_preview(parsed_scales)

    for error in errors:
        st.error(error)

    if st.button("Apply Rating Scales"):
        if errors or not parsed_scales:
            st.error("Fix the rating scale definition before applying it.")
        else:
            df = ensure_rating_columns(df, parsed_scales)
            st.session_state.rating_scales = parsed_scales
            st.session_state.rating_scales_applied_text = updated_text
            st.session_state.selected_scale_labels = {}
            st.session_state.loaded_rating_row = None
            _clear_rating_widget_state()
            st.success(f"Applied {len(parsed_scales)} rating scale(s).")
            st.rerun()

    active_scales = st.session_state.get("rating_scales", rating_scales)
    if active_scales:
        df = ensure_rating_columns(df, active_scales)

    applied_text = st.session_state.get("rating_scales_applied_text", "")
    if active_scales and updated_text != applied_text:
        st.info(
            "You changed the scale definition. Click `Apply Rating Scales` to use the new version."
        )

    if not active_scales:
        st.stop()

    return df, updated_text, active_scales
