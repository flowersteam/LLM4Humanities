"""
Module for handling annotator setup functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
from typing import Tuple


def _reset_annotator_dependent_state() -> None:
    keys_to_clear = (
        "rating_scales",
        "rating_scales_applied_text",
        "selected_scale_labels",
        "loaded_rating_row",
        "current_index",
    )

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("rating_widget_"):
            del st.session_state[key]


def setup_annotator(
    df: pd.DataFrame,
    annotator_name: str,
    confirmed_annotator_name: str,
    annotated_count: int,
    unannotated_count: int,
    total_count: int,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Step 3: Set Annotator Name
    Allows the user to set their name/initials and confirm the annotator profile.

    Args:
        df: The DataFrame to annotate
        annotator_name: Current annotator name
        confirmed_annotator_name: Current confirmed annotator name
        annotated_count: Number of annotated rows
        unannotated_count: Number of unannotated rows
        total_count: Total number of rows

    Returns:
        A tuple containing:
        - df: The updated DataFrame
        - annotator_name: The annotator name
        - confirmed_annotator_name: The confirmed annotator name
    """
    st.header("Step 3: Set Annotator Name")

    annotator = st.text_input(
        "Annotator Name / Initials:",
        value=annotator_name,
        key="annotator_input",
    )

    st.markdown(
        "The tool will create one annotation column per active scale. For example, `clarity` becomes `Rater_YourName_clarity`."
    )

    annotator_name = annotator.strip()
    st.session_state.annotator_name = annotator_name

    is_confirmed = bool(annotator_name) and annotator_name == confirmed_annotator_name

    if is_confirmed:
        st.success(f"Using annotator profile `{confirmed_annotator_name}`.")
        return df, annotator_name, confirmed_annotator_name

    if confirmed_annotator_name and annotator_name != confirmed_annotator_name:
        st.info(
            "The annotator name changed. Click `Confirm Annotator Name` to switch to that annotator's columns."
        )

    if st.button("Confirm Annotator Name"):
        if annotator_name:
            flag_col = f"Invalid_{annotator_name}"
            if flag_col not in df.columns:
                df[flag_col] = False

            confirmed_annotator_name = annotator_name
            st.session_state.confirmed_annotator_name = confirmed_annotator_name

            st.session_state.annotated_count = annotated_count
            st.session_state.unannotated_count = unannotated_count
            st.session_state.total_count = total_count

            _reset_annotator_dependent_state()
            st.success(
                f"Created/Found flag column '{flag_col}' for `{annotator_name}`."
            )
            st.rerun()

        st.error("Please enter an annotator name before confirming.")

    if not annotator_name:
        st.info("Enter an annotator name to continue.")

    if not confirmed_annotator_name or annotator_name != confirmed_annotator_name:
        st.stop()

    return df, annotator_name, confirmed_annotator_name
