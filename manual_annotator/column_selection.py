"""
Module for handling column selection functionality in the Manual Annotation Tool.
"""

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from manual_annotator.rating_scales import RatingScale, get_rating_column_names


def select_columns(
    df: pd.DataFrame, rating_scales: List[RatingScale], annotator_name: str
) -> Tuple[List[str], Optional[str], bool]:
    """
    Step 6: Select columns to display while annotating.
    """
    st.header("Step 6: Select Columns to Display")

    flag_col = f"Invalid_{annotator_name}"
    annotator_prefix = f"Rater_{annotator_name}"
    annotation_columns = set(get_rating_column_names(rating_scales))
    possible_columns = [
        column
        for column in df.columns
        if column not in annotation_columns
        and column != flag_col
        and column != annotator_prefix
        and not column.startswith(f"{annotator_prefix}_")
    ]

    default_value = [
        column
        for column in st.session_state.get("selected_columns", possible_columns)
        if column in possible_columns
    ]
    if not default_value:
        default_value = possible_columns

    selected_columns = st.multiselect(
        "Choose which columns to see while annotating:",
        options=possible_columns,
        default=default_value,
        key="selected_columns",
    )

    if not selected_columns:
        st.info("No columns selected. Please pick at least one.")
        st.stop()

    st.subheader("Sorting Options")
    st.markdown(
        """
        You can choose to sort the data by a specific column. This only changes the display order.
        """
    )

    enable_sorting = st.checkbox(
        "Enable sorting",
        value=st.session_state.get("enable_sorting", False),
        key="enable_sorting",
    )

    sort_column = None
    if enable_sorting and selected_columns:
        current_sort = st.session_state.get("sort_column")
        if current_sort not in selected_columns:
            current_sort = selected_columns[0]
        sort_column = st.selectbox(
            "Select column to sort by:",
            options=selected_columns,
            index=selected_columns.index(current_sort),
            key="sort_column",
        )
    else:
        st.session_state.sort_column = None

    return selected_columns, sort_column, enable_sorting
