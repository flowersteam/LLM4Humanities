"""
Module for handling column selection, renaming, and description functionality in the Streamlit app.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any

from qualitative_analysis import clean_and_normalize, sanitize_dataframe


def select_rename_describe_columns(
    app_instance: Any, data: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Step 2:
    1) Lets the user select which columns contain existing human annotations.
    2) Subsets the dataframe to rows that have non-NA in those annotation columns (if any).
    3) Lets the user select which columns to include for LLM analysis (excluding annotation columns).
    4) Lets user rename and describe those selected columns.
    5) Cleans and normalizes text columns.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        data: The DataFrame to process

    Returns:
        The processed DataFrame or None if no data was provided
    """
    st.markdown("### Step 2: Data Selection", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 2", expanded=True):
        if data is None:
            st.error("No dataset loaded.")
            return None

        columns = data.columns.tolist()

        # --- 2.1: Let user pick the annotation columns
        st.markdown(
            """
            Select column(s) that contain *human annotations*.
            Rows missing those annotations will be filtered out
            so that the dataset in the first part of the analysis only includes fully annotated entries.
            """,
            unsafe_allow_html=True,
        )

        app_instance.annotation_columns = st.multiselect(
            "Annotation Column(s):",
            options=columns,
            default=st.session_state.get("annotation_columns", []),
            key="annotation_columns_selection",
        )

        # Pre-init session state BEFORE creating the checkbox
        if "allow_missing_annotations" not in st.session_state:
            st.session_state["allow_missing_annotations"] = False

        allow_missing = st.checkbox(
            "Allow missing annotations (keep rows annotated by at least one selected annotator)",
            key="allow_missing_annotations",
        )

        if app_instance.annotation_columns:
            app_instance.original_data = data.copy()
            st.session_state["original_data"] = app_instance.original_data
            total_rows = len(data)

            if allow_missing:
                # at least one non-NA among selected columns
                filtered_data = data.dropna(
                    how="all", subset=app_instance.annotation_columns
                )
                st.write(
                    f"**Filtered** dataset to keep rows with at least one annotation in {app_instance.annotation_columns}."
                )
            else:
                # non-NA in all selected columns
                filtered_data = data.dropna(subset=app_instance.annotation_columns)
                st.write(
                    f"**Filtered** dataset to keep only rows with annotations in all of {app_instance.annotation_columns}."
                )

            filtered_rows = len(filtered_data)
            st.write(
                f"Filtered dataset: {filtered_rows} rows kept, {total_rows - filtered_rows} rows filtered out."
            )
            data = filtered_data

            st.info("Evaluation label types are configured per mapping in Step 4.")

        # Store final annotation columns and the toggle in session
        st.session_state["annotation_columns"] = app_instance.annotation_columns

        # 2.2: Select the columns that will be analyzed (exclude annotation columns)
        columns_for_analysis = [
            c for c in data.columns if c not in app_instance.annotation_columns
        ]

        st.markdown(
            """
            Now, select the *analysis columns* (the columns you want the LLM to process).
            You should generally *exclude* your annotation columns here.
            """,
            unsafe_allow_html=True,
        )

        previous_selection = st.session_state.get("selected_columns", [])
        # Filter out invalid columns
        valid_previous_selection = [
            col for col in previous_selection if col in columns_for_analysis
        ]

        app_instance.selected_columns = st.multiselect(
            "Columns to analyze:",
            options=columns_for_analysis,
            default=(
                valid_previous_selection
                if valid_previous_selection
                else columns_for_analysis
            ),
        )
        st.session_state["selected_columns"] = app_instance.selected_columns

        if not app_instance.selected_columns:
            st.info("Select at least one column to proceed.")
            return None

        # 2.3: Rename columns
        # First, create a new column_renames dictionary that only includes selected columns
        filtered_column_renames = {}
        for col in app_instance.selected_columns:
            default_rename = app_instance.column_renames.get(col, col)
            new_name = st.text_input(
                f"Rename '{col}' to:", value=default_rename, key=f"rename_{col}"
            )
            filtered_column_renames[col] = new_name

        # Update app_instance.column_renames to only include selected columns
        app_instance.column_renames = filtered_column_renames
        st.session_state["column_renames"] = app_instance.column_renames

        # 2.4: Descriptions
        st.write("Add a short description for each selected column:")
        for col in app_instance.selected_columns:
            renamed_col = app_instance.column_renames[col]
            default_desc = app_instance.column_descriptions.get(renamed_col, "")
            desc = st.text_area(
                f"Description for '{renamed_col}':",
                height=70,
                value=default_desc,
                key=f"desc_{renamed_col}",
            )
            app_instance.column_descriptions[renamed_col] = desc
        st.session_state["column_descriptions"] = app_instance.column_descriptions

        # 2.5: Cleaning & Normalizing text columns
        st.markdown(
            """
            ### **Normalization of Text Columns**
            Select which columns (among your selected ones) contain textual data to be cleaned & normalized.
            """,
            unsafe_allow_html=True,
        )

        # Build a DataFrame with only the selected columns (renamed)
        processed = data[app_instance.selected_columns].rename(
            columns=app_instance.column_renames
        )

        # Get the default text columns from session state or app instance
        default_text_cols = (
            app_instance.text_columns
            if app_instance.text_columns
            else processed.columns.tolist()
        )
        # Filter out any columns that don't exist in the current processed dataframe
        valid_default_text_cols = [
            col for col in default_text_cols if col in processed.columns.tolist()
        ]

        text_cols: List[str] = st.multiselect(
            "Text columns:",
            processed.columns.tolist(),
            default=valid_default_text_cols,
            key="text_columns_selection",
        )

        # Store text columns in session state and app instance
        st.session_state["text_columns"] = text_cols
        app_instance.text_columns = text_cols

        # Clean and sanitize
        for tcol in text_cols:
            processed[tcol] = clean_and_normalize(processed[tcol])
        processed = sanitize_dataframe(processed)

        # Keep the annotation columns in the processed data, so we can do Step 7 easily
        for ann_col in app_instance.annotation_columns:
            if ann_col not in processed.columns:
                processed[ann_col] = data[ann_col]

        # Store processed data
        app_instance.processed_data = processed
        st.session_state["processed_data"] = processed

        # Rebuild column_descriptions so it only includes the newly renamed analysis columns
        updated_column_descriptions: Dict[str, str] = {}
        renamed_values = list(app_instance.column_renames.values())

        for col in processed.columns:
            # Only include columns that are renamed values of selected columns
            if col in renamed_values:
                updated_column_descriptions[col] = app_instance.column_descriptions.get(
                    col, ""
                )

        app_instance.column_descriptions = updated_column_descriptions
        st.session_state["column_descriptions"] = app_instance.column_descriptions

        st.success("Columns processed successfully!")
        st.write("Processed Data Preview:")
        st.dataframe(processed.head())

    return processed
