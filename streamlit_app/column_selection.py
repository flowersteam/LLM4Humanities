"""
Module for handling column selection, renaming, description, and cleaning in the Streamlit qualitative analysis app.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any

from qualitative_analysis import clean_and_normalize, sanitize_dataframe


def select_rename_describe_columns(
    app_instance: Any, data: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Step 2: Configure the dataset for annotation and LLM analysis.

    This function:
    1. Lets the user select which columns contain existing human annotations.
    2. Optionally filters rows based on the presence of annotations.
    3. Lets the user select which columns to include for LLM analysis (excluding the annotation columns).
    4. Lets the user rename and describe the selected analysis columns.
    5. Cleans and normalizes selected text columns.
    6. Keeps annotation columns in the processed dataframe for later steps.

    Args:
        app_instance: The QualitativeAnalysisApp instance holding state.
        data: The input DataFrame to process.

    Returns:
        A processed DataFrame with selected, renamed, described, and cleaned columns,
        or None if no valid selection is made or no data is provided.
    """
    st.markdown("### Step 2: Data Selection", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 2", expanded=True):
        # Guard clause: no dataset loaded
        if data is None:
            st.error("No dataset loaded.")
            return None

        columns = data.columns.tolist()

        # Step 2.1: Select annotation columns
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

        # Initialize session state for the missing-annotation toggle if needed
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
                # Keep rows with at least one non-NA among selected annotation columns
                filtered_data = data.dropna(
                    how="all", subset=app_instance.annotation_columns
                )
                st.write(
                    f"**Filtered** dataset to keep rows with at least one annotation in {app_instance.annotation_columns}."
                )
            else:
                # Keep rows with non-NA in all selected annotation columns
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

            # Store final annotation columns in session state
            st.session_state["annotation_columns"] = app_instance.annotation_columns

        # Step 2.2: Select analysis columns (exclude annotation columns)
        columns_for_analysis = [
            c for c in data.columns if c not in app_instance.annotation_columns
        ]
        with st.container(border=True):
            st.markdown(
                """
                Now, select the *analysis columns* (the columns you want the LLM to process).
                You should generally *exclude* your annotation columns here.
                """,
                unsafe_allow_html=True,
            )

            previous_selection = st.session_state.get("selected_columns", [])
            # Keep only previously selected columns that still exist
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

            # Step 2.3: Rename selected analysis columns
            # First, create a new column_renames dictionary that only includes selected columns
            filtered_column_renames = {}
            for col in app_instance.selected_columns:
                default_rename = app_instance.column_renames.get(col, col)
                new_name = st.text_input(
                    f"Rename '{col}' to:", value=default_rename, key=f"rename_{col}"
                )
                filtered_column_renames[col] = new_name

            # Only keep renames for selected columns
            app_instance.column_renames = filtered_column_renames
            st.session_state["column_renames"] = app_instance.column_renames

            # 2.4: Add descriptions for selected (renamed) column
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

        # Step 2.5: Cleaning & Normalizing text columns
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

        # Clean and sanitize text columns
        for tcol in text_cols:
            processed[tcol] = clean_and_normalize(processed[tcol])
        processed = sanitize_dataframe(processed)

        # Step 2.6: Re-attach annotation columns in the processed data, so we can do Step 7 easily
        for ann_col in app_instance.annotation_columns:
            if ann_col not in processed.columns:
                processed[ann_col] = data[ann_col]

        # Store processed data in app instance and session state
        app_instance.processed_data = processed
        st.session_state["processed_data"] = processed

        # Step 2.7: Keep only descriptions for the (renamed) analysis columns
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

        # Final feedback and preview
        st.success("Columns processed successfully!")
        st.write("Processed Data Preview:")
        st.dataframe(processed.head())

    return processed
