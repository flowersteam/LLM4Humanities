"""
Module for handling data download functionality in the Manual Annotation Tool.
"""

import streamlit as st
import pandas as pd
import io
from typing import Optional


def download_data(
    df: pd.DataFrame,
    annotated_df: Optional[pd.DataFrame],
    unannotated_df: Optional[pd.DataFrame],
    original_df: Optional[pd.DataFrame],
    annotated_count: int,
    unannotated_count: int,
    total_count: int,
    annotator_name: str,
) -> None:
    """
    Step 8: Download Updated Data
    Allows the user to download the annotated data as an Excel file.

    Args:
        df: The current working DataFrame
        annotated_df: DataFrame with only annotated rows (not used in new approach)
        unannotated_df: DataFrame with only unannotated rows (not used in new approach)
        original_df: The original DataFrame
        annotated_count: Number of annotated rows
        unannotated_count: Number of unannotated rows
        total_count: Total number of rows
        annotator_name: The annotator name
    """
    st.header("Step 8: Download Updated Data")

    st.markdown(
        "When you're done (or want to pause), download your annotated data as an Excel file."
    )

    filename_input = st.text_input(
        "Output filename:", value="annotated_data.xlsx", key="results_filename_input"
    )
    if not filename_input.endswith(".xlsx"):
        filename_input += ".xlsx"

    # Prepare the complete dataset for download
    # We're using the original approach with a single dataframe
    complete_df = df.copy()

    # Use the stored counts for the info message
    if "annotated_indices" in st.session_state and original_df is not None:
        st.info(
            f"Downloading complete dataset with {total_count} rows ({annotated_count} annotated + {unannotated_count} unannotated)"
        )
    else:
        st.info(f"Downloading dataset with {len(complete_df)} rows")

    # Create Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        complete_df.to_excel(writer, index=False)

    # Download button
    st.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name=filename_input,
        key=f"download_button_{filename_input}",  # Dynamic key based on the filename
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
