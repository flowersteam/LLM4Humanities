"""
Module for handling dataset upload functionality in the Streamlit app.
"""

import streamlit as st
from typing import Optional, Any, Dict, Union
import pandas as pd

from qualitative_analysis import load_data
from streamlit_app.session_management import load_previous_session


def upload_dataset(
    app_instance: Any,
    session_state: Union[Dict[Any, Any], "st.runtime.state.SessionStateProxy"],
) -> Optional[pd.DataFrame]:
    """
    Step 1: Uploads a dataset (CSV or XLSX) via Streamlit's file uploader.
    - Validates file type and delimiter.
    - Loads data into data and session_state.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        session_state: Streamlit's session state dictionary

    Returns:
        The loaded DataFrame or None if no file was uploaded
    """

    st.markdown("### Step 1: Upload Your Dataset", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 1", expanded=True):
        # Expected Data Format Explanation
        st.markdown(
            """
            ### Expected Data Format
            Your dataset should be in **CSV** or **Excel (XLSX)** format.

            **Required Structure:**  
            - Each row must have a **unique ID** and represent a single entry.  
            - The dataset should contain at least three columns:  
                - One for the **unique identifier**  
                - One or more **data columns** containing text or information to analyze.  
                - One or more **annotation columns** with human judgments or labels. 
                Those columns will be used to compare the model's predictions and determine if the model can be used on the rest of the data.
                Therefore, you only neeed annotations on a subset of the data.
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

        if uploaded_file is not None:
            file_type = "csv" if uploaded_file.name.endswith(".csv") else "xlsx"
            delimiter = st.text_input("CSV Delimiter (if CSV)", value=";")

            try:
                data = load_data(
                    uploaded_file, file_type=file_type, delimiter=delimiter
                )
                # Reset session states relevant to data
                session_state["selected_columns"] = []
                session_state["column_renames"] = {}
                session_state["column_descriptions"] = {}
                session_state["annotation_columns"] = []
                session_state["evaluation_mappings"] = []
                session_state["evaluation_mappings_initialized"] = False
                session_state["label_column"] = None
                session_state["label_type"] = None

                st.success("Data loaded successfully!")
                st.write("Data Preview:", data.head())

                # Store in session_state
                session_state["data"] = data
                app_instance.data = data

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

        # Add the load previous session functionality at the end of Step 1
        load_previous_session(app_instance)

    return app_instance.data
