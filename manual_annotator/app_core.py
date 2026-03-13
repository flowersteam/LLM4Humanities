"""
Main module for the Manual Annotation Tool.
This module defines the ManualAnnotatorApp class and the main entry point.
"""

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from manual_annotator.annotation_filter import filter_annotations
from manual_annotator.annotator_setup import setup_annotator
from manual_annotator.codebook_upload import upload_codebook
from manual_annotator.column_selection import select_columns
from manual_annotator.data_download import download_data
from manual_annotator.data_upload import upload_dataset
from manual_annotator.label_definition import define_labels
from manual_annotator.rating_scales import RatingScale
from manual_annotator.row_annotation import annotate_rows


class ManualAnnotatorApp:
    def __init__(self) -> None:
        """
        Initializes the ManualAnnotatorApp by pulling default or stored values
        from Streamlit's session_state.
        """
        # Data-related state
        self.df: Optional[pd.DataFrame] = st.session_state.get("df", None)
        self.original_df: Optional[pd.DataFrame] = st.session_state.get(
            "original_df", None
        )
        self.annotated_df: Optional[pd.DataFrame] = st.session_state.get(
            "annotated_df", None
        )
        self.unannotated_df: Optional[pd.DataFrame] = st.session_state.get(
            "unannotated_df", None
        )

        # Counts for display and tracking
        self.annotated_count: int = st.session_state.get("annotated_count", 0)
        self.unannotated_count: int = st.session_state.get("unannotated_count", 0)
        self.total_count: int = st.session_state.get("total_count", 0)

        # Annotation-related state
        self.annotation_columns: List[str] = st.session_state.get(
            "annotation_columns", []
        )
        self.annotator_name: str = st.session_state.get("annotator_name", "")
        self.confirmed_annotator_name: str = st.session_state.get(
            "confirmed_annotator_name", ""
        )
        self.rating_scales_text: str = st.session_state.get(
            "rating_scales_text",
            st.session_state.get("fast_labels_text", ""),
        )
        self.rating_scales: List[RatingScale] = st.session_state.get(
            "rating_scales", []
        )

        # UI-related state
        self.current_index: int = st.session_state.get("current_index", 0)
        self.selected_columns: List[str] = st.session_state.get("selected_columns", [])
        self.sort_column: Optional[str] = st.session_state.get("sort_column", None)
        self.enable_sorting: bool = st.session_state.get("enable_sorting", False)
        self.translated_rows: Dict[int, Dict[str, str]] = st.session_state.get(
            "translated_rows", {}
        )
        self.selected_scale_labels: Dict[str, str] = st.session_state.get(
            "selected_scale_labels", {}
        )

        # Codebook
        self.codebook_text: str = st.session_state.get("codebook_text", "")

    def run(self) -> None:
        """
        Main entry point for the Streamlit app.
        Executes each annotation step in sequence if the required data is available.
        """
        st.title("Manual Annotation Tool")

        st.markdown(
            """
            This application is designed to help you **manually annotate** a dataset.
            Your dataset should be in CSV format, and each row should correspond to a single item to be annotated.
            You can load a codebook, define one or several rating scales, and annotate each row dimension by dimension.
            **Invalid** rows can be flagged (it will create a new column for each annotator).

            **Steps**
            1. **Upload Data** (CSV or XLSX)
            2. **Optionally Filter** rows based on existing annotation columns
            3. **Set Annotator Name** (creates annotator-specific output columns)
            4. **Upload Codebook** (TXT) to display instructions on the sidebar (optional)
            5. **Define Rating Scales**
            6. **Select Columns to Display**
            7. **Annotate Row-by-Row on each scale**
            8. **Download Updated Data**
            """
        )

        # Step 1: Upload Dataset
        self.df, self.original_df = upload_dataset(self.df, self.original_df)

        if self.df is None:
            return

        # Step 2: Filter Annotations
        (
            self.df,
            self.annotation_columns,
            self.annotated_count,
            self.unannotated_count,
            self.total_count,
        ) = filter_annotations(self.df, self.annotation_columns)

        if self.df is None or len(self.df) == 0:
            return

        # Step 3: Set Annotator Name
        self.df, self.annotator_name, self.confirmed_annotator_name = setup_annotator(
            self.df,
            self.annotator_name,
            self.confirmed_annotator_name,
            self.annotated_count,
            self.unannotated_count,
            self.total_count,
        )

        if not self.confirmed_annotator_name:
            return

        # Step 4: Upload Codebook
        self.codebook_text = upload_codebook(self.codebook_text)

        # Step 5: Define Rating Scales
        self.df, self.rating_scales_text, self.rating_scales = define_labels(
            self.df,
            self.confirmed_annotator_name,
            self.rating_scales_text,
            self.rating_scales,
        )

        # Step 6: Select Columns to Display
        self.selected_columns, self.sort_column, self.enable_sorting = select_columns(
            self.df,
            self.rating_scales,
            self.confirmed_annotator_name,
        )

        if not self.selected_columns:
            return

        # Step 7: Annotate Row-by-Row
        (
            self.df,
            self.current_index,
            self.selected_scale_labels,
            self.translated_rows,
        ) = annotate_rows(
            self.df,
            self.current_index,
            self.selected_columns,
            self.rating_scales,
            self.confirmed_annotator_name,
            self.selected_scale_labels,
            self.translated_rows,
            self.sort_column,
            self.enable_sorting,
        )

        # Step 8: Download Data
        download_data(
            self.df,
            self.annotated_df,
            self.unannotated_df,
            self.original_df,
            self.annotated_count,
            self.unannotated_count,
            self.total_count,
            self.confirmed_annotator_name,
        )
