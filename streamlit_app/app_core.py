"""
Main module for the Streamlit app.
This module imports and uses all the other modules to create the complete app.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
import pandas as pd

from streamlit_app.data_upload import upload_dataset
from streamlit_app.column_selection import select_rename_describe_columns
from streamlit_app.codebook_management import codebook_and_examples
from streamlit_app.field_selection import select_fields
from streamlit_app.llm_configuration import configure_llm
from streamlit_app.analysis import run_analysis
from streamlit_app.evaluation import compare_with_external_judgments

# Import generation modules
from streamlit_app.generation import (
    blueprint_input,
    configure_generation,
    run_generation,
    annotate_generated_content,
)


class QualitativeAnalysisApp:
    def __init__(self) -> None:
        """
        Initializes the QualitativeAnalysisApp by pulling default or stored values
        from Streamlit's session_state.
        """
        self.data: Optional[pd.DataFrame] = st.session_state.get("data", None)
        self.original_data: Optional[pd.DataFrame] = st.session_state.get(
            "original_data", None
        )
        self.processed_data: Optional[pd.DataFrame] = st.session_state.get(
            "processed_data", None
        )

        # Keep track of which columns are annotation columns
        self.annotation_columns: List[str] = st.session_state.get(
            "annotation_columns", []
        )

        self.selected_columns: List[str] = st.session_state.get("selected_columns", [])
        self.column_renames: Dict[str, str] = st.session_state.get("column_renames", {})
        self.column_descriptions: Dict[str, str] = st.session_state.get(
            "column_descriptions", {}
        )

        self.codebook: str = st.session_state.get("codebook", "")
        self.examples: str = st.session_state.get("examples", "")

        self.llm_client: Any = None  # Will instantiate later (OpenAI or Together)
        self.selected_model: Optional[str] = st.session_state.get(
            "selected_model", None
        )

        self.selected_fields: List[str] = st.session_state.get("selected_fields", [])
        self.results: List[Dict[str, Any]] = st.session_state.get("results", [])

        # Label configuration
        self.label_type: Optional[str] = st.session_state.get("label_type", None)
        self.label_column: Optional[str] = st.session_state.get("label_column", None)
        self.text_columns: List[str] = st.session_state.get("text_columns", [])

        # Generation workflow attributes
        self.blueprints: Dict[str, str] = st.session_state.get("blueprints", {})
        self.generation_config: Optional[Dict[str, Any]] = st.session_state.get(
            "generation_config", None
        )
        self.generated_content: Optional[pd.DataFrame] = st.session_state.get(
            "generated_content", None
        )
        self.annotation_config: Optional[Dict[str, Any]] = st.session_state.get(
            "annotation_config", None
        )
        self.annotated_content: Optional[pd.DataFrame] = st.session_state.get(
            "annotated_content", None
        )

    def run(self) -> None:
        """
        Main entry point for the Streamlit app.
        Executes each analysis step in sequence if the required data is available.
        """
        st.title("LLM4Humanities")

        # App Purpose Explanation
        st.markdown(
            """
            **LLM4Humanities** helps you analyze qualitative datasets 
            using Large Language Models.
            Choose between **Annotation Mode** (analyze existing data) or **Generation Mode** (generate and annotate new content).
            """,
            unsafe_allow_html=True,
        )

        # Mode Selection
        st.markdown("---")
        mode = st.radio(
            "**Select Mode:**",
            ["Annotation Mode", "Generation Mode"],
            index=st.session_state.get("selected_mode_index", 0),
            key="app_mode_selection",
            help="Annotation Mode: Analyze existing datasets with human annotations. Generation Mode: Generate new content and annotate it.",
        )

        # Store mode selection
        st.session_state["selected_mode_index"] = [
            "Annotation Mode",
            "Generation Mode",
        ].index(mode)
        st.session_state["selected_mode"] = mode

        st.markdown("---")

        if mode == "Annotation Mode":
            self._run_annotation_mode()
        else:  # Generation Mode
            self._run_generation_mode()

    def _run_annotation_mode(self) -> None:
        """
        Run the annotation workflow for analyzing existing datasets.
        """
        st.markdown("## Annotation Mode", unsafe_allow_html=True)
        st.markdown(
            """
            **Annotation Mode** allows you to analyze existing datasets using LLMs and compare results with human annotations.
            You will need a dataset with human annotations, a codebook, and a valid API key.
            """,
            unsafe_allow_html=True,
        )

        # Step 1: Upload Dataset
        self.data = upload_dataset(self, st.session_state)

        # Step 2: Select & Rename Columns, Add Descriptions, plus annotation filtering
        self.processed_data = select_rename_describe_columns(self, self.data)

        # Step 3: Codebook & Examples
        self.codebook, self.examples = codebook_and_examples(self)

        # Step 4: Fields to Extract
        self.selected_fields = select_fields(self)

        # Step 5: Configure LLM (provider & model)
        self.llm_client = configure_llm(self)

        # Step 6: Run Analysis
        run_analysis(self)

        # Step 7: Compare with External Judgments (optionally: alt-test or Cohen's Kappa)
        compare_with_external_judgments(self)

        # Step 8: Run Analysis on Remaining Data
        run_analysis(self, analyze_remaining=True)

    def _run_generation_mode(self) -> None:
        """
        Run the generation workflow for creating and annotating new content.
        """
        st.markdown("## Generation Mode", unsafe_allow_html=True)
        st.markdown(
            """
            **Generation Mode** allows you to generate new content using LLMs based on blueprint examples, then annotate the generated content.
            You will need blueprint examples, generation instructions, and a valid API key.
            """,
            unsafe_allow_html=True,
        )

        # Step 1: Blueprint Input
        blueprints_result = blueprint_input(self)
        if blueprints_result is not None:
            self.blueprints = blueprints_result

        # Step 2: Generation Configuration
        generation_config_result = configure_generation(self)
        if generation_config_result is not None:
            self.generation_config = generation_config_result

        # Step 3: Configure LLM for Generation
        self.llm_client = configure_llm(self, step_number=3, purpose="for Generation")

        # Step 4: Content Generation
        generated_content_result = run_generation(self)
        if generated_content_result is not None:
            self.generated_content = generated_content_result

        # Step 5: Column Selection for Annotation (which columns from generated content to annotate)
        # For now, we'll just use all columns from generated content
        # TODO: Create a proper column selection UI for generation mode
        if self.generated_content is not None:
            # Automatically use all generated columns except generation_id
            generated_columns = [
                col for col in self.generated_content.columns if col != "generation_id"
            ]
            self.selected_columns = generated_columns
            st.session_state["selected_columns"] = generated_columns

        # Step 6: Annotation Codebook (reuse codebook_and_examples from annotation mode)
        self.codebook, self.examples = codebook_and_examples(self, step_number=6)

        # Step 7: Annotation Fields Selection (what fields to extract from annotation)
        self.selected_fields = select_fields(self, step_number=7)

        # Step 8: Configure LLM for Annotation
        self.llm_client = configure_llm(self, step_number=8, purpose="for Annotation")

        # Step 9: Content Annotation
        annotated_content_result = annotate_generated_content(self)
        if annotated_content_result is not None:
            self.annotated_content = annotated_content_result
