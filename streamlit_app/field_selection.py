"""
Module for handling field selection and evaluation mapping configuration.
"""

from typing import Any, List, Mapping, Sequence

import streamlit as st

from streamlit_app.evaluation_mappings import (
    LABEL_TYPE_OPTIONS,
    EvaluationMapping,
    clear_evaluation_result_cache,
    create_default_mapping,
    sanitize_evaluation_mappings,
)
from streamlit_app.session_management import save_session


def _serialize_mappings(mappings: Sequence[Mapping[str, Any]]) -> List[tuple[Any, ...]]:
    return [
        (
            mapping.get("id"),
            mapping.get("name"),
            mapping.get("llm_field"),
            tuple(mapping.get("human_columns", [])),
            mapping.get("label_type"),
        )
        for mapping in mappings
    ]


def select_fields(app_instance: Any, step_number: int = 4) -> List[str]:
    """
    Fields to extract and evaluation mapping setup.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        step_number: The step number to display (default=4 for annotation mode)

    Returns:
        A list of selected fields
    """
    st.markdown(f"### Step {step_number}: Fields to Extract", unsafe_allow_html=True)
    with st.expander(f"Show/hide details of step {step_number}", expanded=True):
        st.markdown(
            """
            Specify the **fields** (or categories) you want the model to generate for each entry.
            The names should match the field names used in your codebook and examples.
            """,
            unsafe_allow_html=True,
        )

        previous_fields = list(app_instance.selected_fields)
        default_fields = ",".join(previous_fields) if previous_fields else ""
        fields_str = st.text_input(
            "Comma-separated fields (e.g. 'Reasoning, Classification')",
            value=default_fields,
            key="fields_input",
        )
        extracted = [field.strip() for field in fields_str.split(",") if field.strip()]

        app_instance.selected_fields = extracted
        st.session_state["selected_fields"] = extracted

        if app_instance.selected_fields and app_instance.annotation_columns:
            st.subheader("Evaluation Mappings")
            st.markdown(
                """
                Configure which LLM field should be evaluated against which human annotation columns.
                Each mapping is evaluated independently in Step 7.
                """,
                unsafe_allow_html=True,
            )

            mappings_initialized = st.session_state.get(
                "evaluation_mappings_initialized", False
            )
            raw_mappings = st.session_state.get(
                "evaluation_mappings", app_instance.evaluation_mappings
            )
            mappings = sanitize_evaluation_mappings(
                raw_mappings=raw_mappings,
                selected_fields=app_instance.selected_fields,
                annotation_columns=app_instance.annotation_columns,
                legacy_label_column=app_instance.label_column,
                legacy_label_type=app_instance.label_type,
                create_default_if_empty=not mappings_initialized,
            )

            updated_mappings: List[EvaluationMapping] = []
            removed_mapping_id = None

            for index, mapping in enumerate(mappings):
                st.markdown(f"**Mapping {index + 1}**")
                col1, col2, col3, col4, col5 = st.columns([1.4, 1.2, 1.8, 1.0, 0.6])

                with col1:
                    mapping_name = st.text_input(
                        "Display name",
                        value=mapping["name"],
                        key=f"evaluation_mapping_name_{mapping['id']}",
                    )

                with col2:
                    llm_options = app_instance.selected_fields
                    llm_index = (
                        llm_options.index(mapping["llm_field"])
                        if mapping["llm_field"] in llm_options
                        else 0
                    )
                    llm_field = st.selectbox(
                        "LLM field",
                        options=llm_options,
                        index=llm_index,
                        key=f"evaluation_mapping_llm_{mapping['id']}",
                    )

                with col3:
                    human_columns = st.multiselect(
                        "Human columns",
                        options=app_instance.annotation_columns,
                        default=[
                            column
                            for column in mapping["human_columns"]
                            if column in app_instance.annotation_columns
                        ],
                        key=f"evaluation_mapping_humans_{mapping['id']}",
                    )

                with col4:
                    label_type_index = LABEL_TYPE_OPTIONS.index(mapping["label_type"])
                    label_type = st.selectbox(
                        "Label type",
                        options=LABEL_TYPE_OPTIONS,
                        index=label_type_index,
                        key=f"evaluation_mapping_type_{mapping['id']}",
                    )

                with col5:
                    remove_clicked = st.button(
                        "Remove",
                        key=f"evaluation_mapping_remove_{mapping['id']}",
                    )

                if remove_clicked:
                    removed_mapping_id = mapping["id"]
                else:
                    updated_mappings.append(
                        {
                            "id": mapping["id"],
                            "name": mapping_name.strip() or llm_field,
                            "llm_field": llm_field,
                            "human_columns": human_columns,
                            "label_type": label_type,
                        }
                    )

                if not human_columns and not remove_clicked:
                    st.warning(
                        f"Mapping `{mapping_name.strip() or llm_field}` needs at least one human annotation column."
                    )

                st.markdown("---")

            if st.button("Add evaluation mapping", key="add_evaluation_mapping_button"):
                updated_mappings.append(
                    create_default_mapping(
                        selected_fields=app_instance.selected_fields,
                        annotation_columns=app_instance.annotation_columns,
                        index=len(updated_mappings),
                    )
                )
                st.session_state["evaluation_mappings"] = updated_mappings
                st.session_state["evaluation_mappings_initialized"] = True
                app_instance.evaluation_mappings = updated_mappings
                clear_evaluation_result_cache(st.session_state)
                st.rerun()

            if removed_mapping_id is not None:
                st.session_state["evaluation_mappings"] = updated_mappings
                st.session_state["evaluation_mappings_initialized"] = True
                app_instance.evaluation_mappings = updated_mappings
                clear_evaluation_result_cache(st.session_state)
                st.rerun()

            mappings_changed = extracted != previous_fields or _serialize_mappings(
                updated_mappings
            ) != _serialize_mappings(st.session_state.get("evaluation_mappings", []))

            app_instance.evaluation_mappings = updated_mappings
            st.session_state["evaluation_mappings"] = updated_mappings
            st.session_state["evaluation_mappings_initialized"] = True

            # Legacy config is no longer the runtime source of truth.
            app_instance.label_column = None
            app_instance.label_type = None
            st.session_state["label_column"] = None
            st.session_state["label_type"] = None

            if mappings_changed:
                clear_evaluation_result_cache(st.session_state)
        else:
            app_instance.evaluation_mappings = []
            st.session_state["evaluation_mappings"] = []
            st.session_state["evaluation_mappings_initialized"] = False

        save_session(app_instance)

    return extracted
