"""
Module for handling session management functionality in the Streamlit app.
"""

import hashlib
import json
from typing import Any

import streamlit as st

from streamlit_app.evaluation_mappings import sanitize_evaluation_mappings


def _build_session_signature(uploaded_file: Any) -> tuple[str, int, str]:
    """
    Create a stable signature so a loaded session is only applied once per file.
    """
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return (uploaded_file.name, len(file_bytes), file_hash)


def load_previous_session(app_instance: Any) -> None:
    """
    Allows the user to upload a previous session configuration and restores the settings.

    Args:
        app_instance: The QualitativeAnalysisApp instance
    """
    st.markdown(
        """
        <h4><b>Load a Previous Session (Optional)</b></h4>
        <p style='font-size:16px'>
        If you've used this app before, you can upload your <b>saved session file (JSON)</b> to automatically restore previous settings.
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload your saved session file (JSON):", type=["json"], key="load_session"
    )

    if uploaded_file is None:
        st.session_state.pop("loaded_session_signature", None)
        return

    session_signature = _build_session_signature(uploaded_file)
    if st.session_state.get("loaded_session_signature") == session_signature:
        return

    try:
        session_payload = uploaded_file.getvalue().decode("utf-8")
        session_data = json.loads(session_payload)

        # Restore session values
        app_instance.selected_columns = session_data.get("selected_columns", [])

        # Get column_renames and ensure it only includes selected columns
        loaded_column_renames = session_data.get("column_renames", {})
        filtered_column_renames = {}
        for col in app_instance.selected_columns:
            if col in loaded_column_renames:
                filtered_column_renames[col] = loaded_column_renames[col]
            else:
                filtered_column_renames[col] = col
        app_instance.column_renames = filtered_column_renames

        # Get column_descriptions and ensure it only includes renamed selected columns
        loaded_column_descriptions = session_data.get("column_descriptions", {})
        filtered_column_descriptions = {}
        renamed_values = list(filtered_column_renames.values())
        for col, desc in loaded_column_descriptions.items():
            if col in renamed_values:
                filtered_column_descriptions[col] = desc
        app_instance.column_descriptions = filtered_column_descriptions
        app_instance.codebook = session_data.get("codebook", "")
        app_instance.examples = session_data.get("examples", "")
        app_instance.selected_fields = session_data.get("selected_fields", [])
        app_instance.selected_model = session_data.get("selected_model", None)
        app_instance.annotation_columns = session_data.get("annotation_columns", [])

        # Get label column, type, and text columns
        label_column = session_data.get("label_column", None)
        label_type = session_data.get("label_type", None)
        text_columns = session_data.get("text_columns", [])
        evaluation_mappings = session_data.get("evaluation_mappings")

        # Store in app instance
        app_instance.label_column = label_column
        app_instance.label_type = label_type
        app_instance.text_columns = text_columns
        app_instance.evaluation_mappings = sanitize_evaluation_mappings(
            raw_mappings=evaluation_mappings,
            selected_fields=app_instance.selected_fields,
            annotation_columns=app_instance.annotation_columns,
            legacy_label_column=label_column,
            legacy_label_type=label_type,
            create_default_if_empty=evaluation_mappings is None,
        )

        # Load generation mode data
        selected_mode = session_data.get("selected_mode", "Annotation Mode")
        blueprints = session_data.get("blueprints", [])
        generation_config = session_data.get("generation_config", None)
        annotation_config = session_data.get("annotation_config", None)

        app_instance.blueprints = blueprints
        app_instance.generation_config = generation_config
        app_instance.annotation_config = annotation_config

        # Update session_state
        st.session_state["selected_columns"] = app_instance.selected_columns
        st.session_state["column_renames"] = app_instance.column_renames
        st.session_state["column_descriptions"] = app_instance.column_descriptions
        st.session_state["codebook"] = app_instance.codebook
        st.session_state["examples"] = app_instance.examples
        st.session_state["selected_fields"] = app_instance.selected_fields
        st.session_state["selected_model"] = app_instance.selected_model
        st.session_state["annotation_columns"] = app_instance.annotation_columns
        st.session_state["label_column"] = label_column
        st.session_state["label_type"] = label_type
        st.session_state["text_columns"] = text_columns
        st.session_state["evaluation_mappings"] = app_instance.evaluation_mappings
        st.session_state["evaluation_mappings_initialized"] = True

        # Update generation mode session state
        st.session_state["selected_mode"] = selected_mode
        st.session_state["selected_mode_index"] = (
            ["Annotation Mode", "Generation Mode"].index(selected_mode)
            if selected_mode in ["Annotation Mode", "Generation Mode"]
            else 0
        )
        st.session_state["blueprints"] = blueprints
        st.session_state["generation_config"] = generation_config
        st.session_state["annotation_config"] = annotation_config

        # Restore generation UI state if applicable
        if generation_config:
            st.session_state["generation_prompt"] = generation_config.get(
                "generation_prompt", ""
            )
            st.session_state["num_items_to_generate"] = generation_config.get(
                "num_items", 5
            )
            st.session_state["generation_temperature"] = generation_config.get(
                "temperature", 0.7
            )
            st.session_state["generation_max_tokens"] = generation_config.get(
                "max_tokens", 500
            )

        if annotation_config:
            st.session_state["annotation_prompt"] = annotation_config.get(
                "annotation_prompt", ""
            )
            st.session_state["annotation_temperature"] = annotation_config.get(
                "annotation_temperature", 0.0
            )
            st.session_state["annotation_max_tokens"] = annotation_config.get(
                "annotation_max_tokens", 300
            )

        st.session_state["loaded_session_signature"] = session_signature
        st.success("Previous session successfully loaded.")

    except Exception as e:
        st.error(f"Failed to load session: {e}")


def save_session(app_instance: Any) -> None:
    """
    Allows the user to save the current session configuration (excluding API key).

    Args:
        app_instance: The QualitativeAnalysisApp instance
    """
    st.markdown(
        """
        <h4><b>Save Your Session</b></h4>
        <p style='font-size:16px'>
        Save your current setup to avoid reconfiguring everything next time.
        """,
        unsafe_allow_html=True,
    )

    filename_input = st.text_input(
        "**Enter a filename for your session:**",
        value="session_config.json",
        key="filename_input",
    )

    if not filename_input.endswith(".json"):
        filename_input += ".json"

    # Ensure column_renames only includes selected columns
    filtered_column_renames = {}
    for col in app_instance.selected_columns:
        if col in app_instance.column_renames:
            filtered_column_renames[col] = app_instance.column_renames[col]

    # Ensure column_descriptions only includes renamed selected columns
    filtered_column_descriptions = {}
    renamed_values = list(filtered_column_renames.values())
    for col, desc in app_instance.column_descriptions.items():
        if col in renamed_values:
            filtered_column_descriptions[col] = desc

    session_data = {
        "selected_columns": app_instance.selected_columns,
        "column_renames": filtered_column_renames,
        "column_descriptions": filtered_column_descriptions,
        "codebook": app_instance.codebook,
        "examples": app_instance.examples,
        "selected_fields": app_instance.selected_fields,
        "selected_model": app_instance.selected_model,
        "annotation_columns": app_instance.annotation_columns,
        "label_column": (
            app_instance.evaluation_mappings[0]["llm_field"]
            if getattr(app_instance, "evaluation_mappings", [])
            else app_instance.label_column
        ),
        "label_type": (
            app_instance.evaluation_mappings[0]["label_type"]
            if getattr(app_instance, "evaluation_mappings", [])
            else app_instance.label_type
        ),
        "evaluation_mappings": getattr(app_instance, "evaluation_mappings", []),
        "text_columns": app_instance.text_columns,
        # Generation mode data
        "selected_mode": st.session_state.get("selected_mode", "Annotation Mode"),
        "blueprints": getattr(app_instance, "blueprints", []),
        "generation_config": getattr(app_instance, "generation_config", None),
        "annotation_config": getattr(app_instance, "annotation_config", None),
    }

    data_json = json.dumps(session_data, indent=4)

    st.download_button(
        label="Save Session Configuration",
        data=data_json,
        file_name=filename_input,
        mime="application/json",
        key="save_session_button",
    )
