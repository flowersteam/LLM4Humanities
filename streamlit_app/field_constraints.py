"""
Gemini structured-output field constraint helpers.
"""

from typing import Any, Dict, List

import streamlit as st


FIELD_TYPE_OPTIONS = ["string", "number", "boolean"]


def render_field_constraints_editor(app_instance: Any, step_number: int) -> None:
    """
    Render the Gemini-only field type and enum editor and persist its state.
    """
    selected_fields = getattr(app_instance, "selected_fields", []) or []
    if not selected_fields:
        st.info(
            "Add at least one field in the previous step to configure Gemini output constraints."
        )
        return

    existing_field_types = getattr(
        app_instance, "field_types", {}
    ) or st.session_state.get("field_types", {})
    existing_field_enums = getattr(
        app_instance, "field_enums", {}
    ) or st.session_state.get("field_enums", {})

    field_types: Dict[str, str] = {}
    field_enums: Dict[str, List[Any]] = {}

    st.markdown("**Structured Output (Gemini only)**")
    st.caption(
        "Define field types and optional allowed values for Gemini's response schema."
    )

    for field in selected_fields:
        with st.container(border=True):
            st.markdown(f"**{field}**")
            col1, col2 = st.columns([2, 4])

            default_type = existing_field_types.get(field, "string")
            with col1:
                selected_type = st.selectbox(
                    "Type",
                    options=FIELD_TYPE_OPTIONS,
                    index=(
                        FIELD_TYPE_OPTIONS.index(default_type)
                        if default_type in FIELD_TYPE_OPTIONS
                        else 0
                    ),
                    help="Choose `string` for text, `number` for numeric values, and `boolean` for True/False outputs.",
                    key=f"field_type_step{step_number}_{field}",
                )
            field_types[field] = selected_type

            existing_enum = existing_field_enums.get(field, None)
            has_enum_default = existing_enum is not None and len(existing_enum) > 0
            with col2:
                if selected_type == "boolean":
                    use_enum = st.checkbox(
                        f"Restrict '{field}' to a list of allowed values?",
                        value=has_enum_default,
                        disabled=True,
                        key=f"field_use_enum_step{step_number}_{field}",
                    )
                else:
                    use_enum = st.checkbox(
                        f"Restrict '{field}' to a list of allowed values?",
                        value=has_enum_default,
                        key=f"field_use_enum_step{step_number}_{field}",
                    )

            if use_enum:
                default_enum_str = (
                    ", ".join(str(value) for value in existing_enum)
                    if existing_enum
                    else ""
                )
                with col2:
                    enum_str = st.text_input(
                        f"Allowed values for '{field}' (comma-separated):",
                        value=default_enum_str,
                        key=f"field_enum_values_step{step_number}_{field}",
                    )
                raw_values = [
                    value.strip() for value in enum_str.split(",") if value.strip()
                ]

                if selected_type == "number":
                    cast_values: List[Any] = []
                    for value in raw_values:
                        try:
                            cast_values.append(float(value))
                        except ValueError:
                            cast_values.append(value)
                    field_enums[field] = cast_values
                else:
                    field_enums[field] = raw_values
            else:
                field_enums[field] = []

    app_instance.field_types = field_types
    app_instance.field_enums = field_enums
    st.session_state["field_types"] = field_types
    st.session_state["field_enums"] = field_enums
