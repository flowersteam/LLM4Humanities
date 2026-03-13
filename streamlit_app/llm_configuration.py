"""
Module for handling LLM configuration functionality in the Streamlit app.

This module handles:
- Selecting an LLM provider (OpenAI, Anthropic, Gemini, Together, OpenRouter, Azure).
- Entering or loading the provider’s API key.
- Choosing a model depending on the provider.
- Configuring advanced parameters such as temperature and max tokens.
- Setting Gemini-specific options (system instruction and thinking credits).

The function `configure_llm(...)` updates `app_instance` with:
    - app_instance.llm_client
    - app_instance.selected_model
    - app_instance.temperature
    - app_instance.max_tokens
    - (Gemini only) system instruction + thinking credits

Returns the instantiated LLM client, or None if configuration is incomplete.
"""

import streamlit as st
from typing import Any, Optional

from qualitative_analysis import get_llm_client
import qualitative_analysis.config as config


def configure_llm(
    app_instance: Any, step_number: int = 5, purpose: str = ""
) -> Optional[Any]:
    """
    Choose the Model
    Lets the user pick the LLM provider, supply an API key (if not in .env),
    and choose a model.

    Args:
        app_instance: The QualitativeAnalysisApp instance
        step_number: The step number to display (default=5 for annotation mode)
        purpose: Optional purpose label (e.g., "for Generation", "for Annotation")

    Returns:
        The LLM client or None if configuration is incomplete
    """
    title = f"### Step {step_number}: Choose the Model"
    if purpose:
        title += f" {purpose}"
    st.markdown(title, unsafe_allow_html=True)
    with st.expander(f"Show/hide details of step {step_number}", expanded=True):
        # Dependency validation based on context
        # For annotation mode (step_number=5), check if fields are defined
        if step_number == 5 and not app_instance.selected_fields:
            st.warning(
                "⚠️ Please specify at least one field to extract in Step 4 before continuing."
            )
            return None

        # For generation mode step 3 (generation LLM), check if generation config exists
        if step_number == 3 and purpose == "for Generation":
            if (
                not hasattr(app_instance, "generation_config")
                or not app_instance.generation_config
            ):
                st.warning(
                    "⚠️ Please configure generation settings in Step 2 before continuing."
                )
                return None

        # For generation mode step 8 (annotation LLM), check if codebook and fields are defined
        if step_number == 8 and purpose == "for Annotation":
            if not app_instance.codebook or not app_instance.codebook.strip():
                st.warning("⚠️ Please provide a codebook in Step 6 before continuing.")
                return None
            if not app_instance.selected_fields:
                st.warning(
                    "⚠️ Please specify fields to extract in Step 7 before continuing."
                )
                return None

        provider_options = [
            "Select Provider",
            "OpenAI",
            "Anthropic",
            "Gemini",
            "Together",
            "OpenRouter",
            "Azure",
        ]

        # Make key unique based on step number
        provider_key = f"llm_provider_select_step{step_number}"

        selected_provider_display = st.selectbox(
            "Select LLM Provider:", provider_options, key=provider_key
        )

        if selected_provider_display == "Select Provider":
            st.info("ℹ️ Please select a provider to continue.")
            return None

        provider_map = {
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "Gemini": "gemini",
            "Together": "together",
            "OpenRouter": "openrouter",
            "Azure": "azure",
        }
        internal_provider = provider_map[selected_provider_display]

        # Check config for an existing API key
        existing_api_key = config.MODEL_CONFIG[internal_provider].get("api_key")

        if existing_api_key:
            st.success(f"🔑 API Key loaded from .env for {selected_provider_display}!")
            final_api_key = existing_api_key
        else:
            st.sidebar.subheader("API Key Configuration")
            api_key_placeholder = {
                "openai": "sk-...",
                "anthropic": "sk-ant-...",
                "gemini": "your-gemini-api-key",
                "together": "together-...",
                "openrouter": "sk-or-...",
                "azure": "azure-...",
            }.get(internal_provider, "Enter API Key")

            api_key = st.sidebar.text_input(
                f"Enter your {selected_provider_display} API Key",
                type="password",
                placeholder=api_key_placeholder,
            )
            st.sidebar.info(
                "🔒 Your API key is used only during this session and is never stored."
            )

            if not api_key:
                st.warning(f"Please provide your {selected_provider_display} API key.")
                st.stop()
            else:
                st.success(f"{selected_provider_display} API Key provided!")
                final_api_key = api_key

        # Update config with the final API key
        provider_config = config.MODEL_CONFIG[internal_provider].copy()
        provider_config["api_key"] = final_api_key

        # Instantiate the LLM client
        app_instance.llm_client = get_llm_client(
            provider=internal_provider, config=provider_config
        )

        # Select model - make keys unique based on step number
        model_key = f"llm_model_select_step{step_number}"

        if selected_provider_display == "OpenRouter":
            # For OpenRouter, use text input to allow any model name
            st.markdown(
                """
                **OpenRouter Model Selection**
                
                Enter the full model name in the format `provider/model-name`. 
                Visit [OpenRouter Models](https://openrouter.ai/models) to see all available models.
                """
            )

            chosen_model = st.text_input(
                "Enter OpenRouter Model Name:",
                value=st.session_state.get(
                    "selected_model", "anthropic/claude-3.5-sonnet"
                ),
                placeholder="anthropic/claude-3.5-sonnet",
                key=f"openrouter_model_input_step{step_number}",
                help="Format: provider/model-name (e.g., anthropic/claude-3.5-sonnet)",
            )

            if not chosen_model.strip():
                st.warning("Please enter a model name to continue.")
                return None

            if "/" not in chosen_model:
                st.warning(
                    "Model name should be in format 'provider/model-name' (e.g., 'anthropic/claude-3.5-sonnet')"
                )
                return None

        elif selected_provider_display == "OpenAI":
            model_options = ["gpt-4o", "gpt-4o-mini"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )
        elif selected_provider_display == "Anthropic":
            model_options = ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )

        elif selected_provider_display == "Gemini":
            model_options = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )
            # System instruction
            system_instruction = st.text_area(
                "System instruction (optional)",
                value="You are a strict and precise annotator who avoids any unnecessary leniency.",
                key=f"gemini_system_instruction_step{step_number}",
                help="High-level instructions: behavior you want Gemini to follow."
            )
            if not system_instruction.strip():
                system_instruction = None

            # Thinking credits
            options = ["Manual", "Let the model decide dynamically"] if chosen_model == "gemini-2.5-pro" else ["Turn off", "Manual", "Let the model decide dynamically"]
            mode = st.radio(
                "Thinking credits definition",
                options,
                key=f"thinking_mode_{step_number}"
            )

            if chosen_model == "gemini-2.5-pro":
                min_credits = 128
                max_credits = 32768
            elif chosen_model == "gemini-2.5-flash-lite":
                min_credits = 512
                max_credits = 24576
            else:  # gemini-2.5-flash ou autres
                min_credits = 1
                max_credits = 24576

            if mode == "Let the model decide dynamically":
                thinking_input = st.number_input(
                    "Thinking credits (dynamic mode enabled)",
                    value=-1,
                    disabled=True,
                    key=f"thinking_dynamic_{step_number}",
                )
                thinking_credits = -1

            elif mode == "Turn off":
                thinking_input = st.number_input(
                    "Thinking credits (thinking disabled)",
                    value=0,
                    disabled=True,
                    key=f"thinking_off_{step_number}",
                )
                thinking_credits = 0

            else:  # Manual
                thinking_input = st.number_input(
                    "Thinking credits",
                    min_value=min_credits,
                    max_value=max_credits,
                    step=128,
                    value=min_credits,
                    key=f"thinking_manual_{step_number}",
                    help=(
                        "Controls how much internal reasoning the model is allowed to perform "
                        "before answering; higher values enable deeper reasoning but cost more tokens.  \n"
                        "⚠️ Thinking cannot be turned off for gemini-2.5-pro."
                    ),
                )
                thinking_credits = thinking_input

            app_instance.gemini_system_instruction = system_instruction
            app_instance.gemini_thinking_credits = thinking_credits

        elif selected_provider_display == "Together":
            model_options = ["gpt-neoxt-chat-20B"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )
        else:  # Azure
            model_options = ["gpt-4o", "gpt-4o-mini"]
            chosen_model = st.selectbox(
                "Select Model:",
                model_options,
                key=model_key,
            )

    # Advanced Settings: temperature and max tokens
    with st.expander("Advanced settings"):
        temperature = st.number_input(f"Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.0,
                                      help="Controls how deterministic the model is: low temperature (< 0.3) gives stable, predictable outputs, while high temperature (> 0.7) produces more creative and varied responses.  \n**For annotation tasks, low temperature is recommended.**"
        )
        max_tokens = st.number_input("Maximum tokens limit", min_value=20, step=40, value=500,
                                     help = "Sets the **maximum number of tokens the model can use for one response**. This includes both internal thinking tokens (when used) and the final visible output. If the limit is too low, the model may stop early, shorten its answer, or produce errors.  \nFor **gemini-2.5-pro**, a max_token value of about **2,500** is recommended to ensure enough space for both internal thinking and output."
        )

    app_instance.selected_model = chosen_model
    app_instance.temperature = temperature
    app_instance.max_tokens = max_tokens

    return app_instance.llm_client
