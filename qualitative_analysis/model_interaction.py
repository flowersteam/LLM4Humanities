"""
model_interaction.py

This module provides a unified interface for interacting with different large language model (LLM) providers,
including:

    - OpenAI
    - Azure OpenAI
    - Anthropic
    - Together AI
    - OpenRouter
    - Google Gemini
    - vLLM (for open-source models)

It abstracts API interactions to simplify sending prompts and retrieving responses across multiple providers.

Dependencies:
    - openai: For interacting with Azure OpenAI, standard OpenAI models, and OpenRouter.
    - anthropic: For interacting with Anthropic Claude models.
    - together: For interacting with Together AI models.
    - google-genai: For interacting with Google Gemini models.
    - vllm: For running inference with open-source models locally.
    - google.api_core.exceptions: For handling Google API errors (Gemini).
    - abc: For defining the abstract base class.
    - types: For using SimpleNamespace to standardize usage object representation.
    - time: For retry backoff with Gemini.
    - re: For normalizing field names for Gemini JSON schema.

Classes:
    - LLMClient: Abstract base class defining the interface for LLM clients.
    - OpenAILLMClient: Client for interacting with standard OpenAI language models.
    - AzureOpenAILLMClient: Client for interacting with Azure OpenAI language models.
    - AnthropicLLMClient: Client for interacting with Anthropic Claude models.
    - TogetherLLMClient: Client for interacting with Together AI language models.
    - GeminiLLMClient: Client for interacting with Google Gemini models.
    - OpenRouterLLMClient: Client for interacting with OpenRouter language models.
    - VLLMLLMClient: Client for interacting with open-source models using vLLM.

Functions:
    - get_llm_client(provider, config, model=None): Factory function to instantiate the appropriate
      LLM client based on the specified provider string.
"""

from abc import ABC, abstractmethod
import openai
from anthropic import Anthropic
from together import Together
from types import SimpleNamespace
from typing import Optional, List, Dict, Any
import time
from google import genai
from google.genai import types
import re


def is_gpt5_model(model_name: str) -> bool:
    """
    Detect if a model is a GPT-5 variant that requires max_completion_tokens instead of max_tokens.

    Parameters
    ----------
    model_name : str
        The name of the model to check

    Returns
    -------
    bool
        True if the model is a GPT-5 variant, False otherwise
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    # Check for GPT-5 variants including mini, nano, and future versions
    gpt5_patterns = [
        "gpt-5",
        "gpt5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt5-mini",
        "gpt5-nano",
    ]

    return any(pattern in model_lower for pattern in gpt5_patterns)


# Try to import vLLM, but handle the case when it's not available
# This could be due to import errors or platform compatibility issues
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except (ImportError, OSError):
    VLLM_AVAILABLE = False
    print("Warning: vLLM is not available. VLLMLLMClient will not be usable.")


class LLMClient(ABC):
    """
    Abstract base class for language model clients.

    This class defines a common interface for interacting with different language model providers.
    Subclasses must implement the `get_response` method to handle API communication with specific models.

    Methods:
        - get_response(prompt, model, **kwargs):
            Sends a prompt to the language model and retrieves the response.
            Must be implemented by all subclasses.

    Usage:
        This class cannot be instantiated directly. Subclasses should implement the `get_response` method.

        Example:
            class CustomLLMClient(LLMClient):
                def get_response(self, prompt, model, **kwargs):
                    # Custom implementation here
                    return "Sample response"

            client = CustomLLMClient()
            response = client.get_response("Hello!", model="custom-model")
    """

    @abstractmethod
    def get_response(self, prompt: str, model: str, **kwargs) -> tuple[str, object]:
        """
        Sends a prompt to the language model and retrieves the response.

        Parameters:
            - prompt (str): The input text prompt to send to the language model.
            - model (str): The identifier of the language model to use.
            - **kwargs: Additional keyword arguments specific to the language model API.

        Returns:
            - tuple[str, object]: A tuple containing:
                - The language model's response to the prompt (str).
                - Usage information or metadata (object).

        Raises:
            - NotImplementedError: If the method is not implemented in the subclass.
        """
        pass


class OpenAILLMClient(LLMClient):
    """
    Client for interacting with OpenAI language models (non-Azure).

    This class manages communication with the standard OpenAI API, enabling
    prompt-based interactions with models like "gpt-3.5-turbo" or "gpt-4".
    It handles authentication via an OpenAI API key (api_key).

    Attributes
    ----------
    client : openai.OpenAI
        The OpenAI client instance used for API calls.

    Methods
    -------
    get_response(prompt, model, **kwargs) -> tuple[str, object]
        Sends a prompt to the specified OpenAI language model and returns the
        response text plus usage metadata.
    """

    def __init__(self, api_key: str):
        """
        Initializes the OpenAI LLM client.

        Parameters
        ----------
        api_key : str
            The OpenAI API key to use (from OPENAI_API_KEY).
        """
        self.client = openai.OpenAI(api_key=api_key)

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the standard OpenAI model and retrieves the response.

        Parameters
        ----------
        prompt : str
            The user prompt to send to the language model.

        model : str
            The OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4", etc.).

        **kwargs : dict, optional
            Additional arguments such as:
            - temperature (float): Controls randomness (default is 0.0).
            - max_tokens (int): Max tokens in the response (default 500).
            - verbose (bool): If True, prints debug info.

        Returns
        -------
        tuple[str, SimpleNamespace]
            A tuple containing:
                - The generated response text (str).
                - The usage object detailing token usage (SimpleNamespace).

        Raises
        ------
        openai.error.OpenAIError
            If the API request fails.
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Handle GPT-5 vs non-GPT-5 models with explicit API calls
        # Type cast to satisfy MyPy - our dict structure matches OpenAI's expected format
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        if is_gpt5_model(model):
            # GPT-5 only supports temperature=1 (default), so don't set temperature parameter
            if verbose:
                print(
                    f"Note: GPT-5 model detected - using default temperature (1) instead of {temperature}"
                )
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_completion_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )

        if verbose:
            print("\n=== LLM Response ===")
            print(f"{response.choices[0].message.content}\n")

        content = response.choices[0].message.content

        # Convert usage object to a SimpleNamespace for consistency with other clients
        if response.usage:
            usage_obj = SimpleNamespace(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            # Fallback if usage is not available
            usage_obj = SimpleNamespace(
                prompt_tokens=len(prompt.split()),  # Very rough estimate
                completion_tokens=(
                    len(content.split()) if content else 0
                ),  # Very rough estimate
                total_tokens=len(prompt.split())
                + (len(content.split()) if content else 0),
            )

        return (content.strip() if content else ""), usage_obj


class AzureOpenAILLMClient(LLMClient):
    """
    Client for interacting with Azure OpenAI language models.

    This class manages communication with the Azure OpenAI API, enabling prompt-based
    interactions with models like GPT-3/GPT-4 deployed on Azure. It handles authentication,
    configuration (endpoint, api_version), and response retrieval.

    Attributes
    ----------
    client : openai.AzureOpenAI
        The Azure OpenAI client instance used for API calls.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Azure OpenAI language model and retrieves the generated response.
    """

    def __init__(self, api_key: str, endpoint: str, api_version: str):
        """
        Initializes the AzureOpenAILLMClient with the required authentication and API configuration.

        Parameters:
        ----------
        - api_key (str):
            The API key for authenticating with the Azure OpenAI service.
        - endpoint (str):
            The endpoint URL for the Azure OpenAI service.
        - api_version (str):
            The API version to use when making API requests.
        """
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the Azure OpenAI language model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The deployment name of the Azure OpenAI model to use.
        - **kwargs:
            Additional keyword arguments for the OpenAI API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, object]:
            A tuple containing:
                - The model's generated response (str).
                - The usage object detailing token usage (object).

        Raises:
        ------
        - openai.error.OpenAIError:
            If the API request fails.
        """
        # Extract parameters or set defaults
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Handle GPT-5 vs non-GPT-5 models with explicit API calls
        # Type cast to satisfy MyPy - our dict structure matches OpenAI's expected format
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        if is_gpt5_model(model):
            # GPT-5 only supports temperature=1 (default), so don't set temperature parameter
            if verbose:
                print(
                    f"Note: GPT-5 model detected - using default temperature (1) instead of {temperature}"
                )
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_completion_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )

        if verbose:
            print("\n=== LLM Response ===")
            print(f"{response.choices[0].message.content}\n")

        content = response.choices[0].message.content

        if response.usage:
            usage_obj = SimpleNamespace(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            usage_obj = SimpleNamespace(
                prompt_tokens=len(prompt.split()),
                completion_tokens=(len(content.split()) if content else 0),
                total_tokens=len(prompt.split())
                + (len(content.split()) if content else 0),
            )

        return (content.strip() if content else ""), usage_obj


class AnthropicLLMClient(LLMClient):
    """
    Client for interacting with Anthropic Claude language models.

    This class manages communication with the Anthropic API, enabling prompt-based
    interactions with Claude models. It handles authentication via an Anthropic API key.

    Attributes:
    ----------
    - client (Anthropic):
        An instance of the Anthropic client initialized with the API key.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Anthropic language model and retrieves the response.
    """

    def __init__(self, api_key: str):
        """
        Initializes the AnthropicLLMClient with the provided API key.

        Parameters:
        ----------
        - api_key (str):
            The API key for the Anthropic service.

        Example:
        -------
        >>> client = AnthropicLLMClient(api_key='your_api_key')
        """
        self.client = Anthropic(api_key=api_key)

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the Anthropic Claude model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The identifier of the Anthropic model to use (e.g., "claude-3-7-sonnet-20250219").
        - **kwargs:
            Additional keyword arguments for the Anthropic API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0.0).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The model's generated response (str).
                - A usage object with token counts (SimpleNamespace).

        Raises:
        ------
        - Exception:
            If the API request fails.
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Create a message with Anthropic's API
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the content from the response
        # The content is a list of ContentBlock objects
        content_text = ""
        for content_block in response.content:
            # Check if the content block has a 'type' attribute and it's 'text'
            if hasattr(content_block, "type") and content_block.type == "text":
                content_text += content_block.text

        if verbose:
            print(f"Generation:\n{content_text}\n")

        # Create a usage object with token counts
        usage_obj = SimpleNamespace(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return content_text.strip(), usage_obj


class TogetherLLMClient(LLMClient):
    """
    Client for interacting with Together AI language models.

    This class handles communication with the Together AI API, allowing you to send prompts
    and receive responses from various language models.

    Attributes:
    ----------
    - client (Together):
        An instance of the Together client initialized with the API key.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Together AI language model and retrieves the response.
    """

    def __init__(self, api_key: str):
        """
        Initializes the TogetherLLMClient with the provided API key.

        Parameters:
        ----------
        - api_key (str):
            The API key for the Together AI service.

        Example:
        -------
        >>> client = TogetherLLMClient(api_key='your_api_key')
        """
        self.client = Together(api_key=api_key)

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the Together AI language model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The identifier of the Together AI model to use.
        - **kwargs:
            Additional keyword arguments for the Together AI API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0.7).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The language model's response to the prompt (str).
                - A usage object with token counts (SimpleNamespace).

        Raises:
        ------
        - Exception:
            If the API request fails.
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if verbose:
            print(f"Generation:\n{response.choices[0].message.content}\n")

        content = response.choices[0].message.content.strip()

        # Create a simple usage object (Together AI doesn't always provide detailed token usage)
        usage_obj = SimpleNamespace(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(prompt.split()) + len(content.split()),
        )

        return content, usage_obj


class GeminiLLMClient(LLMClient):
    """
    Client for interacting with Google Gemini language models.

    This class manages communication with the Google Gemini API, enabling prompt-based
    interactions with Gemini models. It handles authentication via a Google API key.

    Attributes:
    ----------
    - api_key (str):
        The Google API key used for authentication.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the Gemini language model and retrieves the response.
    """

    def __init__(self, api_key: str):
        """
        Initializes the GeminiLLMClient with the provided API key.

        Parameters:
        ----------
        - api_key (str):
            The API key for the Google Gemini service.

        Example:
        -------
        >>> client = GeminiLLMClient(api_key='your_api_key')
        """
        self.api_key = api_key

    def normalize_key(self, field_name: str) -> str:
        key = field_name.strip().lower()
        key = re.sub(r"\W+", "_", key)  # any non-alphanumeric characters with _
        key = key.strip("_")
        return key

    def get_response(
        self, prompt: str, model: str, **kwargs: Any
    ) -> tuple[str, SimpleNamespace]:
        """
        Send a text prompt to a specified Google Gemini model and retrieve the generated response.

        Parameters
        ----------
        prompt : str
            The text prompt to be sent to the Gemini model.
        model : str
            The model identifier (e.g., "gemini-2.0-flash-001").
        app_instance :
            An optional object passed through `kwargs` that carries runtime
            configuration attributes used for this call.
            The following attributes are read if present:
                - temperature (float, default = 0.0) :
                    Controls the randomness of the output.
                - max_tokens (int, default = 500) :
                    Maximum number of tokens to generate.
                - selected_fields (list[str]) :
                    Names of fields to include in the JSON schema for the response.
                - field_types (dict[str, str]) :
                    Mapping from field name to JSON type ("string", "number", "integer", "boolean").
                - field_enums (dict[str, list[str]]) :
                    Optional enum values per field.
                - gemini_system_instruction (str or None) :
                    Optional system-level instruction to control model behavior.
                - gemini_thinking_credits (int) :
                    Optional thinking budget passed to `ThinkingConfig(thinking_budget=...)`.
        **kwargs :
            Optional arguments:
                - verbose (bool, default = False):
                    If True, prints debug information (prompt and generated text).

        Returns
        -------
        tuple[str, SimpleNamespace]
            A tuple containing:
                - The generated response text (str).
                - A SimpleNamespace object with token usage:
                    - prompt_tokens
                    - completion_tokens (output + thoughts tokens)
                    - total_tokens

        Error Handling
        --------------
        - Automatically retries up to 7 times on transient errors, such as
          overload, quota issues, or temporary unavailability.
        - Uses exponential backoff between retries (0.5s, 1s, 2s, 4s, ...).
        - Re-raises the exception immediately if the error is not considered transient,
          or after the final retry attempt fails.
        """
        client = genai.Client()
        verbose = kwargs.get("verbose", False)
        app_instance = kwargs.get("app_instance")

        temperature = getattr(app_instance, "temperature", 0.0)
        max_tokens = getattr(app_instance, "max_tokens", 500)

        selected_fields = getattr(app_instance, "selected_fields", [])
        field_types = getattr(app_instance, "field_types", {})
        field_enums = getattr(app_instance, "field_enums", {})

        system_instruction = getattr(app_instance, "gemini_system_instruction", None)
        thinking_credits = getattr(app_instance, "gemini_thinking_credits", -1)

        # Json schema design
        response_json_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for field in selected_fields:
            field_type = field_types.get(field, "string")
            enum_values = field_enums.get(field, []) or None
            key = self.normalize_key(field)

            prop: Dict[str, Any] = {}
            prop["type"] = field_type
            prop["nullable"] = True

            if enum_values:
                prop["enum"] = enum_values

            response_json_schema["properties"][key] = prop
            response_json_schema["required"].append(key)

        # Prompting with error handling: try at least 7 times with a delay in between
        base_delay = 0.5
        max_retries = 7
        for attempt in range(1, max_retries + 2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=thinking_credits
                        ),
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_json_schema=response_json_schema,
                    ),
                )

                # Tokens counting
                output_token_count = response.usage_metadata.candidates_token_count or 0
                thoughts_token_count = response.usage_metadata.thoughts_token_count or 0
                completion_tokens = output_token_count + thoughts_token_count or 0
                usage_obj = SimpleNamespace(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=completion_tokens,
                    total_tokens=response.usage_metadata.total_token_count,
                )

                break

            except Exception as e:
                msg = str(e).lower()
                print(e)

                # Retry only on specific transient errors inferred from the message
                transient = (
                    "overloaded" in msg
                    or "resource exhausted" in msg
                    or "quota" in msg
                    or "429" in msg
                    or "unavailable" in msg
                    or "try again" in msg
                )

                # Non-transient error or last attempt: re-raise the exception
                if not transient or attempt == max_retries:
                    raise e

                # Exponential backoff between retries
                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"[Gemini] Transient error (attempt {attempt}/{max_retries}): {e}"
                )
                print(f"[Gemini] Retrying in {delay:.2f}s...")
                time.sleep(delay)

        content_text = response.text or ""

        if verbose:
            print(f"Generation:\n{content_text}\n")

        return content_text, usage_obj


class OpenRouterLLMClient(LLMClient):
    """
    Client for interacting with OpenRouter language models.

    This class manages communication with the OpenRouter API, which provides access to
    multiple language model providers through a unified OpenAI-compatible interface.
    It handles authentication via an OpenRouter API key.

    Attributes:
    ----------
    - client (openai.OpenAI):
        An OpenAI client instance configured for OpenRouter's API endpoint.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the OpenRouter language model and retrieves the response.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initializes the OpenRouterLLMClient with the provided API key.

        Parameters:
        ----------
        - api_key (str):
            The API key for the OpenRouter service.
        - base_url (str):
            The base URL for OpenRouter API (default: "https://openrouter.ai/api/v1").

        Example:
        -------
        >>> client = OpenRouterLLMClient(api_key='your_openrouter_api_key')
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the OpenRouter language model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            The identifier of the OpenRouter model to use (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4o").
        - **kwargs:
            Additional keyword arguments for the OpenRouter API call, such as:
                - temperature (float): Controls the randomness of the output (default is 0.0).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The model's generated response (str).
                - A usage object with token counts (SimpleNamespace).

        Raises:
        ------
        - openai.error.OpenAIError:
            If the API request fails (e.g., invalid model name, insufficient credits).
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 5000)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Type cast to satisfy MyPy - our dict structure matches OpenAI's expected format
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if verbose:
                print("\n=== LLM Response ===")
                print(f"{response.choices[0].message.content}\n")

            content = response.choices[0].message.content

            # Convert usage object to a SimpleNamespace for consistency with other clients
            if response.usage:
                usage_obj = SimpleNamespace(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            else:
                # Fallback if usage is not available
                usage_obj = SimpleNamespace(
                    prompt_tokens=len(prompt.split()),  # Very rough estimate
                    completion_tokens=(
                        len(content.split()) if content else 0
                    ),  # Very rough estimate
                    total_tokens=len(prompt.split())
                    + (len(content.split()) if content else 0),
                )

            return (content.strip() if content else ""), usage_obj

        except Exception as e:
            # Provide helpful error messages for common OpenRouter issues
            error_msg = str(e).lower()
            if "model" in error_msg and (
                "not found" in error_msg or "invalid" in error_msg
            ):
                raise ValueError(
                    f"Model '{model}' not found on OpenRouter. "
                    f"Please check the model name or your OpenRouter account access. "
                    f"Visit https://openrouter.ai/models for available models."
                ) from e
            elif "insufficient" in error_msg or "credit" in error_msg:
                raise ValueError(
                    "Insufficient credits on OpenRouter account. "
                    "Please add credits at https://openrouter.ai/credits"
                ) from e
            else:
                # Re-raise the original exception for other errors
                raise


class VLLMLLMClient(LLMClient):
    """
    Client for interacting with open-source language models using vLLM.

    This class manages local inference with open-source models using vLLM,
    which provides efficient inference for large language models. It handles
    model loading, inference, and response formatting.

    Attributes:
    ----------
    - llm (vllm.LLM):
        The vLLM LLM instance used for inference.
    - model_path (str):
        The path or HuggingFace model ID of the loaded model.

    Methods:
    -------
    - get_response(prompt, model, **kwargs):
        Sends a prompt to the local language model and retrieves the response.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Initializes the VLLMLLMClient with the specified model.

        Parameters:
        ----------
        - model_path (str):
            The path to the model or HuggingFace model ID.
        - **kwargs:
            Additional keyword arguments for vLLM initialization, such as:
                - dtype (str): Data type for model weights (e.g., 'float16', 'bfloat16').
                - gpu_memory_utilization (float): Target GPU memory utilization (0.0 to 1.0).
                - max_model_len (int): Maximum sequence length.

        Raises:
        ------
        - ImportError:
            If vLLM is not installed.
        - RuntimeError:
            If model loading fails.

        Example:
        -------
        >>> client = VLLMLLMClient(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="float16")
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm'."
            )

        self.model_path = model_path
        try:
            self.llm = LLM(model=model_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load model with vLLM: {e}")

    def get_response(
        self, prompt: str, model: str, **kwargs
    ) -> tuple[str, SimpleNamespace]:
        """
        Sends a prompt to the vLLM model and retrieves the response.

        Parameters:
        ----------
        - prompt (str):
            The input text prompt to send to the language model.
        - model (str):
            Ignored for vLLM as the model is already loaded during initialization.
        - **kwargs:
            Additional keyword arguments for the vLLM sampling, such as:
                - temperature (float): Controls the randomness of the output (default is 0.7).
                - max_tokens (int): The maximum number of tokens to generate (default is 500).
                - verbose (bool): If True, prints the prompt and response for debugging (default is False).

        Returns:
        -------
        - tuple[str, SimpleNamespace]:
            A tuple containing:
                - The model's generated response (str).
                - A simple usage object with estimated token counts (SimpleNamespace).

        Example:
        -------
        >>> response, usage = client.get_response(
        ...     prompt="What is the capital of France?",
        ...     temperature=0.7,
        ...     max_tokens=100
        ... )
        >>> print(response)
        "The capital of France is Paris."
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Prompt:\n{prompt}\n")

        # Create sampling parameters for vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Generate response with vLLM
        outputs = self.llm.generate(prompt, sampling_params)

        # Extract the generated text
        generated_text = outputs[0].outputs[0].text.strip()

        if verbose:
            print(f"Generation:\n{generated_text}\n")

        # Create a simple usage object (vLLM doesn't provide detailed token usage)
        # This is a rough estimate for compatibility with other clients

        usage_obj = SimpleNamespace(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(generated_text.split()),
            total_tokens=len(prompt.split()) + len(generated_text.split()),
        )

        return generated_text, usage_obj


def get_llm_client(
    provider: str, config: dict, model: Optional[str] = None
) -> LLMClient:
    """
    Factory function to instantiate an LLM client based on the specified provider.

    Parameters
    ----------
    provider : str
        The name of the language model provider. Supported values are:
            - 'azure': For Azure OpenAI models.
            - 'openai': For standard OpenAI models.
            - 'anthropic': For Anthropic Claude models.
            - 'gemini': For Google Gemini models.
            - 'together': For Together AI models.
            - 'openrouter': For OpenRouter models.
            - 'vllm': For open-source models using vLLM.

    config : dict
        A dictionary containing configuration parameters required by the selected provider.

        For **Azure OpenAI**:
            - 'api_key':       Azure API key.
            - 'endpoint':      Azure OpenAI endpoint URL.
            - 'api_version':   API version (e.g., '2023-05-15').

        For **OpenAI**:
            - 'api_key':       OpenAI API key (e.g., from OPENAI_API_KEY environment variable).

        For **Anthropic**:
            - 'api_key':       Anthropic API key (e.g., from ANTHROPIC_API_KEY environment variable).

        For **Gemini**:
            - 'api_key':       Google Gemini API key (e.g., from GEMINI_API_KEY environment variable).

        For **Together AI**:
            - 'api_key':       Together AI API key.

        For **vLLM**:
            - 'model_path':    Path to the model or HuggingFace model ID.
            - 'dtype':         (optional) Data type for model weights (e.g., 'float16').
            - 'gpu_memory_utilization': (optional) Target GPU memory utilization (0.0 to 1.0).
            - 'max_model_len': (optional) Maximum sequence length.

    model : str, optional
        The model name to use. For vLLM, this can be used instead of config["model_path"].
        This allows using the model_name from the scenario directly.

    Returns
    -------
    LLMClient
        An instance of one of the following, depending on provider:
            - AzureOpenAILLMClient
            - OpenAILLMClient (standard OpenAI)
            - AnthropicLLMClient
            - GeminiLLMClient
            - TogetherLLMClient
            - VLLMLLMClient

    Raises
    ------
    ValueError
        If an unknown provider is specified.
    ImportError
        If vLLM is requested but not installed.
    """
    if provider.lower() == "azure":
        return AzureOpenAILLMClient(
            api_key=config["api_key"],
            endpoint=config["endpoint"],
            api_version=config["api_version"],
        )
    elif provider == "openai":
        return OpenAILLMClient(api_key=config["api_key"])
    elif provider == "anthropic":
        return AnthropicLLMClient(api_key=config["api_key"])
    elif provider == "gemini":
        return GeminiLLMClient(api_key=config["api_key"])
    elif provider == "together":
        return TogetherLLMClient(api_key=config["api_key"])
    elif provider == "openrouter":
        return OpenRouterLLMClient(api_key=config["api_key"])
    elif provider == "vllm":
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not available on this system. This could be due to installation issues or platform compatibility.\n"
                "vLLM may not work on Windows without WSL. Please consider using a different provider like 'azure', 'openai', or 'together'."
            )
        # Extract required parameters
        # Use model_name from scenario if provided in the function call
        # This allows users to specify the model in the scenario like other providers
        model_path = model if model else config["model_path"]

        # Extract optional parameters
        kwargs = {}
        # Define supported parameters based on successful Jean Zay configuration
        supported_params = [
            "device",
            "dtype",
            "enforce_eager",
            "disable_async_output_proc",
            "tensor_parallel_size",
            "enable_prefix_caching",
            "worker_cls",
            "distributed_executor_backend",
            "trust_remote_code",
            "revision",
        ]

        for key in supported_params:
            if key in config:
                # Handle type conversions for different parameter types
                if key in [
                    "enforce_eager",
                    "disable_async_output_proc",
                    "enable_prefix_caching",
                ]:
                    # These are boolean parameters
                    if isinstance(config[key], bool):
                        kwargs[key] = config[key]
                    elif isinstance(config[key], str) and config[key].lower() in [
                        "true",
                        "false",
                    ]:
                        kwargs[key] = config[key].lower() == "true"
                    else:
                        # Default to False for invalid values
                        kwargs[key] = False
                elif key == "tensor_parallel_size":
                    # This is an integer parameter
                    if isinstance(config[key], int):
                        kwargs[key] = config[key]
                    elif isinstance(config[key], str) and config[key].isdigit():
                        kwargs[key] = int(config[key])
                    else:
                        # Default to 1 for invalid values
                        kwargs[key] = 1
                else:
                    # Pass other parameters as-is
                    kwargs[key] = config[key]

        return VLLMLLMClient(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
