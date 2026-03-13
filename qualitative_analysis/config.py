"""
config.py

This module manages configuration settings and pricing information for different
LLM providers (OpenAI, Azure OpenAI, Anthropic, Gemini, OpenRouter, vLLM).
It securely loads environment variables and defines pricing data used for
API usage cost estimation in the app.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from a .env file
# This file should contain sensitive keys like API keys and endpoints.
load_dotenv()

# Configuration dictionary for different model providers.
# 'azure' contains settings for accessing Azure's OpenAI services
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "azure": {
        "api_key": os.getenv("AZURE_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_API_VERSION"),
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        # For standard OpenAI usage, you typically just need the API key,
        # no special endpoint or api_version.
        # But you could add more keys if needed (org ID, etc.).
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        # For Anthropic, we just need the API key
    },
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
    },
    "vllm": {
        # Explicitly set device type to fix "Failed to infer device type" error
        "device": os.getenv("VLLM_DEVICE", "cuda"),
        # Support both local paths and HuggingFace model IDs
        # For Jean Zay: set VLLM_LOCAL_MODEL_PATH to the path where you downloaded the model
        "model_path": os.getenv(
            "VLLM_LOCAL_MODEL_PATH",  # First check for a local path
            os.getenv(
                "VLLM_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ),  # Fall back to HF ID
        ),
        # Use half precision (equivalent to float16)
        "dtype": os.getenv("VLLM_DTYPE", "half"),
        # Force eager execution mode (convert string to boolean)
        "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "true").lower() == "true",
        # Disable async output processing (convert string to boolean)
        "disable_async_output_proc": os.getenv("VLLM_DISABLE_ASYNC", "true").lower()
        == "true",
        # Start with tensor parallel size 1, can increase if needed (convert string to int)
        "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        # Enable prefix caching for better performance (convert string to boolean)
        "enable_prefix_caching": os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true").lower()
        == "true",
        # Explicitly set the worker class to fix the "not enough values to unpack" error
        "worker_cls": "vllm.worker.worker.Worker",
        # Set distributed executor backend to None for local execution
        "distributed_executor_backend": None,
        # Prevent HuggingFace from trying to download anything when using local models
        "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "false").lower()
        == "true",
        # Don't try to fetch a specific revision when using local models
        "revision": os.getenv("VLLM_REVISION", None),
    },
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
}


# Pricing information for different models.
# The prices are in dollars per token, where:
#   - 'prompt'     = cost per input token
#   - 'completion' = cost per output token
# Values are approximate / illustrative and are mainly used for relative
# cost estimation inside the app.
MODEL_PRICES: Dict[str, Dict[str, float]] = {
    # OpenAI models - https://openai.com/api/pricing/
    # "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    # "gpt-4o-mini": {
    #     "prompt": 0.00015,
    #     "completion": 0.0006,
    # },
    "gpt-5.1": {
        "prompt": 1.25,  # $1.25 per million input tokens
        "completion": 10,  # $10.00 per million output tokens
    },
    "gpt-5-mini": {
        "prompt": 0.25,  # $0.25 per million input tokens
        "completion": 2,  # $2.00 per million output tokens
    },
    "gpt-5-nano": {
        "prompt": 0.05,  # $0.05 per million input tokens
        "completion": 0.4,  # $0.40 per million output tokens
    },
    "gpt-5-pro": {
        "prompt": 15,  # $15.00 per million input tokens
        "completion": 120,  # $120.00 per million output tokens
    },
    # Anthropic models - https://platform.claude.com/docs/en/about-claude/pricing
    # "claude-3-7-sonnet-20250219": {     /!\DEPRECATED
    #     "prompt": 0.0030,
    #     "completion": 0.0150,
    # },
    "claude-3-5-haiku-20241022": {
        "prompt": 0.8,  # $0.80 per million input tokens
        "completion": 4,  # $4 per million output tokens
    },
    # Gemini models (example pricing - Check on: https://ai.google.dev/gemini-api/docs/pricing?hl=fr)
    "gemini-2.5-flash-lite": {
        "prompt": 0.1,  # $0.10 per million input tokens
        "completion": 0.4,  # $0.40 per million output tokens
    },
    "gemini-2.5-flash": {
        "prompt": 0.3,  # $0.30 per million input tokens
        "completion": 2.5,  # $2.50 per million output tokens
    },
    "gemini-2.5-pro": {
        "prompt": 1.25,  # $1.25 per million input tokens
        "completion": 10,  # $10.00 per million output tokens
    },
    "gemini-3-pro-preview": {
        "prompt": 2,  # $2.00 per million input tokens
        "completion": 12,  # $12.00 per million output tokens
    },
    # Common OpenRouter models (pricing may vary - check https://openrouter.ai/models for current rates)
    # Note: For models not listed here, cost estimation will show a generic message
    "openai/gpt-4o": {
        "prompt": 2.5,  # $2.50/M input tokens
        "completion": 10,  # $10/M output tokens
    },
    "openai/gpt-4o-mini": {
        "prompt": 0.15,  # $0.15/M input tokens
        "completion": 0.6,  # $0.6/M output tokens
    },
    "anthropic/claude-3.5-sonnet": {
        "prompt": 0.8,  # $0.8/M input tokens
        "completion": 4,  # $4/M output tokens
    },
    "anthropic/claude-3.7-sonnet": {
        "prompt": 3,  # $3/M input tokens
        "completion": 15,  # $15/M output tokens
    },
    "anthropic/claude-3-haiku": {
        "prompt": 0.25,  # $0.25/M input tokens
        "completion": 1.25,  # $1.25/M output tokens
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "prompt": 0.02,  # $0.02/M input tokens
        "completion": 0.03,  # $0.03/M output tokens
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "prompt": 0.4,  # $0.40/M input tokens
        "completion": 0.4,  # $0.40/M output tokens
    },
    # "google/gemini-2.0-flash-001": {   RESOURCE UNAVAILABLE
    #     "prompt": 0.000075,
    #     "completion": 0.0003
    # },
}

# NOTE:
# Ensure the .env file is properly set up with your keys:
# .env file might contain:
#   AZURE_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=...
#   AZURE_API_VERSION=...
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...
