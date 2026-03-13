"""
cost_estimation.py

This module calculates the cost of OpenAI API usage based on token consumption
for different models. It uses predefined model pricing data to compute the
total cost of API calls.

Dependencies:
    - qualitative_analysis.config
    - typing.Protocol

Classes:
    - UsageProtocol: Protocol to define the structure of the 'usage' object.

Functions:
    - openai_api_calculate_cost(usage: UsageProtocol, model: str = "gpt-4o") -> float:
        Calculates the total API usage cost in USD based on prompt and completion tokens.
"""

from qualitative_analysis.config import MODEL_PRICES
from typing import Protocol


class UsageProtocol(Protocol):
    """
    Protocol to define the structure of the 'usage' object.
    The object must have integer attributes for token usage details.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def openai_api_calculate_cost(usage: UsageProtocol, model: str = "gpt-4o") -> float:
    """
    Calculate the cost of API usage based on token consumption.

    Parameters:
        usage (UsageProtocol): An object with token usage details, specifically:
            - prompt_tokens (int): Number of tokens used in the prompt.
            - completion_tokens (int): Number of tokens used in the completion.
            - total_tokens (int): Total tokens used (prompt + completion).
        model (str): The model name (default: "gpt-4o"). If not found in `MODEL_PRICES`,
                    uses a default estimate.

    Returns:
        float: The total API usage cost in USD, rounded to 6 decimal places.
                For unknown models, returns a generic estimate based on total tokens.

    Example:
        >>> class MockUsage:
        ...     prompt_tokens = 1000
        ...     completion_tokens = 500
        ...     total_tokens = 1500
        ...
        >>> usage = MockUsage()
        >>> openai_api_calculate_cost(usage, model="gpt-4o")
        0.0075

    Notes:
        - For known models, uses exact pricing from `MODEL_PRICES` dictionary.
        - For unknown models, uses a generic estimate of $1 per 1M tokens.
        - Supports dynamic model usage without requiring pre-configuration.
    """
    pricing = MODEL_PRICES.get(model)
    if not pricing:
        # Handle unknown models gracefully with a generic estimate
        # Using $1 per 1M tokens as a reasonable default for cost estimation
        default_cost_per_1k_tokens = 0.001  # $1 per 1M tokens = $0.001 per 1K tokens
        estimated_cost = round(
            usage.total_tokens * default_cost_per_1k_tokens / 1000, 6
        )
        # Optional: Print warning for debugging (can be removed if too verbose)
        print(
            f"Warning: No pricing data for model '{model}'. Using default estimate of ${estimated_cost:.6f}"
        )

        return estimated_cost

    # Use exact pricing for known models
    # /!\ Prices stored as dollars per million tokens ( $/M tokens )
    prompt_cost: float = usage.prompt_tokens * pricing["prompt"] / 1_000_000
    completion_cost: float = usage.completion_tokens * pricing["completion"] / 1_000_000
    calculated_cost: float = round(prompt_cost + completion_cost, 6)

    return calculated_cost
