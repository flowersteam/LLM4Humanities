"""
# prompt_construction.py

This module provides utility functions for constructing prompts to be used with language models.
Currently it offers:
    - A helper to build a human-readable description of data columns.
    - A helper to build a simple evaluation prompt that includes a codebook, examples, and the entry text to evaluate.

Dependencies:
    - json

Functions:
    - build_data_format_description(column_descriptions):  
      Generates a human-readable description of data columns based on a dictionary of column names and their descriptions.

    - construct_prompt(data_format_description, entry_text, codebook, examples, instructions, selected_fields=None, output_format_example=None, output_format_instructions=None, json_output=True):  
      Builds a comprehensive prompt for a language model by combining data descriptions, input text, coding guidelines, examples, and formatting instructions for the model's response.
"""

import json
from typing import Optional, List, Dict, Any


def build_data_format_description(column_descriptions: dict[str, str]) -> str:
    """
    Builds a textual description of the data format based on column descriptions.

    This function takes a dictionary where keys are column names and values are descriptions
    of those columns. It constructs a formatted string describing the data structure, which
    can be used in prompts for language models.

    Parameters
    ----------
    column_descriptions : dict[str, str]
        A dictionary where:
        - **Key**: Column name (str)
        - **Value**: Description of the column (str)

    Returns
    -------
    str
        A formatted string describing the dataset's columns.
    """
    description = "The data has the following columns:\n"
    for col, desc in column_descriptions.items():
        description += f'- "{col}": {desc}\n'
    return description


def construct_prompt(
    data_format_description: str,
    entry_text: str,
    codebook: str,
    examples: str,
    instructions: str,
    selected_fields: Optional[List[str]] = None,
    output_format_example: Optional[Dict[str, Any]] = None,
    output_format_instructions: Optional[str] = None,
    json_output: bool = True,
) -> str:
    """
    Construct a prompt for a language model from the provided components.

    For JSON workflows, this function explicitly reminds the model to answer
    with the requested fields in valid JSON. Gemini can additionally rely on
    API-level schema enforcement, but OpenAI/OpenRouter and other providers
    still need these textual instructions to stay parseable.
    """
    if selected_fields is None:
        selected_fields = []

    if json_output and output_format_instructions is None:
        output_format_instructions = f"""
- Your response should include the following fields: {', '.join(selected_fields)}.
- **Your response must be in JSON format only. Do not include any explanations, greetings, or additional text.**

**Example response format:**

{json.dumps(output_format_example, ensure_ascii=False, indent=2)}
"""

    prompt = f"""
{instructions}

You are provided with data entries in the following format:

{data_format_description}

Here is an entry to evaluate:

{entry_text}

{codebook}

{examples}

**Instructions:**

- Evaluate the entry according to the codebook and examples.
- Provide your evaluation in the specified format.
{output_format_instructions or ""}
"""
    return prompt
