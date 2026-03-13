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
    Constructs a prompt for gemini, taking into account the automatic json output.

    The prompt includes:
    - The codebook (coding rules or guidelines),
    - One or more worked examples,
    - The entry text to evaluate.

    Parameters
    ----------
    codebook : str
        Set of rules or guidelines the model should follow during evaluation.

    examples : str
        Example evaluations to guide the model's reasoning and output style.

    entry_text : str
        The text entry that the model is tasked to evaluate.

    Returns
    -------
    str
        A prompt string containing the codebook, examples, and the entry to evaluate.
    """
    # Assemble the prompt
    prompt = f"""
**Codebook:**
{codebook}

**Examples:**
{examples}

**Below is the entry to evaluate:**

{entry_text}
    """
    return prompt

# def construct_prompt(
#     data_format_description: str,
#     entry_text: str,
#     codebook: str,
#     examples: str,
#     instructions: str,
#     selected_fields: Optional[List[str]] = None,
#     output_format_example: Optional[Dict[str, Any]] = None,
#     output_format_instructions: Optional[str] = None,
#     json_output: bool = True,
# ) -> str:
#     """
#     Constructs a prompt for a language model based on provided components.

#     This function assembles various elements such as data format descriptions, entry text,
#     codebooks, examples, and instructions into a single prompt string that can be used
#     with a language model to generate structured responses.

#     Parameters:
#     ----------
#     data_format_description : str
#         Description of the dataset's format, usually from `build_data_format_description`.

#     entry_text : str
#         The text entry that the model is tasked to evaluate.

#     codebook : str
#         Set of rules or guidelines the model must follow during evaluation.

#     examples : str
#         Example evaluations to guide the model's output format and logic.

#     instructions : str
#         General task instructions for the language model.

#     selected_fields : Optional[List[str]], optional
#         Fields that must appear in the model's output. Default is `None`.

#     output_format_example : Optional[Dict[str, Any]], optional
#         Example of the expected output structure in dictionary format. Default is `None`.

#     output_format_instructions : Optional[str], optional
#         Specific instructions for formatting the model's output.
#         If `None`, default instructions are generated using `selected_fields`.

#     json_output : bool, optional (default=True)
#         If `True`, enforces JSON-only output. If `False`, allows free text.

#     Returns:
#     -------
#     str
#         A well-structured prompt to guide the language model's response.
#     """
#     # Default to an empty list if no fields are provided
#     if selected_fields is None:
#         selected_fields = []

#     # Generate default JSON output instructions if none are provided
#     if json_output:
#         if output_format_instructions is None:
#             output_format_instructions = f"""
# - Your response should include the following fields: {', '.join(selected_fields)}.
# - **Your response must be in JSON format only. Do not include any explanations, greetings, or additional text.**

# **Example response format:**

# {json.dumps(output_format_example, ensure_ascii=False, indent=2)}
# """

#     # Assemble the prompt
#     prompt = f"""
# {instructions}

# You are provided with data entries in the following format:

# {data_format_description}

# Here is an entry to evaluate:

# {entry_text}

# {codebook}

# {examples}

# **Instructions:**

# - Evaluate the entry according to the codebook and examples.
# - Provide your evaluation in the specified format.
# {output_format_instructions}
# """
#     return prompt