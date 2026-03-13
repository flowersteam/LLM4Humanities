"""
data_processing.py

This module provides utility functions for loading, cleaning, processing, and exporting data. 

Dependencies:
    - pandas
    - unicodedata
    - chardet
    - csv

Functions:
    - load_data(file, file_type="csv", delimiter=",", **kwargs): Loads CSV or Excel data into a DataFrame.

    - detect_file_encoding(file): Detects a file's encoding using the chardet library.

    - clean_and_normalize(series): Cleans and normalizes text data in a pandas Series.

    - sanitize_dataframe(df): Removes line breaks from string entries in a DataFrame.

    - select_and_rename_columns(data, selected_columns, column_renames): Selects and renames DataFrame columns.
    
    - load_results_from_csv(load_path): Loads coding results and verbatims from a CSV file.
"""

import pandas as pd
import unicodedata
import chardet
import csv
from typing import Union, IO, Tuple, List, Dict, Any


def load_data(
    file: Union[str, IO], file_type: str = "csv", delimiter: str = ",", **kwargs: Any
) -> pd.DataFrame:
    """
    Loads data from a CSV or Excel file into a pandas DataFrame with robust encoding detection.

    This function attempts to read the specified file using UTF-8 encoding by default.
    If a `UnicodeDecodeError` occurs, it detects the file's encoding using the `chardet` library.
    If detection fails, it falls back to using 'ISO-8859-1' encoding.

    Supported file types:
        - CSV (.csv): Reads using `pandas.read_csv()`
        - Excel (.xlsx): Reads using `pandas.read_excel()`

    Parameters:
    ----------
    file : str or file-like object
        The file path or file-like object to read.
    file_type : str, optional
        The type of file to load. Accepted values are `'csv'` or `'xlsx'`. Default is `'csv'`.
    delimiter : str, optional
        The delimiter used in the CSV file (ignored for Excel files). Default is `','`.
    **kwargs : dict
        Additional keyword arguments passed to `pd.read_csv()` or `pd.read_excel()`.

    Returns:
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame.

    Raises:
    ------
    ValueError
        If the specified `file_type` is not `'csv'` or `'xlsx'`.
    UnicodeDecodeError
        If the file cannot be decoded using UTF-8, the detected encoding, or ISO-8859-1.
    pd.errors.EmptyDataError
        If the file is empty.
    pd.errors.ParserError
        If there is a parsing error in the file.

    Example:
    -------
    # Load a CSV file with default UTF-8 encoding
    data = load_data("data/sample.csv")

    # Load a CSV file with a custom delimiter (semicolon)
    data = load_data("data/sample.csv", delimiter=";")

    # Load an Excel file
    data = load_data("data/sample.xlsx", file_type="xlsx")
    """
    if file_type == "csv":
        # Attempt UTF-8
        try:
            return pd.read_csv(file, delimiter=delimiter, **kwargs)
        except UnicodeDecodeError:
            pass

        # Reset file pointer if file-like
        if hasattr(file, "seek"):
            file.seek(0)

        # Attempt detected encoding
        encoding = detect_file_encoding(file)
        if hasattr(file, "seek"):
            file.seek(0)

        try:
            return pd.read_csv(file, encoding=encoding, delimiter=delimiter, **kwargs)
        except UnicodeDecodeError:
            pass

        # Reset again if file-like
        if hasattr(file, "seek"):
            file.seek(0)

        # Final attempt with ISO-8859-1
        try:
            return pd.read_csv(
                file, encoding="ISO-8859-1", delimiter=delimiter, **kwargs
            )
        except Exception:
            raise ValueError(
                "Failed to read the file with utf-8, detected encoding, or ISO-8859-1."
            )

    elif file_type == "xlsx":
        return pd.read_excel(file, **kwargs)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'xlsx'.")


def detect_file_encoding(file: Union[str, IO]) -> str:
    """
    Detects the character encoding of a file using the `chardet` library.

    This function reads the first 100,000 bytes of a file or file-like object to
    predict its encoding. If the detection fails, it defaults to `'utf-8'`.

    Parameters:
    ----------
    file : str or file-like object
        The file path as a string or a file-like object opened in binary mode (`rb`).

    Returns:
    -------
    str
        The detected encoding (e.g., `'utf-8'`, `'ISO-8859-1'`). Defaults to `'utf-8'` if detection fails.

    Raises:
    ------
    FileNotFoundError
        If the file path does not exist.
    IOError
        If there is an error opening or reading the file.

    Notes:
    -----
    - For file-like objects, ensure they are opened in binary mode (`rb`) for accurate detection.
    - The function only reads up to the first 100,000 bytes for detection.
    """
    # Read a portion of the file for encoding detection
    if hasattr(file, "read"):
        # If file is a file-like object
        rawdata = file.read(100000)
    else:
        # If file is a file path
        with open(file, "rb") as f:
            rawdata = f.read(100000)

    # Use chardet to detect encoding
    result = chardet.detect(rawdata)
    encoding = result["encoding"]

    # Default to 'utf-8' if detection fails
    if encoding is None:
        encoding = "utf-8"

    return encoding


def clean_and_normalize(series: pd.Series) -> pd.Series:
    """
    Cleans and normalizes a pandas Series of text data.

    This function performs the following operations on each entry in the Series:
        - Converts all values to strings.
        - Removes leading and trailing whitespace.
        - Normalizes Unicode characters using NFKD normalization to decompose accented characters.

    Parameters:
    ----------
    series : pd.Series
        A pandas Series containing text data to clean and normalize.

    Returns:
    -------
    pd.Series
        A pandas Series with cleaned and normalized text data.

    Notes:
    -----
    - **Unicode Normalization (NFKD):** This decomposes characters into their base form and diacritical marks.
      For example, `'é'` becomes `'e' + '́'`.
    - This function does not remove non-ASCII characters. For stricter cleaning, consider combining with `.encode()`.

    Example:
    -------
    >>> import pandas as pd
    >>> data = pd.Series(['  Café ', 'Crème brûlée', 'naïve'])
    >>> clean_and_normalize(data)
    0            Cafe
    1    Creme brulee
    2           naive
    dtype: object
    """
    return (
        series.astype(str)
        .str.strip()
        .apply(
            lambda x: (
                unicodedata.normalize("NFKD", x)
                .replace("⁄", "/")  # otherwise problems with fractions
                .encode("ascii", "ignore")
                .decode("ascii")
                if pd.notnull(x)
                else x
            )
        )
    )


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes a pandas DataFrame by removing line breaks from string columns.

    This function replaces:
        - Newline characters (`\n`)
        - Carriage return characters (`\r`)

    with a single space `" "` in all string entries of the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to sanitize.

    Returns:
    -------
    pd.DataFrame
        The sanitized DataFrame with line breaks replaced by spaces.

    Example:
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'Comments': ['Hello\\nWorld', 'Good\\rMorning', 'No line break']})
    >>> sanitize_dataframe(data)
            Comments
    0    Hello World
    1   Good Morning
    2  No line break
    """
    return df.apply(
        lambda col: (
            col.str.replace(r"\r", " ", regex=True)  # Keep line breaks \n
            if col.dtype == "object"
            else col
        )
    )


def select_and_rename_columns(
    data: pd.DataFrame, selected_columns: List[str], column_renames: Dict[str, str]
) -> pd.DataFrame:
    """
    Selects specific columns from a DataFrame and renames them.

    This function performs two operations:
        1. **Selection:** Filters the DataFrame to include only the specified columns.
        2. **Renaming:** Renames the selected columns according to the provided mapping.

    Parameters:
    ----------
    data : pd.DataFrame
        The original DataFrame from which to select and rename columns.
    selected_columns : List[str]
        A list of column names to retain in the DataFrame.
    column_renames : Dict[str, str]
        A dictionary mapping existing column names to their new names.
        Format: {current_name: new_name}

    Returns:
    -------
    pd.DataFrame
        A new DataFrame containing only the selected and renamed columns.

    Raises:
    ------
    KeyError
        If any of the `selected_columns` are not present in the DataFrame.
    """
    return data[selected_columns].rename(columns=column_renames)


def load_results_from_csv(
    load_path: str,
) -> Union[Tuple[List[str], List[Dict[str, str]]], List[Dict[str, str]]]:
    """
    Loads coding results and verbatims from a CSV file.

    This function reads a CSV file containing coding results and optionally associated verbatims.
    If the 'Verbatim' column exists, it separates the verbatims from the coding data and returns both.
    If not, it returns only the coding results.

    Parameters:
    ----------
    load_path : str
        The file path from which the CSV will be read.

    Returns:
    -------
    Union[Tuple[List[str], List[Dict[str, str]]], List[Dict[str, str]]]
        - If the 'Verbatim' column is present:
            Returns a tuple (`verbatims`, `coding`) where:
                - `verbatims` (List[str]): A list of verbatim texts.
                - `coding` (List[Dict[str, str]]): A list of coding results (excluding the 'Verbatim' column).
        - If the 'Verbatim' column is absent:
            Returns `coding` (List[Dict[str, str]]): A list of coding results.

    Raises:
    ------
    FileNotFoundError
        If the specified file does not exist.
    csv.Error
        If the CSV file is malformed or cannot be parsed.
    """
    verbatims: List[str] = []
    coding: List[Dict[str, str]] = []

    with open(load_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError(f"The file '{load_path}' is empty or corrupted.")

        for row in reader:
            if "Verbatim" in fieldnames:
                verbatims.append(row["Verbatim"])
                code = {k: row[k] for k in fieldnames if k != "Verbatim"}
            else:
                code = row
            coding.append(code)

    print(f"Results loaded from: {load_path}")

    return (verbatims, coding) if verbatims else coding
