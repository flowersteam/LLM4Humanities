"""
Module for handling analysis functionality in the Streamlit app.
"""

import streamlit as st
import pandas as pd
import io
from typing import Any, Optional, Tuple

from qualitative_analysis import (
    parse_llm_response,
    openai_api_calculate_cost,
)

from streamlit_app.prompt_construction import (
    build_data_format_description,
    construct_prompt,
)


def format_value_for_prompt(value: Any) -> Any:
    """
    Format a value for display in the prompt.
    Converts float values to integers if they represent whole numbers.

    Args:
        value: The value to format

    Returns:
        Formatted value as string or original value
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _process_data_with_llm(
    app_instance: Any,
    data_to_process: pd.DataFrame,
    debug_mode: bool = False,
    run_number: int = 1,
) -> Tuple[pd.DataFrame, bool]:
    """
    Process a batch of rows with the LLM and return the results.

    This helper is used by `run_analysis` for both Step 6 and Step 8.
    It iterates over `data_to_process`, builds prompts for each row,
    calls the LLM, parses the model output into a dictionary, and
    aggregates everything into a results DataFrame.

    A simple early-stopping mechanism is applied:
    if more than 20% of the entries cannot be parsed (e.g. truncated
    output due to insufficient max tokens), the run is stopped early.

    Args:
        app_instance: The QualitativeAnalysisApp instance.
        data_to_process: DataFrame containing the rows to process.
        debug_mode: Whether to show constructed prompts for debugging.
        run_number: The current run number (used for multi-run analysis).

    Returns:
        A tuple (results_df, early_stop) where:
            - results_df: DataFrame with the processed results 
              for the rows that were actually run.
            - early_stop: True if the run was stopped early because
              the parse failure rate exceeded 20%, False otherwise.
    """
    results = []
    progress_bar = st.progress(0)
    total = len(data_to_process)
    parse_fail_count = 0
    early_stop=False

    # Get the data format description from session state or build it
    data_format_description = st.session_state.get("data_format_description")
    if not data_format_description:
        data_format_description = build_data_format_description(
            app_instance.column_descriptions
        )
        st.session_state["data_format_description"] = data_format_description

    for i, (idx, row) in enumerate(data_to_process.iterrows()):
        try:
            # Build the text from the selected (renamed) columns in the original order
            # Format numeric values as integers if they're whole numbers
            entry_text_str = ""
            # Get the list of renamed columns from the selected columns only
            selected_renamed_cols = [
                app_instance.column_renames.get(col, col)
                for col in app_instance.selected_columns
            ]
            for col in selected_renamed_cols:
                if col in row:
                    entry_text_str += f"{col}: {format_value_for_prompt(row[col])}\n"
            # Remove trailing newline
            if entry_text_str.endswith("\n"):
                entry_text_str = entry_text_str[:-1]

            # Construct the prompt
            prompt = construct_prompt(
                data_format_description=data_format_description,
                entry_text=entry_text_str,
                codebook=app_instance.codebook,
                examples=app_instance.examples,
                instructions="You are an assistant that evaluates data entries.",
                selected_fields=app_instance.selected_fields,
                output_format_example={
                    field: "Your rating here" for field in app_instance.selected_fields
                },
            )

            if debug_mode:
                st.write("**Constructed Prompt:**")
                st.code(prompt)

            # Call the LLM
            response, usage = app_instance.llm_client.get_response(
                prompt=prompt,
                model=app_instance.selected_model,
                app_instance=app_instance,
                verbose=False,
            )

            # Parse the LLM response into a dictionary
            parsed = parse_llm_response(response, app_instance.selected_fields)
            ## Count parse fails
            failed_parse = (not parsed) or all(v is None for v in parsed.values())
            if failed_parse:
                parse_fail_count += 1
            fail_rate=parse_fail_count / total
            if fail_rate > 0.20:
                st.error(
                f"### 🚨 Run stopped  \nMore than 20% ({parse_fail_count} out of {total} entries = {fail_rate*100:.1f}%) of the LLM responses could not be parsed.  \nThis suggests that the model ran out of tokens while generating its response.  \nTry increasing the **Maximum tokens limit** (Step 5 > Advanced settings) and run the analysis again."
                )
                early_stop=True
                break


            # ------------------------------------------------------------------
            # Optional: partial numeric extraction from label column
            # (If label_type is 'Integer' or 'Float', tries to extract a number)
            # ------------------------------------------------------------------
            label_column = st.session_state.get("label_column")
            label_type = st.session_state.get("label_type")

            if label_column and label_column in parsed:

                def extract_numeric_value(value, target_type):
                    import re

                    if isinstance(value, (int, float)):
                        return value

                    str_value = str(value)
                    # Look for integer first
                    int_match = re.search(r"\b(\d+)\b", str_value)
                    if int_match:
                        try:
                            return (
                                int(int_match.group(1))
                                if target_type == "Integer"
                                else float(int_match.group(1))
                            )
                        except ValueError:
                            pass

                    # Look for decimal
                    float_match = re.search(r"\b(\d+\.\d+)\b", str_value)
                    if float_match:
                        try:
                            return (
                                int(float_match.group(1).split(".")[0])
                                if target_type == "Integer"
                                else float(float_match.group(1))
                            )
                        except ValueError:
                            pass

                    # If no numeric value found, return None
                    return None

                # Attempt best-effort numeric conversion
                if label_type == "Integer":
                    converted_value = extract_numeric_value(
                        parsed[label_column], "Integer"
                    )
                    if converted_value is not None:
                        parsed[label_column] = converted_value
                    else:
                        st.warning(
                            f"Could not extract an integer from '{parsed[label_column]}' at row {idx}. Keeping as text."
                        )
                elif label_type == "Float":
                    converted_value = extract_numeric_value(
                        parsed[label_column], "Float"
                    )
                    if converted_value is not None:
                        parsed[label_column] = converted_value
                    else:
                        st.warning(
                            f"Could not extract a float from '{parsed[label_column]}' at row {idx}. Keeping as text."
                        )
                else:
                    # If "Text", just ensure it's string
                    parsed[label_column] = str(parsed[label_column])

            # ------------------------------------------------------------------
            # Merge in annotation columns that might not be in row.to_dict()
            # If your app_instance.processed_data does NOT have them, fetch from app_instance.data:
            # ------------------------------------------------------------------
            annotation_dict = {}
            if app_instance.data is None:
                # Possibly raise an error or return
                raise ValueError(
                    "app_instance.data is None, cannot merge annotation columns."
                )

            for ann_col in app_instance.annotation_columns:
                if ann_col in app_instance.data.columns:
                    annotation_dict[ann_col] = app_instance.data.loc[idx, ann_col]

            # Combine everything: original row data, LLM parsed fields, annotation columns
            combined = {**row.to_dict(), **parsed, **annotation_dict}
            # Add run number for multi-run analysis
            combined["run"] = run_number
            results.append(combined)

        except Exception as e:
            st.error(f"Error processing row {idx}: {e}")
            continue

        progress_bar.progress((i + 1) / total)
    
    # ------------------------------------------------------------------
    # Report parsing issues to the user
    # ------------------------------------------------------------------
    if parse_fail_count > 0 and not early_stop:
        # Informative message
        st.warning(
            f"{parse_fail_count} out of {total} entries ({fail_rate*100:.1f}%) could not be parsed and has been left empty. This often happens when the model output is truncated. Increasing the **Maximum tokens limit** (Step 5 > Advanced settings) might avoid this."
        )            

    # ------------------------------------------------------------------
    # Build results_df from all processed rows
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Final Type Conversions for annotation cols and label col
    # ------------------------------------------------------------------
    label_type = st.session_state.get("label_type", None)
    label_column = st.session_state.get("label_column", None)

    if label_type and not results_df.empty:
        for ann_col in app_instance.annotation_columns:
            if ann_col in results_df.columns:
                col = results_df[ann_col]
                # nan_count = col.isna().sum()
                # if nan_count > 0:
                #     st.warning(
                #         f"Annotation column '{ann_col}' contains {nan_count} missing values."
                #     )
                try:
                    if label_type == "Integer":
                        # Convert only non-NaN values to avoid useless warnings
                        mask = col.notna()
                        results_df.loc[mask, ann_col] = col[mask].astype(int)
                    elif label_type == "Float":
                        results_df[ann_col] = col.astype(float)
                    else:  # treat as string
                        results_df[ann_col] = col.astype(str)
                except (ValueError, TypeError):
                    st.warning(
                        f"Could not fully convert annotation column '{ann_col}' to {label_type}."
                    )

        if label_column and label_column in results_df.columns:
            col = results_df[label_column]
            nan_count = col.isna().sum()
            if nan_count > 0:
                st.warning(
                    f"LLM label column '{label_column}' contains {nan_count} missing values."
                )
            
            try:
                if label_type == "Integer":
                    # Convert only non-NaN values to avoid useless warnings
                    mask = col.notna()
                    results_df.loc[mask, label_column] = col[mask].astype(int)
                elif label_type == "Float":
                    results_df[label_column] = col.astype(float)
                else:  # treat as string
                    results_df[label_column] = col.astype(str)
            except (ValueError, TypeError):
                st.warning(
                    f"Could not convert LLM label column '{label_column}' to {label_type}."
                )

    return results_df, early_stop


def _process_multiple_runs(
    app_instance: Any,
    data_to_process: pd.DataFrame,
    n_runs: int,
    debug_mode: bool = False,
    is_remaining_data: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """
    Process the same data across multiple runs for consistency analysis.

    This function calls `_process_data_with_llm` `n_runs` times on the
    same `data_to_process` and concatenates the results to allow
    downstream consistency / reliability analysis of LLM outputs.

    If any run triggers early stopping, the loop is interrupted and no
    further runs are executed.

    Args:
        app_instance: The QualitativeAnalysisApp instance.
        data_to_process: DataFrame containing the rows to process.
        n_runs: Number of runs to execute on the same data.
        debug_mode: Whether to show constructed prompts for debugging.
        is_remaining_data: Currently unused in the implementation.
            Reserved for potential future use to distinguish runs on
            remaining data (Step 8) from runs on the main data (Step 6).

    Returns:
        A tuple (combined_results_df, early_stop) where:
            - combined_results_df: DataFrame with results from all
              completed runs (may be partial if early stopping occurs).
            - early_stop: True if at least one run was stopped early
              due to a high parse failure rate, False otherwise.
    """
    all_results = []

    for run in range(1, n_runs + 1):
        st.info(f"Processing run {run} of {n_runs}...")

        # Process the data for this run
        run_results_df, early_stop = _process_data_with_llm(
            app_instance, data_to_process, debug_mode, run_number=run
        )
        if early_stop:
            break
        all_results.append(run_results_df)

    # Combine all runs into a single DataFrame
    combined_results_df = pd.concat(all_results, ignore_index=True)

    return combined_results_df, early_stop


def run_analysis(
    app_instance: Any,
    analyze_remaining: bool = False,
    previous_results_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """
    Run the LLM-based analysis on the dataset (Step 6 or Step 8).

    When `analyze_remaining=False` (default), this runs Step 6 on the
    main processed data. When `analyze_remaining=True`, this runs Step 8
    on the remaining (unprocessed) data.

    Internally, `_process_data_with_llm` and `_process_multiple_runs`
    may stop early if more than 20% of the LLM responses cannot be parsed.
    In that case, partial results are still returned.

    Args:
        app_instance: The QualitativeAnalysisApp instance.
        analyze_remaining: Whether to analyze remaining data (Step 8)
            instead of the main data (Step 6).
        previous_results_df: Optional previous results DataFrame for Step 8.
            If None, the function will try to retrieve it from the session
            state (results from Step 6).

    Returns:
        The final results DataFrame (possibly partial if early stopping
        occurred), or None if the analysis was not run.
    """
    if analyze_remaining:
        st.markdown(
            "### Step 8: Run Analysis on Remaining Data", unsafe_allow_html=True
        )
        with st.expander("Show/hide details of step 8", expanded=True):
            # ------------------------------------------------------------------
            # Validate that we have the necessary data, codebook, fields, etc.
            # ------------------------------------------------------------------
            if app_instance.processed_data is None or app_instance.processed_data.empty:
                st.warning("No processed data. Please go to Step 2.")
                return None

            if not app_instance.codebook.strip():
                st.warning("Please provide a codebook in Step 3.")
                return None

            if not app_instance.selected_fields:
                st.warning("Please specify the fields to extract in Step 4.")
                return None

            if not app_instance.llm_client or not app_instance.selected_model:
                st.warning("Please configure the model in Step 5.")
                return None

            # ------------------------------------------------------------------
            # Build the data format description for the prompt
            # ------------------------------------------------------------------
            data_format_description = build_data_format_description(
                app_instance.column_descriptions
            )
            st.session_state["data_format_description"] = data_format_description

            # ------------------------------------------------------------------
            # For Step 8: Check for previous results and identify remaining data
            # ------------------------------------------------------------------
            # Check if we have results from Step 6
            if previous_results_df is None:
                previous_results_df = st.session_state.get("results_df")
                if previous_results_df is None:
                    st.warning("No results from Step 6. Please run Step 6 first.")
                    return None

            # ------------------------------------------------------------------
            # RUN SELECTION FOR STEP 8
            # ------------------------------------------------------------------
            # Check if we have multiple runs from Step 6
            if "run" in previous_results_df.columns:
                available_runs = sorted(previous_results_df["run"].unique())

                if len(available_runs) > 1:
                    st.subheader("Select Run for Remaining Data Analysis")
                    st.markdown(
                        """
                        Since you ran the analysis using multiple runs on the 
                        manually annotated data, you now need to choose which run's predictions you want to
                        keep for the already processed data.
                        
                        This will not impact the results of the remaining data analysis,
                        as the remaining data will be processed using the same model and configuration.
                        
                        It will only affect the results of the already processed data,
                        so you can choose the run that presented the best results.
                        """
                    )

                    # Create run summary for user decision
                    run_summaries = []
                    for run_id in available_runs:
                        run_data = previous_results_df[
                            previous_results_df["run"] == run_id
                        ]
                        run_summaries.append(
                            {"Run": f"Run {run_id}", "Samples": len(run_data)}
                        )

                    # Let user select which run to use
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        selected_run = st.selectbox(
                            "Choose which run to use for remaining data annotation:",
                            options=available_runs,
                            format_func=lambda x: f"Run {x}",
                            key="step8_selected_run",
                        )

                    with col2:
                        # Show some basic info about the selected run
                        selected_run_data = previous_results_df[
                            previous_results_df["run"] == selected_run
                        ]
                        st.metric("Selected Run Samples", len(selected_run_data))

                    # Filter previous results to only include the selected run
                    previous_results_df = selected_run_data.copy()

                    st.success(
                        f"✅ **Run {selected_run} selected** for remaining data annotation"
                    )

                    # Add some spacing before the next section
                    st.markdown("---")
                else:
                    # Only one run available
                    st.info(
                        f"Using results from the single run in Step 6 ({len(previous_results_df)} samples)"
                    )
            else:
                # No run column (legacy data or single run without run tracking)
                st.info(
                    f"Using results from Step 6 ({len(previous_results_df)} samples)"
                )

            # Get the original full dataset (before annotation filtering)
            original_data = app_instance.original_data
            if original_data is None:
                # If original_data is not available, use the processed_data as fallback
                original_data = app_instance.processed_data
                st.warning(
                    "Original unfiltered data not found. Using filtered data instead."
                )

            # Identify rows that have been processed in Step 6 (now filtered to selected run)
            processed_indices = set(previous_results_df.index)

            # Get all indices from the original data
            all_indices = set(original_data.index)

            # Find indices of rows that haven't been processed yet
            remaining_indices = all_indices - processed_indices

            if len(remaining_indices) == 0:
                st.success(
                    "All data has already been processed in Step 6. No remaining data to analyze."
                )
                return previous_results_df

            # Create a DataFrame with the remaining data
            # We need to ensure the remaining data has the same columns as processed_data
            # First, get the remaining rows from the original data
            remaining_original = original_data.loc[list(remaining_indices)]

            # Then, select only the columns that are in processed_data
            processed_columns = app_instance.processed_data.columns
            remaining_columns = [
                col for col in processed_columns if col in remaining_original.columns
            ]

            # Create the final data subset with the correct columns
            data_subset = remaining_original[remaining_columns]

            # Debug information
            st.write(f"Debug: Total rows in original data: {len(original_data)}")
            st.write(f"Debug: Rows already processed: {len(processed_indices)}")
            st.write(f"Debug: Remaining rows to process: {len(remaining_indices)}")

            # Count annotated vs non-annotated rows in the remaining data
            if app_instance.annotation_columns:
                # Get the total number of rows in the original dataset
                total_original_rows = len(original_data)

                # Get the number of rows with annotations in the original dataset
                has_annotations_original = (
                    original_data[app_instance.annotation_columns].notna().all(axis=1)
                )
                annotated_original_count = has_annotations_original.sum()

                # Get the number of rows with annotations that were already processed
                processed_annotated_count = len(previous_results_df)

                # Calculate remaining rows with annotations
                remaining_annotated_count = (
                    annotated_original_count - processed_annotated_count
                )

                # Calculate rows without annotations (these weren't processed in Step 6)
                non_annotated_count = total_original_rows - annotated_original_count

                # Total remaining rows
                total_remaining = remaining_annotated_count + non_annotated_count

                st.info(
                    f"Found {total_remaining} rows that haven't been processed yet:"
                )
                st.info(f"- {remaining_annotated_count} rows with annotations")
                st.info(f"- {non_annotated_count} rows without annotations")
            else:
                st.info(
                    f"Found {len(data_subset)} rows that haven't been processed yet."
                )

            # ------------------------------------------------------------------
            # Let user pick how many rows to process
            # ------------------------------------------------------------------
            st.subheader("Choose how many rows to analyze")

            process_options = ["All remaining rows", "Subset of remaining rows"]
            radio_key = "remaining_process_option_radio"
            input_key = "remaining_num_rows_input"

            selected_option = st.radio(
                "Process:", process_options, index=0, key=radio_key
            )

            num_rows = len(data_subset)
            if selected_option == process_options[1]:  # "Subset of remaining rows"
                num_rows = st.number_input(
                    "Number of rows to process:",
                    min_value=1,
                    max_value=len(data_subset),
                    value=min(10, len(data_subset)),
                    step=1,
                    key=input_key,
                )

            # ------------------------------------------------------------------
            # Cost Estimation button (optional, for one entry)
            # ------------------------------------------------------------------
            button_key = "remaining_estimate_price_button"
            if st.button(
                "Estimate price before analysis (will run on one entry)", key=button_key
            ):
                # Just process the first row for cost estimation
                first_entry = data_subset.iloc[0]
                # Build the text from the selected (renamed) columns in the original order
                entry_text_str = ""
                # Get the list of renamed columns from the selected columns only
                selected_renamed_cols = [
                    app_instance.column_renames.get(col, col)
                    for col in app_instance.selected_columns
                ]
                for col in selected_renamed_cols:
                    if col in data_subset.columns:
                        entry_text_str += (
                            f"{col}: {format_value_for_prompt(first_entry[col])}\n"
                        )
                # Remove trailing newline
                if entry_text_str.endswith("\n"):
                    entry_text_str = entry_text_str[:-1]

                # Construct the prompt
                prompt = construct_prompt(
                    data_format_description=data_format_description,
                    entry_text=entry_text_str,
                    codebook=app_instance.codebook,
                    examples=app_instance.examples,
                    instructions="You are an assistant that evaluates data entries.",
                    selected_fields=app_instance.selected_fields,
                    output_format_example={
                        field: "Sample text" for field in app_instance.selected_fields
                    },
                )

                try:
                    response, usage = app_instance.llm_client.get_response(
                        prompt=prompt,
                        model=app_instance.selected_model,
                        app_instance=app_instance
                    )
                    cost_for_one = openai_api_calculate_cost(
                        usage, app_instance.selected_model
                    )
                    total_cost_estimate = cost_for_one * num_rows

                    st.info(
                        f"Estimated cost for processing one entry: ${cost_for_one:.4f}"
                    )
                    st.info(
                        f"Estimated total cost for {num_rows} entries: ${total_cost_estimate:.4f}"
                    )

                    st.session_state["cost_for_one"] = cost_for_one
                    st.session_state["total_cost_estimate"] = total_cost_estimate

                except Exception as e:
                    st.error(f"Error estimating cost: {e}")

            # ------------------------------------------------------------------
            # Debug mode (to show the constructed prompt for each row)
            # ------------------------------------------------------------------
            debug_key = "remaining_debug_mode_checkbox"
            debug_mode = st.checkbox(
                "Show constructed prompt for debugging",
                value=False,
                key=debug_key,
            )

            # ------------------------------------------------------------------
            # Run Analysis
            # ------------------------------------------------------------------
            button_text = "Run Analysis on Remaining Data"
            button_key = "run_remaining_analysis_button"

            if st.button(button_text, key=button_key):
                st.info("Processing entries...")

                # Prepare data to process
                data_to_process = data_subset.head(num_rows)

                # Get the selected run number for consistency (default to 1 if not available)
                selected_run_number = st.session_state.get("step8_selected_run", 1)

                # Process the data using the shared helper function with the selected run number
                results_df, early_stop = _process_data_with_llm(
                    app_instance,
                    data_to_process,
                    debug_mode,
                    run_number=selected_run_number,
                )

                # For Step 8: Combine with previous results
                combined_results_df = pd.concat([previous_results_df, results_df])

                # Update the session state with the combined results
                app_instance.results = combined_results_df.to_dict("records")
                st.session_state["results"] = app_instance.results
                st.session_state["results_df"] = combined_results_df

                if not early_stop:
                    st.success("Analysis of remaining data completed!")

                # Display the new results
                st.subheader("New Results")
                st.dataframe(results_df.head())

                # Display the combined results
                st.subheader("Combined Results (Step 6 + Step 8)")
                st.dataframe(combined_results_df.head())

                # Show statistics about the combined results
                st.subheader("Analysis Statistics")

                # Get total rows from original data if available, otherwise use processed_data
                if app_instance.original_data is not None:
                    total_rows = len(app_instance.original_data)
                    annotated_rows = len(app_instance.processed_data)
                    non_annotated_rows = total_rows - annotated_rows
                else:
                    total_rows = len(app_instance.processed_data)
                    annotated_rows = total_rows
                    non_annotated_rows = 0

                analyzed_rows = len(combined_results_df)

                st.info(f"Total rows in original dataset: {total_rows}")
                if app_instance.annotation_columns:
                    st.info(f"- Annotated rows: {annotated_rows}")
                    st.info(f"- Non-annotated rows: {non_annotated_rows}")
                st.info(
                    f"Total rows analyzed by LLM: {analyzed_rows} ({analyzed_rows/total_rows*100:.1f}% of total dataset)"
                )

                # Optional: Provide a download button for combined results
                filename_input = st.text_input(
                    "**Enter a filename for your combined results:**",
                    value="complete_analysis_results.xlsx",
                    key="combined_results_filename_input",
                )
                if not filename_input.endswith(".xlsx"):
                    filename_input += ".xlsx"

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    combined_results_df.to_excel(
                        writer, index=False, sheet_name="Results"
                    )
                data_xlsx = output.getvalue()

                st.download_button(
                    label="💾 **Download Combined Results as Excel**",
                    data=data_xlsx,
                    file_name=filename_input,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_combined_results_button",
                )

                return combined_results_df
    else:
        st.markdown("### Step 6: Run Analysis", unsafe_allow_html=True)
        with st.expander("Show/hide details of step 6", expanded=True):
            # ------------------------------------------------------------------
            # Validate that we have the necessary data, codebook, fields, etc.
            # ------------------------------------------------------------------
            if app_instance.processed_data is None or app_instance.processed_data.empty:
                st.warning("No processed data. Please go to Step 2.")
                return None

            if not app_instance.codebook.strip():
                st.warning("Please provide a codebook in Step 3.")
                return None

            if not app_instance.selected_fields:
                st.warning("Please specify the fields to extract in Step 4.")
                return None

            if not app_instance.llm_client or not app_instance.selected_model:
                st.warning("Please configure the model in Step 5.")
                return None

            # ------------------------------------------------------------------
            # Build the data format description for the prompt
            # ------------------------------------------------------------------
            data_format_description = build_data_format_description(
                app_instance.column_descriptions
            )
            st.session_state["data_format_description"] = data_format_description

            # For Step 6: Use the full processed data
            data_subset = app_instance.processed_data

            # ------------------------------------------------------------------
            # Let user pick how many rows to process
            # ------------------------------------------------------------------
            st.subheader("Choose how many rows to analyze")

            process_options = ["All rows", "Subset of rows"]
            radio_key = "process_option_radio"
            input_key = "num_rows_input"

            selected_option = st.radio(
                "Process:", process_options, index=0, key=radio_key
            )

            num_rows = len(data_subset)
            if selected_option == process_options[1]:  # "Subset of rows"
                num_rows = st.number_input(
                    "Number of rows to process:",
                    min_value=1,
                    max_value=len(data_subset),
                    value=min(10, len(data_subset)),
                    step=1,
                    key=input_key,
                )

            # ------------------------------------------------------------------
            # Multiple runs configuration
            # ------------------------------------------------------------------
            st.subheader("Multiple Runs Configuration")
            st.markdown(
                """
                **Why multiple runs?** Since LLM outputs can vary between runs, running the same analysis 
                multiple times helps assess the consistency and reliability of the model's predictions. 
                This is especially important for research purposes where you want to report robust metrics.
                """
            )

            n_runs = st.number_input(
                "Number of runs:",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="n_runs_input",
                help="Number of times to run the analysis on the same data. More runs provide better consistency estimates but increase cost.",
            )

            # ------------------------------------------------------------------
            # Cost Estimation button (optional, for one entry)
            # ------------------------------------------------------------------
            button_key = "estimate_price_button"
            if st.button(
                "Estimate price before analysis (will run on one entry)", key=button_key
            ):
                # Just process the first row for cost estimation
                first_entry = data_subset.iloc[0]
                # Build the text from the selected (renamed) columns in the original order
                entry_text_str = ""
                # Get the list of renamed columns from the selected columns only
                selected_renamed_cols = [
                    app_instance.column_renames.get(col, col)
                    for col in app_instance.selected_columns
                ]
                for col in selected_renamed_cols:
                    if col in data_subset.columns:
                        entry_text_str += (
                            f"{col}: {format_value_for_prompt(first_entry[col])}\n"
                        )
                # Remove trailing newline
                if entry_text_str.endswith("\n"):
                    entry_text_str = entry_text_str[:-1]

                # Construct the prompt
                prompt = construct_prompt(
                    data_format_description=data_format_description,
                    entry_text=entry_text_str,
                    codebook=app_instance.codebook,
                    examples=app_instance.examples,
                    instructions="You are an assistant that evaluates data entries.",
                    selected_fields=app_instance.selected_fields,
                    output_format_example={
                        field: "Sample text" for field in app_instance.selected_fields
                    },
                )

                try:
                    response, usage = app_instance.llm_client.get_response(
                        prompt=prompt,
                        model=app_instance.selected_model,
                        app_instance=app_instance
                    )
                    cost_for_one = openai_api_calculate_cost(
                        usage, app_instance.selected_model
                    )
                    total_cost_estimate = cost_for_one * num_rows * n_runs

                    st.info(
                        f"Estimated cost for processing one entry: ${cost_for_one:.4f}"
                    )
                    st.info(
                        f"Estimated cost for {num_rows} entries × {n_runs} runs: ${total_cost_estimate:.4f}"
                    )

                    st.session_state["cost_for_one"] = cost_for_one
                    st.session_state["total_cost_estimate"] = total_cost_estimate

                except Exception as e:
                    st.error(f"Error estimating cost: {e}")

            # ------------------------------------------------------------------
            # Debug mode (to show the constructed prompt for each row)
            # ------------------------------------------------------------------
            debug_key = "debug_mode_checkbox"
            debug_mode = st.checkbox(
                "Show constructed prompt for debugging",
                value=False,
                key=debug_key,
            )

            # ------------------------------------------------------------------
            # Run Analysis
            # ------------------------------------------------------------------
            button_text = "Run Analysis"
            button_key = "run_analysis_button"

            if st.button(button_text, key=button_key):
                st.info("Processing entries...")

                # Prepare data to process
                data_to_process = data_subset.head(num_rows)

                # Process the data using multiple runs if specified
                if n_runs > 1:
                    results_df, early_stop = _process_multiple_runs(
                        app_instance, data_to_process, n_runs, debug_mode
                    )
                else:
                    results_df, early_stop = _process_data_with_llm(
                        app_instance, data_to_process, debug_mode, run_number=1
                    )

                # For Step 6: Just store the results
                app_instance.results = results_df.to_dict("records")
                st.session_state["results"] = app_instance.results

                if not early_stop:
                    st.success("Analysis completed!")

                # ------------------------------------------------------------------
                # Store final DataFrame in session and mark analysis as completed
                # ------------------------------------------------------------------
                st.session_state["results_df"] = results_df
                st.session_state["analysis_completed"] = True
                st.session_state["n_runs_used"] = n_runs
                st.session_state["entries_processed"] = len(data_to_process)

            # ------------------------------------------------------------------
            # Display results (outside button block to persist across reruns)
            # ------------------------------------------------------------------
            if st.session_state.get("analysis_completed", False):
                results_df = st.session_state["results_df"]
                n_runs_used = st.session_state.get("n_runs_used", 1)
                entries_processed = st.session_state.get("entries_processed", 0)

                # Display results with run information
                if n_runs_used > 1:
                    st.subheader("Multi-Run Analysis Results")

                    # Show summary statistics
                    st.write(f"**Total entries processed:** {entries_processed}")
                    st.write(f"**Number of runs:** {n_runs_used}")
                    st.write(f"**Total result rows:** {len(results_df)}")

                    # Option to view results by run or aggregated
                    view_option = st.radio(
                        "View results:",
                        ["Aggregated (all runs)", "By individual run"],
                        key="results_view_option",
                    )

                    if view_option == "By individual run":
                        selected_run = st.selectbox(
                            "Select run to view:",
                            options=sorted(results_df["run"].unique()),
                            key="selected_run_view",
                        )
                        display_df = results_df[results_df["run"] == selected_run]
                        st.write(f"**Results for Run {selected_run}:**")
                    else:
                        display_df = results_df
                        st.write("**All Results (All Runs):**")

                    st.dataframe(display_df.head())
                else:
                    st.subheader("Analysis Results")
                    st.dataframe(results_df.head())

                # Provide a download button for results
                format_choice = st.radio(
                    "Choose the output format:",
                    ("Excel (.xlsx)", "CSV (.csv)"),
                    index=0,
                )

                filename_input = st.text_input(
                    "**Enter a filename for your results:**",
                    value="analysis_results",
                    key="results_filename_input",
                )

                if format_choice == "Excel (.xlsx)":
                    if not filename_input.endswith(".xlsx"):
                        filename_input += ".xlsx"

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        results_df.to_excel(writer, index=False, sheet_name="Results")
                    data_file = output.getvalue()
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                else:  # CSV
                    if not filename_input.endswith(".csv"):
                        filename_input += ".csv"

                    data_file = results_df.to_csv(index=False).encode("utf-8")
                    mime_type = "text/csv"

                st.download_button(
                    label="💾 **Download Results as Excel**",
                    data=data_file,
                    file_name=filename_input,
                    mime=mime_type,
                    key="download_results_button",
                )


                return results_df

    return None
