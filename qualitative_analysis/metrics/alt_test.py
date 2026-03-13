"""
alt_test.py

This module provides utility functions for performing the Alternative Annotator Test (alt-test)
on data with model predictions and human annotations. It supports two types of alignment metrics:
    - "accuracy": for discrete classification tasks (default).
    - "rmse": for continuous labels (using negative RMSE so that higher is better).

It also includes functionality for multiple-comparison correction using the Benjamini–Yekutieli procedure,
and computes the winning rate and average advantage probability.

Functions:
    - benjamini_yekutieli_correction(pvals, alpha=0.05)
    - accuracy_alignment(source, others)
    - rmse_alignment(source, others)
    - run_alt_test_general(...)
    - run_alt_test_on_results(...)
"""

import pandas as pd
from typing import List, Dict, Any, Callable, Tuple
import numpy as np
from scipy.stats import ttest_1samp


def convert_labels(labels: List[Any], label_type: str = "auto") -> List[Any]:
    """
    Convert a list of labels to the specified type.

    Parameters
    ----------
    labels : List[Any]
        The list of labels to convert.
    label_type : str, optional
        The type to convert the labels to. Options are:
        - "int": Convert all labels to integers
        - "str": Convert all labels to strings
        - "auto": Infer the best type (default)

    Returns
    -------
    List[Any]
        The list of converted labels.
    """
    if label_type == "int":
        # Convert all labels to integers
        return [
            int(label) if label != -1 and not pd.isna(label) else label
            for label in labels
        ]
    elif label_type == "str":
        # Convert all labels to strings
        return [str(label) if not pd.isna(label) else label for label in labels]
    elif label_type == "float":
        # Convert all labels to floats
        converted = []
        for label in labels:
            if pd.isna(label):
                converted.append(label)
                continue
            try:
                converted.append(float(label))
            except (TypeError, ValueError):
                converted.append(np.nan)
        return converted
    elif label_type == "auto":
        # Try to infer the best type
        try:
            # Check if all labels can be converted to integers
            # Skip NA values and -1 (used for missing values)
            all_int = all(
                isinstance(label, int)
                or (
                    isinstance(label, (str, float))
                    and label != -1
                    and not pd.isna(label)
                    and float(label).is_integer()
                )
                for label in labels
                if not pd.isna(label)
            )
            if all_int:
                return [
                    int(label) if label != -1 and not pd.isna(label) else label
                    for label in labels
                ]
            else:
                return [str(label) if not pd.isna(label) else label for label in labels]
        except (ValueError, TypeError):
            # If conversion fails, return as strings
            return [str(label) if not pd.isna(label) else label for label in labels]
    else:
        # Unknown label_type, return as is
        return labels


def benjamini_yekutieli_correction(
    pvals: List[float], alpha: float = 0.05
) -> List[bool]:
    """
    Applies the Benjamini–Yekutieli procedure to a list of p-values to control the false discovery rate (FDR)
    under arbitrary dependence.

    Parameters
    ----------
    pvals : List[float]
        A list of p-values from multiple hypothesis tests.
    alpha : float, optional
        The desired overall significance level (default is 0.05).

    Returns
    -------
    List[bool]
        A boolean list indicating which null hypotheses are rejected.

    Example
    -------
    >>> pvals = [0.01, 0.04, 0.03, 0.20]
    >>> benjamini_yekutieli_correction(pvals, alpha=0.05)
    [True, True, True, False]
    """
    m = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_indices]
    c_m = sum(1.0 / i for i in range(1, m + 1))

    rejected = [False] * m
    max_k = -1
    for k in range(m):
        threshold_k = (k + 1) * alpha / (m * c_m)
        if sorted_pvals[k] <= threshold_k:
            max_k = k
    if max_k >= 0:
        for i in range(max_k + 1):
            rejected[i] = True
    out = [False] * m
    for i, idx in enumerate(sorted_indices):
        out[idx] = rejected[i]
    return out


def accuracy_alignment(source: Any, others: List[Any]) -> float:
    """
    Computes the alignment score using the "accuracy" metric:
    Returns the average of 1.0 for each element in `others` that equals the string representation of `source`,
    and 0.0 otherwise.

    Parameters
    ----------
    source : Any
        The source value (e.g., model prediction or annotator's value).
    others : List[Any]
        A list of other annotators' values.

    Returns
    -------
    float
        The average accuracy score.

    Example
    -------
    >>> accuracy_alignment("A", ["A", "B", "A"])
    0.6666666666666666
    """
    scores = [1.0 if str(source) == str(other) else 0.0 for other in others]
    return sum(scores) / len(scores) if scores else 0.0


def rmse_alignment(source: Any, others: List[Any]) -> float:
    """
    Computes alignment for continuous values using negative RMSE (root mean squared error).
    Higher is better, so we return the negative of the RMSE.

    Parameters
    ----------
    source : Any
        The source numeric value (e.g., model prediction).
    others : List[Any]
        A list of numeric values from other annotators.

    Returns
    -------
    float
        The negative RMSE between the source and the others.
        If `others` is empty, returns 0.0 by convention.
    """
    if len(others) == 0:
        return 0.0

    # Convert source and others to float
    try:
        src_val = float(source)
    except (TypeError, ValueError):
        # If conversion fails, treat as missing => 0.0
        return 0.0

    float_others = []
    for o in others:
        try:
            float_others.append(float(o))
        except (TypeError, ValueError):
            # If conversion fails, skip or treat as 0
            float_others.append(0.0)

    # Compute RMSE
    differences = [src_val - x for x in float_others]
    mse = np.mean([d**2 for d in differences])
    rmse = np.sqrt(mse)

    # Return negative for "higher-is-better"
    return -rmse


def run_alt_test_general(
    df: pd.DataFrame,
    annotation_columns: List[str],
    model_col: str = "ModelPrediction",
    epsilon: float = 0.1,
    alpha: float = 0.05,
    metric: str = "accuracy",
    verbose: bool = True,
    label_type: str = "auto",
) -> Dict[str, Any]:
    """
    Runs the Alternative Annotator Test (alt-test) on a DataFrame containing model predictions
    and human annotations.

    For each instance and for each annotator (excluded in turn), this function computes:
        - S_llm: The alignment score between the model's prediction and the remaining annotators.
        - S_hum: The alignment score between the excluded annotator's value and the remaining annotators.

    The alignment scores are computed using one of two metrics:

    - "accuracy" (default): Uses exact match (equality) between labels.
      Uses `accuracy_alignment(source, others)`.
    - "rmse": For continuous values. Computes the negative RMSE between the source and the others.
      Uses `rmse_alignment(source, others)`.

    After computing these scores, a binary indicator is set (1.0 if S_llm > S_hum, 0.0 if S_llm < S_hum,
    and 1.0 if tied). A one-sided paired t-test is then performed on the difference (W_h - W_f) for each annotator,
    and a multiple-comparison correction (Benjamini–Yekutieli) is applied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model predictions and human annotations. It must include:
          - A column specified by `model_col` with the model's prediction.
          - At least three annotator columns provided via `annotation_columns`.
    annotation_columns : List[str]
        List of human annotator column names. These columns will be used for the alt-test.
        Some of these columns may be missing values per row; only valid (non-null and non-empty)
        annotations will be used. There must be at least 3 columns in this list.
    model_col : str, optional
        The column name for model predictions (default "ModelPrediction").
    epsilon : float, optional
        The cost-benefit parameter for the t-test (default 0.1).
    alpha : float, optional
        Significance level for the FDR correction (default 0.05).
    metric : str, optional
        The metric used for computing alignment scores. Options are "accuracy" (default) or "rmse".
    verbose : bool, optional
        If True, prints a summary of the alt-test results (default is True).
    label_type : str, optional
        The type to convert labels to before comparison. Options are:
        - "int": Convert all labels to integers
        - "str": Convert all labels to strings
        - "auto": Infer the best type (default)

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
            - 'annotator_columns': List of human annotator column names actually used.
            - 'pvals': List of p-values for each annotator.
            - 'rejections': Boolean list indicating which annotators the model outperforms.
            - 'winning_rate': Fraction of annotators for which the null hypothesis is rejected.
            - 'rho_f': List of advantage probabilities for the model versus each annotator.
            - 'rho_h': List of advantage probabilities for each annotator versus the model.
            - 'average_advantage_probability': The average of the model's advantage probabilities.
            - 'passed_alt_test': Boolean indicating if the model passes the alt-test (winning_rate >= 0.5).
            - 'label_counts': Dictionary with counts of valid labels for each rater.
            - 'label_types': Dictionary with types of labels for each rater.
            - 'mixed_types': Boolean indicating if there are mixed types across raters.
            - 'label_type_used': String indicating which label type conversion was used.

    Raises
    ------
    ValueError
        If there are fewer than 3 annotator columns or if the model column is missing.
    """

    if len(annotation_columns) < 3:
        raise ValueError("Need at least 3 annotator columns for the alt-test.")

    if model_col not in df.columns:
        raise ValueError(f"DataFrame does not have the model column '{model_col}'.")

    # This produces a boolean DataFrame where True means "value is not-null and not an empty string."
    valid_mask = df[annotation_columns].notna() & df[annotation_columns].ne("")

    # DEBUGGING: Count valid labels for each rater
    label_counts = {}
    label_types = {}

    # Count valid labels for the model
    model_valid = df[model_col].notna() & (df[model_col] != "")
    label_counts[model_col] = model_valid.sum()

    # Determine the type of model labels
    model_values = df.loc[model_valid, model_col].values
    model_types = set(type(val) for val in model_values if pd.notna(val))
    label_types[model_col] = [t.__name__ for t in model_types]

    # Count valid labels for each annotator
    for col in annotation_columns:
        col_valid = df[col].notna() & (df[col] != "")
        label_counts[col] = col_valid.sum()

        # Determine the type of annotator labels
        col_values = df.loc[col_valid, col].values
        col_types = set(type(val) for val in col_values if pd.notna(val))
        label_types[col] = [t.__name__ for t in col_types]

    # Check if there are mixed types across all raters
    all_types = set()
    for types in label_types.values():
        all_types.update(types)
    mixed_types = len(all_types) > 1

    if verbose:
        print("=== ALT Test: Label Debugging ===")
        print("Label counts for each rater:")
        for rater, count in label_counts.items():
            print(f"  {rater}: {count} valid labels")

        print("\nLabel types for each rater:")
        for rater, types in label_types.items():
            print(f"  {rater}: {', '.join(types)}")

        print(f"\nMixed types across raters: {mixed_types}")
        if mixed_types:
            print(f"  All types found: {', '.join(all_types)}")
        print("=" * 40)

    # Filter rows to only those with at least 3 valid annotations.
    df_valid = df[valid_mask.sum(axis=1) >= 3].copy()
    if df_valid.empty:
        raise ValueError(
            "No rows with at least 3 valid annotations; cannot perform the alt-test."
        )

    # Convert to numpy arrays.
    llm_vals_raw = df_valid[model_col].values
    # For each annotator column, get the array of values (which may contain missing entries).
    ann_arrays_raw = [df_valid[c].values for c in annotation_columns]

    # Convert labels to consistent types
    if verbose:
        print("\n=== Converting labels to consistent types ===")
        print(f"Using label_type: {label_type}")

    # Convert model predictions
    llm_vals_list = llm_vals_raw.tolist()
    llm_vals_converted = convert_labels(llm_vals_list, label_type)
    llm_vals = np.array(llm_vals_converted)

    # Convert annotator values
    ann_arrays = []
    for j, arr in enumerate(ann_arrays_raw):
        arr_list = arr.tolist()
        arr_converted = convert_labels(arr_list, label_type)
        ann_arrays.append(np.array(arr_converted))

    if verbose:
        print(
            f"Model predictions type after conversion: {type(llm_vals[0]) if len(llm_vals) > 0 else 'empty'}"
        )
        for j, col in enumerate(annotation_columns):
            val_type = (
                type(ann_arrays[j][0])
                if len(ann_arrays[j]) > 0 and not pd.isna(ann_arrays[j][0])
                else "empty/NA"
            )
            print(f"{col} type after conversion: {val_type}")

    n = len(df_valid)
    m = len(annotation_columns)

    # Validate the metric and pick the appropriate alignment function.
    if metric == "accuracy":
        S_func: Callable[[Any, List[Any]], float] = accuracy_alignment
    elif metric == "rmse":
        S_func = rmse_alignment
    else:
        raise ValueError("Unsupported metric. Use 'accuracy' or 'rmse'.")

    # Initialize lists to record wins for each annotator.
    W_f: List[List[float]] = [[] for _ in range(m)]  # LLM wins
    W_h: List[List[float]] = [[] for _ in range(m)]  # Human wins

    # Loop over each instance.
    for i in range(n):
        # For each annotator column, only compute if that annotator's value is valid.
        for j in range(m):
            # Check if the current annotator's value is valid; if not, skip this annotator for row i.
            if pd.isnull(ann_arrays[j][i]) or ann_arrays[j][i] == "":
                continue

            # For "others", include only those annotations that are valid.
            others = [
                ann_arrays[k][i]
                for k in range(m)
                if k != j and (pd.notnull(ann_arrays[k][i]) and ann_arrays[k][i] != "")
            ]
            # We require that at least 2 other annotations are present (so that overall at least 3 exist).
            if len(others) < 2:
                continue

            s_llm = S_func(llm_vals[i], others)
            s_hum = S_func(ann_arrays[j][i], others)

            if s_llm > s_hum:
                W_f[j].append(1.0)
                W_h[j].append(0.0)
            elif s_llm < s_hum:
                W_f[j].append(0.0)
                W_h[j].append(1.0)
            else:
                # In case of a tie, count both as wins.
                W_f[j].append(1.0)
                W_h[j].append(1.0)

    # Compute advantage probabilities for each annotator.
    rho_f_vals = [np.mean(wins) if wins else np.nan for wins in W_f]
    rho_h_vals = [np.mean(wins) if wins else np.nan for wins in W_h]

    # For each annotator, perform a one-sided t-test on the difference (W_h - W_f) versus epsilon.
    pvals = []
    for j in range(m):
        # Only perform the t-test if we have any samples for this annotator.
        if len(W_f[j]) == 0:
            pvals.append(np.nan)
            continue
        d_j = np.array(W_h[j]) - np.array(W_f[j])
        # t-test: alternative='less' tests H0: mean(d_j) >= epsilon vs H1: mean(d_j) < epsilon.
        _, p_val = ttest_1samp(d_j, popmean=epsilon, alternative="less")
        pvals.append(p_val)

    # Apply multiple-comparison correction (Benjamini–Yekutieli).
    rejections = benjamini_yekutieli_correction(pvals, alpha=alpha)
    winning_rate = np.nanmean(np.array(rejections))
    avg_adv_prob = np.nanmean(np.array(rho_f_vals))
    passed_alt_test = winning_rate >= 0.5

    if verbose:
        print("=== Alt-Test: summary ===")
        print("P-values for each comparison:")
        for j in range(m):
            print(
                f"{annotation_columns[j]}: p={pvals[j]:.4f} => rejectH0={rejections[j]} | "
                f"rho_f={rho_f_vals[j]:.3f}, rho_h={rho_h_vals[j]:.3f}"
            )
        print("\nSummary statistics:")
        print(f"Winning Rate (omega) = {winning_rate:.3f}")
        print(f"Average Advantage Probability (rho) = {avg_adv_prob:.3f}")
        print(f"Passed Alt-Test? => {passed_alt_test}")

    return {
        "annotator_columns": annotation_columns,
        "pvals": pvals,
        "rejections": rejections,
        "winning_rate": winning_rate,
        "rho_f": rho_f_vals,
        "rho_h": rho_h_vals,
        "average_advantage_probability": avg_adv_prob,
        "passed_alt_test": passed_alt_test,
        "label_counts": label_counts,
        "label_types": label_types,
        "mixed_types": mixed_types,
        "label_type_used": label_type,
    }


def run_alt_test_on_results(
    detailed_results_df: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    epsilon: float = 0.2,
    alpha: float = 0.05,
    verbose: bool = True,
    show_runs: bool = False,
    metric: str = "accuracy",
    label_type: str = "int",
) -> pd.DataFrame:
    """
    Run ALT test on detailed results DataFrame.

    This function computes ALT test metrics for each run separately and then
    aggregates the results across runs by averaging the p-values for each annotator.

    Parameters
    ----------
    detailed_results_df : pd.DataFrame
        DataFrame containing detailed results from run_scenarios.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.
    epsilon : float, optional
        Epsilon parameter for ALT test, by default 0.2
    alpha : float, optional
        Alpha parameter for ALT test, by default 0.05
    verbose : bool, optional
        Whether to print verbose output, by default True

    Returns
    -------
    pd.DataFrame
        A DataFrame with ALT test metrics, including both per-run metrics and
        aggregated metrics across runs.
    """
    # For the final summary dataframe
    all_results = []

    # Print the columns in the detailed_results_df to debug
    print("\n=== Columns in detailed_results_df (in run_alt_test_on_results) ===")
    print(detailed_results_df.columns.tolist())

    # Check if the required columns are present in the detailed_results_df
    required_columns = [
        "prompt_name",
        "iteration",
        "run",
        "ModelPrediction",
    ] + annotation_columns
    missing_columns = [
        col for col in required_columns if col not in detailed_results_df.columns
    ]

    if missing_columns:
        print(
            f"Warning: The following required columns are missing from the detailed results DataFrame: {missing_columns}"
        )
        print("ALT test metrics will not be computed.")
        # Return empty DataFrame
        return pd.DataFrame()

    # Determine grouping based on show_runs parameter
    if show_runs:
        # For show_runs=True, we need to handle both single and multi-iteration scenarios
        # We'll process them separately to ensure proper grouping

        # Check if we have prompt_iteration column for iterative improvement
        has_prompt_iteration = "prompt_iteration" in detailed_results_df.columns

        if has_prompt_iteration:
            # Split data into single-iteration and multi-iteration scenarios
            single_iteration_data = detailed_results_df[
                detailed_results_df["prompt_iteration"].isna()
            ]
            multi_iteration_data = detailed_results_df[
                detailed_results_df["prompt_iteration"].notna()
            ]

            # Process single-iteration scenarios
            single_grouped = (
                single_iteration_data.groupby(["prompt_name", "iteration", "run"])
                if not single_iteration_data.empty
                else []
            )

            # Process multi-iteration scenarios
            multi_grouped = (
                multi_iteration_data.groupby(["prompt_name", "prompt_iteration", "run"])
                if not multi_iteration_data.empty
                else []
            )

            # Combine both groupings
            all_per_run_groups = []

            # Add single-iteration groups
            for group_key, group in single_grouped:
                prompt_name, iteration, run = group_key
                all_per_run_groups.append((group_key, group, "single"))

            # Add multi-iteration groups
            for group_key, group in multi_grouped:
                prompt_name, prompt_iteration, run = group_key
                iteration = (
                    prompt_iteration  # Use prompt_iteration as iteration for display
                )
                all_per_run_groups.append(
                    ((prompt_name, iteration, run), group, "multi")
                )
        else:
            # No prompt_iteration column, use standard grouping
            per_run_grouped = detailed_results_df.groupby(
                ["prompt_name", "iteration", "run"]
            )
            all_per_run_groups = [
                (group_key, group, "single") for group_key, group in per_run_grouped
            ]
    else:
        # For aggregated results, we still need to process individual runs first
        per_run_grouped = detailed_results_df.groupby(
            ["prompt_name", "iteration", "run"]
        )
        all_per_run_groups = [
            (group_key, group, "single") for group_key, group in per_run_grouped
        ]

    # Store p-values for each annotator for each run
    p_values_by_run: Dict[
        Tuple[str, int, str], List[Tuple[int, List[float], List[str]]]
    ] = {}

    for group_info in all_per_run_groups:
        group_key, group, group_type = group_info
        prompt_name, iteration, run = group_key

        # Split data into train and validation sets
        train_data = group[group["split"] == "train"]
        val_data = group[group["split"] == "val"]

        # Extract validation setting
        use_validation_set = len(val_data) > 0

        # Initialize metrics for this run
        run_metrics = {
            "prompt_name": prompt_name,
            "iteration": int(iteration),  # Ensure iteration is int
            "run": run,
            "use_validation_set": use_validation_set,
            "N_train": len(train_data),
            "N_val": len(val_data) if use_validation_set else 0,
        }

        # Add prompt iteration info if available
        if "prompt_iteration" in detailed_results_df.columns and show_runs:
            run_metrics["prompt_iteration"] = (
                group["prompt_iteration"].iloc[0] if len(group) > 0 else iteration
            )

        # Run ALT test for train data
        try:
            # Ensure all columns have the correct type for the chosen mapping
            train_data_copy = train_data.copy()
            if label_type in {"int", "float"}:
                train_data_copy["ModelPrediction"] = pd.to_numeric(
                    train_data_copy["ModelPrediction"], errors="coerce"
                )
                for col in annotation_columns:
                    train_data_copy[col] = pd.to_numeric(
                        train_data_copy[col], errors="coerce"
                    )

                if label_type == "int":
                    train_data_copy["ModelPrediction"] = train_data_copy[
                        "ModelPrediction"
                    ].astype("Int64")
                    for col in annotation_columns:
                        train_data_copy[col] = train_data_copy[col].astype("Int64")
            else:
                train_data_copy["ModelPrediction"] = train_data_copy[
                    "ModelPrediction"
                ].astype("string")
                for col in annotation_columns:
                    train_data_copy[col] = train_data_copy[col].astype("string")

            # Run the ALT test
            alt_test_res_train = run_alt_test_general(
                df=train_data_copy,
                annotation_columns=annotation_columns,
                model_col="ModelPrediction",
                epsilon=epsilon,
                alpha=alpha,
                metric=metric,
                verbose=verbose,
                label_type=label_type,
            )

            # Add ALT test metrics to run_metrics
            run_metrics["winning_rate_train"] = alt_test_res_train["winning_rate"]
            run_metrics["passed_alt_test_train"] = alt_test_res_train["passed_alt_test"]
            run_metrics["avg_adv_prob_train"] = alt_test_res_train[
                "average_advantage_probability"
            ]
            run_metrics["p_values_train"] = alt_test_res_train["pvals"]

            # Store p-values for aggregation
            key = (prompt_name, iteration, "train")
            if key not in p_values_by_run:
                p_values_by_run[key] = []
            p_values_by_run[key].append(
                (
                    run,
                    alt_test_res_train["pvals"],
                    alt_test_res_train["annotator_columns"],
                )
            )

        except Exception as e:
            print(f"Error running ALT test on train data for run {run}: {e}")

        # Run ALT test for validation data if available
        if use_validation_set:
            try:
                # Ensure all columns have the correct type for the chosen mapping
                val_data_copy = val_data.copy()
                if label_type in {"int", "float"}:
                    val_data_copy["ModelPrediction"] = pd.to_numeric(
                        val_data_copy["ModelPrediction"], errors="coerce"
                    )
                    for col in annotation_columns:
                        val_data_copy[col] = pd.to_numeric(
                            val_data_copy[col], errors="coerce"
                        )

                    if label_type == "int":
                        val_data_copy["ModelPrediction"] = val_data_copy[
                            "ModelPrediction"
                        ].astype("Int64")
                        for col in annotation_columns:
                            val_data_copy[col] = val_data_copy[col].astype("Int64")
                else:
                    val_data_copy["ModelPrediction"] = val_data_copy[
                        "ModelPrediction"
                    ].astype("string")
                    for col in annotation_columns:
                        val_data_copy[col] = val_data_copy[col].astype("string")

                # Run the ALT test
                alt_test_res_val = run_alt_test_general(
                    df=val_data_copy,
                    annotation_columns=annotation_columns,
                    model_col="ModelPrediction",
                    epsilon=epsilon,
                    alpha=alpha,
                    metric=metric,
                    verbose=verbose,
                    label_type=label_type,
                )

                # Add ALT test metrics to run_metrics
                run_metrics["winning_rate_val"] = alt_test_res_val["winning_rate"]
                run_metrics["passed_alt_test_val"] = alt_test_res_val["passed_alt_test"]
                run_metrics["avg_adv_prob_val"] = alt_test_res_val[
                    "average_advantage_probability"
                ]
                run_metrics["p_values_val"] = alt_test_res_val["pvals"]

                # Store p-values for aggregation
                key = (prompt_name, iteration, "val")
                if key not in p_values_by_run:
                    p_values_by_run[key] = []
                p_values_by_run[key].append(
                    (
                        run,
                        alt_test_res_val["pvals"],
                        alt_test_res_val["annotator_columns"],
                    )
                )

            except Exception as e:
                print(f"Error running ALT test on validation data for run {run}: {e}")

        # Add to the list of results
        all_results.append(run_metrics)

    # Now, compute aggregated metrics across runs
    # Group by scenario, prompt_name, and iteration
    scenario_grouped = detailed_results_df.groupby(["prompt_name", "iteration"])

    for (prompt_name, iteration), group in scenario_grouped:
        # Split data into train and validation sets
        train_data = group[group["split"] == "train"]
        val_data = group[group["split"] == "val"]

        # Extract validation setting
        use_validation_set = len(val_data) > 0
        n_runs = len(set(group["run"]))

        # Initialize aggregated metrics
        aggregated_metrics = {
            "prompt_name": prompt_name,
            "iteration": iteration,
            "run": "aggregated",  # Mark as aggregated
            "n_runs": n_runs,
            "use_validation_set": use_validation_set,
            "N_train": len(train_data),
            "N_val": len(val_data) if use_validation_set else 0,
        }

        # Aggregate train metrics across runs
        train_key = (prompt_name, iteration, "train")
        if train_key in p_values_by_run and len(p_values_by_run[train_key]) > 0:
            # Get p-values and annotator columns for each run
            train_runs = p_values_by_run[train_key]

            if len(train_runs) == 1:
                # Only one run, use the original metrics
                run, p_values, annotator_cols = train_runs[0]

                # Find the corresponding run metrics
                for run_metrics in all_results:
                    if (
                        run_metrics["prompt_name"] == prompt_name
                        and run_metrics["iteration"] == iteration
                        and run_metrics["run"] == run
                    ):

                        # Copy metrics from the single run
                        aggregated_metrics["winning_rate_train"] = run_metrics.get(
                            "winning_rate_train"
                        )
                        aggregated_metrics["passed_alt_test_train"] = run_metrics.get(
                            "passed_alt_test_train"
                        )
                        aggregated_metrics["avg_adv_prob_train"] = run_metrics.get(
                            "avg_adv_prob_train"
                        )
                        aggregated_metrics["p_values_train"] = run_metrics.get(
                            "p_values_train"
                        )
                        break
            else:
                # Multiple runs, average p-values and recompute metrics
                # First, ensure all runs have the same annotators in the same order
                all_annotators = train_runs[0][2]  # Annotator columns from first run

                # Initialize arrays to store p-values for each annotator across runs
                avg_p_values_train: List[float] = []
                for i in range(len(all_annotators)):
                    p_values_for_annotator = []
                    for run, p_values, _ in train_runs:
                        if i < len(p_values) and not np.isnan(p_values[i]):
                            p_values_for_annotator.append(p_values[i])

                    # Compute average p-value for this annotator
                    if p_values_for_annotator:
                        avg_p_values_train.append(
                            float(np.mean(p_values_for_annotator))
                        )
                    else:
                        avg_p_values_train.append(float(np.nan))

                # Apply Benjamini-Yekutieli correction to the averaged p-values
                valid_p_values = [p for p in avg_p_values_train if not np.isnan(p)]
                if valid_p_values:
                    rejections = benjamini_yekutieli_correction(
                        valid_p_values, alpha=alpha
                    )
                    winning_rate = np.mean(rejections)

                    # Compute average advantage probability
                    avg_adv_probs = []
                    for run_metrics in all_results:
                        if (
                            run_metrics["prompt_name"] == prompt_name
                            and run_metrics["iteration"] == iteration
                            and "avg_adv_prob_train" in run_metrics
                        ):
                            avg_adv_probs.append(run_metrics["avg_adv_prob_train"])

                    avg_adv_prob = np.mean(avg_adv_probs) if avg_adv_probs else np.nan

                    # Add aggregated metrics
                    aggregated_metrics["winning_rate_train"] = winning_rate
                    aggregated_metrics["passed_alt_test_train"] = winning_rate >= 0.5
                    aggregated_metrics["avg_adv_prob_train"] = avg_adv_prob
                    aggregated_metrics["p_values_train"] = avg_p_values_train

        # Aggregate validation metrics across runs
        val_key = (prompt_name, iteration, "val")
        if (
            use_validation_set
            and val_key in p_values_by_run
            and len(p_values_by_run[val_key]) > 0
        ):
            # Get p-values and annotator columns for each run
            val_runs = p_values_by_run[val_key]

            if len(val_runs) == 1:
                # Only one run, use the original metrics
                run, p_values, annotator_cols = val_runs[0]

                # Find the corresponding run metrics
                for run_metrics in all_results:
                    if (
                        run_metrics["prompt_name"] == prompt_name
                        and run_metrics["iteration"] == iteration
                        and run_metrics["run"] == run
                    ):

                        # Copy metrics from the single run
                        aggregated_metrics["winning_rate_val"] = run_metrics.get(
                            "winning_rate_val"
                        )
                        aggregated_metrics["passed_alt_test_val"] = run_metrics.get(
                            "passed_alt_test_val"
                        )
                        aggregated_metrics["avg_adv_prob_val"] = run_metrics.get(
                            "avg_adv_prob_val"
                        )
                        aggregated_metrics["p_values_val"] = run_metrics.get(
                            "p_values_val"
                        )
                        break
            else:
                # Multiple runs, average p-values and recompute metrics
                # First, ensure all runs have the same annotators in the same order
                all_annotators = val_runs[0][2]  # Annotator columns from first run

                # Initialize arrays to store p-values for each annotator across runs
                avg_p_values: List[float] = []
                for i in range(len(all_annotators)):
                    p_values_for_annotator = []
                    for run, p_values, _ in val_runs:
                        if i < len(p_values) and not np.isnan(p_values[i]):
                            p_values_for_annotator.append(p_values[i])

                    # Compute average p-value for this annotator
                    if p_values_for_annotator:
                        avg_p_values.append(float(np.mean(p_values_for_annotator)))
                    else:
                        avg_p_values.append(float(np.nan))

                # Apply Benjamini-Yekutieli correction to the averaged p-values
                valid_p_values = [p for p in avg_p_values if not np.isnan(p)]
                if valid_p_values:
                    rejections = benjamini_yekutieli_correction(
                        valid_p_values, alpha=alpha
                    )
                    winning_rate = np.mean(rejections)

                    # Compute average advantage probability
                    avg_adv_probs = []
                    for run_metrics in all_results:
                        if (
                            run_metrics["prompt_name"] == prompt_name
                            and run_metrics["iteration"] == iteration
                            and "avg_adv_prob_val" in run_metrics
                        ):
                            avg_adv_probs.append(run_metrics["avg_adv_prob_val"])

                    avg_adv_prob = np.mean(avg_adv_probs) if avg_adv_probs else np.nan

                    # Add aggregated metrics
                    aggregated_metrics["winning_rate_val"] = winning_rate
                    aggregated_metrics["passed_alt_test_val"] = winning_rate >= 0.5
                    aggregated_metrics["avg_adv_prob_val"] = avg_adv_prob
                    aggregated_metrics["p_values_val"] = avg_p_values

        # Always add aggregated metrics to the results
        all_results.append(aggregated_metrics)

    # Create DataFrame from the results
    summary_df = pd.DataFrame(all_results)

    return summary_df
