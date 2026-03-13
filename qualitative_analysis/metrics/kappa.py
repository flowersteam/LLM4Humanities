"""
kappa.py

This module provides functions for computing Cohen's Kappa scores and related metrics
to assess inter-rater reliability between model predictions and human annotations.

Functions:
    - compute_cohens_kappa(judgments_1, judgments_2, labels=None, weights=None): 
      Calculates Cohen's Kappa score between two sets of categorical labels.

    - compute_all_kappas(model_coding, human_annotations, labels=None, weights=None, verbose=False): 
      Computes Cohen's Kappa scores between model predictions and multiple human annotators, 
      as well as between human annotators themselves.

    - compute_detailed_kappa_metrics(model_predictions, human_annotations, labels=None, kappa_weights=None):
      Compute detailed kappa metrics between model and human annotators, and between human annotators.

    - compute_kappa_metrics(detailed_results_df, annotation_columns, labels, kappa_weights=None):
      Compute kappa metrics from detailed results DataFrame.
"""

from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Mapping, Any, Tuple

# Import utility functions
from qualitative_analysis.metrics.utils import (
    compute_majority_vote,
    ensure_numeric_columns,
)


def compute_cohens_kappa(
    judgments_1: Union[List[int], List[str]],
    judgments_2: Union[List[int], List[str]],
    labels: Optional[List[Union[int, str]]] = None,
    weights: Optional[str] = None,
) -> float:
    """
    Computes Cohen's Kappa score between two sets of categorical judgments.

    Parameters:
    ----------
    judgments_1 : List[int] or List[str]
        Labels assigned by the first rater.
    judgments_2 : List[int] or List[str]
        Labels assigned by the second rater.
    labels : List[int] or List[str], optional
        List of unique labels to index the confusion matrix. If not provided,
        the union of labels from both raters is used.
    weights : str, optional
        Weighting scheme for computing Kappa. Options are:
            - `'linear'`: Penalizes disagreements linearly.
            - `'quadratic'`: Penalizes disagreements quadratically.
            - `None`: No weighting (default).

    Returns:
    -------
    float
        Cohen's Kappa score, ranging from:
            - `1.0`: Perfect agreement.
            - `0.0`: Agreement equal to chance.
            - `-1.0`: Complete disagreement.

    Raises:
    ------
    ValueError
        If input lists have different lengths or contain invalid labels.

    Examples:
    --------
    Basic usage without specifying labels or weights:

    >>> judgments_1 = [0, 1, 2, 1, 0]
    >>> judgments_2 = [0, 2, 2, 1, 0]
    >>> round(compute_cohens_kappa(judgments_1, judgments_2), 2)
    0.71

    Using specified labels and linear weights:

    >>> labels = [0, 1, 2]
    >>> round(compute_cohens_kappa(judgments_1, judgments_2, labels=labels, weights='linear'), 2)
    0.78

    Handling perfect agreement:

    >>> judgments_3 = [0, 1, 2, 1, 0]
    >>> round(compute_cohens_kappa(judgments_1, judgments_3), 2)
    1.0

    Example with no agreement beyond chance:

    >>> judgments_4 = [2, 0, 1, 2, 1]
    >>> round(compute_cohens_kappa(judgments_1, judgments_4), 2)
    -0.47

    References:
    ----------
    - Cohen, J. (1960). A coefficient of agreement for nominal scales.
      *Educational and Psychological Measurement*, 20(1), 37–46.
    """
    return cohen_kappa_score(judgments_1, judgments_2, labels=labels, weights=weights)


def compute_all_kappas(
    model_coding: Union[List[int], List[str]],
    human_annotations: Mapping[str, Union[List[int], List[str]]],
    labels: Optional[List[Union[int, str]]] = None,
    weights: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Computes Cohen's Kappa scores for all combinations of model predictions
    and human annotations, as well as between human annotations themselves.

    Parameters:
    ----------
    model_coding : List[int] or List[str]
        Model predictions for each sample.

    human_annotations : Dict[str, List[int] or List[str]]
        Dictionary where keys are rater names and values are lists of annotations.

    labels : List[int] or List[str], optional
        List of unique labels to index the confusion matrix.
        If not provided, the union of all labels is used.

    weights : str, optional
        Weighting scheme for computing Kappa. Options are:
            - `'linear'`: Linear penalty for disagreements.
            - `'quadratic'`: Quadratic penalty for disagreements.
            - `None`: No weighting (default).

    verbose : bool, optional
        If `True`, prints the Kappa scores during computation. Default is `False`.

    Returns:
    -------
    Dict[str, float]
        A dictionary containing Cohen's Kappa scores for all comparisons:
            - `"model_vs_<rater>"`: Kappa between the model and each human rater.
            - `"<rater1>_vs_<rater2>"`: Kappa between each pair of human raters.

    Raises:
    ------
    ValueError
        If the annotations have inconsistent lengths.

    Examples:
    -------
    >>> model_coding = [0, 1, 2, 1, 0]
    >>> human_annotations = {
    ...     "Rater1": [0, 1, 2, 0, 0],
    ...     "Rater2": [0, 2, 2, 1, 0]
    ... }
    >>> labels = [0, 1, 2]
    >>> result = compute_all_kappas(model_coding, human_annotations, labels=labels)
    >>> sorted(result.keys())
    ['Rater1_vs_Rater2', 'model_vs_Rater1', 'model_vs_Rater2']

    >>> invalid_annotations = {
    ...     "Rater1": [0, 1],  # Mismatched length
    ...     "Rater2": [0, 2, 2, 1, 0]
    ... }
    >>> compute_all_kappas(model_coding, invalid_annotations, labels=labels)
    Traceback (most recent call last):
        ...
    ValueError: Length mismatch: model_coding and Rater1's annotations must have the same length.
    """
    results = {}

    # Compare model with each human rater
    for rater, annotations in human_annotations.items():
        if len(model_coding) != len(annotations):
            raise ValueError(
                f"Length mismatch: model_coding and {rater}'s annotations must have the same length."
            )

        kappa = compute_cohens_kappa(
            model_coding, annotations, labels=labels, weights=weights
        )
        results[f"model_vs_{rater}"] = kappa
        if verbose:
            print(f"model vs {rater}: {kappa:.2f}")

    # Compare each human rater with every other human rater
    raters = list(human_annotations.keys())
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            rater1, rater2 = raters[i], raters[j]

            if len(human_annotations[rater1]) != len(human_annotations[rater2]):
                raise ValueError(
                    f"Length mismatch: {rater1} and {rater2}'s annotations must have the same length."
                )

            kappa = compute_cohens_kappa(
                human_annotations[rater1],
                human_annotations[rater2],
                labels=labels,
                weights=weights,
            )
            results[f"{rater1}_vs_{rater2}"] = kappa
            if verbose:
                print(f"{rater1} vs {rater2}: {kappa:.2f}")

    return results


def compute_detailed_kappa_metrics(
    model_predictions: List[Any],
    human_annotations: Dict[str, List[Any]],
    labels: Optional[List[Any]] = None,
    kappa_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute detailed kappa metrics between model and human annotators, and between human annotators.

    Parameters
    ----------
    model_predictions : List[Any]
        List of model predictions.
    human_annotations : Dict[str, List[Any]]
        Dictionary mapping annotator names to lists of annotations.
    labels : Optional[List[Any]], optional
        List of valid labels, by default None
    kappa_weights : Optional[str], optional
        Weighting scheme for kappa calculation ('linear', 'quadratic', or None), by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing detailed kappa metrics:
        - 'mean_llm_human_agreement': Mean kappa between LLM and human annotators
        - 'mean_human_human_agreement': Mean kappa between human annotators
        - 'llm_vs_human_df': DataFrame with kappa between LLM and each human annotator
        - 'human_vs_human_df': DataFrame with kappa between each pair of human annotators
    """
    # Compute kappa between model and each human annotator
    llm_human_kappas = []
    llm_human_data = []

    for annotator, annotations in human_annotations.items():
        kappa = compute_cohens_kappa(
            model_predictions, annotations, labels=labels, weights=kappa_weights
        )
        llm_human_kappas.append(kappa)
        llm_human_data.append({"Human_Annotator": annotator, "Cohens_Kappa": kappa})

    # Compute mean LLM-Human agreement
    mean_llm_human_agreement = np.mean(llm_human_kappas) if llm_human_kappas else None

    # Create DataFrame for LLM vs Human kappas
    llm_human_df = pd.DataFrame(llm_human_data)

    # Compute kappa between each pair of human annotators
    human_human_kappas = []
    human_human_data = []

    annotators = list(human_annotations.keys())
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            annotator1 = annotators[i]
            annotator2 = annotators[j]

            kappa = compute_cohens_kappa(
                human_annotations[annotator1],
                human_annotations[annotator2],
                labels=labels,
                weights=kappa_weights,
            )
            human_human_kappas.append(kappa)
            human_human_data.append(
                {
                    "Annotator_1": annotator1,
                    "Annotator_2": annotator2,
                    "Cohens_Kappa": kappa,
                }
            )

    # Compute mean Human-Human agreement
    mean_human_human_agreement = (
        np.mean(human_human_kappas) if human_human_kappas else None
    )

    # Create DataFrame for Human vs Human kappas
    human_human_df = pd.DataFrame(human_human_data)

    return {
        "mean_llm_human_agreement": mean_llm_human_agreement,
        "mean_human_human_agreement": mean_human_human_agreement,
        "llm_vs_human_df": llm_human_df,
        "human_vs_human_df": human_human_df,
    }


def compute_kappa_metrics(
    detailed_results_df: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    kappa_weights: Optional[str] = None,
    show_runs: bool = False,
    coerce_numeric_columns: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Compute kappa metrics from detailed results DataFrame.

    Parameters
    ----------
    detailed_results_df : pd.DataFrame
        DataFrame containing detailed results from run_scenarios.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.
    kappa_weights : Optional[str], optional
        Weighting scheme for kappa calculation ('linear', 'quadratic', or None), by default None

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        A tuple containing:
        - A DataFrame with aggregated kappa metrics including:
          * accuracy_GT_train/val: Model accuracy against ground truth (majority vote)
          * kappa_GT_train/val: Model kappa against ground truth (majority vote)
          * mean_kappa_llm_human: Mean kappa between LLM and individual human annotators
          * mean_human_human_agreement: Mean kappa between human annotators
          * {annotator}_accuracy_GT: Individual human accuracy against ground truth
          * {annotator}_kappa_GT: Individual human kappa against ground truth
          * For validation data: same metrics with _val suffix
        - A dictionary containing detailed kappa metrics for each scenario
    """
    # For the final summary dataframe
    all_aggregated_results = []

    # For storing detailed kappa metrics
    detailed_kappa_metrics = {}

    # Print the columns in the detailed_results_df to debug
    print("\n=== Columns in detailed_results_df (in compute_kappa_metrics) ===")
    print(detailed_results_df.columns.tolist())

    # Check if the required columns are present in the detailed_results_df
    required_columns = [
        "prompt_name",
        "iteration",
        "ModelPrediction",
    ] + annotation_columns
    missing_columns = [
        col for col in required_columns if col not in detailed_results_df.columns
    ]

    if missing_columns:
        print(
            f"Warning: The following required columns are missing from the detailed results DataFrame: {missing_columns}"
        )
        print("Kappa metrics will not be computed.")
        # Return empty DataFrames
        return pd.DataFrame(), {}

    # Keep legacy coercion for existing notebook workflows, but allow callers that
    # already prepared the data to preserve text or float labels.
    if coerce_numeric_columns:
        detailed_results_df = ensure_numeric_columns(
            detailed_results_df, ["ModelPrediction"] + annotation_columns
        )

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
            all_groups = []

            # Add single-iteration groups
            for group_key, group in single_grouped:
                prompt_name, iteration, run = group_key
                all_groups.append((group_key, group, "single"))

            # Add multi-iteration groups
            for group_key, group in multi_grouped:
                prompt_name, prompt_iteration, run = group_key
                iteration = (
                    prompt_iteration  # Use prompt_iteration as iteration for display
                )
                all_groups.append(((prompt_name, iteration, run), group, "multi"))
        else:
            # No prompt_iteration column, use standard grouping
            grouped = detailed_results_df.groupby(["prompt_name", "iteration", "run"])
            all_groups = [(group_key, group, "single") for group_key, group in grouped]
    else:
        # Group by scenario, prompt_name, and iteration (aggregated)
        grouped = detailed_results_df.groupby(["prompt_name", "iteration"])
        all_groups = [(group_key, group, "aggregated") for group_key, group in grouped]

    for group_info in all_groups:
        if show_runs:
            group_key, group, group_type = group_info
            prompt_name, iteration, run = group_key
        else:
            group_key, group, group_type = group_info
            prompt_name, iteration = group_key
            run = "aggregated"  # Mark as aggregated
        # Split data into train and validation sets
        train_data = group[group["split"] == "train"]
        val_data = group[group["split"] == "val"]

        # Extract validation setting
        use_validation_set = len(val_data) > 0

        # Extract model predictions and human annotations for train data
        train_model_predictions = train_data["ModelPrediction"].tolist()
        train_human_annotations = {
            col: train_data[col].tolist() for col in annotation_columns
        }

        # Extract model predictions and human annotations for validation data if available
        val_model_predictions = (
            val_data["ModelPrediction"].tolist() if use_validation_set else []
        )
        val_human_annotations = (
            {col: val_data[col].tolist() for col in annotation_columns}
            if use_validation_set
            else {}
        )

        # Initialize aggregated metrics
        aggregated_metrics = {
            "prompt_name": prompt_name,
            "iteration": iteration,
            "run": run,  # Include run information
            "use_validation_set": use_validation_set,
            "N_train": len(train_data),
            "N_val": len(val_data) if use_validation_set else 0,
        }

        # Add n_runs only for aggregated results
        if run == "aggregated":
            aggregated_metrics["n_runs"] = len(set(group["run"]))

        # Add prompt iteration info if available
        if "prompt_iteration" in detailed_results_df.columns and show_runs:
            aggregated_metrics["prompt_iteration"] = (
                group["prompt_iteration"].iloc[0] if len(group) > 0 else iteration
            )

        # Compute metrics for train data
        if train_model_predictions and all(
            len(annotations) > 0 for annotations in train_human_annotations.values()
        ):
            try:
                # Compute accuracy and kappa for train data
                train_ground_truth = compute_majority_vote(train_human_annotations)
                accuracy_train = accuracy_score(
                    train_ground_truth, train_model_predictions
                )
                kappa_train = compute_cohens_kappa(
                    train_ground_truth,
                    train_model_predictions,
                    labels=labels,
                    weights=kappa_weights,
                )

                # Add basic metrics to aggregated_metrics with clearer naming
                aggregated_metrics["accuracy_GT_train"] = accuracy_train
                aggregated_metrics["kappa_GT_train"] = kappa_train

                # Compute detailed kappa metrics for train data
                train_kappa_metrics = compute_detailed_kappa_metrics(
                    model_predictions=train_model_predictions,
                    human_annotations=train_human_annotations,
                    labels=labels,
                    kappa_weights=kappa_weights,
                )

                # Add mean agreement scores to aggregated metrics with clearer naming
                aggregated_metrics["mean_kappa_llm_human"] = train_kappa_metrics[
                    "mean_llm_human_agreement"
                ]
                aggregated_metrics["mean_human_human_agreement"] = train_kappa_metrics[
                    "mean_human_human_agreement"
                ]

                # Compute individual human annotator metrics against ground truth (majority vote)
                for annotator_name, annotations in train_human_annotations.items():

                    # Compute kappa of this human annotator vs ground truth
                    human_kappa_gt = compute_cohens_kappa(
                        train_ground_truth,
                        annotations,
                        labels=labels,
                        weights=kappa_weights,
                    )
                    aggregated_metrics[f"{annotator_name}_kappa_GT"] = human_kappa_gt

                # Store DataFrames for detailed reporting
                scenario_key = f"{prompt_name}_iteration_{iteration}"
                detailed_kappa_metrics[scenario_key] = {
                    "llm_vs_human_df": train_kappa_metrics["llm_vs_human_df"],
                    "human_vs_human_df": train_kappa_metrics["human_vs_human_df"],
                }

            except Exception as e:
                print(f"Error computing train kappa metrics: {e}")

        # Compute metrics for validation data if available
        if (
            use_validation_set
            and val_model_predictions
            and all(
                len(annotations) > 0 for annotations in val_human_annotations.values()
            )
        ):
            try:
                # Compute accuracy and kappa for validation data
                val_ground_truth = compute_majority_vote(val_human_annotations)
                accuracy_val = accuracy_score(val_ground_truth, val_model_predictions)
                kappa_val = compute_cohens_kappa(
                    val_ground_truth,
                    val_model_predictions,
                    labels=labels,
                    weights=kappa_weights,
                )

                # Add basic metrics to aggregated_metrics with clearer naming
                aggregated_metrics["accuracy_GT_val"] = accuracy_val
                aggregated_metrics["kappa_GT_val"] = kappa_val

                # Compute detailed kappa metrics for validation data
                val_kappa_metrics = compute_detailed_kappa_metrics(
                    model_predictions=val_model_predictions,
                    human_annotations=val_human_annotations,
                    labels=labels,
                    kappa_weights=kappa_weights,
                )

                # Add mean agreement scores to aggregated metrics with clearer naming
                aggregated_metrics["mean_kappa_llm_human_val"] = val_kappa_metrics[
                    "mean_llm_human_agreement"
                ]
                aggregated_metrics["mean_human_human_agreement_val"] = (
                    val_kappa_metrics["mean_human_human_agreement"]
                )

                # Compute individual human annotator metrics against ground truth (majority vote) for validation
                for annotator_name, annotations in val_human_annotations.items():

                    # Compute kappa of this human annotator vs ground truth
                    human_kappa_gt = compute_cohens_kappa(
                        val_ground_truth,
                        annotations,
                        labels=labels,
                        weights=kappa_weights,
                    )
                    aggregated_metrics[f"{annotator_name}_kappa_GT_val"] = (
                        human_kappa_gt
                    )

            except Exception as e:
                print(f"Error computing validation kappa metrics: {e}")

        # Add to the list of aggregated results
        all_aggregated_results.append(aggregated_metrics)

    # If show_runs=True, we also need to create aggregated results for each scenario
    if show_runs:
        # Group by scenario to create aggregated results
        scenario_grouped = detailed_results_df.groupby(["prompt_name", "iteration"])

        for (prompt_name, iteration), scenario_group in scenario_grouped:
            # Split data into train and validation sets
            train_data = scenario_group[scenario_group["split"] == "train"]
            val_data = scenario_group[scenario_group["split"] == "val"]

            # Extract validation setting
            use_validation_set = len(val_data) > 0
            n_runs = len(set(scenario_group["run"]))

            # Extract model predictions and human annotations for train data
            train_model_predictions = train_data["ModelPrediction"].tolist()
            train_human_annotations = {
                col: train_data[col].tolist() for col in annotation_columns
            }

            # Extract model predictions and human annotations for validation data if available
            val_model_predictions = (
                val_data["ModelPrediction"].tolist() if use_validation_set else []
            )
            val_human_annotations = (
                {col: val_data[col].tolist() for col in annotation_columns}
                if use_validation_set
                else {}
            )

            # Initialize aggregated metrics
            aggregated_metrics = {
                "prompt_name": prompt_name,
                "iteration": int(iteration),  # Ensure iteration is int
                "run": "aggregated",  # Mark as aggregated
                "n_runs": n_runs,
                "use_validation_set": use_validation_set,
                "N_train": len(train_data),
                "N_val": len(val_data) if use_validation_set else 0,
            }

            # Compute metrics for train data
            if train_model_predictions and all(
                len(annotations) > 0 for annotations in train_human_annotations.values()
            ):
                try:
                    # Compute accuracy and kappa for train data
                    train_ground_truth = compute_majority_vote(train_human_annotations)
                    accuracy_train = accuracy_score(
                        train_ground_truth, train_model_predictions
                    )
                    kappa_train = compute_cohens_kappa(
                        train_ground_truth,
                        train_model_predictions,
                        labels=labels,
                        weights=kappa_weights,
                    )

                    # Add basic metrics to aggregated_metrics with clearer naming
                    aggregated_metrics["accuracy_GT_train"] = accuracy_train
                    aggregated_metrics["kappa_GT_train"] = kappa_train

                    # Compute detailed kappa metrics for train data
                    train_kappa_metrics = compute_detailed_kappa_metrics(
                        model_predictions=train_model_predictions,
                        human_annotations=train_human_annotations,
                        labels=labels,
                        kappa_weights=kappa_weights,
                    )

                    # Add mean agreement scores to aggregated metrics with clearer naming
                    aggregated_metrics["mean_kappa_llm_human"] = train_kappa_metrics[
                        "mean_llm_human_agreement"
                    ]
                    aggregated_metrics["mean_human_human_agreement"] = (
                        train_kappa_metrics["mean_human_human_agreement"]
                    )

                    # Compute individual human annotator metrics against ground truth (majority vote)
                    for annotator_name, annotations in train_human_annotations.items():
                        # Compute kappa of this human annotator vs ground truth
                        human_kappa_gt = compute_cohens_kappa(
                            train_ground_truth,
                            annotations,
                            labels=labels,
                            weights=kappa_weights,
                        )
                        aggregated_metrics[f"{annotator_name}_kappa_GT"] = (
                            human_kappa_gt
                        )

                except Exception as e:
                    print(f"Error computing aggregated train kappa metrics: {e}")

            # Compute metrics for validation data if available
            if (
                use_validation_set
                and val_model_predictions
                and all(
                    len(annotations) > 0
                    for annotations in val_human_annotations.values()
                )
            ):
                try:
                    # Compute accuracy and kappa for validation data
                    val_ground_truth = compute_majority_vote(val_human_annotations)
                    accuracy_val = accuracy_score(
                        val_ground_truth, val_model_predictions
                    )
                    kappa_val = compute_cohens_kappa(
                        val_ground_truth,
                        val_model_predictions,
                        labels=labels,
                        weights=kappa_weights,
                    )

                    # Add basic metrics to aggregated_metrics with clearer naming
                    aggregated_metrics["accuracy_GT_val"] = accuracy_val
                    aggregated_metrics["kappa_GT_val"] = kappa_val

                    # Compute detailed kappa metrics for validation data
                    val_kappa_metrics = compute_detailed_kappa_metrics(
                        model_predictions=val_model_predictions,
                        human_annotations=val_human_annotations,
                        labels=labels,
                        kappa_weights=kappa_weights,
                    )

                    # Add mean agreement scores to aggregated metrics with clearer naming
                    aggregated_metrics["mean_kappa_llm_human_val"] = val_kappa_metrics[
                        "mean_llm_human_agreement"
                    ]
                    aggregated_metrics["mean_human_human_agreement_val"] = (
                        val_kappa_metrics["mean_human_human_agreement"]
                    )

                    # Compute individual human annotator metrics against ground truth (majority vote) for validation
                    for annotator_name, annotations in val_human_annotations.items():
                        # Compute kappa of this human annotator vs ground truth
                        human_kappa_gt = compute_cohens_kappa(
                            val_ground_truth,
                            annotations,
                            labels=labels,
                            weights=kappa_weights,
                        )
                        aggregated_metrics[f"{annotator_name}_kappa_GT_val"] = (
                            human_kappa_gt
                        )

                except Exception as e:
                    print(f"Error computing aggregated validation kappa metrics: {e}")

            # Add aggregated results to the list
            all_aggregated_results.append(aggregated_metrics)

    # Create DataFrame from the results
    summary_df = pd.DataFrame(all_aggregated_results)

    # Ensure iteration column is int
    if "iteration" in summary_df.columns:
        summary_df["iteration"] = summary_df["iteration"].astype(int)

    return summary_df, detailed_kappa_metrics
