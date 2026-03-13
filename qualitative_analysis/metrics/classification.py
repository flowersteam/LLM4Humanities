"""
classification.py

This module provides functions for computing classification metrics.

Classes:
    - ClassMetrics (TypedDict): Type definition for class-level metrics
    - GlobalMetrics (TypedDict): Type definition for global metrics
    - ClassificationResults (TypedDict): Type definition for classification results

Functions:
    - compute_classification_metrics(model_coding, human_annotations, labels=None):
      Computes detailed classification metrics for a model's predictions against human annotations.

    - compute_classification_metrics_from_results(detailed_results_df, annotation_columns, labels):
      Compute classification metrics from detailed results DataFrame.
"""

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any, TypedDict

# Import utility functions
from qualitative_analysis.metrics.utils import (
    compute_majority_vote,
    ensure_numeric_columns,
)


# TypedDict definitions for classification metrics
class ClassMetrics(TypedDict):
    recall: float
    error_rate: float
    correct_count: int  # TP
    missed_count: int  # FN
    false_positives: int  # FP


class GlobalMetrics(TypedDict):
    accuracy: float
    recall: float
    error_rate: float


class ClassificationResults(TypedDict):
    class_distribution: Dict[Union[int, str], int]
    global_metrics: Dict[str, GlobalMetrics]
    per_class_metrics: Dict[Union[int, str], Dict[str, ClassMetrics]]
    ground_truth: List[Any]
    confusion_matrices: Dict[str, np.ndarray]


def compute_classification_metrics(
    model_coding: List[Any],
    human_annotations: Dict[str, List[Any]],
    labels: Optional[List[Any]] = None,
) -> ClassificationResults:
    """
    Computes detailed classification metrics for a model's predictions against human annotations.

    This function calculates a variety of metrics including:
    - Global metrics: accuracy, recall, error rate
    - Per-class metrics: recall, error rate, TP, FN, FP
    - Confusion matrices

    Parameters:
    ----------
    model_coding : List[Any]
        List of model predictions for each sample.

    human_annotations : Dict[str, List[Any]]
        Dictionary where keys are rater names and values are lists of annotations.

    labels : List[Any], optional
        List of unique labels to consider. If not provided, the unique values from
        the ground truth (majority vote of human annotations) will be used.

    Returns:
    -------
    ClassificationResults
        A dictionary containing:
        - 'class_distribution': Distribution of classes in the ground truth.
        - 'global_metrics': Global metrics for the model.
        - 'per_class_metrics': Per-class metrics for the model.
        - 'ground_truth': The ground truth labels (majority vote of human annotations).
        - 'confusion_matrices': Confusion matrices for the model.
    """
    # Compute ground truth as majority vote of human annotations
    ground_truth = compute_majority_vote(human_annotations)

    # If labels not provided, use unique values from ground truth
    if labels is None:
        labels = sorted(set(ground_truth))

    # Compute class distribution
    class_distribution = {}
    for label in labels:
        class_distribution[label] = sum(1 for gt in ground_truth if gt == label)

    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, model_coding, labels=labels)

    # Compute global metrics
    accuracy = accuracy_score(ground_truth, model_coding)
    recall = recall_score(
        ground_truth, model_coding, labels=labels, average="macro", zero_division=0
    )
    error_rate = 1.0 - accuracy

    # Create the global_metrics dictionary with the model metrics
    global_metrics: Dict[str, GlobalMetrics] = {
        "model": {
            "accuracy": accuracy,
            "recall": recall,
            "error_rate": error_rate,
        }
    }

    # Compute per-class metrics
    per_class_metrics: Dict[Union[int, str], Dict[str, ClassMetrics]] = {}
    for i, label in enumerate(labels):
        # True positives: diagonal element for this class
        tp = cm[i, i]
        # False negatives: sum of row minus true positives
        fn = np.sum(cm[i, :]) - tp
        # False positives: sum of column minus true positives
        fp = np.sum(cm[:, i]) - tp

        # Compute recall and error rate for this class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        error_rate = 1.0 - recall

        # Add the model metrics to the per_class_metrics dictionary
        per_class_metrics[label] = {
            "model": {
                "recall": recall,
                "error_rate": error_rate,
                "correct_count": int(tp),
                "missed_count": int(fn),
                "false_positives": int(fp),
            }
        }

    # Return all metrics
    return {
        "class_distribution": class_distribution,
        "global_metrics": global_metrics,
        "per_class_metrics": per_class_metrics,
        "ground_truth": ground_truth,
        "confusion_matrices": {"model": cm},
    }


def compute_classification_metrics_from_results(
    detailed_results_df: pd.DataFrame,
    annotation_columns: List[str],
    labels: List[Any],
    show_runs: bool = False,
    coerce_numeric_columns: bool = True,
) -> pd.DataFrame:
    """
    Compute classification metrics from detailed results DataFrame.

    Parameters
    ----------
    detailed_results_df : pd.DataFrame
        DataFrame containing detailed results from run_scenarios.
    annotation_columns : List[str]
        List of column names containing human annotations.
    labels : List[Any]
        List of valid labels.

    Returns
    -------
    pd.DataFrame
        A DataFrame with classification metrics
    """
    # For the final summary dataframe
    all_aggregated_results = []

    # Print the columns in the detailed_results_df to debug
    print(
        "\n=== Columns in detailed_results_df (in compute_classification_metrics_from_results) ==="
    )
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
        print("Classification metrics will not be computed.")
        # Return empty DataFrame
        return pd.DataFrame()

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
                # Compute classification metrics for train data
                train_classification_results = compute_classification_metrics(
                    model_coding=train_model_predictions,
                    human_annotations=train_human_annotations,
                    labels=labels,
                )

                # Add global metrics to aggregated_metrics
                for metric_name, metric_value in train_classification_results[
                    "global_metrics"
                ]["model"].items():
                    aggregated_metrics[f"global_{metric_name}_train"] = metric_value

                # Add per-class metrics
                for label, class_metrics in train_classification_results[
                    "per_class_metrics"
                ].items():
                    for metric_name, metric_value in class_metrics["model"].items():
                        aggregated_metrics[f"class_{label}_{metric_name}_train"] = (
                            metric_value
                        )

            except Exception as e:
                print(f"Error computing train classification metrics: {e}")

        # Compute metrics for validation data if available
        if (
            use_validation_set
            and val_model_predictions
            and all(
                len(annotations) > 0 for annotations in val_human_annotations.values()
            )
        ):
            try:
                # Compute classification metrics for validation data
                val_classification_results = compute_classification_metrics(
                    model_coding=val_model_predictions,
                    human_annotations=val_human_annotations,
                    labels=labels,
                )

                # Add global metrics to aggregated_metrics
                for metric_name, metric_value in val_classification_results[
                    "global_metrics"
                ]["model"].items():
                    aggregated_metrics[f"global_{metric_name}_val"] = metric_value

                # Add per-class metrics
                for label, class_metrics in val_classification_results[
                    "per_class_metrics"
                ].items():
                    for metric_name, metric_value in class_metrics["model"].items():
                        aggregated_metrics[f"class_{label}_{metric_name}_val"] = (
                            metric_value
                        )

            except Exception as e:
                print(f"Error computing validation classification metrics: {e}")

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
                    # Compute classification metrics for train data
                    train_classification_results = compute_classification_metrics(
                        model_coding=train_model_predictions,
                        human_annotations=train_human_annotations,
                        labels=labels,
                    )

                    # Add global metrics to aggregated_metrics
                    for metric_name, metric_value in train_classification_results[
                        "global_metrics"
                    ]["model"].items():
                        aggregated_metrics[f"global_{metric_name}_train"] = metric_value

                    # Add per-class metrics
                    for label, class_metrics in train_classification_results[
                        "per_class_metrics"
                    ].items():
                        for metric_name, metric_value in class_metrics["model"].items():
                            aggregated_metrics[f"class_{label}_{metric_name}_train"] = (
                                metric_value
                            )

                except Exception as e:
                    print(
                        f"Error computing aggregated train classification metrics: {e}"
                    )

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
                    # Compute classification metrics for validation data
                    val_classification_results = compute_classification_metrics(
                        model_coding=val_model_predictions,
                        human_annotations=val_human_annotations,
                        labels=labels,
                    )

                    # Add global metrics to aggregated_metrics
                    for metric_name, metric_value in val_classification_results[
                        "global_metrics"
                    ]["model"].items():
                        aggregated_metrics[f"global_{metric_name}_val"] = metric_value

                    # Add per-class metrics
                    for label, class_metrics in val_classification_results[
                        "per_class_metrics"
                    ].items():
                        for metric_name, metric_value in class_metrics["model"].items():
                            aggregated_metrics[f"class_{label}_{metric_name}_val"] = (
                                metric_value
                            )

                except Exception as e:
                    print(
                        f"Error computing aggregated validation classification metrics: {e}"
                    )

            # Add aggregated results to the list
            all_aggregated_results.append(aggregated_metrics)

    # Create DataFrame from the results
    summary_df = pd.DataFrame(all_aggregated_results)

    # Ensure iteration column is int
    if "iteration" in summary_df.columns:
        summary_df["iteration"] = summary_df["iteration"].astype(int)

    return summary_df
