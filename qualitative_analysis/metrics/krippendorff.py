"""
krippendorff.py

This module provides functions for computing Krippendorff's alpha and related metrics
for assessing inter-rater reliability and non-inferiority testing.

Functions:
    - compute_krippendorff_non_inferiority(detailed_results_df, annotation_columns, model_column, 
      level_of_measurement, non_inferiority_margin, n_bootstrap, confidence_level, random_seed, verbose):
      Test if model annotations are non-inferior to human annotations using Krippendorff's alpha 
      with configurable confidence intervals.

    - print_non_inferiority_results(non_inferiority_results, show_per_run):
      Print non-inferiority test results in a formatted way.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import krippendorff
import itertools

# Import utility functions
from qualitative_analysis.metrics.utils import ensure_numeric_columns


def compute_krippendorff_non_inferiority(
    detailed_results_df: pd.DataFrame,
    annotation_columns: List[str],
    model_column: str = "ModelPrediction",
    level_of_measurement: str = "ordinal",
    non_inferiority_margin: float = -0.05,
    n_bootstrap: int = 2000,
    confidence_level: float = 90.0,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Test if model annotations are non-inferior to human annotations using Krippendorff's alpha.

    This function implements a flexible non-inferiority test comparing all human annotators to
    groups that include the model. It uses bootstrap resampling to compute confidence intervals
    and determine if the model is non-inferior to human annotators within a specified margin.

    The test compares:
    - Human group: All available human annotators together
    - Model groups: All possible combinations of (n-1) human annotators + model

    Parameters
    ----------
    detailed_results_df : pd.DataFrame
        DataFrame containing detailed results
    annotation_columns : List[str]
        List of column names containing human annotations
    model_column : str, optional
        Column name containing model predictions, by default "ModelPrediction"
    level_of_measurement : str, optional
        Level of measurement for Krippendorff's alpha, by default 'ordinal'
    non_inferiority_margin : float, optional
        Non-inferiority margin (delta), by default -0.05
    n_bootstrap : int, optional
        Number of bootstrap samples, by default 2000
    confidence_level : float, optional
        Confidence level for the confidence interval (e.g., 90.0 for 90% CI), by default 90.0
    random_seed : int, optional
        Random seed for reproducibility, by default 42
    verbose : bool, optional
        Whether to print detailed results, by default True

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing test results for each scenario
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Ensure numeric columns
    columns_to_check = annotation_columns + [model_column]
    detailed_results_df = ensure_numeric_columns(detailed_results_df, columns_to_check)

    # Group by scenario (prompt_name, iteration)
    scenario_grouped = detailed_results_df.groupby(["prompt_name", "iteration"])

    # Store results for each scenario
    scenario_results = {}

    # Helper function to compute alpha for a subset of columns
    def alpha_cols(frame, cols, row_idx=None):
        """
        Compute Krippendorff's alpha for specified columns and rows,
        keeping missing values (NaNs) so that partial annotations are allowed.

        Krippendorff's alpha naturally handles missing data, as long as
        at least two annotators share at least one commonly annotated item.
        """
        sub = frame.iloc[row_idx] if row_idx is not None else frame
        data = sub[cols].to_numpy(dtype=float).T  # keep NaNs for krippendorff.alpha

        # Detect annotators with no data at all
        annotator_mask = ~np.all(np.isnan(data), axis=1)
        if not np.all(annotator_mask):
            empty_annotators = [
                cols[i] for i, ok in enumerate(annotator_mask) if not ok
            ]
            # if verbose:
            #     print(
            #         f"⚠️ Warning: {empty_annotators} have no annotations and were excluded."
            #     )
            data = data[annotator_mask, :]

        # Check for minimum conditions: at least 2 valid annotators and 2 annotated items
        if data.shape[0] < 2:
            if verbose:
                print(
                    "⚠️ Skipping α computation: fewer than 2 annotators with valid data."
                )
            return np.nan

        valid_items_mask = ~np.all(np.isnan(data), axis=0)
        data = data[:, valid_items_mask]

        if data.shape[1] < 2:
            if verbose:
                print(
                    "⚠️ Skipping α computation: fewer than 2 items with overlapping annotations."
                )
            return np.nan

        try:
            return krippendorff.alpha(data, level_of_measurement=level_of_measurement)
        except Exception as e:
            if verbose:
                print(f"⚠️ Error computing α for {cols}: {e}")
            return np.nan

    for (prompt_name, iteration), scenario_group in scenario_grouped:
        scenario_key = f"{prompt_name}_iteration_{iteration}"

        if verbose:
            print(f"\n=== Non-inferiority Test: {scenario_key} ===")

        # Process each run separately
        run_results = []

        for run_id in sorted(scenario_group["run"].unique()):
            run_data = scenario_group[scenario_group["run"] == run_id]

            # Split into train and validation
            train_data = run_data[run_data["split"] == "train"]

            # Skip if no training data
            if len(train_data) == 0:
                continue

            try:
                # Check minimum requirements
                n_human_annotators = len(annotation_columns)
                if n_human_annotators < 3:
                    if verbose:
                        print(
                            f"  Run {run_id}: Skipping - need at least 3 human annotators, got {n_human_annotators}"
                        )
                    continue

                # Define the groups to compare using flexible approach
                # Human group: all human annotators together
                human_group = annotation_columns

                # Model groups: all possible combinations of (n-1) human annotators + model
                model_groups = []
                for human_subset in itertools.combinations(
                    annotation_columns, n_human_annotators - 1
                ):
                    model_group = list(human_subset) + [model_column]
                    model_groups.append(model_group)

                if verbose:
                    print(f"    Using {n_human_annotators} human annotators")
                    print(f"    Human group: all {n_human_annotators} annotators")
                    print(
                        f"    Model groups: {len(model_groups)} combinations of LLM + {n_human_annotators-1} humans"
                    )

                # Compute alpha for human-only group (all human annotators)
                alpha_H = alpha_cols(train_data, human_group)

                # Compute alpha for each model group and take the mean
                alpha_model_groups = [
                    alpha_cols(train_data, group) for group in model_groups
                ]
                alpha_M = np.mean(alpha_model_groups)

                # Observed difference
                observed_difference = alpha_M - alpha_H

                # Bootstrap test
                n_items = len(train_data)
                deltas = []

                for _ in range(n_bootstrap):
                    # Sample with replacement
                    idx = rng.integers(0, n_items, n_items)

                    # Compute alphas for bootstrap samples
                    # Human group: all human annotators
                    alpha_H_b = alpha_cols(train_data, human_group, idx)

                    # Model groups: all combinations of (n-1) humans + model
                    alpha_M_b = np.mean(
                        [alpha_cols(train_data, group, idx) for group in model_groups]
                    )

                    # Store difference
                    deltas.append(alpha_M_b - alpha_H_b)

                # Compute confidence interval based on confidence_level
                alpha_level = (100 - confidence_level) / 2
                ci_low = np.percentile(deltas, alpha_level)
                ci_high = np.percentile(deltas, 100 - alpha_level)

                # Determine if non-inferiority is demonstrated
                non_inferiority = ci_low > non_inferiority_margin

                # Store results for this run
                run_result = {
                    "run": run_id,
                    "alpha_human_groups": alpha_H,
                    "alpha_model_groups": alpha_M,
                    "difference": observed_difference,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "non_inferiority_margin": non_inferiority_margin,
                    "non_inferiority_demonstrated": non_inferiority,
                    "n_bootstrap_samples": n_bootstrap,
                    "human_group": human_group,
                    "model_groups": model_groups,
                    "n_human_annotators": n_human_annotators,
                    "bootstrap_differences": np.array(deltas),
                }

                run_results.append(run_result)

                if verbose:
                    print(f"\n  Run {run_id}:")
                    print(
                        f"    Human group ({n_human_annotators} annotators) α: {alpha_H:.4f}"
                    )
                    print(
                        f"    Model groups (LLM + {n_human_annotators-1} humans) α: {alpha_M:.4f}"
                    )
                    print(f"    Δ = model − human = {observed_difference:+.4f}")
                    print(
                        f"    {confidence_level:.0f}% CI: [{ci_low:.4f}, {ci_high:.4f}]"
                    )
                    if non_inferiority:
                        print(
                            f"    ✅ Non-inferiority demonstrated (margin = {non_inferiority_margin})"
                        )
                    else:
                        print(
                            f"    ❌ Non-inferiority NOT demonstrated (margin = {non_inferiority_margin})"
                        )

            except Exception as e:
                if verbose:
                    print(f"  Run {run_id}: Error - {str(e)}")

        # Aggregate results across runs
        if run_results:
            # Calculate means and standard deviations
            mean_alpha_H = np.mean([r["alpha_human_groups"] for r in run_results])
            std_alpha_H = np.std([r["alpha_human_groups"] for r in run_results])

            mean_alpha_M = np.mean([r["alpha_model_groups"] for r in run_results])
            std_alpha_M = np.std([r["alpha_model_groups"] for r in run_results])

            mean_difference = np.mean([r["difference"] for r in run_results])
            std_difference = np.std([r["difference"] for r in run_results])

            mean_ci_low = np.mean([r["ci_lower"] for r in run_results])
            mean_ci_high = np.mean([r["ci_upper"] for r in run_results])

            # Count non-inferiority demonstrations
            n_non_inferior = sum(
                [r["non_inferiority_demonstrated"] for r in run_results]
            )

            # Store aggregated results
            aggregated_metrics = {
                "n_runs": len(run_results),
                "alpha_human_trios_mean": mean_alpha_H,
                "alpha_human_trios_std": std_alpha_H,
                "alpha_model_trios_mean": mean_alpha_M,
                "alpha_model_trios_std": std_alpha_M,
                "difference_mean": mean_difference,
                "difference_std": std_difference,
                "ci_lower_mean": mean_ci_low,
                "ci_upper_mean": mean_ci_high,
                "confidence_level": confidence_level,
                "non_inferiority_margin": non_inferiority_margin,
                "n_non_inferior": n_non_inferior,
                "non_inferiority_ratio": n_non_inferior / len(run_results),
            }

            # Store all results for this scenario
            scenario_results[scenario_key] = {
                "run_results": run_results,
                "aggregated_metrics": aggregated_metrics,
            }

            # Print summary
            if verbose:
                print(f"\n  Summary across {len(run_results)} runs:")
                print(f"    Human trios α: {mean_alpha_H:.4f} ± {std_alpha_H:.4f}")
                print(f"    Model trios α: {mean_alpha_M:.4f} ± {std_alpha_M:.4f}")
                print(
                    f"    Δ = model − human = {mean_difference:+.4f} ± {std_difference:.4f}"
                )
                print(
                    f"    {confidence_level:.0f}% CI: [{mean_ci_low:.4f}, {mean_ci_high:.4f}]"
                )
                print(
                    f"    Non-inferiority demonstrated in {n_non_inferior}/{len(run_results)} runs"
                )

                if n_non_inferior == len(run_results):
                    print(
                        f"    ✅ Non-inferiority consistently demonstrated across all runs (margin = {non_inferiority_margin})"
                    )
                elif n_non_inferior > 0:
                    print(
                        f"    ⚠️ Non-inferiority demonstrated in some but not all runs (margin = {non_inferiority_margin})"
                    )
                else:
                    print(
                        f"    ❌ Non-inferiority NOT demonstrated in any run (margin = {non_inferiority_margin})"
                    )

    return scenario_results


def print_non_inferiority_results(
    non_inferiority_results: Dict[str, Dict[str, Any]], show_per_run: bool = False
) -> None:
    """
    Print non-inferiority test results in a formatted way.

    Parameters
    ----------
    non_inferiority_results : Dict[str, Dict[str, Any]]
        Results from compute_krippendorff_non_inferiority
    show_per_run : bool, optional
        Whether to show individual run details, by default False
    """
    for scenario_key, results in non_inferiority_results.items():
        print(f"\n=== Non-inferiority Test: {scenario_key} ===")

        # Access the aggregated metrics
        agg = results["aggregated_metrics"]

        print(
            f"Human trios α: {agg['alpha_human_trios_mean']:.4f} ± {agg['alpha_human_trios_std']:.4f}"
        )
        print(
            f"Model trios α: {agg['alpha_model_trios_mean']:.4f} ± {agg['alpha_model_trios_std']:.4f}"
        )
        print(
            f"Δ = model − human = {agg['difference_mean']:+.4f} ± {agg['difference_std']:.4f}"
        )
        print(
            f"{agg['confidence_level']:.0f}% CI: [{agg['ci_lower_mean']:.4f}, {agg['ci_upper_mean']:.4f}]"
        )
        print(
            f"Non-inferiority demonstrated in {agg['n_non_inferior']}/{agg['n_runs']} runs"
        )

        if agg["n_non_inferior"] == agg["n_runs"]:
            print(
                f"✅ Non-inferiority consistently demonstrated across all runs (margin = {agg['non_inferiority_margin']})"
            )
        elif agg["n_non_inferior"] > 0:
            print(
                f"⚠️ Non-inferiority demonstrated in some but not all runs (margin = {agg['non_inferiority_margin']})"
            )
        else:
            print(
                f"❌ Non-inferiority NOT demonstrated in any run (margin = {agg['non_inferiority_margin']})"
            )

        # Show individual run details if requested
        if show_per_run:
            print("\nDetailed per-run results:")
            for run_result in results["run_results"]:
                run_id = run_result["run"]
                confidence_level = agg.get("confidence_level", 90.0)
                print(f"  Run {run_id}:")
                print(f"    Human trios α: {run_result['alpha_human_trios']:.4f}")
                print(f"    Model trios α: {run_result['alpha_model_trios']:.4f}")
                print(f"    Δ = model − human = {run_result['difference']:+.4f}")
                print(
                    f"    {confidence_level:.0f}% CI: [{run_result['ci_lower']:.4f}, {run_result['ci_upper']:.4f}]"
                )
                if run_result["non_inferiority_demonstrated"]:
                    print(
                        f"    ✅ Non-inferiority demonstrated (margin = {run_result['non_inferiority_margin']})"
                    )
                else:
                    print(
                        f"    ❌ Non-inferiority NOT demonstrated (margin = {run_result['non_inferiority_margin']})"
                    )
