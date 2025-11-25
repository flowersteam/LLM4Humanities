"""
Module for handling evaluation functionality in the Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, List, Optional

from qualitative_analysis.metrics import (
    compute_kappa_metrics,
    run_alt_test_on_results,
    compute_classification_metrics_from_results,
)
from qualitative_analysis.metrics.kappa import compute_cohens_kappa


def compare_with_external_judgments(app_instance: Any) -> None:
    """
    Step 7: Compare with External Judgments (Annotation Columns)
    """
    st.markdown("### Step 7: Evaluate Model Performance", unsafe_allow_html=True)
    with st.expander("Show/hide details of step 7", expanded=True):
        if not app_instance.results:
            st.warning("No analysis results. Please run the analysis first.")
            return

        if not app_instance.annotation_columns:
            st.info("No annotation columns were selected in Step 2.")
            return

        results_df: pd.DataFrame = st.session_state["results_df"]

        st.markdown(
            """
        This step provides three options to measure how closely your LLM's outputs align with existing
        manually annotated labels. If the alignment is sufficiently high, you could rely on the
        model-generated labels for annotating the rest of your unannotated data.

        We provide three options for comparing the LLM's outputs with human annotations: Cohen's Kappa,
        Classification Metrics, and the Alternative Annotator Test (Alt-Test).

        **Which Should I Choose?**
        - If you want to measure agreement between annotators:
            Cohen's Kappa is the simpler approach.
        - If you have multiple manual annotations and want a more robust metric,
            you can choose the Krippendorff's Alpha test.
            It will give you an interval confidence interval for the agreement using bootstrapping,
            and a possibility to check the risk that the real value of the Krippendorff's Alpha is outside this interval.
        - If you need detailed per-class performance metrics (recall, true positives, false positives):
            Classification Metrics provides a breakdown of model performance for each class.
        - If you have multiple annotation columns (≥ 3), want to see if the model
            "outperforms" or "can replace" humans, and can afford 50–100 annotated items:
            use the Alt-Test. This is more stringent because it compares against each
            annotator in a leave-one-out manner.

        In all cases, ideally 50+ annotated instances to get a stable estimate

        The ultimate decision of whether a metric is “good enough” depends on your
        research domain and practical considerations like cost, effort, and the
        consequences of annotation mistakes.

        If you are not satisfied with the model’s performance, you can go back to
        Step 3 and adjust the codebook and examples.

        Below, you can select your method, configure any needed parameters, and run
        the computation.
        """
        )

        # Choose the comparison method
        method: str = st.radio(
            "Select Comparison Method:",
            (
                "Cohen's Kappa (Agreement Analysis)",
                "Classification Metrics (Balanced Acc, TP%, FP%)",
                "Alt-Test (Model Viability)",
                "Krippendorff's Alpha (Non-Inferiority Test)",
            ),
            index=0,
        )

        # --------------------------------------------------------------------------------
        # OPTION 1: COHEN'S KAPPA (AGREEMENT ANALYSIS)
        # --------------------------------------------------------------------------------
        if method == "Cohen's Kappa (Agreement Analysis)":
            st.markdown(
                """
                Analyze agreement between LLM and human annotators, as well as agreement among human annotators.
                
                This analysis provides:
                - Mean agreement between LLM and all human annotators (when multiple annotators are available)
                - Mean agreement among human annotators (when multiple annotators are available)
                - Individual agreement scores for all comparisons
                
                **Weighting Options:**
                - **Unweighted**: Treats all disagreements equally (e.g., disagreeing between 0 and 1 is the same as between 0 and 2)
                - **Linear**: Weights disagreements by their distance (e.g., disagreeing between 0 and 2 is twice as bad as between 0 and 1)
                - **Quadratic**: Weights disagreements by the square of their distance (e.g., disagreeing between 0 and 2 is four times as bad as between 0 and 1)
                
                Use weighting when your categories have a meaningful order (e.g., 0, 1, 2) and the magnitude of disagreement matters.
                """
            )

            # Add weighting option
            weights_option = st.radio(
                "Weighting Scheme:",
                ["Unweighted", "Linear", "Quadratic"],
                index=0,
                key="kappa_weights_option",
                help="Select how to weight disagreements between categories. Use weighting when categories have a meaningful order.",
            )

            # Convert the option to the format expected by the function
            weights_map = {
                "Unweighted": None,
                "Linear": "linear",
                "Quadratic": "quadratic",
            }
            weights = weights_map[weights_option]

            # LLM columns presumably the ones in selected_fields
            llm_columns: list[str] = [
                col for col in app_instance.selected_fields if col in results_df.columns
            ]

            if not llm_columns:
                st.warning("No LLM-generated columns found in the results to compare.")
                return

            if len(app_instance.annotation_columns) < 1:
                st.warning(
                    "At least one annotation column is required for agreement analysis."
                )
                return

            # Figure out a default index that points to our label column if possible
            default_index = 0
            if "label_column" in st.session_state:
                label_col = st.session_state["label_column"]
                if label_col in llm_columns:
                    default_index = llm_columns.index(label_col)

            llm_judgment_col: str = st.selectbox(
                "Select LLM Judgment Column:",
                llm_columns,
                index=default_index,
                key="llm_judgment_col_select",
            )

            if st.button("Compute Agreement Scores", key="compute_agreement_button"):
                if llm_judgment_col not in results_df.columns:
                    st.error("The chosen LLM column is not in the results dataframe.")
                    return

                # Check if annotation columns exist in the dataframe
                missing_columns = [
                    col
                    for col in app_instance.annotation_columns
                    if col not in results_df.columns
                ]
                if missing_columns:
                    st.error(
                        f"The following annotation columns are not in the results dataframe: {', '.join(missing_columns)}"
                    )
                    return

                # Prepare data for compute_kappa_metrics
                analysis_data = results_df.copy()

                # Add required columns for compute_kappa_metrics if they don't exist
                if "prompt_name" not in analysis_data.columns:
                    analysis_data["prompt_name"] = "streamlit_analysis"
                if "iteration" not in analysis_data.columns:
                    analysis_data["iteration"] = 1
                if "split" not in analysis_data.columns:
                    analysis_data["split"] = "train"
                if "run" not in analysis_data.columns:
                    analysis_data["run"] = 1

                # Rename the selected LLM column to ModelPrediction for consistency
                if llm_judgment_col != "ModelPrediction":
                    analysis_data = analysis_data.rename(
                        columns={llm_judgment_col: "ModelPrediction"}
                    )

                # Drop rows with missing values in essential columns
                essential_cols = ["ModelPrediction"] + app_instance.annotation_columns
                analysis_data = analysis_data.dropna(subset=essential_cols)

                if analysis_data.empty:
                    st.error(
                        "No valid (non-NA) rows found after filtering for these columns."
                    )
                    return

                # Get unique labels from the data
                all_labels = set()
                for col in ["ModelPrediction"] + app_instance.annotation_columns:
                    if col in analysis_data.columns:
                        all_labels.update(analysis_data[col].unique())
                labels = sorted(list(all_labels))

                # Use compute_kappa_metrics for unified handling of single/multiple runs
                try:
                    summary_df, detailed_kappa_metrics = compute_kappa_metrics(
                        detailed_results_df=analysis_data,
                        annotation_columns=app_instance.annotation_columns,
                        labels=labels,
                        kappa_weights=weights,
                    )

                    if summary_df.empty:
                        st.error("Could not compute kappa metrics from the data.")
                        return

                    # Get the first (and likely only) row of results
                    result_row = summary_df.iloc[0]

                    # Check if we have multiple runs
                    n_runs = result_row.get("n_runs", 1)
                    has_multiple_runs = n_runs > 1

                    # Display run information if multiple runs
                    if has_multiple_runs:
                        st.info(
                            f"**Multiple Runs Detected**: Analysis computed across {n_runs} runs"
                        )
                        st.markdown(
                            """
                            **Note**: With multiple runs, metrics show the consistency of LLM performance. 
                            Lower variance across runs indicates more reliable model behavior.
                            """
                        )

                    # Display main agreement scores
                    if len(app_instance.annotation_columns) > 1:
                        # Multiple annotators - show mean scores with standard deviations if multiple runs
                        st.subheader("Mean Agreement Scores")

                        mean_llm_human = result_row.get("mean_kappa_llm_human", 0)
                        mean_human_human = result_row.get(
                            "mean_human_human_agreement", 0
                        )

                        if has_multiple_runs:
                            # Calculate standard deviations across runs
                            llm_human_stds = []
                            human_human_stds = []
                            unique_runs = sorted(analysis_data["run"].unique())

                            for run_id in unique_runs:
                                run_data = analysis_data[analysis_data["run"] == run_id]
                                run_model_predictions = run_data[
                                    "ModelPrediction"
                                ].tolist()
                                run_human_annotations = {
                                    col: run_data[col].tolist()
                                    for col in app_instance.annotation_columns
                                }

                                # Compute detailed kappa metrics for this run
                                from qualitative_analysis.metrics.kappa import (
                                    compute_detailed_kappa_metrics,
                                )

                                run_detailed_metrics = compute_detailed_kappa_metrics(
                                    model_predictions=run_model_predictions,
                                    human_annotations=run_human_annotations,
                                    labels=labels,
                                    kappa_weights=weights,
                                )
                                llm_human_stds.append(
                                    run_detailed_metrics["mean_llm_human_agreement"]
                                )
                                human_human_stds.append(
                                    run_detailed_metrics["mean_human_human_agreement"]
                                )

                            std_llm_human = (
                                np.std(llm_human_stds) if len(llm_human_stds) > 1 else 0
                            )
                            std_human_human = (
                                np.std(human_human_stds)
                                if len(human_human_stds) > 1
                                else 0
                            )

                            st.write(
                                f"**Mean LLM-Human Agreement**: {mean_llm_human:.4f} ± {std_llm_human:.4f}"
                            )
                            st.write(
                                f"**Mean Human-Human Agreement**: {mean_human_human:.4f} ± {std_human_human:.4f}"
                            )
                            st.write(f"**Computed across {n_runs} runs**")
                        else:
                            st.write(
                                f"**Mean LLM-Human Agreement**: {mean_llm_human:.4f}"
                            )
                            st.write(
                                f"**Mean Human-Human Agreement**: {mean_human_human:.4f}"
                            )
                    else:
                        # Single annotator - show individual score
                        st.subheader("Agreement Score")
                        human_annotator = app_instance.annotation_columns[0]

                        # Get kappa from the basic metrics
                        kappa_score = result_row.get("kappa_train", 0)

                        if has_multiple_runs:
                            # Calculate standard deviation across runs for single annotator
                            kappa_values = []
                            unique_runs = sorted(analysis_data["run"].unique())

                            for run_id in unique_runs:
                                run_data = analysis_data[analysis_data["run"] == run_id]
                                run_model_predictions = run_data[
                                    "ModelPrediction"
                                ].tolist()
                                run_human_annotations = {
                                    col: run_data[col].tolist()
                                    for col in app_instance.annotation_columns
                                }

                                # Compute majority vote for this run
                                from qualitative_analysis.metrics.utils import (
                                    compute_majority_vote,
                                )

                                run_ground_truth = compute_majority_vote(
                                    run_human_annotations
                                )

                                # Compute kappa for this run
                                run_kappa = compute_cohens_kappa(
                                    run_ground_truth,
                                    run_model_predictions,
                                    labels=labels,
                                    weights=weights,
                                )
                                kappa_values.append(run_kappa)

                            std_kappa = (
                                np.std(kappa_values) if len(kappa_values) > 1 else 0
                            )
                            st.write(
                                f"**Cohen's Kappa (LLM vs {human_annotator})**: {kappa_score:.4f} ± {std_kappa:.4f}"
                            )
                            st.write(f"**Computed across {n_runs} runs**")
                        else:
                            st.write(
                                f"**Cohen's Kappa (LLM vs {human_annotator})**: {kappa_score:.4f}"
                            )

                    # Display detailed scores if available and multiple annotators
                    if (
                        len(app_instance.annotation_columns) > 1
                        and detailed_kappa_metrics
                    ):
                        scenario_key = list(detailed_kappa_metrics.keys())[0]
                        kappa_details = detailed_kappa_metrics[scenario_key]

                        st.subheader("Individual Agreement Scores")

                        # Display LLM vs Human scores
                        if not kappa_details["llm_vs_human_df"].empty:
                            st.write("**LLM vs Human Annotators**")
                            llm_human_display = kappa_details["llm_vs_human_df"].copy()
                            llm_human_display["Cohens_Kappa"] = llm_human_display[
                                "Cohens_Kappa"
                            ].apply(lambda x: f"{x:.4f}")
                            st.table(llm_human_display)

                        # Display Human vs Human scores
                        if not kappa_details["human_vs_human_df"].empty:
                            st.write("**Human vs Human Annotators**")
                            human_human_display = kappa_details[
                                "human_vs_human_df"
                            ].copy()
                            human_human_display["Cohens_Kappa"] = human_human_display[
                                "Cohens_Kappa"
                            ].apply(lambda x: f"{x:.4f}")
                            st.table(human_human_display)

                    # Store evaluation data in session state for per-run analysis (if multiple runs)
                    if has_multiple_runs:
                        st.session_state["evaluation_completed"] = True
                        st.session_state["evaluation_analysis_data"] = analysis_data
                        st.session_state["evaluation_labels"] = labels
                        st.session_state["evaluation_weights"] = weights
                        st.session_state["evaluation_annotation_columns"] = (
                            app_instance.annotation_columns
                        )

                except Exception as e:
                    st.error(f"Error computing kappa metrics: {e}")
                    st.error("Please check your data format and try again.")

        # --------------------------------------------------------------------------------
        # OPTION 2: CLASSIFICATION METRICS
        # --------------------------------------------------------------------------------
        elif method == "Classification Metrics (Balanced Acc, TP%, FP%)":
            st.markdown(
                """
                Analyze detailed classification metrics for each class, focusing on recall and confusion matrix elements.
                
                This analysis uses majority vote from human annotations as ground truth and provides:
                - Class distribution (number of instances per class)
                - Global metrics for the LLM and each human annotator (balanced accuracy and F1 score)
                - Per-class metrics showing:
                  - TP%: Percentage of instances of this class that were correctly identified
                  - FP%: Percentage of predictions for this class that were incorrect
                """
            )

            # LLM columns presumably the ones in selected_fields
            metrics_llm_columns: list[str] = [
                col for col in app_instance.selected_fields if col in results_df.columns
            ]

            if not metrics_llm_columns:
                st.warning("No LLM-generated columns found in the results to compare.")
                return

            if len(app_instance.annotation_columns) < 1:
                st.warning(
                    "At least one annotation column is required for classification metrics."
                )
                return

            # Figure out a default index that points to our label column if possible
            metrics_default_index = 0
            if "label_column" in st.session_state:
                label_col = st.session_state["label_column"]
                if label_col in metrics_llm_columns:
                    metrics_default_index = metrics_llm_columns.index(label_col)

            metrics_llm_judgment_col: str = st.selectbox(
                "Select LLM Judgment Column:",
                metrics_llm_columns,
                index=metrics_default_index,
                key="llm_metrics_col_select",
            )

            if st.button(
                "Compute Classification Metrics", key="compute_metrics_button"
            ):
                if metrics_llm_judgment_col not in results_df.columns:
                    st.error("The chosen LLM column is not in the results dataframe.")
                    return

                # Check if annotation columns exist in the dataframe
                missing_columns = [
                    col
                    for col in app_instance.annotation_columns
                    if col not in results_df.columns
                ]
                if missing_columns:
                    st.error(
                        f"The following annotation columns are not in the results dataframe: {', '.join(missing_columns)}"
                    )
                    return

                # Prepare data for compute_classification_metrics_from_results
                analysis_data = results_df.copy()

                # Add required columns for compute_classification_metrics_from_results if they don't exist
                if "prompt_name" not in analysis_data.columns:
                    analysis_data["prompt_name"] = "streamlit_analysis"
                if "iteration" not in analysis_data.columns:
                    analysis_data["iteration"] = 1
                if "split" not in analysis_data.columns:
                    analysis_data["split"] = "train"
                if "run" not in analysis_data.columns:
                    analysis_data["run"] = 1

                # Rename the selected LLM column to ModelPrediction for consistency
                if metrics_llm_judgment_col != "ModelPrediction":
                    analysis_data = analysis_data.rename(
                        columns={metrics_llm_judgment_col: "ModelPrediction"}
                    )

                # Drop rows with missing values in essential columns
                essential_cols = ["ModelPrediction"] + app_instance.annotation_columns
                analysis_data = analysis_data.dropna(subset=essential_cols)

                if analysis_data.empty:
                    st.error(
                        "No valid (non-NA) rows found after filtering for these columns."
                    )
                    return

                # Get unique labels from the data
                all_labels = set()
                for col in ["ModelPrediction"] + app_instance.annotation_columns:
                    if col in analysis_data.columns:
                        all_labels.update(analysis_data[col].unique())
                labels = sorted(list(all_labels))

                # Use compute_classification_metrics_from_results for unified handling of single/multiple runs
                try:
                    summary_df = compute_classification_metrics_from_results(
                        detailed_results_df=analysis_data,
                        annotation_columns=app_instance.annotation_columns,
                        labels=labels,
                    )

                    if summary_df.empty:
                        st.error(
                            "Could not compute classification metrics from the data."
                        )
                        return

                    # Get the first (and likely only) row of results
                    result_row = summary_df.iloc[0]

                    # Check if we have multiple runs
                    n_runs = result_row.get("n_runs", 1)
                    has_multiple_runs = n_runs > 1

                    # Display run information if multiple runs
                    if has_multiple_runs:
                        st.info(
                            f"**Multiple Runs Detected**: Classification metrics computed across {n_runs} runs"
                        )
                        st.markdown(
                            """
                            **Note**: With multiple runs, metrics show the aggregated performance across all runs. 
                            This provides a more robust assessment of model performance.
                            """
                        )

                    # Extract global metrics
                    global_accuracy = result_row.get("global_accuracy_train", 0)
                    global_recall = result_row.get("global_recall_train", 0)
                    global_error_rate = result_row.get("global_error_rate_train", 0)

                    # Display global metrics
                    st.subheader("Global Metrics")

                    global_metrics_data = [
                        {"Metric": "Accuracy", "Value": f"{global_accuracy:.4f}"},
                        {
                            "Metric": "Balanced Accuracy (Macro Recall)",
                            "Value": f"{global_recall:.4f}",
                        },
                        {"Metric": "Error Rate", "Value": f"{global_error_rate:.4f}"},
                    ]

                    if has_multiple_runs:
                        global_metrics_data.append(
                            {
                                "Metric": "Training samples",
                                "Value": f"{result_row.get('N_train', 'N/A')} (across {n_runs} runs)",
                            }
                        )
                    else:
                        global_metrics_data.append(
                            {
                                "Metric": "Training samples",
                                "Value": f"{result_row.get('N_train', 'N/A')}",
                            }
                        )

                    st.table(pd.DataFrame(global_metrics_data))

                    # Note: Per-class metrics are only shown in per-run analysis
                    if has_multiple_runs:
                        st.info(
                            "💡 **Per-class metrics are available in the per-run analysis below** (check the box to view detailed breakdown by individual runs)"
                        )

                    # Store classification data in session state for per-run analysis (if multiple runs)
                    if has_multiple_runs:
                        st.session_state["classification_completed"] = True
                        st.session_state["classification_analysis_data"] = analysis_data
                        st.session_state["classification_labels"] = labels
                        st.session_state["classification_annotation_columns"] = (
                            app_instance.annotation_columns
                        )

                except Exception as e:
                    st.error(f"Error computing classification metrics: {e}")
                    st.error("Please check your data format and try again.")

        # --------------------------------------------------------------------------------
        # OPTION 3: ALT-TEST
        # --------------------------------------------------------------------------------
        elif method == "Alt-Test (Model Viability)":
            st.markdown(
                """
                **Alternative Annotator Test** (requires >= 3 annotation columns).  
                Compares the model's predictions to human annotators by excluding one human at a time 
                and measuring alignment with the remaining humans.
                
                - The test yields a "winning rate" (the proportion of annotators for
                   which the LLM outperforms or is at least as good as that annotator,
                   given the cost/benefit trade-off).
                 - "Epsilon" (ε), represents how much we adjust the human advantage to account for time/cost/effort
                   savings when using an LLM. Larger ε values make it easier for the LLM
                   to "pass" because it reflects that human annotations are costlier (if your human are experts, the original article recommend 0.2, if they are crowdworker, 0.1).
                 - If the LLM's winning rate ≥ 0.5, the test concludes that the LLM is
                   (statistically) as viable as a human annotator for that dataset (the LLM is "better" than half the humans).
                """
            )

            if len(app_instance.annotation_columns) < 3:
                st.warning(
                    "You must have at least 3 annotation columns to run the alt-test."
                )
                return

            alt_llm_columns: list[str] = [
                col for col in app_instance.selected_fields if col in results_df.columns
            ]
            if not alt_llm_columns:
                st.warning("No valid LLM columns found in the results.")
                return

            # Default to the label column if possible
            alt_default_index = 0
            if "label_column" in st.session_state:
                label_col = st.session_state["label_column"]
                if label_col in alt_llm_columns:
                    alt_default_index = alt_llm_columns.index(label_col)

            alt_model_col: str = st.selectbox(
                "Choose model column for alt-test:",
                alt_llm_columns,
                index=alt_default_index,
                key="alt_test_model_col_select",
            )

            epsilon_val: float = st.number_input(
                "Epsilon (cost-benefit margin)",
                min_value=0.0,
                value=0.1,
                step=0.01,
                key="alt_test_epsilon",
            )
            alpha_val: float = st.number_input(
                "Alpha (significance level)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                key="alt_test_alpha",
            )

            if st.button("Run Alternative Annotator Test", key="run_alt_test_button"):
                try:
                    # Prepare data for run_alt_test_on_results
                    analysis_data = results_df.copy()

                    # Add required columns for run_alt_test_on_results if they don't exist
                    if "prompt_name" not in analysis_data.columns:
                        analysis_data["prompt_name"] = "streamlit_analysis"
                    if "iteration" not in analysis_data.columns:
                        analysis_data["iteration"] = 1
                    if "split" not in analysis_data.columns:
                        analysis_data["split"] = "train"
                    if "run" not in analysis_data.columns:
                        analysis_data["run"] = 1

                    # Rename the selected LLM column to ModelPrediction for consistency
                    if alt_model_col != "ModelPrediction":
                        analysis_data = analysis_data.rename(
                            columns={alt_model_col: "ModelPrediction"}
                        )

                    # Convert to integers if possible
                    try:
                        for col in [
                            "ModelPrediction"
                        ] + app_instance.annotation_columns:
                            if col in analysis_data.columns:
                                analysis_data[col] = analysis_data[col].astype(int)
                    except ValueError:
                        st.error("Could not convert columns to integer for ALT test.")
                        return

                    # Drop rows with missing values in essential columns
                    essential_cols = [
                        "ModelPrediction"
                    ] + app_instance.annotation_columns
                    analysis_data = analysis_data.dropna(subset=essential_cols)

                    if analysis_data.empty:
                        st.error(
                            "No valid (non-NA) rows found after filtering for these columns."
                        )
                        return

                    # Get unique labels from the data
                    all_labels = set()
                    for col in ["ModelPrediction"] + app_instance.annotation_columns:
                        if col in analysis_data.columns:
                            all_labels.update(analysis_data[col].unique())
                    labels = sorted(list(all_labels))

                    # Use run_alt_test_on_results for unified handling of single/multiple runs
                    alt_results_df = run_alt_test_on_results(
                        detailed_results_df=analysis_data,
                        annotation_columns=app_instance.annotation_columns,
                        labels=labels,
                        epsilon=epsilon_val,
                        alpha=alpha_val,
                        verbose=False,
                    )

                    if alt_results_df.empty:
                        st.error("Could not compute ALT test metrics from the data.")
                        return

                    # Get aggregated results (row with run="aggregated")
                    aggregated_results = alt_results_df[
                        alt_results_df["run"] == "aggregated"
                    ]

                    if aggregated_results.empty:
                        # If no aggregated results, use the first row (single run case)
                        aggregated_results = alt_results_df.iloc[[0]]

                    result_row = aggregated_results.iloc[0]

                    # Check if we have multiple runs
                    n_runs = result_row.get("n_runs", 1)
                    has_multiple_runs = n_runs > 1

                    # Display run information if multiple runs
                    if has_multiple_runs:
                        st.info(
                            f"**Multiple Runs Detected**: ALT test computed across {n_runs} runs"
                        )

                    # Display aggregated results
                    st.subheader("Alt-Test Results")

                    # Extract data for the aggregated table
                    p_values_train = result_row.get("p_values_train", [])
                    winning_rate = result_row.get("winning_rate_train", 0)
                    passed_test = result_row.get("passed_alt_test_train", False)
                    avg_adv_prob = result_row.get("avg_adv_prob_train", 0)

                    if p_values_train and len(p_values_train) > 0:
                        # Create table data for aggregated results
                        table_data = []
                        for i, annotator in enumerate(app_instance.annotation_columns):
                            if i < len(p_values_train):
                                p_val = p_values_train[i]
                                reject_h0 = (
                                    p_val < alpha_val if not pd.isna(p_val) else False
                                )

                                # For aggregated results, we don't have individual rho values
                                # So we'll show the average advantage probability
                                table_data.append(
                                    {
                                        "Annotator": annotator,
                                        "p-value": (
                                            f"{p_val:.4f}"
                                            if not pd.isna(p_val)
                                            else "NaN"
                                        ),
                                        "RejectH0?": reject_h0,
                                        "rho_f (LLM advantage)": f"{avg_adv_prob:.3f}",
                                        "rho_h (Human advantage)": f"{1 - avg_adv_prob:.3f}",
                                    }
                                )

                        # Display the table
                        st.table(pd.DataFrame(table_data))

                        # Display summary metrics
                        st.write(f"**Winning Rate (omega)**: {winning_rate:.3f}")
                        st.write(f"**Average LLM Advantage (rho)**: {avg_adv_prob:.3f}")

                        if passed_test:
                            st.success(
                                "✅ The model **passed** the alt-test (winning rate ≥ 0.5)."
                            )
                        else:
                            st.warning(
                                "❌ The model **did not pass** the alt-test (winning rate < 0.5)."
                            )

                        # Store ALT test data in session state for per-run analysis
                        if has_multiple_runs:
                            st.session_state["alt_test_completed"] = True
                            st.session_state["alt_test_results_df"] = alt_results_df
                            st.session_state["alt_test_annotation_columns"] = (
                                app_instance.annotation_columns
                            )
                            st.session_state["alt_test_alpha_value"] = alpha_val

                    else:
                        st.error("No valid ALT test results found.")

                except ValueError as ve:
                    st.error(f"Alt-Test Error: {ve}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    return

        # --------------------------------------------------------------------------------
        # OPTION 4: KRIPPENDORFF'S ALPHA (NON-INFERIORITY TEST)
        # --------------------------------------------------------------------------------
        else:  # method == "Krippendorff's Alpha (Non-Inferiority Test)"
            st.markdown(
                """
                Krippendorff’s α Non‑Inferiority Test  
                *(requires ≥ 3 human‑annotation columns)*

                **Purpose**  
                Show that the model’s labels are *no worse* than a fully human‑annotated baseline, so you can safely let the model take over the remaining, unlabeled items.

                How the test works

                1. **Baseline reliability (`α_human`)**  
                *Compute Krippendorff’s α across all *n* human annotators.*  

                2. **Model‑augmented reliability (`α_model`)**  
                *For each possible panel of *(n − 1) humans + the model*, compute α, then take the mean.*  

                3. **Effect size (`Δ`)**  
                \[
                Δ = α_{\text{model}} - α_{\text{human}}
                \]  
                Positive Δ = model improves reliability; negative Δ = potential drop.

                4. **Uncertainty (bootstrap CI)**  
                *Resample the dataset thousands of times, recomputing Δ each time.*  
                The resulting 90 % (configurable) confidence interval shows where the *true* Δ is likely to fall.

                5. **Non‑inferiority margin (δ)**  
                You set δ (often −0.05) to be the largest drop in α you’re willing to accept.  
                *Decision rule*: **If the entire CI lies above δ, the model is declared *non‑inferior*.**  
                With a 90 % CI this corresponds to a one‑sided 5 % risk of mistakenly approving a model that is actually worse than δ.

                Interpretation cheatsheet

                | Outcome | What it means for deployment |
                |---------|------------------------------|
                | CI fully **above 0** | Model is **statistically superior** to humans—use it with confidence. |
                | CI fully **above δ but crosses 0** | Model is **non‑inferior** (at worst a small, acceptable loss). Annotate the rest with the model if that loss is tolerable. |
                | CI **touches or falls below δ** | Insufficient evidence—keep humans or gather more data. |

                > **Why “5 % risk”?**  
                > With a 90 % CI you’re running a *one‑sided* non‑inferiority test at α = 0.05.  
                > That 5 % error rate is anchored to the **margin δ**—not to zero.  
                > • If the CI just touches δ, there is ≈ 5 % chance the true Δ is ≤ δ.  
                > • If the CI sits well above δ, the risk the true Δ is even ≤ 0 is **smaller** than 5 %.

                | Setting | What happens when you **increase** it | What happens when you **decrease** it |
                |---------|---------------------------------------|---------------------------------------|
                | **Confidence level** (e.g. 90 % → 95 %) | – CI gets **wider**.<br>– Test becomes **more conservative**: harder for the model to pass.<br>– One‑sided Type I error drops (5 % → 2.5 %). | – CI gets **narrower**.<br>– Easier to declare non‑inferiority, but higher chance of a false positive. |
                | **Non‑inferiority margin δ** (e.g. ‑0.05 → ‑0.10) | – You tolerate a **larger performance drop**.<br>– Much easier for the model to pass, but you may accept poorer quality. | – You demand the model stay **closer to (or above) human level**.<br>– Harder to pass; stronger guarantee of quality. |

                """
            )

            if len(app_instance.annotation_columns) < 3:
                st.warning(
                    "You must have at least 3 annotation columns to run Krippendorff's alpha test."
                )
                return

            kripp_llm_columns: list[str] = [
                col for col in app_instance.selected_fields if col in results_df.columns
            ]
            if not kripp_llm_columns:
                st.warning("No valid LLM columns found in the results.")
                return

            # Default to the label column if possible
            kripp_default_index = 0
            if "label_column" in st.session_state:
                label_col = st.session_state["label_column"]
                if label_col in kripp_llm_columns:
                    kripp_default_index = kripp_llm_columns.index(label_col)

            kripp_model_col: str = st.selectbox(
                "Choose model column for Krippendorff's alpha test:",
                kripp_llm_columns,
                index=kripp_default_index,
                key="kripp_test_model_col_select",
            )

            # Configuration parameters
            st.subheader("Test Configuration")

            col1, col2 = st.columns(2)

            with col1:
                level_of_measurement: str = st.radio(
                    "Level of Measurement:",
                    ["ordinal", "nominal", "interval", "ratio"],
                    index=0,
                    key="kripp_level_measurement",
                )

                non_inferiority_margin: float = st.number_input(
                    "Non-inferiority margin (δ)",
                    min_value=-1.0,
                    max_value=0.0,
                    value=-0.05,
                    step=0.01,
                    key="kripp_margin",
                )

            with col2:
                n_bootstrap: int = st.number_input(
                    "Bootstrap samples",
                    min_value=100,
                    max_value=10000,
                    value=2000,
                    step=100,
                    key="kripp_bootstrap",
                    help="More samples = more accurate confidence intervals but slower computation",
                )

                confidence_level: float = st.number_input(
                    "Confidence level (%)",
                    min_value=80.0,
                    max_value=99.0,
                    value=90.0,
                    step=1.0,
                    key="kripp_confidence",
                )

            if st.button("Run Krippendorff's Alpha Test", key="run_kripp_test_button"):
                try:
                    # Import the Krippendorff functions
                    from qualitative_analysis.metrics.krippendorff import (
                        compute_krippendorff_non_inferiority,
                    )

                    # Prepare data for compute_krippendorff_non_inferiority
                    analysis_data = results_df.copy()

                    # Add required columns if they don't exist
                    if "prompt_name" not in analysis_data.columns:
                        analysis_data["prompt_name"] = "streamlit_analysis"
                    if "iteration" not in analysis_data.columns:
                        analysis_data["iteration"] = 1
                    if "split" not in analysis_data.columns:
                        analysis_data["split"] = "train"
                    if "run" not in analysis_data.columns:
                        analysis_data["run"] = 1

                    # Rename the selected LLM column to ModelPrediction for consistency
                    if kripp_model_col != "ModelPrediction":
                        analysis_data = analysis_data.rename(
                            columns={kripp_model_col: "ModelPrediction"}
                        )

                    # --- Prepare data for Krippendorff's alpha test ---
                    analysis_data = results_df.copy()

                    # Add required columns if they don't exist
                    for col, default_value in {
                        "prompt_name": "streamlit_analysis",
                        "iteration": 1,
                        "split": "train",
                        "run": 1,
                    }.items():
                        if col not in analysis_data.columns:
                            analysis_data[col] = default_value

                    # Rename the selected LLM column to ModelPrediction for consistency
                    if kripp_model_col != "ModelPrediction":
                        analysis_data = analysis_data.rename(
                            columns={kripp_model_col: "ModelPrediction"}
                        )

                    # 🧹 Clean and convert columns safely
                    essential_cols = [
                        "ModelPrediction"
                    ] + app_instance.annotation_columns

                    for col in essential_cols:
                        if col in analysis_data.columns:
                            # Replace blanks and pseudo-NaNs
                            analysis_data[col] = (
                                analysis_data[col]
                                .replace(["", " ", "None", "nan", "NaN"], np.nan)
                                .astype(float)
                            )
                            # Warn if the column is fully empty
                            if analysis_data[col].isna().all():
                                st.warning(
                                    f"Column '{col}' has no valid numeric entries and will be ignored."
                                )

                    # Drop only rows where *all* essential columns are NaN
                    analysis_data = analysis_data.dropna(
                        how="all", subset=essential_cols
                    )

                    if analysis_data.empty:
                        st.error(
                            "No valid rows found after filtering for essential columns."
                        )
                        return

                    # Run Krippendorff's alpha non-inferiority test
                    with st.spinner(
                        "Running Krippendorff's alpha test... This may take a moment due to bootstrap resampling."
                    ):
                        kripp_results = compute_krippendorff_non_inferiority(
                            detailed_results_df=analysis_data,
                            annotation_columns=app_instance.annotation_columns,
                            model_column="ModelPrediction",
                            level_of_measurement=level_of_measurement,
                            non_inferiority_margin=non_inferiority_margin,
                            n_bootstrap=n_bootstrap,
                            confidence_level=confidence_level,
                            verbose=False,
                        )

                    if not kripp_results:
                        st.error("Could not compute Krippendorff's alpha test results.")
                        return

                    # Display results
                    st.subheader("Krippendorff's Alpha Non-Inferiority Test Results")

                    # Get the first scenario results
                    scenario_key = list(kripp_results.keys())[0]
                    scenario_results = kripp_results[scenario_key]
                    agg_metrics = scenario_results["aggregated_metrics"]

                    # Check if we have multiple runs
                    n_runs = agg_metrics["n_runs"]
                    has_multiple_runs = n_runs > 1

                    # Display run information if multiple runs
                    if has_multiple_runs:
                        st.info(
                            f"**Multiple Runs Detected**: Krippendorff's alpha test computed across {n_runs} runs"
                        )

                    # Display the main results in the requested format
                    st.write(
                        f"**Human trios α**: {agg_metrics['alpha_human_trios_mean']:.4f} ± {agg_metrics['alpha_human_trios_std']:.4f}"
                    )
                    st.write(
                        f"**Model trios α**: {agg_metrics['alpha_model_trios_mean']:.4f} ± {agg_metrics['alpha_model_trios_std']:.4f}"
                    )
                    st.write(
                        f"**Δ = model − human**: {agg_metrics['difference_mean']:+.4f} ± {agg_metrics['difference_std']:.4f}"
                    )
                    st.write(
                        f"**{confidence_level:.0f}% CI**: [{agg_metrics['ci_lower_mean']:.4f}, {agg_metrics['ci_upper_mean']:.4f}]"
                    )
                    st.write(
                        f"**Non-inferiority demonstrated in {agg_metrics['n_non_inferior']}/{n_runs} runs**"
                    )

                    # Display final verdict
                    if agg_metrics["n_non_inferior"] == n_runs:
                        st.success(
                            f"✅ **Non-inferiority consistently demonstrated across all runs** (margin = {non_inferiority_margin})"
                        )
                    elif agg_metrics["n_non_inferior"] > 0:
                        st.warning(
                            f"⚠️ **Non-inferiority demonstrated in some but not all runs** (margin = {non_inferiority_margin})"
                        )
                    else:
                        st.error(
                            f"❌ **Non-inferiority NOT demonstrated in any run** (margin = {non_inferiority_margin})"
                        )

                    # Store Krippendorff test data in session state for per-run analysis
                    if has_multiple_runs:
                        st.session_state["kripp_test_completed"] = True
                        st.session_state["kripp_test_results"] = kripp_results
                        st.session_state["kripp_test_confidence_level"] = (
                            confidence_level
                        )

                except ImportError:
                    st.error(
                        "Krippendorff library not available. Please install it with: pip install krippendorff"
                    )
                    return
                except Exception as e:
                    st.error(f"Error running Krippendorff's alpha test: {e}")
                    return

        # --------------------------------------------------------------------------------
        # PER-RUN ANALYSIS (outside button blocks to persist across reruns)
        # --------------------------------------------------------------------------------
        if (
            st.session_state.get("evaluation_completed", False)
            and method == "Cohen's Kappa (Agreement Analysis)"
        ):
            per_run_analysis_data: Optional[pd.DataFrame] = st.session_state.get(
                "evaluation_analysis_data"
            )
            per_run_labels: Optional[List[Any]] = st.session_state.get(
                "evaluation_labels"
            )
            per_run_weights: Optional[str] = st.session_state.get("evaluation_weights")
            per_run_annotation_columns: Optional[List[str]] = st.session_state.get(
                "evaluation_annotation_columns"
            )

            if (
                per_run_analysis_data is not None
                and per_run_annotation_columns is not None
                and "run" in per_run_analysis_data.columns
                and len(per_run_analysis_data["run"].unique()) > 1
            ):
                st.subheader("Per-Run Analysis")
                show_per_run = st.checkbox(
                    "Show metrics by individual runs",
                    key="show_per_run_metrics",
                    help="Display kappa and accuracy metrics computed separately for each run to assess consistency",
                )

                if show_per_run:
                    st.markdown(
                        """
                        **Per-Run Metrics**: These metrics are computed separately for each run to show 
                        the consistency of LLM performance. Large variations between runs indicate 
                        less reliable model behavior.
                        """
                    )

                    # Compute per-run metrics
                    per_run_results = []
                    unique_runs = sorted(per_run_analysis_data["run"].unique())

                    for run_id in unique_runs:
                        run_data = per_run_analysis_data[
                            per_run_analysis_data["run"] == run_id
                        ]

                        # Get model predictions and human annotations for this run
                        run_model_predictions = run_data["ModelPrediction"].tolist()
                        run_human_annotations = {
                            col: run_data[col].tolist()
                            for col in per_run_annotation_columns
                        }

                        # Compute majority vote for this run
                        from qualitative_analysis.metrics.utils import (
                            compute_majority_vote,
                        )

                        run_ground_truth = compute_majority_vote(run_human_annotations)

                        # Compute kappa for this run
                        run_kappa = compute_cohens_kappa(
                            run_ground_truth,
                            run_model_predictions,
                            labels=per_run_labels,
                            weights=per_run_weights,
                        )

                        # Compute detailed kappa metrics for this run if multiple annotators
                        if len(per_run_annotation_columns) > 1:
                            from qualitative_analysis.metrics.kappa import (
                                compute_detailed_kappa_metrics,
                            )

                            run_detailed_metrics = compute_detailed_kappa_metrics(
                                model_predictions=run_model_predictions,
                                human_annotations=run_human_annotations,
                                labels=per_run_labels,
                                kappa_weights=per_run_weights,
                            )
                            run_mean_llm_human = run_detailed_metrics[
                                "mean_llm_human_agreement"
                            ]
                            run_mean_human_human = run_detailed_metrics[
                                "mean_human_human_agreement"
                            ]
                        else:
                            run_mean_llm_human = run_kappa
                            run_mean_human_human = None

                        per_run_results.append(
                            {
                                "Run": run_id,
                                "Mean LLM-Human Kappa": (
                                    f"{run_mean_llm_human:.4f}"
                                    if run_mean_llm_human is not None
                                    else "N/A"
                                ),
                                "Mean Human-Human Kappa": (
                                    f"{run_mean_human_human:.4f}"
                                    if run_mean_human_human is not None
                                    else "N/A"
                                ),
                                "Samples": len(run_data),
                            }
                        )

                    # Display per-run results table
                    per_run_df = pd.DataFrame(per_run_results)
                    st.table(per_run_df)

        # --------------------------------------------------------------------------------
        # ALT TEST PER-RUN ANALYSIS (outside button blocks to persist across reruns)
        # --------------------------------------------------------------------------------
        if (
            st.session_state.get("alt_test_completed", False)
            and method == "Alt-Test (Model Viability)"
        ):
            per_run_alt_results_df: Optional[pd.DataFrame] = st.session_state.get(
                "alt_test_results_df"
            )
            per_run_alt_annotation_columns: Optional[List[str]] = st.session_state.get(
                "alt_test_annotation_columns"
            )
            per_run_alt_alpha_val: float = st.session_state.get(
                "alt_test_alpha_value", 0.05
            )

            if (
                per_run_alt_results_df is not None
                and per_run_alt_annotation_columns is not None
                and not per_run_alt_results_df.empty
            ):
                # Check if we have individual run data (not just aggregated)
                individual_runs = per_run_alt_results_df[
                    per_run_alt_results_df["run"] != "aggregated"
                ]

                if not individual_runs.empty and len(individual_runs) > 1:
                    st.subheader("Per-Run ALT Test Analysis")
                    show_per_run_alt = st.checkbox(
                        "Show ALT test results by individual runs",
                        key="show_per_run_alt_metrics",
                        help="Display ALT test metrics computed separately for each run to assess consistency",
                    )

                    if show_per_run_alt:
                        st.markdown(
                            """
                            **Per-Run ALT Test Results**: These results are computed separately for each run to show 
                            the consistency of LLM performance in the ALT test. Variations between runs indicate 
                            different levels of model reliability.
                            """
                        )

                        # Display results for each individual run
                        unique_runs = sorted(individual_runs["run"].unique())

                        for run_id in unique_runs:
                            run_data = individual_runs[
                                individual_runs["run"] == run_id
                            ].iloc[0]

                            st.write(f"**Run {run_id}:**")

                            # Extract data for this run
                            p_values_train = run_data.get("p_values_train", [])
                            winning_rate = run_data.get("winning_rate_train", 0)
                            passed_test = run_data.get("passed_alt_test_train", False)
                            avg_adv_prob = run_data.get("avg_adv_prob_train", 0)

                            if p_values_train and len(p_values_train) > 0:
                                # Create table data for this run
                                table_data = []
                                for i, annotator in enumerate(
                                    per_run_alt_annotation_columns
                                ):
                                    if i < len(p_values_train):
                                        p_val = p_values_train[i]
                                        reject_h0 = (
                                            p_val < per_run_alt_alpha_val
                                            if not pd.isna(p_val)
                                            else False
                                        )

                                        table_data.append(
                                            {
                                                "Annotator": annotator,
                                                "p-value": (
                                                    f"{p_val:.4f}"
                                                    if not pd.isna(p_val)
                                                    else "NaN"
                                                ),
                                                "RejectH0?": reject_h0,
                                                "rho_f (LLM advantage)": f"{avg_adv_prob:.3f}",
                                                "rho_h (Human advantage)": f"{1 - avg_adv_prob:.3f}",
                                            }
                                        )

                                # Display the table for this run
                                st.table(pd.DataFrame(table_data))

                                # Display summary metrics for this run
                                st.write(f"Winning Rate (omega): {winning_rate:.3f}")
                                st.write(
                                    f"Average LLM Advantage (rho): {avg_adv_prob:.3f}"
                                )

                                if passed_test:
                                    st.success(
                                        "✅ The model **passed** the alt-test for this run."
                                    )
                                else:
                                    st.warning(
                                        "❌ The model **did not pass** the alt-test for this run."
                                    )

                                st.write("---")  # Separator between runs
                            else:
                                st.warning(
                                    f"No valid ALT test results found for Run {run_id}."
                                )

        # --------------------------------------------------------------------------------
        # CLASSIFICATION METRICS PER-RUN ANALYSIS (outside button blocks to persist across reruns)
        # --------------------------------------------------------------------------------
        if (
            st.session_state.get("classification_completed", False)
            and method == "Classification Metrics (Balanced Acc, TP%, FP%)"
        ):
            classification_analysis_data: Optional[pd.DataFrame] = st.session_state.get(
                "classification_analysis_data"
            )
            classification_labels: Optional[List[Any]] = st.session_state.get(
                "classification_labels"
            )
            classification_annotation_columns: Optional[List[str]] = (
                st.session_state.get("classification_annotation_columns")
            )

            if (
                classification_analysis_data is not None
                and classification_labels is not None
                and classification_annotation_columns is not None
                and "run" in classification_analysis_data.columns
                and len(classification_analysis_data["run"].unique()) > 1
            ):
                st.subheader("Per-Run Classification Analysis")
                show_per_run_classification = st.checkbox(
                    "Show classification metrics by individual runs",
                    key="show_per_run_classification_metrics",
                    help="Display detailed per-class metrics computed separately for each run",
                )

                if show_per_run_classification:
                    st.markdown(
                        """
                        **Per-Run Classification Metrics**: These metrics are computed separately for each run to show 
                        the consistency of LLM performance across different runs. Each table shows detailed per-class 
                        breakdown for that specific run.
                        """
                    )

                    # Display results for each individual run
                    unique_runs = sorted(classification_analysis_data["run"].unique())

                    for run_id in unique_runs:
                        run_data = classification_analysis_data[
                            classification_analysis_data["run"] == run_id
                        ]

                        st.write(f"**Run {run_id} - Per-Class Metrics:**")

                        # Get model predictions and human annotations for this run
                        run_model_predictions = run_data["ModelPrediction"].tolist()
                        run_human_annotations = {
                            col: run_data[col].tolist()
                            for col in classification_annotation_columns
                        }

                        # Compute classification metrics for this run
                        from qualitative_analysis.metrics.classification import (
                            compute_classification_metrics,
                        )

                        run_classification_results = compute_classification_metrics(
                            model_coding=run_model_predictions,
                            human_annotations=run_human_annotations,
                            labels=classification_labels,
                        )

                        # Create per-class metrics table for this run
                        per_class_data = []
                        for label in classification_labels:
                            if label in run_classification_results["per_class_metrics"]:
                                class_metrics = run_classification_results[
                                    "per_class_metrics"
                                ][label]["model"]

                                recall = class_metrics["recall"]
                                tp_count = class_metrics["correct_count"]
                                fp_count = class_metrics["false_positives"]

                                # Calculate FP% = FP / (TP + FP)
                                total_predictions = tp_count + fp_count
                                fp_percentage = (
                                    fp_count / total_predictions
                                    if total_predictions > 0
                                    else 0
                                )

                                per_class_data.append(
                                    {
                                        "Class": label,
                                        "TP% (Recall)": f"{recall:.2%}",
                                        "FP%": f"{fp_percentage:.2%}",
                                        "True Positives": int(tp_count),
                                        "False Positives": int(fp_count),
                                    }
                                )

                        if per_class_data:
                            st.table(pd.DataFrame(per_class_data))
                        else:
                            st.warning(f"No per-class metrics found for Run {run_id}.")

                        # Add some spacing between runs
                        if (
                            run_id != unique_runs[-1]
                        ):  # Don't add separator after the last run
                            st.write("---")
