import unittest
import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


evaluation_mappings = _load_module(
    "evaluation_mappings_module", "streamlit_app/evaluation_mappings.py"
)


class EvaluationMappingsTests(unittest.TestCase):
    def test_sanitize_migrates_legacy_label_configuration(self) -> None:
        mappings = evaluation_mappings.sanitize_evaluation_mappings(
            raw_mappings=None,
            selected_fields=["clarity", "creativity"],
            annotation_columns=["human_a", "human_b"],
            legacy_label_column="clarity",
            legacy_label_type="Integer",
            create_default_if_empty=True,
        )

        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0]["llm_field"], "clarity")
        self.assertEqual(mappings[0]["human_columns"], ["human_a", "human_b"])
        self.assertEqual(mappings[0]["label_type"], "Integer")

    def test_prepare_evaluation_data_coerces_integer_values(self) -> None:
        results_df = pd.DataFrame(
            {
                "clarity": ["1 - clear", "0 unclear", "missing"],
                "human_a": ["1", "0", "1"],
                "human_b": ["1", "0", "oops"],
            }
        )
        mapping = {
            "id": "map1",
            "name": "Clarity",
            "llm_field": "clarity",
            "human_columns": ["human_a", "human_b"],
            "label_type": "Integer",
        }

        prepared = evaluation_mappings.prepare_evaluation_data(
            results_df, mapping, row_filter="complete"
        )

        self.assertEqual(len(prepared.analysis_data), 2)
        self.assertEqual(prepared.analysis_data["ModelPrediction"].tolist(), [1, 0])
        self.assertEqual(prepared.analysis_data["human_a"].tolist(), [1, 0])
        self.assertEqual(prepared.labels, [0, 1])
        self.assertTrue(
            any(
                "could not be coerced to Integer" in warning
                for warning in prepared.warnings
            )
        )

    def test_prepare_evaluation_data_preserves_text_labels(self) -> None:
        results_df = pd.DataFrame(
            {
                "creativity": ["novel", "expected", None],
                "human_a": ["novel", "expected", "novel"],
                "human_b": ["novel", "expected", None],
            }
        )
        mapping = {
            "id": "map2",
            "name": "Creativity",
            "llm_field": "creativity",
            "human_columns": ["human_a", "human_b"],
            "label_type": "Text",
        }

        prepared = evaluation_mappings.prepare_evaluation_data(
            results_df, mapping, row_filter="complete"
        )

        self.assertEqual(len(prepared.analysis_data), 2)
        self.assertEqual(
            prepared.analysis_data["ModelPrediction"].astype(str).tolist(),
            ["novel", "expected"],
        )
        self.assertEqual(prepared.labels, ["expected", "novel"])

    def test_validate_krippendorff_mapping_rejects_text_ordinal(self) -> None:
        mapping = {
            "id": "map3",
            "name": "Creativity",
            "llm_field": "creativity",
            "human_columns": ["human_a", "human_b", "human_c"],
            "label_type": "Text",
        }

        error = evaluation_mappings.validate_krippendorff_mapping(mapping, "ordinal")

        self.assertEqual(
            error,
            "Text mappings can only be used with nominal Krippendorff measurement.",
        )

    def test_alt_test_supports_float_label_type(self) -> None:
        try:
            alt_test_module = _load_module(
                "alt_test_module", "qualitative_analysis/metrics/alt_test.py"
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional dependency missing for alt-test: {exc}")

        results_df = pd.DataFrame(
            {
                "prompt_name": ["prompt"] * 4,
                "iteration": [1] * 4,
                "run": [1] * 4,
                "split": ["train"] * 4,
                "ModelPrediction": [0.10, 0.20, 0.80, 0.90],
                "human_a": [0.12, 0.18, 0.82, 0.88],
                "human_b": [0.11, 0.21, 0.79, 0.91],
                "human_c": [0.09, 0.22, 0.81, 0.89],
            }
        )

        alt_df = alt_test_module.run_alt_test_on_results(
            detailed_results_df=results_df,
            annotation_columns=["human_a", "human_b", "human_c"],
            labels=[0.1, 0.2, 0.8, 0.9],
            epsilon=0.1,
            alpha=0.05,
            verbose=False,
            metric="rmse",
            label_type="float",
        )

        self.assertFalse(alt_df.empty)
        aggregated = alt_df[alt_df["run"] == "aggregated"].iloc[0]
        self.assertIn("winning_rate_train", aggregated.index)


if __name__ == "__main__":
    unittest.main()
