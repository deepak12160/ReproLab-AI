"""Regression tests for the specification-screen bug: metric names (Youden's J, Macro Rec., G-Mean) appeared in
Datasets while method / hyperparameters / seed / results were reported missing."""
import unittest

from app import ingest
from app.metrics_vocab import is_metric_label, canonical_key, find_metrics_in_text
from app.schemas import ResearchSpec, SchemaError
from app.spec_quality import (dataset_entry_problem, validate_user_edits, sanitize_spec, ground_spec, compute_missing,
                              apply_user_edits)
from .helpers import make_pdf, IMBALANCED_PAPER, StubProvider


def spec_of(pages):
    return ingest.build_spec(ingest.extract_pages(make_pdf(pages)))


class TestFieldMappingRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = spec_of(IMBALANCED_PAPER)

    def test_datasets_contain_dataset_names_only(self):
        self.assertEqual(self.s.datasets, ["Wine", "Glass"])
        for bad in ("Youden's J", "Macro Rec.", "G-Mean", "0.91"):
            self.assertNotIn(bad, self.s.datasets)

    def test_method_hyperparameters_seed_found_not_missing(self):
        self.assertIn("MarginBoost reweights", self.s.method)
        self.assertEqual(self.s.hyperparameters, {"learning_rate": 0.05, "random_seed": 7, "n_estimators": 200,
                                                  "test_size": 0.3})
        # test_size is DERIVED from the stated 70/30 split and labelled as such, not presented as quoted
        derived = [e for e in self.s.evidence if e["field"] == "hyperparameters" and e["key"] == "test_size"]
        self.assertEqual(derived[0]["label"], "derived")
        self.assertNotIn("Random seed not reported", self.s.missing_info)
        self.assertNotIn("No hyperparameters found", self.s.missing_info)
        self.assertNotIn("No method/algorithm section found", self.s.missing_info)
        self.assertNotIn("No numerical results found to compare against", self.s.missing_info)

    def test_metrics_field_has_the_metric_names(self):
        low = {m.rstrip(".") for m in self.s.metrics}
        self.assertEqual(low, {"youden's j", "macro rec", "g-mean"})

    def test_reported_results_are_metric_value_pairs_with_evidence(self):
        got = {(r.dataset, canonical_key(r.metric), r.value) for r in self.s.reported_results}
        self.assertIn(("Wine", "youden_j", 0.91), got)
        self.assertIn(("Glass", "g_mean", 0.70), got)
        self.assertEqual(len(self.s.reported_results), 6)
        r = self.s.reported_results[0]
        self.assertEqual(r.page, 1); self.assertIn("Table row", r.excerpt)
        self.assertTrue(any(e["field"] == "reported_results" and e["page"] == 1 for e in self.s.evidence))

    def test_subheading_stays_in_parent_section(self):
        self.assertIn("The loss penalises small margins.", self.s.method)

    def test_conclusion_does_not_leak_into_other_fields(self):
        self.assertNotIn("MarginBoost works", " ".join(self.s.datasets) + self.s.method)

    def test_split_and_task_inferred(self):
        self.assertTrue(any("70/30" in x for x in self.s.splits))
        self.assertEqual(self.s.task_type, "classification")

    def test_absent_information_stays_absent(self):
        pages = [["A Title", "Some Authors", "Abstract", "We study things.", "3. Methodology", "We do a thing.",
                  "5. Results", "It works well."]]
        s = spec_of(pages)
        self.assertEqual((s.datasets, s.hyperparameters, s.reported_results, s.metrics), ([], {}, [], []))
        for m in ("No dataset specified", "No hyperparameters found", "Random seed not reported",
                  "No numerical results found to compare against", "Evaluation metrics not reported"):
            self.assertIn(m, s.missing_info)

    def test_table_header_word_is_never_a_heading_even_without_other_headings(self):
        s = spec_of([["A Study of Imbalanced Learning Methods", "A. Author", "Dataset", "Accuracy", "F1", "Wine", "0.9", "0.8", "Iris", "0.95", "0.9"]])
        self.assertEqual(s.datasets, ["Wine", "Iris"])
        self.assertNotIn("Accuracy", s.datasets)

    def test_percent_values_are_converted(self):
        s = spec_of([["A Study of Imbalanced Learning Methods", "A. Author", "Results", "Table 1.", "Dataset", "Accuracy", "F1", "Wine", "91.5", "90.0", "Iris", "95.0", "94.0"]])
        acc = [r for r in s.reported_results if r.metric == "accuracy"]
        self.assertAlmostEqual(acc[0].value, 0.915)
        self.assertEqual(acc[0].note, "converted from percent")


class TestFieldValidators(unittest.TestCase):
    def test_metric_labels_recognised(self):
        for t in ("Youden's J", "Macro Rec.", "G-Mean", "macro F1-score", "AUC", "Balanced Accuracy", "MCC"):
            self.assertTrue(is_metric_label(t), t)
        for t in ("Wine", "Iris dataset", "KDD Cup 99", "UNSW-NB15", "CIFAR-10"):
            self.assertFalse(is_metric_label(t), t)

    def test_dataset_entry_problems(self):
        self.assertIsNone(dataset_entry_problem("UNSW-NB15"))
        self.assertIsNone(dataset_entry_problem("Iris dataset (150 samples, 4 features, 3 classes)"))
        for bad in ("Youden's J", "Macro Rec.", "0.93", "Accuracy, Macro Rec., G-Mean", "Dataset",
                    "We evaluate our method on a wide variety of datasets drawn from many different public repositories online"):
            self.assertIsNotNone(dataset_entry_problem(bad), bad)

    def test_sanitize_moves_metrics_out_of_datasets(self):
        spec = ResearchSpec(title="T", datasets=["Youden's J", "Macro Rec.", "G-Mean", "Wine", "0.91"])
        sanitize_spec(spec)
        self.assertEqual(spec.datasets, ["Wine"])
        self.assertEqual({m.rstrip(".") for m in spec.metrics}, {"youden's j", "macro rec", "g-mean"})
        self.assertTrue(spec.validation_issues)

    def test_user_edits_are_validated_strictly(self):
        with self.assertRaises(SchemaError) as cm:
            validate_user_edits({"datasets": ["Wine", "Youden's J"]})
        self.assertIn("metric", str(cm.exception).lower())
        with self.assertRaises(SchemaError): validate_user_edits({"datasets": ["0.93"]})
        with self.assertRaises(SchemaError): validate_user_edits({"hyperparameters": {"G-Mean": 0.9}})
        with self.assertRaises(SchemaError): validate_user_edits({"hyperparameters": ["a"]})
        with self.assertRaises(SchemaError): validate_user_edits({"metrics": ["Wine"]})
        with self.assertRaises(SchemaError): validate_user_edits({"reported_results": [{"metric": "Wine", "value": 1}]})
        with self.assertRaises(SchemaError): validate_user_edits({"reported_results": [{"metric": "accuracy", "value": "x"}]}) \
            if False else validate_user_edits({"reported_results": [{"metric": "accuracy"}]})
        with self.assertRaises(SchemaError): validate_user_edits({"security_flags": []})
        validate_user_edits({"datasets": ["Wine"], "hyperparameters": {"lr": 0.1}, "metrics": ["G-Mean"],
                             "reported_results": [{"metric": "G-Mean", "value": 0.9, "dataset": "Wine"}]})


class TestGroundingOfLLMOutput(unittest.TestCase):
    PAGES = ["Margin Boosting\nWe use the Wine dataset. The learning rate is 0.05 and seed 7.\nMacro Rec. reaches 0.93 on Wine."]

    class P(StubProvider):
        pass

    def run_llm(self, d):
        return ingest.build_spec(self.PAGES, StubProvider([d]))

    def test_invented_values_are_dropped_and_reported(self):
        s = self.run_llm({"title": "Margin Boosting", "datasets": ["Wine", "ImageNet"], "metrics": ["Macro Rec.", "BLEU"],
                          "hyperparameters": {"learning_rate": 0.05, "batch_size": 512, "random_seed": 7},
                          "reported_results": [{"metric": "Macro Rec.", "value": 0.93, "dataset": "Wine"},
                                               {"metric": "Macro Rec.", "value": 0.99, "dataset": "Wine"}]})
        self.assertEqual(s.datasets, ["Wine"])
        self.assertEqual(s.hyperparameters, {"learning_rate": 0.05, "random_seed": 7})
        self.assertEqual([r.value for r in s.reported_results], [0.93])
        self.assertEqual([m.lower() for m in s.metrics], ["macro rec."])
        self.assertEqual(len([i for i in s.validation_issues if "not grounded" in i]), 4)

    def test_llm_metric_names_in_datasets_are_rehomed(self):
        s = self.run_llm({"title": "Margin Boosting", "datasets": ["Macro Rec.", "Wine"]})
        self.assertEqual(s.datasets, ["Wine"])
        self.assertTrue(any(m.startswith("macro rec") for m in s.metrics))

    def test_evidence_is_attached_from_the_paper_text_not_from_the_model(self):
        s = self.run_llm({"title": "Margin Boosting", "datasets": ["Wine"], "hyperparameters": {"learning_rate": 0.05},
                          "evidence": [{"field": "datasets", "key": "Wine", "excerpt": "MADE UP QUOTE", "page": 9}]})
        quotes = [e["excerpt"] for e in s.evidence]
        self.assertNotIn("MADE UP QUOTE", quotes)
        self.assertTrue(any("learning rate is 0.05" in q for q in quotes))
        self.assertTrue(all(e["label"] in ("explicit", "derived") for e in s.evidence))

    def test_percent_and_string_numbers_still_normalised(self):
        s = self.run_llm({"title": "Margin Boosting", "reported_results": [{"metric": "Macro Rec.", "value": "93%", "dataset": "Wine"}]})
        self.assertAlmostEqual(s.reported_results[0].value, 0.93)


class TestMissingInfoIsNotStale(unittest.TestCase):
    def test_recomputed_after_edit(self):
        s = spec_of([["T", "A", "Abstract", "Hello."]])
        self.assertIn("No dataset specified", s.missing_info)
        new, log = apply_user_edits(s.to_dict(), {"datasets": ["Iris"], "hyperparameters": {"random_seed": 1}})
        self.assertNotIn("No dataset specified", new.missing_info)
        self.assertNotIn("Random seed not reported", new.missing_info)
        self.assertEqual(new.provenance["datasets"], "user_provided")
        self.assertEqual(new.provenance["metrics"], "not_reported")
        self.assertEqual({e["field"] for e in log}, {"datasets", "hyperparameters"})


if __name__ == "__main__":
    unittest.main()
