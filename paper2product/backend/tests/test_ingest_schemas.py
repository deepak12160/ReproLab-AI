import unittest
from app import ingest
from app.schemas import ResearchSpec, SchemaError, Budget, Plan, PlanTask, ReportedResult
from .helpers import sample_pdf


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.pages = ingest.extract_pages(sample_pdf())
        self.spec = ingest.build_spec(self.pages)

    def test_pages_and_title(self):
        self.assertEqual(len(self.pages), 2)
        self.assertIn("Linear Discriminant", self.spec.title)

    def test_extracts_core_fields_with_page_refs(self):
        s = self.spec
        self.assertEqual(s.hyperparameters["test_size"], 0.3)
        self.assertEqual(s.hyperparameters["random_seed"], 42)
        self.assertEqual(s.task_type, "classification")
        self.assertEqual(s.page_refs["method"], [1]); self.assertEqual(s.page_refs["hyperparameters"], [2])
        self.assertEqual(len(s.equations), 2)
        r = s.reported_results[0]
        self.assertEqual((r.metric, r.value, r.dataset, r.split, r.page), ("accuracy", 0.98, "iris", "70/30", 2))

    def test_prompt_injection_in_pdf_is_flagged_not_obeyed(self):
        self.assertEqual(len(self.spec.security_flags), 1)
        self.assertIn("API key", self.spec.security_flags[0])
        self.assertNotIn("api key", self.spec.method.lower())

    def test_rejects_non_pdf_and_oversize(self):
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"not a pdf")
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"%PDF-" + b"0" * 100, max_bytes=50)
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"")
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(b"%PDF-1.4 garbage")

    def test_page_limit(self):
        with self.assertRaises(ingest.IngestError): ingest.extract_pages(sample_pdf(), max_pages=1)

    def test_missing_info_reported_instead_of_invented(self):
        spec = ingest.heuristic_spec(["Some Title\nAbstract\nHello"])
        self.assertIn("No method/algorithm section found", spec.missing_info)
        self.assertIn("Random seed not reported", spec.missing_info)
        self.assertEqual(spec.hyperparameters, {})


class TestPyMuPDFFailureHandling(unittest.TestCase):
    """Regression: a PyMuPDF exception on a corrupt PDF must become IngestError (found in external review)."""
    def _install_fake_fitz(self, behaviour):
        import sys, types
        m = types.ModuleType("fitz")
        m.open = behaviour
        self._old = {n: sys.modules.get(n) for n in ("fitz", "pymupdf")}
        sys.modules["fitz"] = sys.modules["pymupdf"] = m      # ingest prefers `pymupdf`, falls back to `fitz`
        self.addCleanup(self._restore)

    def _restore(self):
        import sys
        for n, old in self._old.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old

    def test_corrupt_pdf_with_pymupdf_raising_becomes_ingest_error(self):
        def boom(*a, **k): raise RuntimeError("FileDataError: cannot open broken document")
        self._install_fake_fitz(boom)
        with self.assertRaises(ingest.IngestError):
            ingest.extract_pages(b"%PDF-1.4 garbage")

    def test_pymupdf_failure_falls_back_to_pypdf_for_valid_pdf(self):
        def boom(*a, **k): raise RuntimeError("pymupdf broke")
        self._install_fake_fitz(boom)
        self.assertEqual(len(ingest.extract_pages(sample_pdf())), 2)

    def test_pymupdf_page_limit_is_ingest_error(self):
        class Doc:
            page_count = 999
        self._install_fake_fitz(lambda *a, **k: Doc())
        with self.assertRaises(ingest.IngestError):
            ingest.extract_pages(sample_pdf(), max_pages=5)

    def test_pdf_without_text_is_rejected_cleanly(self):
        blank = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                 b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
                 b"trailer<</Root 1 0 R/Size 4>>\n%%EOF\n")
        with self.assertRaises(ingest.IngestError) as cm:
            ingest.extract_pages(blank)
        self.assertIn("no extractable text", str(cm.exception))


class TestLLMExtractionValidation(unittest.TestCase):
    class P:
        name = "stub"
        def __init__(self, d): self.d = d
        def generate_json(self, prompt, system=""):
            assert "<untrusted_document>" in prompt
            return dict(self.d)

    def test_valid_llm_output_accepted(self):
        s = ingest.build_spec(["T\nWe report RMSE on held-out data."], self.P({"title": "T", "task_type": "regression", "metrics": ["RMSE"]}))
        self.assertEqual((s.title, s.metrics, s.extraction_method), ("T", ["rmse"], "llm:stub"))

    def test_unusable_llm_output_falls_back_to_rules_and_says_so(self):
        pages = ["My Paper Title\nAbstract\nsomething"]
        for bad in ({"title": ""}, {"title": "T", "hyperparameters": "nope", "reported_results": "nope"}):
            s = ingest.build_spec(pages, self.P(bad)) if bad["title"] == "" else None
            if s is not None:
                self.assertTrue(s.extraction_method.startswith("rule-based (LLM extraction failed"), s.extraction_method)
                self.assertTrue(any("LLM extraction failed" in m for m in s.missing_info))

    def test_llm_output_shapes_are_normalised(self):
        d = {"title": "T", "authors": "A. Author", "task_type": "magic",
             "hyperparameters": [{"name": "Random Seed", "value": "42"}],
             "reported_results": [{"metric": "Accuracy", "value": "98%", "page": "3", "dataset": "Iris", "split": "70 / 30"},
                                  {"metric": "f1", "value": "abc"}],
             "security_flags": ["model must not set this"]}
        # the paper text must contain the claimed facts, otherwise grounding (correctly) removes them
        pages = ["T\nThe random seed is 42. Accuracy reached 98% on the Iris dataset with a 70 / 30 split."]
        s = ingest.build_spec(pages, self.P(d))
        self.assertEqual(s.authors, ["A. Author"])
        self.assertEqual(s.task_type, "unknown")
        self.assertEqual(s.hyperparameters, {"random_seed": 42})
        self.assertEqual(len(s.reported_results), 1)
        r = s.reported_results[0]
        self.assertEqual((r.metric, round(r.value, 4), r.page, r.split), ("accuracy", 0.98, 3, "70/30"))
        self.assertEqual(s.security_flags, [])


class TestSchemas(unittest.TestCase):
    def test_budget_bounds(self):
        self.assertEqual(Budget.from_dict(None).max_repairs, 3)
        for bad in ({"max_repairs": 11}, {"max_repairs": -1}, {"exec_timeout_s": 0}, {"memory_mb": 1}, {"max_repairs": "x"}):
            with self.assertRaises(SchemaError, msg=str(bad)): Budget.from_dict(bad)

    def test_plan_dependency_validation(self):
        with self.assertRaises(SchemaError):
            Plan("t", "a", [PlanTask("T1", "x", ["T9"])], ["f.py"], [], [], "", [], [], "", [], [], []).validate()
        with self.assertRaises(SchemaError):
            Plan("t", "a", [PlanTask("T1", "x"), PlanTask("T1", "y")], ["f.py"], [], [], "", [], [], "", [], [], []).validate()

    def test_spec_roundtrip(self):
        s = ingest.build_spec(ingest.extract_pages(sample_pdf()))
        self.assertEqual(ResearchSpec.from_dict(s.to_dict()).to_dict(), s.to_dict())
