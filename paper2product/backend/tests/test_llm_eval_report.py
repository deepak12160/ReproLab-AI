"""LLM robustness (mocked - no paid calls), safe demo fallback, independent metrics, comparison validity, honest reports."""
import math
import sys
import types
import unittest
from unittest import mock

from app import evaluation as ev, codegen
from app.config import Settings
from app.graph import S
from app.llm.provider import (GeminiProvider, LLMProvider, LLMOutputError, LLMRateLimitError, LLMTimeoutError, LLMConfigError,
                              get_provider, CURRENT_PROJECT)
from app.llm import prompts
from app.schemas import Budget, ReportedResult as R, SchemaError
from app.services import Service
from .helpers import make_engine, sample_pdf, StubProvider, tmp_settings, make_pdf, IMBALANCED_PAPER


class Err(Exception):
    def __init__(self, code, msg="boom"): super().__init__(msg); self.code = code


class Resp:
    def __init__(self, text, um=None): self.text, self.usage_metadata = text, um


def gemini(behaviour, **kw):
    google, genai = types.ModuleType("google"), types.ModuleType("google.genai")
    genai.Client = lambda **k: object(); google.genai = genai
    with mock.patch.dict(sys.modules, {"google": google, "google.genai": genai}):
        p = GeminiProvider(Settings(llm_api_key="AIza" + "x" * 30, llm_model="m-1", llm_max_retries=1, **kw))
    p._call_model = lambda model, prompt, system: behaviour(model)
    return p


class TestProviderRobustness(unittest.TestCase):
    def test_llm_codegen_merges_repeated_chunk_files(self):
        class RepeatProvider(LLMProvider):
            name = "repeat-test"
            is_mock = False
            def __init__(self):
                super().__init__()
                self.calls = 0
            def generate_json(self, prompt, system=""):
                self.calls += 1
                if self.calls == 1:
                    return {"files": {"gfb_kan/__init__.py": "", "gfb_kan/model.py": "MODEL_V1\n"}}
                if self.calls == 2:
                    return {"files": {"gfb_kan/__init__.py": "DIFFERENT\n", "gfb_kan/model.py": "MODEL_V2\n",
                                      "run_experiment.py": "print('run')\n"}}
                if self.calls == 3:
                    return {"files": {"tests/test_model.py": "import unittest\n"}}
                return {"files": {"README.md": "# Reproduction\n", "config.json": "{}\n",
                                  "requirements.txt": "numpy\n"}}

        spec = __import__("app.schemas", fromlist=["ResearchSpec"]).ResearchSpec(
            title="Test", method="test method", datasets=["iris"], metrics=["accuracy"])
        from app.schemas import Plan, PlanTask
        plan = Plan(template="llm_generic", approach="test", tasks=[PlanTask("T1", "Run")],
                    files=["run_experiment.py"], packages=["numpy"], dataset_requirements=["iris"],
                    test_strategy="unit tests", metrics=["accuracy"], expected_outputs=[],
                    compute="CPU", risks=[], assumptions=[], open_questions=[]).validate()
        files = codegen.generate_llm_project(spec, plan, RepeatProvider())
        self.assertEqual(files["gfb_kan/model.py"], "MODEL_V1\n")
        self.assertIn("run_experiment.py", files)
        self.assertIn("tests/test_model.py", files)

    def test_extract_json_repairs_literal_control_characters_in_strings(self):
        from app.llm.provider import extract_json
        raw = '{"files":{"run.py":"import sys\nprint(\'ok\')\t# done"}}'
        obj = extract_json(raw)
        self.assertIn("\n", obj["files"]["run.py"])
        self.assertIn("\t", obj["files"]["run.py"])

    def test_llm_codegen_is_split_into_small_calls(self):
        class ChunkProvider(LLMProvider):
            name = "chunk-test"
            is_mock = False
            def __init__(self):
                super().__init__()
                self.calls = []
            def generate_json(self, prompt, system=""):
                self.calls.append(prompt)
                if len(self.calls) == 1:
                    return {"files": {"pkg/model.py": "import numpy as np\n"}}
                if len(self.calls) == 2:
                    return {"files": {"run_experiment.py": "print('run')\n"}}
                if len(self.calls) == 3:
                    return {"files": {"tests/test_model.py": "import unittest\n"}}
                return {"files": {"README.md": "# Reproduction\n", "config.json": "{}\n",
                                  "requirements.txt": "numpy\n", "experiment_metadata.json": "{}\n"}}

        spec = __import__("app.schemas", fromlist=["ResearchSpec"]).ResearchSpec(
            title="Test", method="test method", datasets=["iris"], metrics=["accuracy"])
        from app.schemas import Plan, PlanTask
        plan = Plan(template="llm_generic", approach="test", tasks=[PlanTask("T1", "Run")],
                    files=["run_experiment.py"], packages=["numpy"], dataset_requirements=["iris"],
                    test_strategy="unit tests", metrics=["accuracy"], expected_outputs=[],
                    compute="CPU", risks=[], assumptions=[], open_questions=[]).validate()
        provider = ChunkProvider()
        progress = []
        files = codegen.generate_llm_project(spec, plan, provider, progress=progress.append)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(len(progress), 8)
        self.assertIn("pkg/model.py", files)
        self.assertIn("tests/test_model.py", files)

    def test_quota_exhaustion_fails_fast_without_trying_fallback_models(self):
        calls = []
        def exhausted(model):
            calls.append(model)
            raise Err(429, "You exceeded your current quota; check your plan and billing details")
        p = gemini(exhausted)
        with mock.patch("app.llm.provider.time.sleep"), self.assertRaises(LLMRateLimitError) as cm:
            p.generate_json("x")
        self.assertEqual(len(calls), 1)
        self.assertIn("quota is exhausted", str(cm.exception))

    def test_codegen_timeout_override_is_temporary(self):
        p = gemini(lambda m: Resp('{"ok": true}'), llm_timeout_s=30)
        with p.request_timeout(90, retries=0, fallback=False):
            self.assertEqual(p._timeout, 90)
            self.assertEqual(p._retries, 0)
            self.assertFalse(p._allow_fallback)
        self.assertEqual(p._timeout, 30)
        self.assertTrue(p._allow_fallback)

    def test_codegen_timeout_mode_makes_one_model_attempt(self):
        calls = []
        def unavailable(model):
            calls.append(model)
            raise Err(504, "deadline exceeded")
        p = gemini(unavailable)
        with p.request_timeout(90, retries=0, fallback=False):
            with mock.patch("app.llm.provider.time.sleep"), self.assertRaises(LLMTimeoutError):
                p.generate_json("x")
        self.assertEqual(len(calls), 1)

    def test_missing_key_modes(self):
        self.assertIsNone(get_provider(Settings(llm_provider="auto")))                 # -> DEMO MODE
        with self.assertRaises(LLMConfigError): get_provider(Settings(llm_provider="gemini"))

    def test_rate_limit_exhausted_gives_typed_clear_error(self):
        p = gemini(lambda m: (_ for _ in ()).throw(Err(429, "RESOURCE_EXHAUSTED quota")))
        with mock.patch("app.llm.provider.time.sleep"), self.assertRaises(LLMRateLimitError) as cm:
            p.generate_json("x")
        self.assertIn("rate limit", str(cm.exception).lower())

    def test_timeout_gives_typed_clear_error(self):
        p = gemini(lambda m: (_ for _ in ()).throw(TimeoutError("request timed out")))
        with mock.patch("app.llm.provider.time.sleep"), self.assertRaises(LLMTimeoutError):
            p.generate_json("x")

    def test_retries_are_bounded(self):
        calls = []
        def b(m): calls.append(m); raise Err(503, "unavailable")
        p = gemini(b)
        with mock.patch("app.llm.provider.time.sleep"), self.assertRaises(Exception):
            p.generate_json("x")
        self.assertLessEqual(len(calls), 4)

    def test_malformed_json_is_an_error_never_a_success(self):
        p = gemini(lambda m: Resp("I am sorry, here is some prose"))
        with mock.patch("app.llm.provider.time.sleep"), self.assertRaises(LLMOutputError):
            p.generate_json("x")

    def test_validated_generation_retries_once_with_the_error_then_gives_up(self):
        class P(LLMProvider):
            name = "t"
            def __init__(self): super().__init__(); self.prompts = []
            def generate_json(self, prompt, system=""):
                self.prompts.append(prompt); return {"bad": True}
        p = P()
        def validate(d): raise SchemaError("needs 'tasks'")
        with self.assertRaises(LLMOutputError): p.generate_validated("do it", validate)
        self.assertEqual(len(p.prompts), 2); self.assertIn("rejected", p.prompts[1]); self.assertIn("needs 'tasks'", p.prompts[1])

    def test_usage_is_tracked_per_project(self):
        um = types.SimpleNamespace(prompt_token_count=100, candidates_token_count=20)
        p = gemini(lambda m: Resp('{"ok": true}', um), llm_price_per_1k_tokens=0.5)
        CURRENT_PROJECT.set("proj_a"); p.generate_json("x")
        CURRENT_PROJECT.set("proj_b"); p.generate_json("x"); p.generate_json("x")
        self.assertEqual(p.usage_for("proj_a").total, 120); self.assertEqual(p.usage_for("proj_b").total, 240)
        self.assertEqual(p.usage.total, 360); self.assertAlmostEqual(p.usage_for("proj_a").est_cost_usd, 0.06)
        CURRENT_PROJECT.set("")

    def test_key_never_appears_in_errors(self):
        p = gemini(lambda m: (_ for _ in ()).throw(Err(500, "failure for key " + "AIza" + "x" * 30)))
        with mock.patch("app.llm.provider.time.sleep"), self.assertRaises(Exception) as cm:
            p.generate_json("x")
        self.assertNotIn("AIza", str(cm.exception))

    def test_prompts_wrap_untrusted_text_and_select_relevant_pages(self):
        pages = ["Title page"] + [f"filler {i} " * 400 for i in range(30)] + ["The dataset and learning rate are listed here."] + ["References\n[1] x"]
        body = prompts.select_relevant_text(pages, max_chars=6000)
        self.assertLessEqual(len(body), 6000); self.assertIn("[PAGE 1]", body); self.assertIn("learning rate", body)
        self.assertNotIn("[1] x", body)
        self.assertIn("<untrusted_document>", prompts.extraction_prompt(["x"]))


class TestModeAndFallback(unittest.TestCase):
    def test_demo_mode_is_labelled_and_needs_no_key(self):
        sv = Service(tmp_settings(), sync_jobs=True, provider=None)
        i = sv.integrations()["llm"]
        self.assertEqual((i["configured"], i["mode"]), (False, "demo")); self.assertIn("DEMO MODE", i["label"])
        self.assertFalse(sv.test_llm()["ok"])

    def test_llm_mode_label_names_provider_and_model(self):
        sv = Service(tmp_settings(), sync_jobs=True, provider=StubProvider([]))
        self.assertIn("LLM MODE", sv.integrations()["llm"]["label"])

    def test_failed_llm_extraction_falls_back_visibly_not_silently(self):
        class Broken(StubProvider):
            def generate_json(self, prompt, system=""): raise LLMOutputError("model returned invalid JSON")
        e, s, db = make_engine(provider=Broken([]))
        pid = e.create("x", sample_pdf(), Budget()); e.run_ingest(pid)
        st = e.load(pid)
        self.assertIn("LLM extraction failed", st.spec["extraction_method"])
        self.assertTrue(st.degraded)
        self.assertIn("LLM extraction failed", " ".join(st.spec["missing_info"]))
        e.approve_spec(pid, None); e.run_plan(pid); e.approve_plan(pid, True); e.run_execution(pid)
        st = e.load(pid)
        self.assertIn("Degraded step", st.report_md)
        self.assertIn("DEGRADED", st.manifest["honesty"]["note"])

    def test_llm_failure_at_planning_stops_the_run_with_a_message(self):
        class FailPlan(StubProvider):
            def generate_json(self, prompt, system=""):
                if "implementation plan" in prompt: raise LLMRateLimitError("rate limit")
                raise LLMOutputError("x")
        e, s, db = make_engine(provider=FailPlan([]))
        pid = e.create("x", make_pdf(IMBALANCED_PAPER), Budget()); e.run_ingest(pid); e.approve_spec(pid, None); e.run_plan(pid)
        self.assertEqual(e.status(pid), S.FAILED); self.assertIn("rate limit", e.load(pid).blocked_reason)


class TestIndependentMetrics(unittest.TestCase):
    def test_binary_known_values(self):
        yt = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]; yp = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        m = ev.classification_metrics(yt, yp)
        self.assertAlmostEqual(m["per_class"]["1"]["recall"], 0.75)
        self.assertAlmostEqual(m["per_class"]["1"]["specificity"], 4 / 6)
        self.assertAlmostEqual(m["youden_j"], 0.75 + 4 / 6 - 1)
        self.assertAlmostEqual(m["g_mean"], math.sqrt(0.75 * 4 / 6))
        self.assertAlmostEqual(m["balanced_accuracy"], (0.75 + 4 / 6) / 2)
        tp, tn, fp, fn = 3, 4, 2, 1
        self.assertAlmostEqual(m["mcc"], (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    def test_multiclass_averagings(self):
        m = ev.classification_metrics([0, 0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 1, 0])
        self.assertAlmostEqual(m["micro_f1"], m["accuracy"])
        self.assertNotAlmostEqual(m["weighted_recall"], m["macro_recall"])
        self.assertEqual(m["g_mean"], 0.0)                          # class 2 never predicted correctly

    def test_regression_has_mse(self):
        self.assertAlmostEqual(ev.regression_metrics([1, 2, 3], [1, 2, 4])["mse"], 1 / 3)


class TestComparisonValidity(unittest.TestCase):
    ctx = {"dataset_name": "wine", "test_size": 0.3}
    meas = ev.measured_values(ev.classification_metrics([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 1]))

    def test_paper_metric_labels_map_to_measured_values(self):
        out = ev.compare_results([R("youden's j", self.meas["youden_j"], "Wine", "70/30"), R("macro rec", self.meas["recall"], "Wine", "70/30"),
                                  R("g-mean", self.meas["g_mean"], "Wine", "70/30")], self.meas, self.ctx)
        self.assertTrue(all(c.status == "reproduced" and c.comparable for c in out))
        self.assertEqual({c.direction for c in out}, {"higher_is_better"})
        self.assertIn("one-vs-rest", " ".join(out[0].warnings))      # multi-class J definition caveat

    def test_unstated_averaging_is_flagged(self):
        c = ev.compare_results([R("recall", self.meas["recall"], "Wine", "70/30")], self.meas, self.ctx)[0]
        self.assertTrue(any("averaging" in w for w in c.warnings))

    def test_micro_and_weighted_are_compared_with_the_right_average(self):
        c = ev.compare_results([R("weighted f1", self.meas["weighted_f1"], "Wine", "70/30")], self.meas, self.ctx)[0]
        self.assertEqual((c.metric_key, c.status), ("weighted_f1", "reproduced"))

    def test_baseline_rows_are_never_compared_with_our_run(self):
        c = ev.compare_results([R("accuracy", 0.5, "Wine", "70/30", variant="Random Forest")], self.meas, self.ctx, ["MarginBoost"])[0]
        self.assertFalse(c.comparable); self.assertEqual(c.status, "unable_to_verify")
        ok = ev.compare_results([R("accuracy", self.meas["accuracy"], "Wine", "70/30", variant="MarginBoost")], self.meas, self.ctx, ["MarginBoost"])[0]
        self.assertTrue(ok.comparable)

    def test_better_than_reported_is_flagged_not_celebrated(self):
        c = ev.compare_results([R("accuracy", 0.5, "Wine", "70/30")], self.meas, self.ctx)[0]
        self.assertEqual(c.status, "not_reproduced"); self.assertTrue(any("BETTER" in w for w in c.warnings))

    def test_tolerance_rule_is_explicit(self):
        c = ev.compare_results([R("accuracy", 0.5, "Wine", "70/30")], self.meas, self.ctx)[0]
        self.assertIn("2%", c.rule); self.assertIn("10%", c.rule)

    def test_user_supplied_dataset_is_caveated(self):
        c = ev.compare_results([R("accuracy", 0.8, "Wine", "70/30")], self.meas, {**self.ctx, "user_dataset": True})[0]
        self.assertTrue(any("user-supplied" in w for w in c.warnings))


class TestReportHonesty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .test_services import svc, run_all
        cls.sv, _ = svc(); cls.pid = cls.sv.create_demo(); run_all(cls.sv, cls.pid)
        cls.md = cls.sv.report_md(cls.pid); cls.m = cls.sv.manifest(cls.pid)

    def test_headings_render_and_no_literal_backslash_n(self):
        self.assertNotIn("\\n##", self.md)
        for h in ("## 1. Paper", "## 3. Evidence", "## 4. User corrections and approvals", "## 6. Dataset provenance",
                  "## 7. Environment, seeds and hardware", "## 11. Comparison with reported results",
                  "## 13. What is verified, claimed, assumed and unverified"):
            self.assertIn(h, self.md)

    def test_demo_fixture_is_prominently_labelled(self):
        self.assertIn("SYNTHETIC DEMO FIXTURE", self.md); self.assertTrue(self.m["is_synthetic_fixture"])
        self.assertIn("DEMO MODE", self.md)

    def test_paper_claims_and_measured_values_are_separated(self):
        self.assertIn("Claimed by the paper (not verified by this system)", self.md)
        self.assertIn("Verified by this system", self.md)
        self.assertIn("Reported (page)", self.md)

    def test_status_comes_from_comparison_not_from_a_clean_run(self):
        self.assertEqual(self.m["reproduction_status"], "partially_reproduced")      # 0.9556 vs 0.98 -> 2.5% rel diff
        self.assertIn("Partially reproduced", self.md)

    def test_blocked_run_report_says_unable_to_verify(self):
        e, s, db = make_engine()
        pid = e.create("x", sample_pdf(), Budget()); e.run_ingest(pid)
        e.approve_spec(pid, {"method": "A transformer.", "datasets": ["mnist"]}); e.run_plan(pid)
        st = e.load(pid)
        self.assertEqual(st.reproduction_status, "unable_to_verify")


if __name__ == "__main__":
    unittest.main()
