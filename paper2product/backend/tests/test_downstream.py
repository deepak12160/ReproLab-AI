"""User corrections must be persisted and used by every later stage; dataset blockers/substitutions need approval."""
import csv
import json
import os
import unittest

from app.graph import S, GateError
from app.schemas import Budget, SchemaError
from app.integrations import datasets as ds
from app.services import Service
from .helpers import make_engine, sample_pdf, make_pdf, StubProvider, tmp_settings, drive


def fixture_engine(**kw):
    e, s, db = make_engine(**kw)
    pid = e.create("fixture", sample_pdf(), Budget(), is_fixture=True)
    e.run_ingest(pid)
    return e, pid


class TestEditsAreUsedDownstream(unittest.TestCase):
    def test_edited_seed_reaches_config_run_meta_manifest_and_report(self):
        e, pid = fixture_engine()
        e.approve_spec(pid, {"hyperparameters": {"test_size": 0.3, "random_seed": 7, "regularization": 1e-9}})
        e.run_plan(pid); e.approve_plan(pid, True); e.run_execution(pid)
        st = e.load(pid)
        self.assertEqual(e.status(pid), S.COMPLETED, st.blocked_reason)
        self.assertEqual(json.loads(st.files["config.json"])["seed"], 7)
        self.assertEqual(st.run_meta["seed"], 7)
        self.assertEqual(st.manifest["seed"], 7)
        self.assertEqual(st.manifest["user_spec_edits"][0]["field"], "hyperparameters")
        self.assertIn("Edited **hyperparameters**", st.report_md)
        self.assertEqual(st.spec["provenance"]["hyperparameters"], "user_provided")
        self.assertEqual(st.spec_original["hyperparameters"]["random_seed"], 42)       # parser output kept for audit

    def test_edited_reported_result_is_what_gets_compared(self):
        e, pid = fixture_engine()
        e.approve_spec(pid, {"reported_results": [{"metric": "accuracy", "value": 0.9556, "dataset": "iris", "split": "70/30", "page": 2}]})
        e.run_plan(pid); e.approve_plan(pid, True); e.run_execution(pid)
        c = e.load(pid).comparisons[0]
        self.assertEqual(c["reported"], 0.9556)
        self.assertEqual(c["status"], "reproduced")

    def test_user_edit_not_overwritten_by_stale_parser_output(self):
        e, pid = fixture_engine()
        e.approve_spec(pid, {"method": "Fisher linear discriminant analysis with a custom prior.", "datasets": ["Iris"]})
        e.run_plan(pid)
        st = e.load(pid)
        self.assertEqual(st.spec["method"], "Fisher linear discriminant analysis with a custom prior.")
        self.assertEqual(st.spec["datasets"], ["Iris"])
        self.assertNotIn("No dataset specified", st.plan["assumptions"])
        self.assertFalse(any("No dataset" in a for a in st.plan["assumptions"]))

    def test_invalid_edit_is_rejected_and_nothing_changes(self):
        e, pid = fixture_engine()
        before = e.load(pid).spec
        with self.assertRaises(SchemaError):
            e.approve_spec(pid, {"datasets": ["Youden's J", "Macro Rec.", "G-Mean"]})
        self.assertEqual(e.status(pid), S.AWAITING_SPEC)
        self.assertEqual(e.load(pid).spec, before)

    def test_plan_is_bound_to_the_approved_spec(self):
        e, pid = fixture_engine()
        e.approve_spec(pid, None); e.run_plan(pid); e.approve_plan(pid, True)
        st = e.load(pid)
        self.assertEqual(st.plan["spec_sha256"], st.spec_sha256)
        st.spec["hyperparameters"]["random_seed"] = 999          # spec changes after the plan was approved
        e.save(st)
        e.run_execution(pid)
        self.assertEqual(e.status(pid), S.BLOCKED)
        self.assertIn("plan is stale", e.load(pid).blocked_reason)


def _kdd_paper():
    return make_pdf([["KDD Cup 99 Intrusion Detection with Boosting", "A. Author", "Abstract", "We study intrusion detection.",
                      "3. Methodology", "A boosted classifier is trained.", "4. Datasets",
                      "We use the KDD Cup 99 dataset.", "Hyperparameters", "- random_seed: 1"]])


def _llm_spec():
    return {"title": "KDD Cup 99 Intrusion Detection with Boosting", "method": "A boosted classifier is trained.",
            "datasets": ["KDD Cup 99"], "hyperparameters": {"random_seed": 1}, "metrics": [], "reported_results": [],
            "task_type": "classification"}


def _llm_plan():
    return {"approach": "boost", "tasks": [{"id": "T1", "title": "t", "depends_on": [], "files": ["run_experiment.py"]}],
            "files": ["run_experiment.py"], "packages": ["numpy"], "dataset_requirements": [], "test_strategy": "x",
            "metrics": ["accuracy"], "expected_outputs": [], "compute": "cpu", "risks": [], "assumptions": [], "open_questions": []}


class TestDatasetResolution(unittest.TestCase):
    def planned(self, hint=""):
        e, s, db = make_engine(provider=StubProvider([_llm_spec(), _llm_plan()]))
        pid = e.create("kdd", _kdd_paper(), Budget(), hint)
        e.run_ingest(pid); e.approve_spec(pid, None); e.run_plan(pid)
        return e, pid

    def test_unavailable_dataset_is_a_plan_blocker_not_a_silent_substitution(self):
        e, pid = self.planned()
        plan = e.load(pid).plan
        self.assertTrue(any("KDD Cup 99" in b for b in plan["blockers"]))
        self.assertEqual(plan["reproduction_type"], "partial")
        with self.assertRaises(GateError) as cm:
            e.approve_plan(pid, True)
        self.assertIn("blockers", str(cm.exception))

    def test_user_chosen_substitute_becomes_an_assumption_that_needs_approval(self):
        e, pid = self.planned()
        plan = e.set_dataset(pid, "sklearn:wine")
        self.assertEqual(plan["blockers"], [])
        self.assertTrue(any("SUBSTITUTION" in a and "KDD Cup 99" in a for a in plan["assumptions"]))
        with self.assertRaises(GateError):
            e.approve_plan(pid, False)                      # assumptions not accepted
        e.approve_plan(pid, True)
        self.assertEqual(e.load(pid).dataset_hint, "sklearn:wine")

    def test_dataset_can_only_be_chosen_at_the_plan_gate_and_must_exist(self):
        e, pid = self.planned()
        with self.assertRaises(GateError): e.set_dataset(pid, "sklearn:nope")
        with self.assertRaises(GateError): e.set_dataset(pid, "hf:some/dataset")        # metadata-only -> not usable

    def test_resolution_metadata_is_recorded(self):
        info = ds.resolve_for_spec(["Iris dataset"])
        for f in ("version", "license", "url", "split", "expected_files", "access"):
            self.assertTrue(getattr(info, f), f)
        self.assertFalse(info.differs_from_paper)
        self.assertTrue(ds.resolve_for_spec(["MNIST"], "sklearn:wine").differs_from_paper)

    def test_upload_names_cannot_traverse(self):
        for bad in ("../x.csv", "/etc/passwd.csv", "a/b.csv", "x.exe", ".hidden.csv", "a b.csv"):
            with self.assertRaises(ds.ResourceError, msg=bad): ds.safe_upload_name(bad)
        self.assertEqual(ds.safe_upload_name("glass.csv"), "glass.csv")
        with self.assertRaises(ds.ResourceError): ds.resolve_for_spec([], "upload:../../etc/x.csv", upload_dir="/tmp")


UPLOAD_PROJECT = {
    "README.md": "# upload project\n", "requirements.txt": "numpy\n", "config.json": '{"seed": 3, "test_size": 0.3}\n',
    "clf/__init__.py": "",
    "run_experiment.py": (
        "import csv, json, os\nimport numpy as np\n"
        "cfg = json.load(open('config.json', encoding='utf-8'))\n"
        "rows = list(csv.reader(open('data/points.csv', encoding='utf-8')))[1:]\n"
        "X = np.array([[float(v) for v in r[:-1]] for r in rows]); y = np.array([int(r[-1]) for r in rows])\n"
        "rng = np.random.default_rng(cfg['seed']); te = []\ntr = []\n"
        "for c in np.unique(y):\n    idx = np.where(y == c)[0]; rng.shuffle(idx); n = int(round(len(idx) * cfg['test_size']))\n"
        "    te += idx[:n].tolist(); tr += idx[n:].tolist()\n"
        "cent = {c: X[tr][y[tr] == c].mean(axis=0) for c in np.unique(y[tr])}\n"
        "pred = np.array([min(cent, key=lambda c: np.linalg.norm(x - cent[c])) for x in X[te]])\n"
        "os.makedirs('artifacts', exist_ok=True)\n"
        "json.dump({'y_true': y[te].tolist(), 'y_pred': pred.tolist(), 'labels': [0, 1]}, open('artifacts/predictions.json', 'w', encoding='utf-8'))\n"
        "json.dump({'seed': cfg['seed'], 'test_size': cfg['test_size'], 'dataset': 'points.csv'}, open('artifacts/run_meta.json', 'w', encoding='utf-8'))\n"),
    "tests/__init__.py": "",
    "tests/test_basic.py": "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
}


class TestUserSuppliedDataset(unittest.TestCase):
    def test_uploaded_csv_flows_into_sandbox_run_and_is_labelled_user_supplied(self):
        plan = _llm_plan(); plan["files"] = sorted(UPLOAD_PROJECT)
        e, s, db = make_engine(provider=StubProvider([_llm_spec(), plan, {"files": UPLOAD_PROJECT}]))
        pid = e.create("kdd", _kdd_paper(), Budget())
        e.run_ingest(pid); e.approve_spec(pid, None); e.run_plan(pid)
        os.makedirs(e.upload_dir(pid), exist_ok=True)
        with open(os.path.join(e.upload_dir(pid), "points.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["a", "b", "label"])
            for i in range(60):
                w.writerow([i % 2 * 5 + (i % 7) * .1, i % 2 * 5 + (i % 5) * .1, i % 2])
        e.set_dataset(pid, "upload:points.csv")
        e.approve_plan(pid, True); e.run_execution(pid)
        st = e.load(pid)
        self.assertEqual(e.status(pid), S.COMPLETED, st.blocked_reason)
        self.assertTrue(st.dataset["user_supplied"])
        self.assertTrue(st.dataset["differs_from_paper"])
        self.assertIn("DIFFERS FROM THE PAPER", st.report_md)
        self.assertGreater(st.metrics["accuracy"], 0.9)


class TestServiceDatasetApi(unittest.TestCase):
    def test_service_rejects_bad_sources_and_names(self):
        sv = Service(tmp_settings(), sync_jobs=True, provider=None)
        pid = sv.create_demo()
        with self.assertRaises(SchemaError): sv.set_dataset(pid, "ftp://x")
        with self.assertRaises(SchemaError): sv.upload_dataset(pid, "evil.exe", b"a,b\n1,2\n")


if __name__ == "__main__":
    unittest.main()
