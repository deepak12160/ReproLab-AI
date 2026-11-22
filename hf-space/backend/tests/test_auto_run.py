import csv
import os
import tempfile
import unittest

from app.config import Settings
from app.services import Service
from app.schemas import SchemaError
from app.sandbox import LocalSandbox


def csv_bytes(rows):
    td = tempfile.mkdtemp()
    path = os.path.join(td, "points.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    with open(path, "rb") as f:
        return f.read()


class TestAutoRun(unittest.TestCase):
    def make_service(self):
        td = tempfile.mkdtemp(prefix="p2p_auto_test_")
        settings = Settings(data_dir=td, db_path=os.path.join(td, "t.db"), llm_provider="demo")
        return Service(settings, sync_jobs=True, sandbox=LocalSandbox())

    def test_default_still_waits_at_spec_gate(self):
        svc = self.make_service()
        pdf = open(os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers",
                                "fisher_lda_iris_fixture.pdf"), "rb").read()
        pid = svc.create_project("demo.pdf", pdf, {})
        self.assertEqual(svc._get(pid)["status"], "awaiting_spec_approval")

    def test_auto_demo_completes_and_audit_records_auto(self):
        svc = self.make_service()
        pdf = open(os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers",
                                "fisher_lda_iris_fixture.pdf"), "rb").read()
        pid = svc.create_project("demo.pdf", pdf, {}, auto_run=True)
        st = svc.engine.load(pid)
        self.assertTrue(st.auto_run)
        self.assertIn(svc._get(pid)["status"], ("completed", "failed", "blocked"))
        audit = svc.db.audit_log(pid)
        actors = [x["actor"] for x in audit]
        self.assertIn("auto", actors)

    def test_valid_attached_csv_is_stored(self):
        svc = self.make_service()
        pdf = open(os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers",
                                "fisher_lda_iris_fixture.pdf"), "rb").read()
        rows = [["x", "y", "label"]]
        rows += [[i, i + 1, i % 2] for i in range(12)]
        raw = csv_bytes(rows)
        pid = svc.create_project("demo.pdf", pdf, {}, dataset_file=("points.csv", raw), auto_run=False)
        st = svc.engine.load(pid)
        self.assertEqual(st.dataset_hint, "upload:points.csv")
        self.assertTrue(os.path.isfile(os.path.join(svc.engine.upload_dir(pid), "points.csv")))

    def test_bad_csv_rejected_before_project_exists(self):
        svc = self.make_service()
        pdf = open(os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers",
                                "fisher_lda_iris_fixture.pdf"), "rb").read()
        before = len(svc.db.list_projects())
        with self.assertRaises(SchemaError):
            svc.create_project("demo.pdf", pdf, {}, dataset_file=("bad.csv", b"x\n1\n"))
        self.assertEqual(len(svc.db.list_projects()), before)

    def test_upload_name_cannot_escape_directory(self):
        svc = self.make_service()
        pdf = open(os.path.join(os.path.dirname(__file__), "..", "..", "sample_papers",
                                "fisher_lda_iris_fixture.pdf"), "rb").read()
        raw = csv_bytes([["x", "label"], [1, 0], [2, 1], [3, 0]])
        with self.assertRaises(SchemaError):
            svc.create_project("demo.pdf", pdf, {}, dataset_file=("../../x.csv", raw))


if __name__ == "__main__":
    unittest.main()
