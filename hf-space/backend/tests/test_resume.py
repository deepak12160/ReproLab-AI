import os
import tempfile
import unittest

from app.graph import S
from app.sandbox import LocalSandbox
from app.services import Service
from app.config import Settings
from app.graph import GateError


class TestCodegenResume(unittest.TestCase):
    def make_engine_state(self):
        td = tempfile.mkdtemp(prefix="p2p_resume_test_")
        settings = Settings(data_dir=td, db_path=os.path.join(td, "t.db"), llm_provider="demo")
        svc = Service(settings, sync_jobs=False, sandbox=LocalSandbox())
        pid = svc.engine.db.create_project("resume", {})
        st = svc.engine.load(pid)
        st.codegen_partial = {"pkg/model.py": "import numpy as np\n"}
        st.codegen_missing = ["unit tests"]
        st.blocked_reason = "quota"
        svc.engine.save(st)
        svc.db.set_status(pid, S.FAILED)
        return svc, pid

    def test_failed_project_exposes_partial_resume_transition(self):
        svc, pid = self.make_engine_state()
        svc.engine.resume(pid)
        self.assertIn(svc.engine.status(pid), (S.RESOLVING, S.GENERATING))
        st = svc.engine.load(pid)
        self.assertTrue(st.codegen_resumed)

    def test_completed_project_cannot_resume(self):
        svc, pid = self.make_engine_state()
        svc.db.set_status(pid, S.COMPLETED)
        with self.assertRaises(GateError):
            svc.resume(pid)


if __name__ == "__main__":
    unittest.main()
