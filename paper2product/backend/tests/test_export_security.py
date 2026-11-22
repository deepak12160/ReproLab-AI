import io, json, unittest, zipfile
from app.security import sanitize_export
from tests.helpers import make_engine, sample_pdf, drive
from app.schemas import Budget
from app.services import Service
from app.config import Settings

KEY = "AIza" + "Z" * 35


class TestSanitize(unittest.TestCase):
    def test_secret_files_dropped_and_keys_scrubbed(self):
        files = {".env": f"GEMINI_API_KEY={KEY}\n", "sub/.env.local": "X=1", "deploy/id_rsa": "x", "a.pem": "x", ".git/config": "x",
                 "run.py": f"print('hi')  # {KEY}\n", "notes.txt": "GITHUB_TOKEN=abcdef123456\nok line\n", "README.md": "fine"}
        clean, dropped = sanitize_export(files, ["abcdef123456"])
        self.assertEqual(sorted(dropped), sorted([".env", "sub/.env.local", "deploy/id_rsa", "a.pem", ".git/config"]))
        self.assertEqual(sorted(clean), ["README.md", "notes.txt", "run.py"])
        blob = json.dumps(clean)
        self.assertNotIn(KEY, blob); self.assertNotIn("abcdef123456", blob); self.assertIn("ok line", blob)

    def test_windows_style_paths_are_checked(self):
        _, dropped = sanitize_export({"sub\\.env": "x"})
        self.assertEqual(dropped, ["sub\\.env"])


class TestExportsOfRealProject(unittest.TestCase):
    def test_zip_and_report_never_contain_configured_keys(self):
        e, s, db = make_engine()
        pid = e.create("x", sample_pdf(), Budget(), "")
        drive(e, pid)
        st = e.load(pid)
        # simulate a hostile/accidental leak inside generated content and the report
        st.files[".env"] = f"GEMINI_API_KEY={KEY}"
        st.files["README.md"] += f"\nkey={KEY}\n"
        st.report_md += f"\nleaked {KEY}\n"
        e.save(st)
        svc = Service.__new__(Service)
        svc.engine, svc.db = e, db
        svc.s = Settings(llm_api_key=KEY)
        z = zipfile.ZipFile(io.BytesIO(svc.zip_bytes(pid)))
        names = z.namelist()
        self.assertFalse(any(n.endswith(".env") for n in names))
        for n in ("REPORT.md", "reproducibility_manifest.json", "project/run_experiment.py", "project/config.json", "project/README.md"):
            self.assertIn(n, names)
        self.assertTrue(all(KEY not in z.read(n).decode("utf-8", "replace") for n in names))
        self.assertNotIn(KEY, svc.report_md(pid)); self.assertNotIn(KEY, json.dumps(svc.manifest(pid)))


if __name__ == "__main__":
    unittest.main()
