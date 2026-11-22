import os, tempfile, unittest
import unittest.mock as um
from app import config
from app.repair import diagnose, rule_based_patch


class TestDotEnv(unittest.TestCase):
    def test_loads_values_without_overriding_real_environment(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, ".env")
        with open(p, "w", encoding="utf-8-sig") as f:       # BOM = what Windows Notepad may write
            f.write('# comment\nP2P_T_A=hello  # trailing comment\nP2P_T_B="quoted value"\nP2P_T_C=\nP2P_T_D=fromfile\n')
        with um.patch.dict(os.environ, {"P2P_T_D": "already-set"}, clear=False):
            for k in ("P2P_T_A", "P2P_T_B", "P2P_T_C"): os.environ.pop(k, None)
            config.load_dotenv(p)
            self.assertEqual(os.environ["P2P_T_A"], "hello"); self.assertEqual(os.environ["P2P_T_B"], "quoted value")
            self.assertNotIn("P2P_T_C", os.environ); self.assertEqual(os.environ["P2P_T_D"], "already-set")
            for k in ("P2P_T_A", "P2P_T_B"): os.environ.pop(k, None)

    def test_missing_env_file_is_fine(self):
        config.load_dotenv(os.path.join(tempfile.gettempdir(), "does-not-exist.env"))

    def test_default_model_is_not_the_restricted_2_5_model(self):
        self.assertEqual(config.DEFAULT_MODEL, "gemini-3.8-flash")
        self.assertIn("gemini-2.5-flash", config.FALLBACK_MODELS)

    def test_settings_paths_are_absolute(self):
        with um.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("P2P_DATA_DIR", None); os.environ.pop("P2P_DB_PATH", None)
            s = config.Settings.from_env()
        self.assertTrue(os.path.isabs(s.data_dir)); self.assertTrue(os.path.isabs(s.db_path))


class TestWindowsTracebacks(unittest.TestCase):
    def test_backslash_paths_are_mapped_and_repaired(self):
        files = {"lda_repro/lda.py": "import numpy as np\nSw_inv = 1\ncoef = Sw_inverse\n"}
        stderr = ('Traceback (most recent call last):\n'
                  '  File "C:\\Users\\me\\data\\projects\\p1\\work\\lda_repro\\lda.py", line 3, in <module>\n'
                  "NameError: name 'Sw_inverse' is not defined\n")
        diag = diagnose("", stderr, files)
        self.assertEqual((diag.category, diag.file, diag.line, diag.repairable_by_rule), ("name_error", "lda_repro/lda.py", 3, True))
        patch = rule_based_patch(diag, files)
        self.assertIn("coef = Sw_inv\n", patch.new_content)


if __name__ == "__main__":
    unittest.main()
