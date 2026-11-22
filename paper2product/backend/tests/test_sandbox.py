import os, sys, tempfile, time, unittest
import unittest.mock as um
from app.sandbox import LocalSandbox, SandboxResult


def run(script, **kw):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "t.py"), "w", encoding="utf-8") as f:
        f.write(script)
    args = dict(timeout_s=5, memory_mb=512, max_output_kb=64); args.update(kw)
    return LocalSandbox().run_python(d, ["t.py"], **args)


class TestLocalSandbox(unittest.TestCase):
    def test_captures_stdout_stderr_exit(self):
        r = run("import sys; print('hi'); print('err', file=sys.stderr); sys.exit(3)")
        self.assertEqual((r.exit_code, r.stdout.strip(), r.stderr.strip()), (3, "hi", "err"))
        self.assertFalse(r.ok)

    def test_timeout_kills_runaway(self):
        t = time.time()
        r = run("while True: pass", timeout_s=1)
        self.assertTrue(r.timed_out); self.assertLess(time.time() - t, 6); self.assertFalse(r.ok)

    @unittest.skipIf(os.name == "nt", "memory rlimits are not available on Windows (documented limitation; use Docker mode)")
    def test_memory_limit(self):
        r = run("x = bytearray(900*1024*1024)\nprint('allocated')", memory_mb=200)
        self.assertNotEqual(r.exit_code, 0); self.assertNotIn("allocated", r.stdout)

    def test_output_truncation(self):
        r = run("print('x' * 500000)", max_output_kb=1)
        self.assertTrue(r.truncated); self.assertLess(len(r.stdout), 2000)

    def test_secrets_not_inherited(self):
        os.environ["P2P_TEST_SECRET"] = "topsecret"
        try:
            r = run("import os; print(os.environ.get('P2P_TEST_SECRET')); print(sorted(os.environ))")
            self.assertIn("None", r.stdout); self.assertNotIn("topsecret", r.stdout)
        finally:
            del os.environ["P2P_TEST_SECRET"]

    def test_local_mode_is_labelled_not_secure(self):
        r = run("print(1)")
        self.assertFalse(r.isolated); self.assertIn("NOT SECURE", r.note); self.assertEqual(r.mode, "local")

    def test_no_stdin(self):
        r = run("import sys; print(repr(sys.stdin.read()))")
        self.assertIn("''", r.stdout)


class TestWindowsCompatibility(unittest.TestCase):
    """These run on every OS: they simulate the Windows branches without needing Windows."""
    def test_sandbox_module_does_not_require_resource_on_windows(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "app", "sandbox.py"), encoding="utf-8").read()
        self.assertIn("if not IS_WINDOWS:", src)
        top = src.split("class SandboxResult")[0]
        self.assertNotIn("\nimport resource\n", top)          # only imported inside the non-Windows guard

    def test_windows_env_keeps_systemroot_but_no_secrets(self):
        import app.sandbox as sbx
        with um.patch.object(sbx, "IS_WINDOWS", True), um.patch.dict(os.environ, {"SYSTEMROOT": "C:\\Windows", "GEMINI_API_KEY": "secret", "PATH": "C:\\x"}):
            env = sbx.LocalSandbox._env("C:\\work")
        self.assertEqual(env["SYSTEMROOT"], "C:\\Windows")
        self.assertNotIn("GEMINI_API_KEY", env); self.assertEqual(env["TEMP"], "C:\\work")

    def test_unicode_output_roundtrips(self):
        r = run("print('caf\u00e9 \u2713')")
        self.assertIn("caf\u00e9", r.stdout)

    def test_docker_mount_uses_mount_flag_not_colon_volume(self):
        from app.sandbox import DockerSandbox
        cmd = DockerSandbox().build_command(os.getcwd(), ["run_experiment.py"], 30, 512, "n")
        self.assertIn("--mount", cmd); self.assertNotIn("-v", cmd)
        self.assertTrue(any(c.startswith("type=bind,source=") and c.endswith("target=/work") for c in cmd))

    def test_docker_mount_rejects_unsafe_path(self):
        from app.sandbox import DockerSandbox
        from app.security import SecurityError
        with self.assertRaises(SecurityError):
            DockerSandbox().build_command("/tmp/a,b", ["run_experiment.py"], 30, 512, "n")

    def test_auto_mode_never_downgrades_explicit_docker(self):
        from app.sandbox import get_sandbox
        with um.patch("app.sandbox.DockerSandbox.available", return_value=False):
            with self.assertRaises(RuntimeError):
                get_sandbox("docker")
            self.assertEqual(get_sandbox("auto").mode, "local")      # auto without Docker: labelled local sandbox

    def test_auto_sandbox_is_local_and_labelled_until_image_ready(self):
        from app.sandbox import AutoSandbox
        with um.patch("app.sandbox.DockerSandbox.ensure_image", side_effect=lambda: time.sleep(0.3) or True):
            a = AutoSandbox("img")
            self.assertEqual((a.mode, a.isolated), ("local", False)); self.assertIn("still being built", a.description)
            for _ in range(30):
                if a.build_state == "ready": break
                time.sleep(0.1)
            self.assertEqual((a.mode, a.isolated), ("docker", True))
