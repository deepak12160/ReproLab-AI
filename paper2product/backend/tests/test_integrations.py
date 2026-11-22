import csv, io, json, os, tempfile, unittest, urllib.error
from app.integrations import datasets as ds
from app.integrations.github import GitHubClient, GitHubError
from app.llm.provider import (get_provider, LLMConfigError, LLMOutputError, extract_json, GeminiProvider, Usage)
from app.config import Settings


class FakeResp:
    def __init__(self, body): self.body = json.dumps(body).encode()
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b"{}"))


class TestDatasets(unittest.TestCase):
    def test_iris_validated_with_provenance(self):
        d = ds.resolve_sklearn_iris()
        self.assertEqual((d.n_rows, d.n_features, d.n_classes, d.validated), (150, 4, 3, True))
        self.assertEqual(len(d.sha256), 64); self.assertIn("Fisher", d.provenance); self.assertIn("uci", d.url)

    def test_resolution_rules(self):
        self.assertEqual(ds.resolve_for_spec(["Iris dataset (150 samples)"]).id, "sklearn:iris")
        with self.assertRaises(ds.ResourceError) as cm: ds.resolve_for_spec(["ImageNet"])
        self.assertIn("supply a dataset source", str(cm.exception))
        with self.assertRaises(ds.ResourceError): ds.resolve_for_spec([], "ftp://evil")

    def _csv(self, rows, header=("a", "b", "y"), name="d.csv"):
        p = os.path.join(tempfile.mkdtemp(), name)
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        return p

    def test_upload_validation(self):
        ok = ds.resolve_upload(self._csv([[i, i * 2, i % 2] for i in range(20)]))
        self.assertEqual((ok.n_rows, ok.n_features, ok.validated), (20, 2, True))
        with self.assertRaises(ds.ResourceError): ds.resolve_upload(self._csv([[1, 2, 3]] * 3))               # too few rows
        with self.assertRaises(ds.ResourceError): ds.resolve_upload(self._csv([[1, 2]] * 20))                 # ragged
        with self.assertRaises(ds.ResourceError): ds.resolve_upload(self._csv([[1, 2, 3]] * 20, name="d.exe"))  # type
        with self.assertRaises(ds.ResourceError): ds.resolve_upload("/nonexistent.csv")

    def test_huggingface_input_validation_and_missing_lib(self):
        with self.assertRaises(ds.ResourceError): ds.resolve_huggingface("a; rm -rf /")
        try:
            import huggingface_hub  # noqa
        except ImportError:
            with self.assertRaises(ds.ResourceError) as cm: ds.resolve_huggingface("scikit-learn/iris")
            self.assertIn("huggingface_hub", str(cm.exception))


class TestMoreBundledDatasets(unittest.TestCase):
    def test_wine_breast_cancer_digits_validate(self):
        for name, shape in (("wine", (178, 13, 3)), ("breast_cancer", (569, 30, 2)), ("digits", (1797, 64, 10))):
            d = ds.resolve_sklearn(name)
            self.assertEqual((d.n_rows, d.n_features, d.n_classes, d.validated), (*shape, True))
        self.assertEqual(ds.resolve_for_spec(["Wine recognition data"]).id, "sklearn:wine")
        self.assertEqual(ds.resolve_for_spec([], "sklearn:digits").id, "sklearn:digits")
        with self.assertRaises(ds.ResourceError):
            ds.resolve_for_spec([], "sklearn:nope")


class TestGitHub(unittest.TestCase):
    def test_status_without_token(self):
        s = GitHubClient("").status()
        self.assertFalse(s["configured"]); self.assertIn("GITHUB_TOKEN", s["detail"])

    def test_inspect_success_parses_metadata(self):
        calls = []
        def opener(req, timeout=0):
            calls.append(req.full_url)
            if req.full_url.endswith("/contents"): return FakeResp([{"name": "README.md", "type": "file", "size": 5}])
            return FakeResp({"full_name": "o/r", "stargazers_count": 3, "license": {"spdx_id": "MIT"}, "default_branch": "main"})
        r = GitHubClient("tok", opener).inspect_repo("o/r")
        self.assertEqual((r["license"], r["top_level_files"][0]["name"]), ("MIT", "README.md"))
        self.assertIn("never executed", r["note"])

    def test_http_errors_become_safe_messages(self):
        for code, text in [(401, "rejected the token"), (403, "rate limit"), (404, "not found"), (500, "error 500")]:
            def opener(req, timeout=0, c=code): raise http_error(c)
            with self.assertRaises(GitHubError) as cm: GitHubClient("secret-token-123", opener).inspect_repo("o/r")
            self.assertIn(text, str(cm.exception)); self.assertNotIn("secret-token-123", str(cm.exception))

    def test_network_failure(self):
        def opener(req, timeout=0): raise urllib.error.URLError("dns")
        with self.assertRaises(GitHubError) as cm: GitHubClient("", opener).inspect_repo("o/r")
        self.assertIn("Could not reach", str(cm.exception))

    def test_repo_name_validation(self):
        for bad in ["", "no-slash", "a/b/c", "../../etc", "o/r; rm", "o/" + "x" * 200]:
            with self.assertRaises(GitHubError, msg=bad): GitHubClient("t", lambda *a, **k: None).inspect_repo(bad)

    def test_export_requires_confirmation_and_token_and_makes_no_calls(self):
        called = []
        def opener(req, timeout=0): called.append(req); return FakeResp({})
        with self.assertRaises(GitHubError): GitHubClient("tok", opener).export_project("n", {"a.py": "x"}, confirm=False)
        with self.assertRaises(GitHubError): GitHubClient("tok", opener).export_project("n", {"a.py": "x"}, confirm="yes")
        with self.assertRaises(GitHubError): GitHubClient("", opener).export_project("n", {"a.py": "x"}, confirm=True)
        with self.assertRaises(GitHubError): GitHubClient("tok", opener).export_project("bad name!", {"a.py": "x"}, confirm=True)
        self.assertEqual(called, [])

    def test_confirmed_export_creates_private_repo_then_uploads(self):
        seen = []
        def opener(req, timeout=0):
            seen.append((req.get_method(), req.full_url, json.loads(req.data) if req.data else None))
            return FakeResp({"full_name": "me/n", "html_url": "https://github.com/me/n"})
        r = GitHubClient("tok", opener).export_project("n", {"a.py": "x = 1"}, confirm=True)
        self.assertEqual(seen[0][0:2], ("POST", "https://api.github.com/user/repos")); self.assertTrue(seen[0][2]["private"])
        self.assertEqual(seen[1][0], "PUT"); self.assertEqual(r["files_uploaded"], 1)


class TestLLMProvider(unittest.TestCase):
    def test_missing_credentials_message(self):
        with self.assertRaises(LLMConfigError) as cm: get_provider(Settings(llm_provider="gemini", llm_api_key=""))
        self.assertIn("GEMINI_API_KEY", str(cm.exception))

    def test_auto_without_key_is_demo_mode(self):
        self.assertIsNone(get_provider(Settings(llm_provider="auto", llm_api_key="")))
        self.assertIsNone(get_provider(Settings(llm_provider="demo", llm_api_key="k")))

    def test_unknown_provider(self):
        with self.assertRaises(LLMConfigError): get_provider(Settings(llm_provider="skynet"))

    def test_gemini_needs_sdk(self):
        try:
            import google.genai  # noqa
            self.skipTest("google-genai installed")
        except ImportError:
            with self.assertRaises(LLMConfigError) as cm: GeminiProvider(Settings(llm_api_key="k"))
            self.assertIn("google-genai", str(cm.exception))

    def test_json_extraction_and_validation(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! {"a": 2} done'), {"a": 2})
        for bad in ["no json", "[1,2]", '{"a": ']:
            with self.assertRaises(LLMOutputError): extract_json(bad)

    def test_usage_and_cost_tracking(self):
        u = Usage(); u.add(1000, 500, 0.002); u.add(500, 0, 0.002)
        self.assertEqual((u.total, u.calls), (2000, 2)); self.assertAlmostEqual(u.est_cost_usd, 0.004)


class TestGeminiModelFallbackAndErrors(unittest.TestCase):
    """Offline tests of GeminiProvider's retry / model-fallback logic (the SDK call itself is replaced)."""

    class Err(Exception):
        def __init__(self, code, msg="boom"): super().__init__(msg); self.code = code

    class Resp:
        usage_metadata = None
        def __init__(self, text): self.text = text

    def make(self, behaviour):
        # Provider retry/fallback tests replace the network call, so they do
        # not need the optional Google SDK installed in the test environment.
        import sys, types
        from unittest import mock
        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = lambda **kwargs: object()
        google.genai = genai
        with mock.patch.dict(sys.modules, {"google": google, "google.genai": genai}):
            p = GeminiProvider(Settings(llm_api_key="AIza" + "x" * 30, llm_model="gemini-old-model", llm_max_retries=1))
        p.calls = []
        def fake(model, prompt, system):
            p.calls.append(model)
            return behaviour(model)
        p._call_model = fake
        return p

    def test_retired_model_falls_back_to_next_model_and_remembers_it(self):
        def b(model):
            if model == "gemini-old-model": raise self.Err(404, "models/gemini-old-model is not found")
            return self.Resp('{"ok": true}')
        p = self.make(b)
        self.assertEqual(p.generate_json("x"), {"ok": True})
        self.assertEqual(p.calls[0], "gemini-old-model"); self.assertEqual(len(p.calls), 2)
        self.assertNotEqual(p.info()["model"], "gemini-old-model")
        p.calls.clear(); p.generate_json("y")
        self.assertEqual(len(p.calls), 1)                 # remembered the working model

    def test_invalid_key_fails_fast_with_clear_message(self):
        p = self.make(lambda m: (_ for _ in ()).throw(self.Err(401, "API key not valid")))
        with self.assertRaises(Exception) as cm:
            p.generate_json("x")
        self.assertIn("rejected the API key", str(cm.exception)); self.assertEqual(len(p.calls), 1)

    def test_rate_limit_is_retried_then_succeeds(self):
        state = {"n": 0}
        def b(model):
            state["n"] += 1
            if state["n"] == 1: raise self.Err(429, "quota")
            return self.Resp('```json\n{"a": 1}\n```')
        import unittest.mock as um
        with um.patch("app.llm.provider.time.sleep"):
            self.assertEqual(self.make(b).generate_json("x"), {"a": 1})

    def test_service_unavailable_retries_then_uses_one_alternate_model(self):
        state = {"primary": 0, "alternate": 0}
        def b(model):
            if model == "gemini-old-model":
                state["primary"] += 1
                raise self.Err(503, "temporarily unavailable")
            state["alternate"] += 1
            return self.Resp('{"ok": true}')
        import unittest.mock as um
        p = self.make(b)
        with um.patch("app.llm.provider.time.sleep"):
            self.assertEqual(p.generate_json("x"), {"ok": True})
        self.assertEqual(state["primary"], 2)
        self.assertEqual(state["alternate"], 1)

    def test_error_message_never_contains_the_key(self):
        key = "AIza" + "x" * 30
        p = self.make(lambda m: (_ for _ in ()).throw(self.Err(500, f"server said {key}")))
        import unittest.mock as um
        with um.patch("app.llm.provider.time.sleep"), self.assertRaises(Exception) as cm:
            p.generate_json("x")
        self.assertNotIn(key, str(cm.exception))


@unittest.skipUnless(os.environ.get("P2P_LIVE_TESTS") == "1" and os.environ.get("GEMINI_API_KEY"), "live test: set P2P_LIVE_TESTS=1 and GEMINI_API_KEY")
class TestLiveGemini(unittest.TestCase):
    def test_real_structured_call(self):
        p = get_provider(Settings.from_env())
        self.assertEqual(p.generate_json('Return JSON {"ok": true}')["ok"], True)


@unittest.skipUnless(os.environ.get("P2P_LIVE_TESTS") == "1" and os.environ.get("GITHUB_TOKEN"), "live test: set P2P_LIVE_TESTS=1 and GITHUB_TOKEN")
class TestLiveGitHub(unittest.TestCase):
    def test_inspect_public_repo(self):
        self.assertEqual(GitHubClient(os.environ["GITHUB_TOKEN"]).inspect_repo("scikit-learn/scikit-learn")["full_name"], "scikit-learn/scikit-learn")
