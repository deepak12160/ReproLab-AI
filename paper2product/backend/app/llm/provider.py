"""Provider-agnostic LLM layer.

* GeminiProvider  - real calls through the official Google Gen AI SDK (`google-genai`).
* No credentials  - `get_provider` returns None and the pipeline runs in clearly labelled DEMO MODE
                    using deterministic code paths. Demo mode never claims LLM reasoning happened.
"""
from __future__ import annotations
import contextvars
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Settings, FALLBACK_MODELS
from ..security import redact


class LLMError(Exception):
    """Provider failure (network, quota, malformed output...). Message is safe to show to users."""


class LLMConfigError(LLMError):
    """Credentials or SDK missing."""


class LLMOutputError(LLMError):
    """Model returned output that failed schema validation."""


class LLMRateLimitError(LLMError):
    """Provider quota / rate limit still exceeded after the bounded retries."""


class LLMTimeoutError(LLMError):
    """Provider did not answer within the configured timeout (after bounded retries)."""


class LLMAuthError(LLMError):
    """API key rejected."""


# project id the current LLM call is made for (set by the engine) so token usage is tracked per project
CURRENT_PROJECT: contextvars.ContextVar[str] = contextvars.ContextVar("p2p_llm_project", default="")


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    est_cost_usd: float = 0.0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def add(self, p: int, o: int, price_per_1k: float) -> None:
        self.prompt_tokens += p
        self.output_tokens += o
        self.calls += 1
        self.est_cost_usd += (p + o) / 1000.0 * price_per_1k

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.total, "calls": self.calls, "est_cost_usd": round(self.est_cost_usd, 6)}


from .prompts import SYSTEM_GUARD  # noqa: E402  (kept importable from here for backward compatibility)


def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()

    def _loads(candidate: str) -> dict:
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise LLMOutputError("model JSON must be an object")
        return obj

    try:
        return _loads(t)
    except json.JSONDecodeError as first:
        # Gemini sometimes emits unescaped newline/tab/control chars inside
        # JSON string values (for example generated Python source). Escape those
        # characters without changing ordinary JSON outside string values.
        candidate = t
        in_string = False
        escaped = False
        out = []
        for ch in candidate:
            if in_string:
                if escaped:
                    out.append(ch)
                    escaped = False
                elif ch == "\\":
                    out.append(ch)
                    escaped = True
                elif ch == '"':
                    out.append(ch)
                    in_string = False
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                elif ord(ch) < 32:
                    out.append("\\u%04x" % ord(ch))
                else:
                    out.append(ch)
            else:
                out.append(ch)
                if ch == '"':
                    in_string = True
        candidate = "".join(out)
        try:
            return _loads(candidate)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", candidate, re.S)
            if not m:
                raise LLMOutputError("model did not return JSON")
            try:
                return _loads(m.group(0))
            except json.JSONDecodeError as e:
                raise LLMOutputError(f"model returned invalid JSON: {e.msg}")


class LLMProvider:
    name = "base"
    is_mock = False

    def __init__(self) -> None:
        self.usage = Usage()                         # process-wide total
        self.by_project: dict[str, Usage] = {}

    def record_usage(self, p: int, o: int, price: float) -> None:
        self.usage.add(p, o, price)
        key = CURRENT_PROJECT.get()
        if key:
            self.by_project.setdefault(key, Usage()).add(p, o, price)

    def usage_for(self, project_id: str) -> Usage:
        return self.by_project.get(project_id, Usage())

    def generate_json(self, prompt: str, system: str = SYSTEM_GUARD, timeout_s: int | None = None) -> dict:
        old_timeout = self._timeout
        if timeout_s is not None:
            self._timeout = max(1, int(timeout_s))
        raise NotImplementedError

    def generate_validated(self, prompt: str, validate, attempts: int = 2, system: str = SYSTEM_GUARD) -> dict:
        """generate_json + local schema validation.  If validation fails the model gets ONE more try with the error
        message (bounded by `attempts`); persistent failure raises LLMOutputError - never a silent pass."""
        last = ""
        for i in range(max(1, attempts)):
            p = prompt if not last else (prompt + f"\n\nYour previous answer was rejected: {last}\nReturn corrected JSON only.")
            raw = self.generate_json(p, system)
            try:
                return validate(raw)
            except Exception as e:                   # SchemaError / ValueError / KeyError from the caller's validator
                last = f"{type(e).__name__}: {str(e)[:300]}"
        raise LLMOutputError(f"model output failed validation after {attempts} attempt(s): {last}")

    def info(self) -> dict:
        return {"provider": self.name, "mock": self.is_mock}


def _status_code(e: Exception) -> int | None:
    """HTTP status carried by google-genai errors (APIError.code), if any."""
    for attr in ("code", "status_code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


def _quota_exhausted(e: Exception) -> bool:
    text = str(e).lower()
    return ("exceeded your current quota" in text
            or "check your plan and billing details" in text
            or "billing_not_active" in text
            or "quota exceeded" in text)


def _rate_limit_kind(e: Exception) -> tuple[str, int]:
    text = str(e)
    low = text.lower()
    if ("per day" in low or "daily" in low or "quota exceeded" in low
            or "billing" in low or "check your plan" in low):
        return "daily", 0
    delay = 0
    for pattern in (r"retryDelay'?[^0-9]*(\d+(?:\.\d+)?)", r"retry in\s*(\d+(?:\.\d+)?)\s*s",
                    r"retry[- ]after[:=]?\s*(\d+)"):
        m = re.search(pattern, text, re.I)
        if m:
            delay = int(float(m.group(1)))
            break
    if delay or "per minute" in low or "per-minute" in low or "rate limit" in low:
        return "minute", min(delay or 2, 60)
    return "unknown", 0


class GeminiProvider(LLMProvider):
    """Real Gemini calls through the official `google-genai` SDK.

    * If the configured model is retired / unknown / not allowed for the key (HTTP 404/403/400-model errors)
      the next model in FALLBACK_MODELS is tried automatically and the one that worked is remembered.
    * Bad API key (401/403 on every model) fails fast with a clear message instead of retrying.
    * Rate limits (429) and server errors (5xx) are retried with backoff.
    """
    name = "gemini"
    supports_timeout_override = True

    def __init__(self, settings: Settings):
        super().__init__()
        if not settings.llm_api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set. Put it in the .env file or run in demo mode (P2P_LLM_PROVIDER=demo).")
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise LLMConfigError("google-genai is not installed: pip install google-genai") from e
        self._genai = genai
        self._client = genai.Client(api_key=settings.llm_api_key)
        # Keep one alternate model available for temporary overloads as well as
        # retired-model errors.  The configured model is always tried first.
        self._models = [settings.llm_model] + [m for m in FALLBACK_MODELS if m != settings.llm_model]
        self._model = self._models[0]
        self._retries = max(0, settings.llm_max_retries)
        self._timeout = settings.llm_timeout_s
        self._allow_fallback = True
        self._cancel_event = None
        self._warn_callback = None
        self._price = settings.llm_price_per_1k_tokens
        self._secret = settings.llm_api_key

    def set_cancel_event(self, event) -> None:
        self._cancel_event = event

    def set_warn_callback(self, callback) -> None:
        self._warn_callback = callback

    def timeout_for(self, minimum_seconds: int) -> int:
        """Return the per-attempt timeout for an operation, preserving the provider setting as an optional floor."""
        return max(self._timeout, int(minimum_seconds))

    @contextmanager
    def request_timeout(self, minimum_seconds: int, retries: int | None = None, fallback: bool = True):
        """Temporarily override timeout/retries for one large operation, then restore them."""
        old = self._timeout
        old_retries = self._retries
        old_fallback = self._allow_fallback
        self._timeout = self.timeout_for(minimum_seconds)
        if retries is not None:
            self._retries = max(0, int(retries))
        self._allow_fallback = fallback
        try:
            yield
        finally:
            self._timeout = old
            self._retries = old_retries
            self._allow_fallback = old_fallback

    def _call_model(self, model: str, prompt: str, system: str):
        from google.genai import types  # type: ignore
        return self._client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=self._timeout * 1000)))

    def generate_json(self, prompt: str, system: str = SYSTEM_GUARD) -> dict:
        last: Exception | None = None
        tried_models = 0
        models = ([self._model] if not getattr(self, "_allow_fallback", True)
                  else [self._model] + [m for m in self._models if m != self._model])
        for model_index, model in enumerate(models):
            tried_models += 1
            model_unusable = False
            # Spend retries on the configured/last-known-good model. If it is
            # overloaded (429/5xx), try one alternate model once rather than
            # multiplying a long timeout by every fallback model.
            attempts = self._retries + 1 if model_index == 0 else 1
            retryable_server_error = False
            for attempt in range(attempts):
                try:
                    resp = self._call_model(model, prompt, system)
                    um = getattr(resp, "usage_metadata", None)
                    if um:
                        self.record_usage(getattr(um, "prompt_token_count", 0) or 0,
                                          getattr(um, "candidates_token_count", 0) or 0, self._price)
                    out = extract_json(resp.text or "")
                    self._model = model                      # remember the model that worked
                    return out
                except LLMOutputError as e:                  # model answered but not with valid JSON: retry same model
                    last = e
                except Exception as e:                       # SDK raises many types; surface a safe message
                    code = _status_code(e)
                    safe = redact(f"{type(e).__name__}: {str(e)[:200]}", [self._secret])
                    last = LLMError(safe)
                    if code == 429:
                        kind, delay = _rate_limit_kind(e)
                        if kind == "daily" or _quota_exhausted(e):
                            raise LLMRateLimitError(
                                "Gemini daily/billing quota is exhausted. Wait for the daily reset, enable billing, "
                                "use a key from another project, or set P2P_LLM_MODEL to a lighter model."
                            )
                        if kind in ("minute", "unknown"):
                            for retry in range(1, 4):
                                wait = min(delay or (2 ** retry), 60)
                                if self._warn_callback:
                                    self._warn_callback(f"Gemini rate limit; retrying in {wait} s")
                                elapsed = 0
                                while elapsed < wait:
                                    if self._cancel_event and self._cancel_event.is_set():
                                        raise LLMError("LLM rate-limit wait cancelled")
                                    time.sleep(min(1, wait - elapsed))
                                    elapsed += 1
                                try:
                                    resp = self._call_model(model, prompt, system)
                                    um = getattr(resp, "usage_metadata", None)
                                    if um:
                                        self.record_usage(getattr(um, "prompt_token_count", 0) or 0,
                                                          getattr(um, "candidates_token_count", 0) or 0, self._price)
                                    return extract_json(resp.text or "")
                                except Exception as retry_error:
                                    code = _status_code(retry_error)
                                    kind, delay = _rate_limit_kind(retry_error)
                                    if code == 429 and kind == "daily":
                                        raise LLMRateLimitError(
                                            "Gemini daily/billing quota is exhausted. Wait for the daily reset, enable billing, "
                                            "use a key from another project, or set P2P_LLM_MODEL to a lighter model."
                                        )
                            raise LLMRateLimitError("Gemini per-minute rate limit persisted after 3 retries.")
                    if code in (401,) or "API key not valid" in str(e) or "API_KEY_INVALID" in str(e):
                        raise LLMAuthError("Gemini rejected the API key (invalid or expired). Create a new key at "
                                           "https://aistudio.google.com/apikey and put it in the .env file.")
                    if code in (404, 403) or (code == 400 and "model" in str(e).lower()):
                        model_unusable = True                # retired / unknown / not allowed for this key
                        break
                    if code == 429 or (code is not None and code >= 500):
                        retryable_server_error = True
                    elif code is None:
                        # Network/transport errors can be transient too; after
                        # exhausting retries, allow one alternate model.
                        retryable_server_error = True
                    if code is not None and code < 500 and code != 429:
                        break                                # other client error: retrying will not help
                if attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, 8))
            if not model_unusable and not retryable_server_error and not isinstance(last, LLMOutputError):
                break                                         # not a model-availability problem: stop trying models
            if model_index >= 1 and retryable_server_error:
                break                                         # only one alternate is attempted for transient failures
        msg = f"LLM call failed ({tried_models} model(s) tried, last model '{self._model}'): {last}"
        low = str(last).lower()
        if isinstance(last, LLMOutputError):
            raise last
        if "429" in low or "resource_exhausted" in low or "quota" in low or "rate" in low:
            raise LLMRateLimitError(msg + " (rate limit/quota - wait and retry, or check your Gemini quota)")
        if "timeout" in low or "timed out" in low or "deadline" in low:
            raise LLMTimeoutError(msg + f" (no answer within {self._timeout}s; raise P2P_LLM_TIMEOUT_S)")
        raise LLMError(msg)

    def check(self) -> dict:
        """Cheap live probe used by the UI 'Test Gemini' button."""
        out = self.generate_json('Return exactly this JSON object: {"ok": true}')
        return {"ok": out.get("ok") is True, "model": self._model, "usage": self.usage.to_dict()}

    def info(self) -> dict:
        return {"provider": self.name, "model": self._model, "mock": False}


def get_provider(settings: Settings) -> Optional[LLMProvider]:
    """Return a real provider, or None meaning DEMO MODE (no LLM)."""
    if settings.llm_provider == "demo":
        return None
    if settings.llm_provider in ("auto", "gemini"):
        if not settings.llm_api_key:
            if settings.llm_provider == "gemini":
                raise LLMConfigError("P2P_LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
            return None
        return GeminiProvider(settings)
    raise LLMConfigError(f"unknown provider '{settings.llm_provider}'")
