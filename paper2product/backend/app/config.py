"""Configuration loaded from environment variables (and an optional .env file). No secrets are hardcoded."""
from __future__ import annotations
import os
from dataclasses import dataclass

# <repo>/backend/app/config.py -> <repo>
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Gemini models are tried in this order if the configured one is missing / retired / not allowed for the key.
# (Google's deprecation page, updated 2026-10-22, recommends gemini-3.8-flash for new projects and says
#  gemini-2.5-* is now limited to accounts that already used it.)
DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash")


def load_dotenv(path: str) -> None:
    """Tiny .env reader (no extra dependency). Never overrides variables that are already set."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.split(" #")[0].strip().strip('"').strip("'")   # allow trailing "  # comment"
                if k and k not in os.environ and v != "":
                    os.environ[k] = v
    except OSError:
        pass


def load_env_files() -> None:
    for p in (os.path.join(ROOT_DIR, ".env"), os.path.join(ROOT_DIR, "backend", ".env"), ".env"):
        load_dotenv(p)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: str = "./data"
    db_path: str = "./data/paper2product.db"
    llm_provider: str = "auto"          # auto | gemini | demo
    llm_model: str = DEFAULT_MODEL
    llm_api_key: str = ""
    llm_timeout_s: int = 120
    llm_max_retries: int = 1
    llm_price_per_1k_tokens: float = 0.0  # optional, for cost estimates
    sandbox_mode: str = "auto"          # auto | docker | local
    sandbox_image: str = "paper2product-sandbox:1"
    github_token: str = ""
    hf_token: str = ""
    max_upload_mb: int = 20
    max_pdf_pages: int = 60
    api_key: str = ""                   # optional shared key protecting the API
    rate_limit_per_min: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_files()
        e = os.environ.get
        data_dir = e("P2P_DATA_DIR") or os.path.join(ROOT_DIR, "data")
        data_dir = os.path.abspath(data_dir)
        key = (e("GEMINI_API_KEY") or e("GOOGLE_API_KEY") or "").strip()
        return cls(
            data_dir=data_dir,
            db_path=e("P2P_DB_PATH") or os.path.join(data_dir, "paper2product.db"),
            llm_provider=(e("P2P_LLM_PROVIDER") or "auto").lower(),
            llm_model=(e("P2P_LLM_MODEL") or DEFAULT_MODEL).strip(),
            llm_api_key=key,
            llm_timeout_s=_int("P2P_LLM_TIMEOUT_S", 60),
            llm_max_retries=_int("P2P_LLM_MAX_RETRIES", 1),
            llm_price_per_1k_tokens=_float("P2P_LLM_PRICE_PER_1K", 0.0),
            sandbox_mode=(e("P2P_SANDBOX") or "auto").lower(),
            sandbox_image=(e("P2P_SANDBOX_IMAGE") or "paper2product-sandbox:1").strip(),
            github_token=(e("GITHUB_TOKEN") or "").strip(),
            hf_token=(e("HF_TOKEN") or "").strip(),
            max_upload_mb=_int("P2P_MAX_UPLOAD_MB", 20),
            max_pdf_pages=_int("P2P_MAX_PDF_PAGES", 60),
            api_key=(e("P2P_API_KEY") or "").strip(),
            rate_limit_per_min=_int("P2P_RATE_LIMIT_PER_MIN", 60),
        )
