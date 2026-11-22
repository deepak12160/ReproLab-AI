"""Stage A - research ingestion: validated PDF text extraction + specification extraction."""
from __future__ import annotations
import io
import re
from typing import Optional

from .extract_rules import heuristic_spec, infer_task, KNOWN_DATASETS  # noqa: F401  (re-exported for callers/tests)
from .llm import prompts
from .metrics_vocab import FRACTION_METRICS, canonical_key
from .schemas import ResearchSpec, ReportedResult
from .security import scan_prompt_injection
from .spec_quality import finalize_spec, ground_spec, attach_evidence


class IngestError(Exception):
    pass


def extract_pages(pdf_bytes: bytes, max_bytes: int = 20 * 1024 * 1024, max_pages: int = 60) -> list[str]:
    """Return text per page (index 0 = page 1).

    Validates magic bytes, size and page count. PyMuPDF is preferred; ANY failure inside it (corrupt or
    malicious PDF) falls back to pypdf, and every parser failure is surfaced as IngestError - never a raw
    third-party exception.
    """
    if not pdf_bytes or len(pdf_bytes) > max_bytes:
        raise IngestError(f"PDF is empty or larger than {max_bytes // (1024 * 1024)} MB")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise IngestError("file is not a PDF (bad magic bytes)")
    pages: list[str] | None = None
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3 (avoids the deprecated `fitz` name)
    except ImportError:
        try:
            import fitz  # type: ignore  # older PyMuPDF
        except ImportError:
            fitz = None
    if fitz is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                if doc.page_count > max_pages:
                    raise IngestError(f"PDF has more than {max_pages} pages")
                pages = [p.get_text() for p in doc]
            finally:
                try:
                    doc.close()
                except Exception:
                    pass
        except IngestError:
            raise
        except Exception:          # PyMuPDF parse error -> try the fallback parser
            pages = None
    if pages is None:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            if len(reader.pages) > max_pages:
                raise IngestError(f"PDF has more than {max_pages} pages")
            pages = [(p.extract_text() or "") for p in reader.pages]
        except IngestError:
            raise
        except Exception as e:
            raise IngestError(f"could not parse PDF ({type(e).__name__})") from e
    if not any(t.strip() for t in pages):
        raise IngestError("PDF contains no extractable text (scanned image? OCR is not supported)")
    return pages


def _as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_as_text(x)}" for k, x in v.items())
    return str(v)


def _as_list(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, (str, int, float)):
        return [str(v)]
    if isinstance(v, dict):
        return [f"{k}: {_as_text(x)}" for k, x in v.items()]
    return [t for t in (_as_text(x).strip() for x in v) if t]


def _num(v: str):
    v = v.strip().rstrip(".")
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def coerce_llm_spec(d: dict) -> dict:
    """LLMs often return numbers as strings, a string where a list is expected, nested objects, etc.
    Normalise the shape (never the meaning) so the strict schema check does not reject usable output."""
    out = dict(d)
    for k in ("title", "abstract", "problem", "method", "architecture", "training", "hardware"):
        out[k] = _as_text(d.get(k)).strip()
    for k in ("authors", "equations", "datasets", "preprocessing", "limitations", "metrics", "baselines", "splits",
              "dependencies", "ambiguities"):
        out[k] = _as_list(d.get(k))
    # LLM free-text "missing_info" becomes ambiguity notes; the authoritative missing list is recomputed from the spec
    out["ambiguities"] = list(dict.fromkeys(out["ambiguities"] + _as_list(d.get("missing_info"))))
    out.pop("missing_info", None)
    hp = d.get("hyperparameters")
    if isinstance(hp, list):                      # [{"name": "seed", "value": 42}, ...]
        conv = {}
        for item in hp:
            if isinstance(item, dict) and "name" in item and "value" in item:
                conv[str(item["name"])] = item["value"]
        hp = conv
    out["hyperparameters"] = {str(k).strip().lower().replace(" ", "_"): (_num(v) if isinstance(v, str) else v)
                              for k, v in (hp or {}).items()} if isinstance(hp, dict) else {}
    rr = []
    for r in d.get("reported_results") or []:
        if not isinstance(r, dict):
            continue
        try:
            val = float(str(r.get("value")).strip().rstrip("%"))
        except (TypeError, ValueError):
            continue
        mk = canonical_key(_as_text(r.get("metric")))
        if str(r.get("value", "")).strip().endswith("%") or (val > 1.5 and mk in FRACTION_METRICS):
            val = val / 100.0                     # 98 or "98%" -> 0.98 for fraction-valued metrics
        page = r.get("page")
        try:
            page = int(page) if page not in (None, "") else None
        except (TypeError, ValueError):
            page = None
        rr.append({"metric": _as_text(r.get("metric")).lower().strip(), "value": val, "dataset": _as_text(r.get("dataset")).strip(),
                   "split": _as_text(r.get("split")).replace(" ", ""), "page": page, "note": _as_text(r.get("note"))[:200],
                   "variant": _as_text(r.get("variant")).strip()[:80]})
    out["reported_results"] = [x for x in rr if x["metric"]]
    refs = {}
    for k, v in (d.get("page_refs") or {}).items() if isinstance(d.get("page_refs"), dict) else []:
        vals = v if isinstance(v, list) else [v]
        pages_ = []
        for p in vals:
            try:
                pages_.append(int(p))
            except (TypeError, ValueError):
                pass
        refs[str(k)] = pages_
    out["page_refs"] = refs
    ev = []
    for e in d.get("evidence") or []:
        if isinstance(e, dict) and e.get("excerpt"):
            ev.append({"field": _as_text(e.get("field")), "key": _as_text(e.get("key")), "excerpt": _as_text(e.get("excerpt")),
                       "page": e.get("page"), "label": _as_text(e.get("label")) or "explicit"})
    out["evidence"] = ev
    tt = _as_text(d.get("task_type")).lower().strip()
    out["task_type"] = tt if tt in ("classification", "regression", "clustering", "unknown") else "unknown"
    for k in ("security_flags", "provenance", "validation_issues"):
        out.pop(k, None)                          # never trust the model for security findings or provenance
    return out


def llm_spec(pages: list[str], provider) -> ResearchSpec:
    """LLM-based extraction. Document text is wrapped as untrusted data; the output is schema-validated,
    sanitised (wrong-field entries) and *grounded* (claims that are not in the paper text are removed)."""
    prompt = prompts.extraction_prompt(pages)

    def validate(raw: dict) -> dict:
        d = coerce_llm_spec(raw)
        d["extraction_method"] = f"llm:{provider.name}"
        ResearchSpec.from_dict(d)                  # raises SchemaError on structural problems
        return d

    if hasattr(provider, "generate_validated"):
        d = provider.generate_validated(prompt, validate)
    else:                                          # simple providers / test doubles
        d = validate(provider.generate_json(prompt))
    spec = ResearchSpec.from_dict(d)
    ground_spec(spec, pages)
    attach_evidence(spec, pages)
    return finalize_spec(spec)


def build_spec(pages: list[str], provider=None) -> ResearchSpec:
    full = "\n".join(pages)
    flags = scan_prompt_injection(full)
    if provider is not None:
        try:
            spec = llm_spec(pages, provider)
        except Exception as e:      # LLM unavailable / bad output: fall back to the rules, and SAY so
            spec = heuristic_spec(pages)
            spec.extraction_method = f"rule-based (LLM extraction failed: {type(e).__name__}: {str(e)[:120]})"
            spec.ambiguities.append("LLM extraction failed, so rule-based extraction was used - please review carefully")
            finalize_spec(spec)
    else:
        spec = heuristic_spec(pages)
    if not spec.title:
        raise IngestError("could not determine a paper title")
    spec.security_flags = flags
    return spec
