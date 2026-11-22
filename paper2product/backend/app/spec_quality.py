"""Field-level quality control for extracted (or user-edited) specifications.

Why this module exists
----------------------
The original extractor assigned *whatever text followed a heading* to that field, and nothing checked the result.
A table header cell "Dataset" became a section heading, so the metric names after it ("Youden's J", "Macro Rec.",
"G-Mean") ended up in the Datasets field while method / hyperparameters / results stayed empty.

Every spec - whether produced by the rule-based extractor, by an LLM, or edited by the user - now passes through
here:

* `sanitize_spec`   : moves/removes entries that are in the wrong field (metric names in datasets, numbers in
                      datasets, sentences in datasets, junk hyperparameters) and records an issue for each.
* `ground_spec`     : (LLM output) drops claims whose values cannot be found in the paper text. Missing stays missing.
* `validate_user_edits`: strict version for user edits - wrong-field entries are rejected with a clear message.
* `compute_missing` : missing-information list derived from the *current* spec (never stale after an edit).
* `compute_provenance`: label per field (explicit / derived / user_provided / assumed / not_reported).
"""
from __future__ import annotations
import hashlib
import json
import re
from typing import Any

from .metrics_vocab import is_metric_label, parse_number, find_metrics_in_text, canonical_key, FRACTION_METRICS
from .schemas import ResearchSpec, ReportedResult, SchemaError, PROVENANCE

# ----------------------------------------------------------------------------------------------------------
# dataset entries
# ----------------------------------------------------------------------------------------------------------
MAX_DATASET_WORDS = 14
_GENERIC_NOT_DATASET = {"dataset", "datasets", "data", "table", "figure", "method", "methods", "model", "models",
                        "results", "result", "metric", "metrics", "baseline", "baselines", "proposed", "ours",
                        "the", "our", "this", "these", "all", "each", "both", "n/a", "na", "none", "-", "--"}
_TABLE_HEADER_WORDS = {"dataset", "datasets", "data set", "data", "method", "methods", "model", "models",
                       "algorithm", "approach", "classifier", "metric", "metrics", "system", "baseline"}


def dataset_entry_problem(text: str) -> str | None:
    """None if `text` can be a dataset name; otherwise a short reason (used in messages and issues)."""
    t = (text or "").strip()
    if not t:
        return "empty"
    low = t.lower().strip(" .:;")
    if is_metric_label(t):
        return "is an evaluation-metric name, not a dataset"
    if parse_number(t) is not None:
        return "is a number, not a dataset"
    if low in _GENERIC_NOT_DATASET or low in _TABLE_HEADER_WORDS:
        return "is a generic/table-header word, not a dataset name"
    if len(t.split()) > MAX_DATASET_WORDS or len(t) > 160:
        return "is a sentence/paragraph, not a dataset name"
    if not re.search(r"[A-Za-z]", t):
        return "contains no letters"
    # a list of metrics ("Accuracy, Macro Rec., G-Mean") is not a dataset either
    parts = [p.strip() for p in re.split(r"[,;/]| and ", t) if p.strip()]
    if len(parts) >= 2 and all(is_metric_label(p) for p in parts):
        return "is a list of metric names, not a dataset"
    return None


def clean_dataset_list(entries: list[str], issues: list[str]) -> tuple[list[str], list[str]]:
    """Returns (datasets, metrics_found_in_dataset_field). Metric names found in the dataset field are returned
    separately so they can be re-homed in the metrics field instead of silently lost."""
    keep: list[str] = []
    moved: list[str] = []
    for e in entries:
        e = re.sub(r"^[-•*]\s*", "", str(e)).strip()
        why = dataset_entry_problem(e)
        if why is None:
            if e.lower() not in {k.lower() for k in keep}:
                keep.append(e)
            continue
        if why.startswith("is an evaluation-metric") or why.startswith("is a list of metric"):
            moved.append(e)
        if why not in ("empty",):
            issues.append(f"Removed {e[:60]!r} from datasets: {why}.")
    return keep, moved


# ----------------------------------------------------------------------------------------------------------
# hyperparameters
# ----------------------------------------------------------------------------------------------------------
def _hp_key(k: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(k).strip().lower()).strip("_")


def hyperparameter_problem(key: str, value: Any) -> str | None:
    if not key:
        return "empty name"
    if len(key) > 48:
        return "name too long to be a hyperparameter"
    if is_metric_label(key.replace("_", " ")):
        return "is a metric name, not a hyperparameter"
    if isinstance(value, bool) or value is None:
        return None if isinstance(value, bool) else "no value"
    if isinstance(value, (int, float)):
        return None
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return "no value"
        if len(v) > 80 or len(v.split()) > 8:
            return "value is a sentence, not a hyperparameter value"
        return None
    if isinstance(value, list):
        return None if all(isinstance(x, (int, float, str)) and not isinstance(x, bool) for x in value) and len(value) <= 16 \
            else "list value must contain only numbers/strings"
    return "unsupported value type"


def clean_hyperparameters(hp: dict, issues: list[str]) -> dict:
    out: dict = {}
    for k, v in (hp or {}).items():
        key = _hp_key(k)
        if isinstance(v, str):
            n = parse_number(v)
            v = n if n is not None and re.fullmatch(r"-?\d*\.?\d+", v.strip().replace(",", "")) else v.strip()
            if isinstance(v, float) and v.is_integer() and "." not in str(hp[k]):
                v = int(v)
        why = hyperparameter_problem(key, v)
        if why:
            issues.append(f"Removed hyperparameter {str(k)[:40]!r}: {why}.")
            continue
        out[key] = v
    return out


# ----------------------------------------------------------------------------------------------------------
# reported results
# ----------------------------------------------------------------------------------------------------------
def clean_results(results: list[ReportedResult], issues: list[str]) -> list[ReportedResult]:
    out: list[ReportedResult] = []
    seen: set[tuple] = set()
    for r in results:
        if not is_metric_label(r.metric):
            issues.append(f"Removed reported result {r.metric[:40]!r}={r.value}: label is not a recognised metric.")
            continue
        ds = r.dataset
        if ds and dataset_entry_problem(ds) is not None:
            issues.append(f"Reported result {r.metric!r}: dataset {ds[:40]!r} is not a valid dataset name; cleared.")
            r.dataset = ""
        key = (canonical_key(r.metric), r.value, r.dataset.lower(), r.variant.lower(), r.split)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------------------
# whole-spec sanitiser
# ----------------------------------------------------------------------------------------------------------
def sanitize_spec(spec: ResearchSpec) -> ResearchSpec:
    """Idempotent. Applies the field validators and re-homes misplaced metric names."""
    issues: list[str] = []
    spec.datasets, moved = clean_dataset_list(spec.datasets, issues)
    if moved:
        have = {m.lower() for m in spec.metrics}
        for e in moved:
            for part in [p.strip() for p in re.split(r"[,;/]| and ", e) if p.strip()]:
                if is_metric_label(part) and part.lower() not in have:
                    spec.metrics.append(part.lower())
                    have.add(part.lower())
        issues.append("Metric names found in the dataset field were moved to the metrics field.")
    spec.hyperparameters = clean_hyperparameters(spec.hyperparameters, issues)
    spec.reported_results = clean_results(spec.reported_results, issues)
    metrics, seen = [], set()
    for m in spec.metrics:
        if not is_metric_label(m):
            issues.append(f"Removed {m[:40]!r} from metrics: not a recognised evaluation metric.")
            continue
        if m.lower() not in seen:
            seen.add(m.lower()); metrics.append(m.lower())
    spec.metrics = metrics
    spec.validation_issues = list(dict.fromkeys(list(spec.validation_issues) + issues))
    return spec


# ----------------------------------------------------------------------------------------------------------
# user edits (strict)
# ----------------------------------------------------------------------------------------------------------
EDITABLE_FIELDS = {"title", "authors", "abstract", "problem", "method", "equations", "architecture", "datasets",
                   "preprocessing", "hyperparameters", "training", "metrics", "reported_results", "hardware",
                   "limitations", "task_type", "baselines", "splits", "dependencies"}


def validate_user_edits(edits: dict) -> dict:
    """Reject edits that put the wrong kind of value in a field, with a message the user can act on.
    Returns the edits unchanged when valid."""
    if not isinstance(edits, dict):
        raise SchemaError("edits must be an object")
    unknown = sorted(set(edits) - EDITABLE_FIELDS)
    if unknown:
        raise SchemaError(f"fields cannot be edited: {unknown}")
    if "datasets" in edits:
        if not isinstance(edits["datasets"], list) or not all(isinstance(x, str) for x in edits["datasets"]):
            raise SchemaError("'datasets' must be a list of dataset names")
        for d in edits["datasets"]:
            if not d.strip():
                continue
            why = dataset_entry_problem(d)
            if why:
                raise SchemaError(f"Dataset entry {d.strip()[:60]!r} {why}. Datasets must be dataset names only; "
                                  f"put metric names in 'metrics'.")
    if "hyperparameters" in edits:
        hp = edits["hyperparameters"]
        if not isinstance(hp, dict):
            raise SchemaError("'hyperparameters' must be an object of name -> value")
        for k, v in hp.items():
            why = hyperparameter_problem(_hp_key(k), v if not isinstance(v, str) else v)
            if why:
                raise SchemaError(f"Hyperparameter {str(k)[:40]!r} {why}.")
    if "metrics" in edits:
        if not isinstance(edits["metrics"], list) or not all(isinstance(x, str) for x in edits["metrics"]):
            raise SchemaError("'metrics' must be a list of metric names")
        for m in edits["metrics"]:
            if m.strip() and not is_metric_label(m):
                raise SchemaError(f"Metric {m.strip()[:60]!r} is not a recognised evaluation metric.")
    if "reported_results" in edits:
        rr = edits["reported_results"]
        if not isinstance(rr, list):
            raise SchemaError("'reported_results' must be a list of {metric, value, dataset, split}")
        for r in rr:
            if not isinstance(r, dict) or "metric" not in r or "value" not in r:
                raise SchemaError("each reported result needs a 'metric' and a numeric 'value'")
            if not is_metric_label(str(r["metric"])):
                raise SchemaError(f"Reported result metric {str(r['metric'])[:60]!r} is not a recognised metric name.")
            if r.get("dataset") and dataset_entry_problem(str(r["dataset"])):
                raise SchemaError(f"Reported result dataset {str(r['dataset'])[:60]!r} is not a dataset name.")
    return edits


# ----------------------------------------------------------------------------------------------------------
# grounding (LLM output must be traceable to the paper)
# ----------------------------------------------------------------------------------------------------------
def _norm_text(t: str) -> str:
    t = t.replace("’", "'").replace("−", "-").replace("­", "")
    t = re.sub(r"-\s*\n\s*", "", t)               # hyphenation at line ends
    return re.sub(r"\s+", " ", t).lower()


def _number_forms(v: float) -> set[str]:
    """Textual spellings of v (or v as a percentage) that really equal it - a rounded form such as '1' for 0.96
    would match unrelated digits in the paper and make an invented number look grounded."""
    forms = set()
    for x in (v, v * 100):
        if abs(x) > 1e9:
            continue
        for nd in range(0, 7):
            s = f"{x:.{nd}f}"
            if abs(float(s) - x) > 1e-9 * max(1.0, abs(x)):
                continue
            for f in {s, s.rstrip("0").rstrip(".") if "." in s else s}:
                forms.add(f)
                if f.startswith("0."):
                    forms.add(f[1:])
    forms.add(repr(v))
    forms.discard("")
    forms.discard(".")
    return forms


def _contains_number(text_norm: str, v: float) -> bool:
    for f in _number_forms(v):
        if re.search(rf"(?<![\d.]){re.escape(f.lower())}(?![\d])", text_norm):
            return True
        # scientific notation e.g. 1e-9 / 1e-09
    m = re.search(r"e-?\d+", repr(float(v)))
    if m and re.search(re.escape(repr(float(v)).replace("e-0", "e-")), text_norm):
        return True
    return False


def ground_spec(spec: ResearchSpec, pages: list[str]) -> ResearchSpec:
    """Drop LLM-claimed values that cannot be found in the paper. Idempotent.

    This is the main defence against invented datasets / hyperparameters / reported numbers: a value that
    does not occur anywhere in the document text is removed and listed in `validation_issues`."""
    text = _norm_text("\n".join(pages))
    issues: list[str] = []
    ds_keep = []
    for d in spec.datasets:
        toks = [t for t in re.split(r"[^a-z0-9\-]+", d.lower()) if len(t) > 2 and t not in ("dataset", "datasets", "data", "the")]
        if toks and all(t in text for t in toks[:3]):
            ds_keep.append(d)
        else:
            issues.append(f"Dropped dataset {d[:60]!r}: name not found in the paper text (not grounded).")
    spec.datasets = ds_keep
    hp_keep = {}
    for k, v in spec.hyperparameters.items():
        if isinstance(v, bool):
            hp_keep[k] = v
        elif isinstance(v, (int, float)):
            if _contains_number(text, float(v)):
                hp_keep[k] = v
            else:
                issues.append(f"Dropped hyperparameter {k}={v}: value not found in the paper text (not grounded).")
        elif isinstance(v, str):
            if v.lower() in text:
                hp_keep[k] = v
            else:
                issues.append(f"Dropped hyperparameter {k}={v!r}: value not found in the paper text (not grounded).")
        else:
            hp_keep[k] = v
    spec.hyperparameters = hp_keep
    res_keep = []
    for r in spec.reported_results:
        if _contains_number(text, r.value):
            res_keep.append(r)
        else:
            issues.append(f"Dropped reported result {r.metric}={r.value}: number not found in the paper text (not grounded).")
    spec.reported_results = res_keep
    found = {canonical_key(m) for m in find_metrics_in_text("\n".join(pages))}
    m_keep = []
    for m in spec.metrics:
        if canonical_key(m) in found or m.lower() in text:
            m_keep.append(m)
        else:
            issues.append(f"Dropped metric {m[:40]!r}: not mentioned in the paper text (not grounded).")
    spec.metrics = m_keep
    # evidence excerpts must really occur in the paper
    ev_keep = []
    for e in spec.evidence:
        ex = _norm_text(e.get("excerpt", ""))[:200]
        if ex and ex in text:
            ev_keep.append(e)
        elif ex:
            issues.append(f"Dropped evidence excerpt for {e.get('field')}/{e.get('key')}: not found verbatim in the paper.")
    spec.evidence = ev_keep
    spec.validation_issues = list(dict.fromkeys(list(spec.validation_issues) + issues))
    return spec


def attach_evidence(spec: ResearchSpec, pages: list[str]) -> ResearchSpec:
    """Give every dataset / hyperparameter / reported result an evidence excerpt + page, found *deterministically*
    in the paper text (we do not rely on the model's own excerpts).  Items that have no locatable source line are
    labelled 'derived' rather than 'explicit'."""
    have = {(e["field"], e["key"]) for e in spec.evidence}
    lines = [(p, ln.strip()) for p, t in enumerate(pages, 1) for ln in t.splitlines() if ln.strip()]

    def find(pred) -> tuple[str, int] | tuple[None, None]:
        for p, ln in lines:
            if pred(_norm_text(ln)):
                return ln[:240], p
        return None, None

    def add(field_: str, key: str, ex: str | None, page: int | None) -> None:
        if (field_, key) in have:
            return
        spec.evidence.append({"field": field_, "key": key, "excerpt": ex or "", "page": page,
                              "label": "explicit" if ex else "derived"})
        have.add((field_, key))

    for d in spec.datasets:
        toks = [t for t in re.split(r"[^a-z0-9\-]+", d.lower()) if len(t) > 2 and t not in ("dataset", "datasets", "data", "the")]
        add("datasets", d[:60], *find(lambda l, toks=toks: bool(toks) and all(t in l for t in toks[:3])))
    for k, v in spec.hyperparameters.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            add("hyperparameters", k, *find(lambda l, v=v: _contains_number(l, float(v))))
        else:
            add("hyperparameters", k, *find(lambda l, v=str(v).lower(): v in l))
    for r in spec.reported_results:
        if r.excerpt:
            have.add(("reported_results", r.metric))
            spec.evidence.append({"field": "reported_results", "key": f"{r.metric}={r.value}", "excerpt": r.excerpt[:240],
                                  "page": r.page, "label": "explicit"})
        else:
            ex, pg = find(lambda l, v=r.value: _contains_number(l, v))
            if ex:
                r.excerpt = ex
                r.page = r.page or pg
            add("reported_results", f"{r.metric}={r.value}", ex, r.page or pg)
    return spec


# ----------------------------------------------------------------------------------------------------------
# missing info + provenance
# ----------------------------------------------------------------------------------------------------------
def compute_missing(spec: ResearchSpec) -> list[str]:
    """Missing-information list derived from the *current* content of the spec."""
    hp = spec.hyperparameters
    m: list[str] = []
    if not spec.method:
        m.append("No method/algorithm section found")
    if not spec.datasets:
        m.append("No dataset specified")
    if not hp:
        m.append("No hyperparameters found")
    if not any(k in hp for k in ("random_seed", "seed", "random_state")):
        m.append("Random seed not reported")
    if not spec.metrics:
        m.append("Evaluation metrics not reported")
    if not spec.reported_results:
        m.append("No numerical results found to compare against")
    if not spec.splits and not any(k in hp for k in ("test_size", "train_test_split", "split", "folds", "n_folds")):
        m.append("Train/test split not reported")
    return m + [a for a in spec.ambiguities if a not in m]


def compute_provenance(spec: ResearchSpec, user_fields: set[str] | None = None) -> dict[str, str]:
    """Label for each top-level field.  user_fields = fields the user edited at the review gate."""
    user_fields = user_fields or set()
    by_field: dict[str, set[str]] = {}
    for e in spec.evidence:
        by_field.setdefault(e["field"], set()).add(e["label"])
    out: dict[str, str] = {}
    for f in ("method", "problem", "architecture", "datasets", "preprocessing", "hyperparameters", "training", "metrics",
              "reported_results", "hardware", "splits", "baselines", "dependencies"):
        v = getattr(spec, f)
        has = bool(v)
        if f in user_fields:
            out[f] = "user_provided" if has else "not_reported"
        elif not has:
            out[f] = "not_reported"
        elif "derived" in by_field.get(f, set()) and "explicit" not in by_field.get(f, set()):
            out[f] = "derived"
        elif f in by_field or f in spec.provenance and spec.provenance[f] == "explicit":
            out[f] = "explicit"
        else:
            out[f] = spec.provenance.get(f, "derived")
    return out


def spec_sha256(spec_dict: dict) -> str:
    """Stable hash of the parts of a spec that drive the experiment (not of review-only metadata)."""
    keys = ("title", "method", "datasets", "hyperparameters", "metrics", "reported_results", "preprocessing",
            "splits", "task_type", "training")
    payload = {k: spec_dict.get(k) for k in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def apply_user_edits(before: dict, edits: dict) -> tuple[ResearchSpec, list[dict]]:
    """Merge validated user edits into a spec dict.

    Returns (new_spec, edit_log).  Only fields whose value really changed are logged and labelled `user_provided`;
    items the user left alone keep their paper evidence.  The result is sanitised, its missing-info list is
    recomputed from the *edited* content (never stale), and provenance is recomputed."""
    validate_user_edits(edits)
    merged = {**before, **edits}
    spec = ResearchSpec.from_dict(merged)
    log: list[dict] = []
    changed: set[str] = set()
    for f, new in edits.items():
        old = before.get(f)
        cur = getattr(spec, f)
        cur_plain = [x.__dict__ if isinstance(x, ReportedResult) else x for x in cur] if isinstance(cur, list) else cur
        if cur_plain == old or (isinstance(old, list) and not old and not cur):
            continue
        changed.add(f)
        log.append({"field": f, "before": old, "after": cur_plain})
    # evidence bookkeeping: user-touched items get a user_provided label instead of stale paper evidence
    old_hp = before.get("hyperparameters") or {}
    old_ds = {d.lower() for d in before.get("datasets") or []}
    old_rr = {(r.get("metric"), r.get("value"), r.get("dataset", "")) for r in before.get("reported_results") or []}
    keep = []
    for e in spec.evidence:
        f, k = e["field"], e["key"]
        if f == "hyperparameters" and "hyperparameters" in changed and k in old_hp and spec.hyperparameters.get(k) != old_hp.get(k):
            continue
        if f == "hyperparameters" and "hyperparameters" in changed and k not in spec.hyperparameters:
            continue
        if f == "datasets" and "datasets" in changed and k.lower() not in {d.lower()[:60] for d in spec.datasets}:
            continue
        if f in changed and f not in ("hyperparameters", "datasets", "reported_results"):
            continue
        keep.append(e)
    spec.evidence = keep
    if "hyperparameters" in changed:
        for k, v in spec.hyperparameters.items():
            if v != old_hp.get(k):
                spec.evidence.append({"field": "hyperparameters", "key": k, "excerpt": "", "page": None, "label": "user_provided"})
    if "datasets" in changed:
        for d in spec.datasets:
            if d.lower() not in old_ds:
                spec.evidence.append({"field": "datasets", "key": d[:60], "excerpt": "", "page": None, "label": "user_provided"})
    if "reported_results" in changed:
        spec.evidence = [e for e in spec.evidence if e["field"] != "reported_results"]
        for r in spec.reported_results:
            if (r.metric, r.value, r.dataset) not in old_rr:
                r.provenance, r.excerpt = "user_provided", ""
                spec.evidence.append({"field": "reported_results", "key": f"{r.metric}={r.value}", "excerpt": "", "page": None,
                                      "label": "user_provided"})
            else:
                spec.evidence.append({"field": "reported_results", "key": f"{r.metric}={r.value}", "excerpt": r.excerpt, "page": r.page,
                                      "label": "explicit"})
    finalize_spec(spec, changed)
    return spec, log


def finalize_spec(spec: ResearchSpec, user_fields: set[str] | None = None) -> ResearchSpec:
    """sanitize -> missing -> provenance.  Call after any change to the spec."""
    sanitize_spec(spec)
    spec.missing_info = compute_missing(spec)
    spec.provenance = compute_provenance(spec, user_fields)
    return spec


__all__ = ["apply_user_edits", "sanitize_spec", "ground_spec", "attach_evidence", "validate_user_edits", "compute_missing", "compute_provenance",
           "spec_sha256", "finalize_spec", "dataset_entry_problem", "hyperparameter_problem", "EDITABLE_FIELDS"]
