"""Stage G - deterministic metric computation and honest comparison with reported results.
No LLM is involved here: numbers come from predictions, not from model-written prose."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict, field

from .schemas import ReportedResult

REL_REPRODUCED = 0.02    # <= 2% relative difference
REL_PARTIAL = 0.10       # <= 10% relative difference


def confusion_matrix(y_true, y_pred, labels) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels)
    n = len(y_true)
    per, ps, rs, fs = {}, [], [], []
    for i, l in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        fn = sum(cm[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[str(l)] = {"precision": p, "recall": r, "f1": f, "support": sum(cm[i])}
        ps.append(p); rs.append(r); fs.append(f)
    return {"task": "classification", "n": n, "accuracy": sum(cm[i][i] for i in range(len(labels))) / n,
            "precision": sum(ps) / len(ps), "recall": sum(rs) / len(rs), "f1": sum(fs) / len(fs),
            "averaging": "macro", "per_class": per, "labels": labels, "confusion_matrix": cm}


def regression_metrics(y_true, y_pred) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    n = len(y_true)
    err = [t - p for t, p in zip(y_true, y_pred)]
    mae = sum(abs(e) for e in err) / n
    rmse = math.sqrt(sum(e * e for e in err) / n)
    mean = sum(y_true) / n
    ss_tot = sum((t - mean) ** 2 for t in y_true)
    r2 = 1 - sum(e * e for e in err) / ss_tot if ss_tot else float("nan")
    return {"task": "regression", "n": n, "mae": mae, "rmse": rmse, "r2": r2}


@dataclass
class Comparison:
    metric: str
    reported: float
    measured: float | None
    abs_diff: float | None
    rel_diff: float | None
    status: str                       # reproduced | partially_reproduced | not_reproduced | unable_to_verify
    comparable: bool
    warnings: list[str] = field(default_factory=list)
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _frac(split: str) -> float | None:
    """'70/30' -> test fraction 0.30."""
    try:
        a, b = [int(x) for x in split.replace(" ", "").split("/")]
        return b / (a + b)
    except Exception:
        return None


def _norm_ds(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def _same_dataset(reported: str, measured: str) -> bool:
    a, b = _norm_ds(reported), _norm_ds(measured)
    return bool(a and b and (a in b or b in a or set(b.split()) <= set(a.split())))


def canonical_metric(name: str) -> str:
    """Normalize common paper/LLM metric labels to evaluator keys."""
    key = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    aliases = {
        "accuracy": "accuracy", "acc": "accuracy",
        "precision": "precision", "macroprecision": "precision", "precisionmacro": "precision",
        "recall": "recall", "macrorecall": "recall", "recallmacro": "recall",
        "f1": "f1", "f1score": "f1", "macrof1": "f1", "macrof1score": "f1",
        "fscore": "f1", "f1mac": "f1",
    }
    return aliases.get(key, key)


def compare_results(reported: list[ReportedResult], measured: dict, run: dict) -> list[Comparison]:
    """run = {'dataset_name': 'iris', 'test_size': 0.3}"""
    out: list[Comparison] = []
    for r in reported:
        warns: list[str] = []
        comparable = True
        metric_key = canonical_metric(r.metric)
        m = measured.get(metric_key)
        if m is None or (isinstance(m, float) and math.isnan(m)):
            out.append(Comparison(r.metric, r.value, None, None, None, "unable_to_verify", False,
                                  [f"metric '{r.metric}' (normalized as '{metric_key}') was not measured in this run"], r.page))
            continue
        if not r.dataset:
            warns.append("Paper does not state which dataset the number refers to; comparison is not validated.")
            comparable = False
        elif not _same_dataset(r.dataset, run.get("dataset_name", "")):
            warns.append(f"Dataset mismatch: reported on '{r.dataset}', measured on '{run.get('dataset_name')}'. Not comparable.")
            comparable = False
        if r.split:
            rf, af = _frac(r.split), run.get("test_size")
            if rf is not None and af is not None and abs(rf - af) > 0.01:
                warns.append(f"Split mismatch: reported {r.split}, measured test fraction {af:.2f}. Not comparable.")
                comparable = False
        else:
            warns.append("Evaluation split/protocol not stated in the paper; agreement may be coincidental.")
        warns.append("Exact original data split/seed usually unknown; small-sample metrics are high-variance.")
        absd = abs(m - r.value)
        reld = absd / abs(r.value) if r.value else None
        if not comparable:
            status = "unable_to_verify"
        elif reld is None:
            status = "unable_to_verify"
        elif reld <= REL_REPRODUCED:
            status = "reproduced"
        elif reld <= REL_PARTIAL:
            status = "partially_reproduced"
        else:
            status = "not_reproduced"
        out.append(Comparison(r.metric, r.value, m, absd, reld, status, comparable, warns, r.page))
    return out


def overall_status(comparisons: list[Comparison], tests_ok: bool, experiment_ok: bool) -> tuple[str, str]:
    if not experiment_ok:
        return "unable_to_verify", "The experiment did not complete, so nothing can be verified."
    valid = [c for c in comparisons if c.comparable and c.measured is not None]
    if not valid:
        return "unable_to_verify", "No reported result could be validly compared (missing or incompatible dataset/split/metric)."
    sts = {c.status for c in valid}
    note = "" if tests_ok else " Unit tests did not all pass - treat with caution."
    skipped = [c for c in comparisons if not (c.comparable and c.measured is not None)]
    if sts == {"reproduced"} and (not tests_ok or skipped):
        # never claim full reproduction when tests failed or some reported numbers could not be verified
        why = []
        if not tests_ok:
            why.append("not all unit tests passed")
        if skipped:
            why.append(f"{len(skipped)} reported metric(s) could not be validly compared")
        return "partially_reproduced", ("Comparable metrics are within tolerance, but " + " and ".join(why) + ", so this is not reported as fully reproduced.")
    if sts == {"reproduced"}:
        return "reproduced", "All comparable reported metrics are within tolerance." + note
    if sts == {"not_reproduced"}:
        return "not_reproduced", "Measured metrics differ from reported ones beyond tolerance." + note
    if sts == {"partially_reproduced"}:
        return "partially_reproduced", (f"Comparable metrics deviate modestly from reported values (beyond {REL_REPRODUCED:.0%} "
                                        f"but within {REL_PARTIAL:.0%} relative difference)." + note)
    return "partially_reproduced", "Some comparable metrics match within tolerance, others deviate." + note
