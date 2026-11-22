"""Stage G - deterministic metric computation and honest comparison with reported results.
No LLM is involved here: numbers come from predictions, not from model-written prose.

Comparison rules (explicit, and recorded in every Comparison):
  * a reported number is compared only if the dataset, split, metric (incl. averaging) and system variant are compatible;
  * tolerance is a relative difference: <= 2% reproduced, <= 10% partially reproduced, beyond that not reproduced;
  * metric direction (higher/lower is better) is recorded; a result that is *better* than reported is flagged, not hidden.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict, field

from .metrics_vocab import canonical_key, metric_spec, HIGHER_IS_BETTER
from .schemas import ReportedResult

# metric keys the independent evaluator can compute from predictions (AUC needs scores, which runs do not emit)
SUPPORTED_KEYS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "g_mean", "youden_j", "mcc",
                  "micro_precision", "micro_recall", "micro_f1", "weighted_precision", "weighted_recall", "weighted_f1",
                  "mae", "mse", "rmse", "r2"}

REL_REPRODUCED = 0.02    # <= 2% relative difference
REL_PARTIAL = 0.10       # <= 10% relative difference
RULE_TEXT = (f"relative difference |measured - reported| / |reported|: <= {REL_REPRODUCED:.0%} reproduced, "
             f"<= {REL_PARTIAL:.0%} partially reproduced, otherwise not reproduced")


def confusion_matrix(y_true, y_pred, labels) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def _mcc(cm: list[list[int]]) -> float:
    k = len(cm)
    s = sum(sum(r) for r in cm)
    c = sum(cm[i][i] for i in range(k))
    t = [sum(cm[i]) for i in range(k)]
    p = [sum(cm[r][i] for r in range(k)) for i in range(k)]
    num = c * s - sum(pi * ti for pi, ti in zip(p, t))
    den = math.sqrt(max(s * s - sum(x * x for x in p), 0) * max(s * s - sum(x * x for x in t), 0))
    return num / den if den else 0.0


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels)
    n = len(y_true)
    K = len(labels)
    per, ps, rs, fs, specs, supports = {}, [], [], [], [], []
    for i, l in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(K)) - tp
        fn = sum(cm[i]) - tp
        tn = n - tp - fp - fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        sp = tn / (tn + fp) if tn + fp else 0.0
        per[str(l)] = {"precision": p, "recall": r, "f1": f, "specificity": sp, "support": sum(cm[i])}
        ps.append(p); rs.append(r); fs.append(f); specs.append(sp); supports.append(sum(cm[i]))
    correct = sum(cm[i][i] for i in range(K))
    acc = correct / n
    macro_p, macro_r, macro_f = sum(ps) / K, sum(rs) / K, sum(fs) / K
    w = [s / n for s in supports]
    g_mean = math.prod(rs) ** (1.0 / K) if all(x > 0 for x in rs) else 0.0
    youden = sum(r + s - 1 for r, s in zip(rs, specs)) / K          # one-vs-rest, macro-averaged
    out = {"task": "classification", "n": n, "accuracy": acc,
           "precision": macro_p, "recall": macro_r, "f1": macro_f, "averaging": "macro",
           "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f,
           "weighted_precision": sum(a * b for a, b in zip(ps, w)), "weighted_recall": sum(a * b for a, b in zip(rs, w)),
           "weighted_f1": sum(a * b for a, b in zip(fs, w)),
           "micro_precision": acc, "micro_recall": acc, "micro_f1": acc,      # single-label multi-class: micro == accuracy
           "balanced_accuracy": macro_r, "specificity": sum(specs) / K, "g_mean": g_mean, "youden_j": youden,
           "mcc": _mcc(cm), "n_classes": K,
           "per_class": per, "labels": labels, "confusion_matrix": cm}
    return out


def regression_metrics(y_true, y_pred) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    n = len(y_true)
    err = [t - p for t, p in zip(y_true, y_pred)]
    mae = sum(abs(e) for e in err) / n
    mse = sum(e * e for e in err) / n
    rmse = math.sqrt(mse)
    mean = sum(y_true) / n
    ss_tot = sum((t - mean) ** 2 for t in y_true)
    r2 = 1 - sum(e * e for e in err) / ss_tot if ss_tot else float("nan")
    return {"task": "regression", "n": n, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def measured_values(m: dict) -> dict:
    """Flat {metric_key: number} of everything the evaluator computed (what reported numbers can be compared to)."""
    return {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


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
    dataset: str = ""
    split: str = ""
    variant: str = ""
    direction: str = ""               # higher_is_better | lower_is_better | unknown
    rule: str = RULE_TEXT
    metric_key: str = ""

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
    s = re.sub(r"\.(csv|data|txt)$", "", s.lower())
    return " ".join(s.replace("_", " ").replace("-", " ").split())


def _same_dataset(reported: str, measured: str) -> bool:
    a, b = _norm_ds(reported), _norm_ds(measured)
    return bool(a and b and (a in b or b in a or set(b.split()) <= set(a.split())))


def canonical_metric(name: str) -> str:
    """Normalize paper/LLM metric labels to evaluator keys ('Macro Rec.' -> 'recall', 'G-Mean' -> 'g_mean')."""
    return canonical_key(name)


_PROPOSED = re.compile(r"\b(proposed|ours?|our (?:method|model|approach)|full model|this (?:work|paper))\b", re.I)


def is_implemented_variant(variant: str, implemented_names: list[str] | None) -> bool:
    """A reported number belongs to the implemented method if it is unlabeled, labeled 'proposed/ours', or
    names the paper's own method.  Baseline rows must never be compared with our run of the proposed method."""
    v = (variant or "").strip()
    if not v or _PROPOSED.search(v):
        return True
    low = v.lower()
    return any(n and (low in n.lower() or n.lower() in low) for n in (implemented_names or []) if len(n) > 2)


def compare_results(reported: list[ReportedResult], measured: dict, run: dict,
                    implemented_names: list[str] | None = None) -> list[Comparison]:
    """run = {'dataset_name': 'iris', 'test_size': 0.3, 'user_dataset': False}"""
    out: list[Comparison] = []
    for r in reported:
        warns: list[str] = []
        comparable = True
        spec_ = metric_spec(r.metric)
        base, avg = spec_ if spec_ else (None, None)
        metric_key = canonical_key(r.metric)
        direction = ("higher_is_better" if HIGHER_IS_BETTER.get(base) else "lower_is_better") if base in HIGHER_IS_BETTER else "unknown"
        common = dict(page=r.page, dataset=r.dataset, split=r.split, variant=r.variant, direction=direction, metric_key=metric_key)
        m = measured.get(metric_key)
        if m is None or (isinstance(m, float) and math.isnan(m)):
            out.append(Comparison(r.metric, r.value, None, None, None, "unable_to_verify", False,
                                  [f"metric '{r.metric}' (normalized as '{metric_key}') was not measured in this run"], **common))
            continue
        if base in ("precision", "recall", "f1", "specificity") and avg is None:
            warns.append("Paper does not state the averaging for this metric; compared with the MACRO average.")
        if base in ("youden_j", "g_mean", "mcc") and measured.get("n_classes", 2) != 2:
            warns.append("Multi-class definition is not stated in the paper; evaluator uses the one-vs-rest mean "
                         "(Youden's J) / geometric mean of per-class recalls (G-Mean).")
        if not is_implemented_variant(r.variant, implemented_names):
            warns.append(f"Row/system '{r.variant}' is not the implemented method (looks like a baseline or variant); "
                         "not compared with this run.")
            comparable = False
        if not r.dataset:
            warns.append("Paper does not state which dataset the number refers to; comparison is not validated.")
            comparable = False
        elif not _same_dataset(r.dataset, run.get("dataset_name", "")):
            warns.append(f"Dataset mismatch: reported on '{r.dataset}', measured on '{run.get('dataset_name')}'. Not comparable.")
            comparable = False
        elif run.get("user_dataset"):
            warns.append("Measured on a user-supplied dataset; its identity/version is asserted by the user and not verified.")
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
        if not comparable or reld is None:
            status = "unable_to_verify"
        elif reld <= REL_REPRODUCED:
            status = "reproduced"
        elif reld <= REL_PARTIAL:
            status = "partially_reproduced"
        else:
            status = "not_reproduced"
        if comparable and direction != "unknown" and absd > 0:
            better = (m > r.value) if direction == "higher_is_better" else (m < r.value)
            if better and status != "reproduced":
                warns.append("Measured value is BETTER than reported (outside tolerance); this is still a difference, not a success.")
        out.append(Comparison(r.metric, r.value, m, absd, reld, status, comparable, warns, **common))
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
