"""Metric vocabulary shared by extraction, validation and evaluation.

One place decides what a metric label means, so a metric name can never be mistaken for a dataset, and
the evaluator and the extractor agree on names ("Macro Rec." == macro-averaged recall).
"""
from __future__ import annotations
import re

AVERAGINGS = ("macro", "micro", "weighted")

# (regex on the normalised label, base key).  Order matters: more specific phrases first.
_BASE_PATTERNS: list[tuple[str, str]] = [
    (r"\bbalanced acc(?:uracy)?\b|\bbacc\b", "balanced_accuracy"),
    (r"\byouden'?s? ?(?:j|index|statistic)?\b|\bj ?statistic\b|\binformedness\b", "youden_j"),
    (r"\bg ?-? ?means?\b|\bgmean\b|\bgeometric mean\b", "g_mean"),
    (r"\bmatthews\b|\bmcc\b", "mcc"),
    (r"\bspecificity\b|\btnr\b|\btrue negative rate\b", "specificity"),
    (r"\bf ?-? ?1(?: ?-? ?(?:score|measure))?\b|\bf ?-? ?(?:score|measure)\b|\bf1\b", "f1"),
    (r"\bprecision\b|\bprec\b|\bppv\b", "precision"),
    (r"\brecall\b|\brec\b|\bsensitivity\b|\btpr\b|\bhit rate\b", "recall"),
    (r"\broc ?-? ?auc\b|\bauroc\b|\bauc\b", "auc"),
    (r"\baccuracy\b|\bacc\b", "accuracy"),
    (r"\brmse\b|\broot mean squared? error\b", "rmse"),
    (r"\bmae\b|\bmean absolute error\b", "mae"),
    (r"\bmse\b|\bmean squared? error\b", "mse"),
    (r"\br2\b|\br ?\^ ?2\b|\br ?-? ?squared\b|\bcoefficient of determination\b", "r2"),
]
_BASE_RX = [(re.compile(p), k) for p, k in _BASE_PATTERNS]

# Metrics whose natural range is [0, 1] (so "98" / "98%" means 0.98).
FRACTION_METRICS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "g_mean", "auc"}
# Direction: True = higher is better.
HIGHER_IS_BETTER = {"accuracy": True, "balanced_accuracy": True, "precision": True, "recall": True, "f1": True,
                    "specificity": True, "g_mean": True, "youden_j": True, "mcc": True, "auc": True, "r2": True,
                    "mae": False, "mse": False, "rmse": False}
CLASSIFICATION_METRICS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "g_mean",
                          "youden_j", "mcc", "auc"}
REGRESSION_METRICS = {"mae", "mse", "rmse", "r2"}


def _norm(label: str) -> str:
    s = str(label).lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[_‐-―]", " ", s)          # underscores and unicode dashes -> space
    s = re.sub(r"[^a-z0-9' ^.\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def metric_spec(label: str) -> tuple[str, str | None] | None:
    """('recall', 'macro') for 'Macro Rec.'; None if the label is not a known metric.

    The averaging is None when the paper does not state it (the caller must treat that as ambiguous)."""
    s = _norm(label)
    if not s or len(s) > 60:
        return None
    avg = next((a for a in AVERAGINGS if re.search(rf"\b{a}\b", s)), None)
    for rx, key in _BASE_RX:
        if rx.search(s):
            return key, avg
    return None


def is_metric_label(text: str) -> bool:
    """True if the text *is* (or is mostly) a metric name, e.g. "Youden's J", "Macro Rec.", "G-Mean".

    Used to keep metric names out of the dataset field.  Long sentences are not metric labels."""
    s = _norm(text)
    if not s or len(s.split()) > 6:
        return False
    return metric_spec(s) is not None


def canonical_key(label: str) -> str:
    """Evaluator key for a label: 'macro recall' -> 'recall', 'weighted f1' -> 'weighted_f1', 'G-Mean' -> 'g_mean'.

    macro averaging and "unstated" both map to the unprefixed key, as the evaluator's macro metrics do; the
    comparison step adds a warning when averaging was not stated."""
    spec = metric_spec(label)
    if spec is None:
        return re.sub(r"[^a-z0-9]+", "", str(label).lower())
    base, avg = spec
    if base in ("precision", "recall", "f1", "specificity") and avg in ("micro", "weighted"):
        return f"{avg}_{base}"
    return base


def find_metrics_in_text(text: str) -> list[str]:
    """Surface forms of metric names found in running text, in order of appearance (lower-cased, de-duplicated by key).

    Matches phrases like "Macro Rec.", "G-Mean", "Youden's J", "macro F1 score"."""
    out: list[tuple[int, str, str]] = []
    low = text.replace("’", "'")
    surface = re.compile(
        r"(?:\b(?:macro|micro|weighted)[ -]?(?:averaged[ -]?)?)?"
        r"(?:balanced accuracy|youden'?s? ?(?:j|index|statistic)\b|\bj statistic|g[ -]?mean|geometric mean|"
        r"matthews correlation(?: coefficient)?|\bmcc\b|specificity|"
        r"f ?-? ?1(?:[ -]?(?:score|measure))?|f[ -]?(?:score|measure)|precision|recall|sensitivity|"
        r"roc[ -]?auc|auroc|\bauc\b|accuracy|rmse|root mean squared? error|mean absolute error|\bmae\b|"
        r"mean squared? error|\bmse\b|\br\^?2\b|r[ -]squared)"
        r"(?:\.(?=\s|$|[,;)]))?", re.I)
    seen: set[str] = set()
    for m in surface.finditer(low):
        raw = m.group(0).strip()
        # also accept the abbreviated forms "Rec." / "Prec." immediately after macro/micro/weighted
        k = canonical_key(raw)
        if metric_spec(raw) is None:
            continue
        key = f"{raw.lower()}"
        if k in seen and not re.match(r"(macro|micro|weighted)", raw.lower()):
            continue
        out.append((m.start(), raw.lower().rstrip("."), k))
        seen.add(k)
    # abbreviations such as "Macro Rec." / "Macro Prec." that the long-form pattern misses
    for m in re.finditer(r"\b(macro|micro|weighted)[ -]?(rec|prec)\b\.?", low, re.I):
        raw = m.group(0).strip().lower().rstrip(".")
        out.append((m.start(), raw, canonical_key(raw)))
    out.sort()
    res, keys = [], set()
    for _, raw, k in out:
        if (raw, k) in keys:
            continue
        keys.add((raw, k))
        res.append(raw)
    return res


def parse_number(text: str) -> float | None:
    """'0.91', '.91', '91.2%', '0.91 ± 0.02' -> first numeric value; None if the cell is not numeric."""
    t = str(text).strip().replace(",", "").replace("−", "-")
    m = re.fullmatch(r"(-?\d*\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?)(?:\s*%)?(?:\s*(?:±|\+/-)\s*\d*\.?\d+)?", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
