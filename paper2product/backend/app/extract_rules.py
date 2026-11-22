"""Deterministic (no LLM) specification extraction.

Design rules, each one a fix for a real failure of the previous extractor:

1. *Tables are not headings.*  PDF text extraction emits every table cell as its own line.  A header cell such
   as "Dataset" used to look like a section heading, so all following cells (metric names, numbers) were filed
   under Datasets.  Table-like runs of short cells are now detected first and parsed as tables.
2. *Unrecognised headings end the current section.*  "3. Methodology" / "4. Experimental Setup" are mapped by
   keyword; headings that map to nothing close the section instead of letting later text leak into it.
3. *Values are validated, never trusted by position.*  Everything goes through spec_quality afterwards.
4. *Absent stays absent.*  Nothing is guessed; missing items are listed in `missing_info`.
5. *Every extracted item carries an evidence excerpt and page number.*
"""
from __future__ import annotations
import re
from bisect import bisect_right
from typing import Optional

from .metrics_vocab import (is_metric_label, canonical_key, find_metrics_in_text, parse_number, metric_spec,
                            FRACTION_METRICS, CLASSIFICATION_METRICS, REGRESSION_METRICS)
from .schemas import ResearchSpec, ReportedResult
from .spec_quality import finalize_spec

Line = tuple[int, str]          # (page number starting at 1, text)

# ------------------------------------------------------------------------------------------------ headings
SECTION_ALIASES = {
    "abstract": ["abstract"],
    "problem": ["research problem", "problem statement", "problem"],
    "method": ["proposed method", "method", "algorithm", "approach"],
    "equations": ["equations", "equation", "key equations"],
    "architecture": ["model architecture", "architecture"],
    "datasets": ["datasets", "dataset", "data"],
    "preprocessing": ["data preprocessing", "preprocessing"],
    "hyperparameters": ["hyperparameters", "hyper-parameters", "experimental settings"],
    "training": ["training procedure", "training"],
    "metrics": ["evaluation metrics", "metrics", "evaluation protocol"],
    "results": ["reported results", "results"],
    "hardware": ["hardware and software", "hardware", "software requirements"],
    "limitations": ["limitations"],
    "notes": ["notes", "additional notes"],
}
_ALIAS = {a: k for k, v in SECTION_ALIASES.items() for a in v}

# keyword -> section; first match wins, so order is significant.
_KEYWORDS: list[tuple[str, str]] = [
    (r"^abstract$", "abstract"),
    (r"datasets? (?:and|&) (?:evaluation )?metrics|data (?:and|&) metrics", "data_metrics"),
    (r"results? (?:and|&) (?:discussion|analysis)|experimental results?|empirical results?|\bresults?\b|comparison|performance", "results"),
    (r"pre-?processing|data preparation|feature engineering", "preprocessing"),
    (r"evaluation (?:metrics?|protocol|measures?|criteria|setup)|metrics?|performance measures?", "metrics"),
    (r"hyper-?parameters?|parameter settings?|configurations?", "hyperparameters"),
    (r"experimental (?:setup|settings?|details|setting|design)|experiment(?:s|al)?(?: settings)?$|implementation details|training details|training setup|experiments?$|setup$", "setup"),
    (r"datasets?|data ?sets?|benchmarks?|corpus|corpora|data collection|\bdata\b", "datasets"),
    (r"architecture|network design", "architecture"),
    (r"problem (?:statement|formulation|definition|setting)|research problem|task definition|preliminaries", "problem"),
    (r"training|optimi[sz]ation", "training"),
    (r"hardware|software|environment|computing", "hardware"),
    (r"limitations?|threats to validity", "limitations"),
    (r"equations?", "equations"),
    (r"proposed (?:method|approach|model|framework|algorithm|methodology)|methodology|methods?\b|approach|algorithm|framework|model\b", "method"),
]
_KW = [(re.compile(p, re.I), k) for p, k in _KEYWORDS]
_OTHER = re.compile(r"introduction|related work|background|conclusions?|future work|discussion|acknowledg|references|bibliography|appendix|summary", re.I)
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{1,80})$")
_ROMAN = re.compile(r"^\s*([IVX]+)\.\s+([A-Z][A-Z \-&]{2,60})$")
_STOP = {"and", "of", "the", "for", "in", "on", "to", "with", "a", "an", "&", "or", "vs"}

BIB_KEYS = {"references"}

KNOWN_DATASETS = ["iris", "wine", "digits", "breast cancer", "diabetes", "mnist", "cifar-10", "cifar-100",
                  "imagenet", "california housing", "titanic", "glass", "ionosphere", "sonar", "ecoli", "yeast",
                  "vehicle", "pima", "adult", "kdd cup 99", "unsw-nb15", "nsl-kdd", "cora", "citeseer", "pubmed",
                  "imdb", "sst-2", "ag news", "20 newsgroups", "fashion-mnist", "svhn", "coco", "squad", "glue"]


def _titleish(words: list[str]) -> bool:
    return all(w[:1].isupper() or w.lower() in _STOP or not w[:1].isalpha() for w in words)


def classify_heading(line: str) -> Optional[tuple[str, Optional[str], Optional[tuple[str, int]]]]:
    """Returns (kind, section_key, (top_number, depth)) for heading-like lines, else None.

    kind is 'known' (maps to a section) or 'other' (a heading that maps to no extractable section)."""
    text = line.strip()
    if not text or len(text) > 90 or text.endswith((".", ",", ";", "?")) or text[:1] in "-•*•(":
        return None
    num: Optional[tuple[str, int]] = None
    m = _NUMBERED.match(text)
    title = text
    if m:
        parts = m.group(1).split(".")
        num = (parts[0], len(parts))
        title = m.group(2).strip()
    else:
        r = _ROMAN.match(text)
        if r:
            title, num = r.group(2).strip().title(), (r.group(1), 1)
    title = title.rstrip(":").strip()
    words = title.split()
    if not words or len(words) > 9:
        return None
    low = title.lower()
    if low in _ALIAS:
        return "known", _ALIAS[low], num
    allcaps = title.isupper() and len(title) >= 4
    if num is None and not (allcaps or (_titleish(words) and len(words) <= 6)):
        return None
    if re.search(r"[=<>{}|]|\d{3,}", title):                    # equations / numbers are not headings
        return None
    for rx, key in _KW:
        if rx.search(low):
            return "known", key, num
    if _OTHER.search(low) or num is not None or allcaps:
        if num is None and not _OTHER.search(low):
            return None if len(words) > 4 else ("other", None, None)
        return "other", ("references" if re.search(r"references|bibliography", low) else None), num
    return None


# ------------------------------------------------------------------------------------------------ tables
def _cellish(text: str) -> bool:
    t = text.strip()
    if not t or t[:1] in "-•*•" or ":" in t or "=" in t:
        return False
    words = t.split()
    if len(words) > 4 or len(t) > 32:
        return False
    if t.endswith(".") and (len(t) > 14 or len(words) > 3):
        return False
    return True


def find_tables(lines: list[Line]) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] index ranges of table-like runs: >= 6 consecutive short cell-like lines on one page,
    with numeric cells and either metric-like headers or at least three numbers."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        if not _cellish(lines[i][1]):
            i += 1
            continue
        j = i
        while j < n and lines[j][0] == lines[i][0] and _cellish(lines[j][1]):
            j += 1
        cells = [t for _, t in lines[i:j]]
        nums = sum(1 for c in cells if parse_number(c) is not None)
        metrics = sum(1 for c in cells if is_metric_label(c))
        if j - i >= 6 and nums >= 2 and (metrics >= 1 or nums >= 3):
            out.append((i, j))
        i = max(j, i + 1)
    return out


def _context_dataset(ctx: list[str]) -> tuple[str, str]:
    """dataset name and split mentioned in the few lines before a table (caption / lead-in sentence)."""
    text = " ".join(ctx)
    return _dataset_in_sentence(text), _split_in_sentence(text)


def parse_table(cells_with_pages: list[Line], ctx: list[str], issues: list[str]):
    """Returns (metrics, [ReportedResult], datasets, baselines, evidence)."""
    cells = [t.strip() for _, t in cells_with_pages]
    pages = [p for p, _ in cells_with_pages]
    first = next((i for i, c in enumerate(cells) if is_metric_label(c)), None)
    if first is None:
        issues.append("A table was found but no metric column headers were recognised; its numbers were not used.")
        return [], [], [], [], []
    k = 0
    while first + k < len(cells) and is_metric_label(cells[first + k]):
        k += 1
    head_all = cells[:first]
    rest_cells = cells[first + k:]

    def _fits(rr_: int) -> bool:
        n_ = len(rest_cells)
        if n_ == 0 or n_ % (rr_ + k):
            return False
        for i_ in range(0, n_, rr_ + k):
            ch = rest_cells[i_:i_ + rr_ + k]
            if any(parse_number(x) is not None for x in ch[:rr_]) or any(parse_number(x) is None for x in ch[rr_:]):
                return False
        return True

    # Stray short lines directly above the table (an author line, a caption fragment) can look like extra header
    # cells.  The row-label columns are the header cells nearest the metrics: pick the smallest count that parses.
    r = next((c for c in range(1, max(1, len(head_all)) + 1) if _fits(c)), max(1, len(head_all)))
    head = head_all[len(head_all) - r:] if head_all else []
    hdr_kinds = []
    for h in (head or ["?"]):
        hl = h.lower()
        hdr_kinds.append("dataset" if re.search(r"data ?sets?|data|benchmark|corpus", hl) else "variant")
    metric_names = cells[first:first + k]
    ctx_ds, ctx_split = _context_dataset(ctx)
    rest = list(zip(cells[first + k:], pages[first + k:]))
    results: list[ReportedResult] = []
    datasets: list[str] = []
    baselines: list[str] = []
    evidence: list[dict] = []
    pos = 0
    stopped = ""
    while pos + r + k <= len(rest):
        chunk = rest[pos:pos + r + k]
        labels = [c for c, _ in chunk[:r]]
        nums = [parse_number(c) for c, _ in chunk[r:]]
        if any(parse_number(l) is not None for l in labels) or any(x is None for x in nums):
            stopped = chunk[0][0]
            break
        page = chunk[0][1]
        ds, variant = ctx_ds, ""
        for lab, kind in zip(labels, hdr_kinds + ["variant"] * r):
            if kind == "dataset":
                ds = lab
                datasets.append(lab)
            else:
                variant = lab
        if variant:
            baselines.append(variant)
        for mname, val in zip(metric_names, nums):
            ck = canonical_key(mname)
            note = ""
            if ck in FRACTION_METRICS and 1.5 < val <= 100:
                val, note = val / 100.0, "converted from percent"
            results.append(ReportedResult(mname.lower().rstrip("."), float(val), ds, ctx_split, page, note, variant,
                                          f"Table row: {' | '.join(labels)} -> {mname} = {chunk[r + metric_names.index(mname)][0]}",
                                          "explicit"))
        for lab, kind in zip(labels, hdr_kinds + ["variant"] * r):
            if kind == "dataset":
                evidence.append({"field": "datasets", "key": lab[:60], "excerpt": f"Table row: {' | '.join(labels)}", "page": page, "label": "explicit"})
        evidence.append({"field": "reported_results", "key": " / ".join(labels), "excerpt":
                         f"Table row: {' | '.join(labels)} | " + " | ".join(c for c, _ in chunk[r:]), "page": page, "label": "explicit"})
        pos += r + k
    leftover = len(rest) - pos
    if leftover and (stopped or leftover >= 1):
        issues.append(f"Table rows could not be fully parsed (stopped near {(stopped or rest[pos][0])[:30]!r}); "
                      "check reported results manually.")
    mets = [m.lower().rstrip(".") for m in metric_names]
    return mets, results, datasets, baselines, evidence


# ------------------------------------------------------------------------------------------------ sentence helpers
_NAME_STOP = {"the", "our", "this", "these", "those", "we", "a", "an", "in", "on", "for", "each", "all", "both", "of",
              "and", "with", "two", "three", "four", "five", "several", "many", "other", "same", "benchmark", "public",
              "standard", "following", "used", "using", "evaluate", "evaluated", "tested", "test", "experiments",
              "datasets", "dataset", "real", "world", "synthetic", "multi", "class", "binary", "imbalanced", "various",
              "different", "popular", "well", "known", "commonly", "widely", "from", "to", "is", "are", "was", "were"}
_DS_WORD = re.compile(r"\b(?:data ?sets?|corpus|corpora|benchmarks?|databases?)\b", re.I)


def _name_tokens_before(words: list[str], idx: int) -> list[str]:
    """Capitalised / alphanumeric tokens immediately before position idx, allowing 'and'/',' connectors."""
    names: list[str] = []
    cur: list[str] = []
    i = idx - 1
    while i >= 0:
        w = words[i]
        bare = w.strip(",;:()").lstrip("-•*\u2022")
        if bare.lower() in ("and", "&") and cur:
            names.insert(0, " ".join(reversed(cur))); cur = []
            i -= 1
            continue
        ok = bool(bare) and bare.lower() not in _NAME_STOP and (bare[:1].isupper() or bare[:1].isdigit() or "-" in bare)
        if not ok:
            break
        cur.append(bare)
        if w.endswith(","):
            names.insert(0, " ".join(reversed(cur))); cur = []
        i -= 1
    if cur:
        names.insert(0, " ".join(reversed(cur)))
    return [n for n in names if n and n.lower() not in _NAME_STOP]


def _dataset_names_in_sentence(sent: str) -> list[str]:
    out: list[str] = []
    low = sent.lower()
    for k in KNOWN_DATASETS:
        if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", low):
            out.append(k)
    words = sent.split()
    for i, w in enumerate(words):
        if _DS_WORD.fullmatch(w.strip(".,;:()")):
            for n in _name_tokens_before(words, i):
                if not any(n.lower() == o.lower() or n.lower() in o.lower() for o in out):
                    out.append(n)
    m = re.search(r"data ?sets?\s*(?:include|are|:)\s*([^.]{3,160})", sent, re.I)
    if m:
        for part in re.split(r",|;| and ", m.group(1)):
            part = part.strip(" .()")
            if part and 1 <= len(part.split()) <= 5 and part[:1].isupper():
                out.append(part)
    return out


def _dataset_in_sentence(sent: str) -> str:
    names = _dataset_names_in_sentence(sent)
    return names[0] if names else ""


def _split_in_sentence(sent: str) -> str:
    m = re.search(r"\b(\d{2})\s*/\s*(\d{2})\b", sent)
    return f"{m.group(1)}/{m.group(2)}" if m else ""


class _Text:
    """Joined section text with a character -> (page) map so a regex match can be traced back to its page."""

    def __init__(self, lines: list[Line]):
        self.parts, self.starts, self.pages = [], [], []
        pos = 0
        for p, t in lines:
            self.starts.append(pos)
            self.pages.append(p)
            self.parts.append(t)
            pos += len(t) + 1
        self.text = " ".join(self.parts)

    def page_at(self, idx: int) -> Optional[int]:
        if not self.pages:
            return None
        return self.pages[max(0, bisect_right(self.starts, idx) - 1)]

    def sentence_at(self, a: int, b: int) -> str:
        lo = max(self.text.rfind(". ", 0, a) + 2, self.text.rfind("\n", 0, a) + 1, 0)
        hi = self.text.find(". ", b)
        hi = len(self.text) if hi < 0 else hi + 1
        return self.text[lo:hi].strip()[:260]


def _is_bullet(t: str) -> bool:
    return bool(re.match(r"^\s*[-•*•]\s+", t))


# ------------------------------------------------------------------------------------------------ hyperparameters
_HP_PROSE: list[tuple[str, str, str]] = [
    ("learning_rate", r"learning[ -]rate(?:\s*\(?(?:lr|η|eta)\)?)?(?:\s+(?:of|is|was|set to|to|=|:))*\s*(\d*\.?\d+(?:[eE]-?\d+)?)", "num"),
    ("batch_size", r"batch[ -]size(?:\s+(?:of|is|was|set to|to|=|:))*\s*(\d+)", "int"),
    ("epochs", r"(?:(\d+)\s+(?:training\s+)?epochs?)|(?:epochs?(?:\s+(?:of|is|was|set to|=|:))+\s*(\d+))", "int"),
    ("random_seed", r"(?:random\s+)?(?:seed|random[ _]state)(?:\s+(?:of|is|was|set to|to|=|:))*\s*(\d+)", "int"),
    ("n_estimators", r"(\d+)\s+(?:estimators|trees|boosting rounds|weak learners|base learners)", "int"),
    ("max_depth", r"max(?:imum)?[ -]depth(?:\s+(?:of|is|was|set to|=|:))*\s*(\d+)", "int"),
    ("dropout", r"dropout(?:\s+(?:rate|probability))?(?:\s+(?:of|is|was|set to|=|:))*\s*(\d*\.\d+)", "num"),
    ("weight_decay", r"weight[ -]decay(?:\s+(?:of|is|was|set to|=|:))*\s*(\d*\.?\d+(?:[eE]-?\d+)?)", "num"),
    ("momentum", r"momentum(?:\s+(?:of|is|was|set to|=|:))*\s*(\d*\.\d+)", "num"),
    ("n_folds", r"(\d+)[ -]fold\b", "int"),
    ("optimizer", r"\b(adamw|adam|sgd|rmsprop|adagrad|l-?bfgs)\b", "str"),
]
_HP_COMPILED = [(k, re.compile(p, re.I), t) for k, p, t in _HP_PROSE]


def _cast(v: str, kind: str):
    if kind == "int":
        return int(v)
    if kind == "num":
        f = float(v)
        return f
    return v.lower().replace("-", "")


def extract_hyperparameters(sections: list[Line], issues: list[str], evidence: list[dict]) -> dict:
    hp: dict = {}
    for page, line in sections:
        m = re.match(r"^[-•*•]?\s*([A-Za-z_][\w \-/]{1,40}?)\s*[:=]\s*(.+?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        val = m.group(2).strip().rstrip(".")
        if is_metric_label(key.replace("_", " ")) or len(val.split()) > 6:
            continue
        n = parse_number(val)
        hp[key] = (n if n is not None and not re.search(r"[%]", val) else val)
        if isinstance(hp[key], float) and hp[key].is_integer() and "." not in val and "e" not in val.lower():
            hp[key] = int(hp[key])
        evidence.append({"field": "hyperparameters", "key": key, "excerpt": line[:200], "page": page, "label": "explicit"})
    tx = _Text(sections)
    for key, rx, kind in _HP_COMPILED:
        if key in hp:
            continue
        m = rx.search(tx.text)
        if not m:
            continue
        val = next((g for g in m.groups() if g), None)
        if val is None:
            continue
        try:
            hp[key] = _cast(val, kind)
        except ValueError:
            continue
        evidence.append({"field": "hyperparameters", "key": key, "excerpt": tx.sentence_at(m.start(), m.end()),
                         "page": tx.page_at(m.start()), "label": "explicit"})
    return hp


# ------------------------------------------------------------------------------------------------ prose results
_METRIC_PHRASE = re.compile(
    r"(?:\b(?:macro|micro|weighted)[ -]?(?:averaged[ -]?)?)?"
    r"(?:balanced accuracy|youden'?s? ?(?:j|index|statistic)\b|g[ -]?mean|geometric mean|matthews correlation(?: coefficient)?|"
    r"\bmcc\b|specificity|f ?-? ?1(?:[ -]?(?:score|measure))?|f[ -]?(?:score|measure)|precision|recall|sensitivity|"
    r"roc[ -]?auc|auroc|\bauc\b|accuracy|rmse|root mean squared? error|mean absolute error|\bmae\b|"
    r"mean squared? error|\bmse\b|\br\^?2\b|r[ -]squared)", re.I)
_NUM_AFTER = re.compile(r"[^\d\n]{0,40}?(?<![\w.])(\d*\.\d+|\d+(?:\.\d+)?)\s*(%?)")
_NOT_RESULT = re.compile(r"improv|increase|gain|outperform|\bby\b|reduc|higher than|lower than|margin|over the|relative", re.I)


def extract_prose_results(lines: list[Line], default_dataset: str = "") -> list[ReportedResult]:
    tx = _Text(lines)
    out: list[ReportedResult] = []
    for m in _METRIC_PHRASE.finditer(tx.text):
        if metric_spec(m.group(0)) is None:
            continue
        tail = tx.text[m.end(): m.end() + 70]
        n = _NUM_AFTER.match(tail)
        if not n:
            continue
        between = tail[:n.start(1)]
        before = tx.text[max(0, m.start() - 18): m.start()]
        if _NOT_RESULT.search(between) or re.search(r"\bby\s*$|improv\w*\s*$", before, re.I):
            continue
        if re.search(r"(?:and|,|or)\s*$", between.strip()) and not re.search(r"[:=]|\bis\b|\bof\b|\bwas\b|\breach", between):
            continue
        raw, pct = float(n.group(1)), n.group(2) == "%"
        ck = canonical_key(m.group(0))
        note = ""
        if ck in FRACTION_METRICS and (pct or 1.5 < raw <= 100):
            raw, note = raw / 100.0, "converted from percent"
        sent = tx.sentence_at(m.start(), m.end())
        ds = _dataset_in_sentence(sent) or default_dataset
        out.append(ReportedResult(m.group(0).lower().strip().rstrip("."), raw, ds, _split_in_sentence(sent),
                                  tx.page_at(m.start()), note, "", sent[:240], "explicit"))
    return out


# ------------------------------------------------------------------------------------------------ main
def _join(items: list[Line]) -> str:
    return " ".join(t for _, t in items).strip()


def _bullets(items: list[Line]) -> list[str]:
    return [re.sub(r"^[-•*•]\s*", "", t).strip() for _, t in items if t.strip()]


def _num(v: str):
    v = v.strip().rstrip(".")
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def infer_task(metrics: list[str]) -> str:
    keys = {canonical_key(m) for m in metrics}
    keys = {k.split("_", 1)[1] if k.startswith(("micro_", "weighted_")) else k for k in keys}
    if keys & CLASSIFICATION_METRICS:
        return "classification"
    if keys & REGRESSION_METRICS:
        return "regression"
    return "unknown"


def segment(lines: list[Line], tables: list[tuple[int, int]]):
    """Section segmentation. Returns (sections, preamble, table_blocks) with table lines excluded from sections."""
    in_table = {}
    for ti, (a, b) in enumerate(tables):
        for i in range(a, b):
            in_table[i] = ti
    sections: dict[str, list[Line]] = {}
    preamble: list[Line] = []
    table_ctx: dict[int, list[str]] = {}
    current: Optional[str] = None
    cur_top: Optional[str] = None
    recent: list[str] = []
    for i, (pno, line) in enumerate(lines):
        if i in in_table:
            table_ctx.setdefault(in_table[i], list(recent[-4:]))
            continue
        h = classify_heading(line)
        if h and i == 0 and line.strip().lower().rstrip(":") not in _ALIAS:
            h = None                                      # the first line of a paper is its title, even if it reads like a heading
        if h:
            kind, key, num = h
            if key is None and num and num[1] >= 2 and cur_top == num[0] and current not in (None, "other"):
                continue                                  # sub-heading inside the current numbered section
            current = key if kind == "known" else ("references" if key == "references" else "other")
            cur_top = num[0] if num else None
            sections.setdefault(current, [])
            recent.append(line)
            continue
        (sections[current] if current else preamble).append((pno, line.strip()))
        recent.append(line)
    return sections, preamble, table_ctx


def heuristic_spec(pages: list[str]) -> ResearchSpec:
    """Deterministic, rule-based extraction (no LLM)."""
    lines: list[Line] = [(pno, ln.strip()) for pno, text in enumerate(pages, start=1) for ln in text.splitlines() if ln.strip()]
    tables = find_tables(lines)
    sec, pre, table_ctx = segment(lines, tables)
    issues: list[str] = []
    evidence: list[dict] = []

    pl = [t for _, t in pre]
    title = pl[0] if pl else ""
    k = 1
    while title and k < len(pl) and re.search(r"\b(?:with|for|of|and|in|on|via|using|to|a|the|by|from)$|[-:–]$", title, re.I) and k < 3:
        title, k = f"{title} {pl[k]}", k + 1
    auth_line = pl[k] if len(pl) > k else ""
    authors = ([a.strip() for a in re.split(r",| and ", auth_line) if a.strip()]
               if auth_line and not auth_line.upper().startswith("NOTICE") and len(auth_line) < 200 else [])

    # text that may contain experimental details (never the bibliography)
    body_keys = [k_ for k_ in sec if k_ not in ("references", "other")]
    exp_lines: list[Line] = [l for k_ in ("hyperparameters", "setup", "training", "data_metrics", "datasets", "preprocessing",
                                          "method", "metrics") for l in sec.get(k_, [])]

    hp = extract_hyperparameters(exp_lines, issues, evidence)

    # metrics ------------------------------------------------------------------------------------------------
    metrics: list[str] = []
    metric_text = _join([l for k_ in ("metrics", "data_metrics", "setup", "results") for l in sec.get(k_, [])])
    for m in find_metrics_in_text(metric_text):
        if m not in metrics:
            metrics.append(m)
    # results ------------------------------------------------------------------------------------------------
    results: list[ReportedResult] = []
    datasets: list[str] = []
    baselines: list[str] = []
    for ti, (a, b) in enumerate(tables):
        mets, rr, dss, bl, ev = parse_table(lines[a:b], table_ctx.get(ti, []), issues)
        for m in mets:
            if not any(canonical_key(m) == canonical_key(x) for x in metrics):
                metrics.append(m)
        results += rr
        datasets += dss
        baselines += bl
        evidence += ev
    results += extract_prose_results(sec.get("results", []))
    if not metrics:
        metrics = sorted({r.metric for r in results})

    # datasets (prose / bullets) -------------------------------------------------------------------------------
    for p, l in (sec.get("datasets", []) + sec.get("data_metrics", [])):
        prose = len(l.split()) > 12 or (l.endswith(".") and len(l.split()) >= 5 and not _is_bullet(l))
        if not prose:
            for part in [re.sub(r"^[-•*•]\s*", "", l).strip()]:
                if part:
                    datasets.append(part)
                    evidence.append({"field": "datasets", "key": part[:60], "excerpt": l[:200], "page": p, "label": "explicit"})
    ds_lines = sec.get("datasets", []) + sec.get("data_metrics", []) + sec.get("setup", []) + sec.get("results", [])
    for p_, ln in ds_lines:
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"^[-•*\u2022]\s*", "", ln)):
            for n in _dataset_names_in_sentence(sent):
                if any(n.lower() in d.lower() or d.lower() in n.lower() for d in datasets):
                    continue
                datasets.append(n)
                evidence.append({"field": "datasets", "key": n[:60], "excerpt": ln[:240], "page": p_, "label": "explicit"})
    # splits ----------------------------------------------------------------------------------------------------
    splits: list[str] = []
    sp_text = _Text(exp_lines + sec.get("results", []))
    for rx in (r"\b\d{2}\s*/\s*\d{2}\b(?:\s+(?:stratified\s+)?(?:train(?:ing)?[/-]test\s+)?(?:split|ratio|partition))?",
               r"\b\d+[ -]fold(?:\s+(?:stratified\s+)?cross[ -]validation)?",
               r"\btrain(?:ing)?\s*/\s*val(?:idation)?\s*/\s*test\b(?:\s+split)?"):
        for m in re.finditer(rx, sp_text.text, re.I):
            s_ = re.sub(r"\s+", " ", m.group(0)).strip()
            sig = re.sub(r"\D+", "", s_)[:4] + re.sub(r"[\d\s/]+", "", s_.lower())[:4]
            if all(sig != re.sub(r"\D+", "", x)[:4] + re.sub(r"[\d\s/]+", "", x.lower())[:4] for x in splits) \
                    and not any(re.sub(r"\D+", "", s_) == re.sub(r"\D+", "", x) for x in splits):
                splits.append(s_)
                src = next(((pg, ln) for pg, ln in exp_lines + sec.get("results", []) if m.group(0) in ln), (None, ""))
                evidence.append({"field": "splits", "key": s_, "excerpt": src[1][:240] or sp_text.sentence_at(m.start(), m.end()),
                                 "page": src[0] or sp_text.page_at(m.start()), "label": "explicit"})
    m = re.search(r"\b(\d{2})\s*/\s*(\d{2})\b", " ".join(splits))
    if m and "test_size" not in hp and int(m.group(1)) + int(m.group(2)) == 100:
        hp["test_size"] = int(m.group(2)) / 100.0
        evidence.append({"field": "hyperparameters", "key": "test_size", "excerpt": f"derived from the stated {m.group(0)} split",
                         "page": next((e["page"] for e in evidence if e["field"] == "splits"), None), "label": "derived"})
    # baselines in prose
    for p, l in sec.get("results", []) + sec.get("setup", []):
        mm = re.search(r"baselines?\s*(?:include|are|:|such as)\s*([^.]{3,200})", l, re.I)
        if mm:
            baselines += [b.strip(" .()") for b in re.split(r",|;| and ", mm.group(1)) if 0 < len(b.split()) <= 4]
    # dependencies / hardware ----------------------------------------------------------------------------------
    dep_text = _join(sec.get("hardware", []) + sec.get("setup", []) + sec.get("training", []))
    deps: list[str] = []
    for mm in re.finditer(r"\b(PyTorch|TensorFlow|scikit-learn|sklearn|NumPy|SciPy|pandas|Keras|JAX|XGBoost|LightGBM|CUDA|Python)\b(?:\s*v?(\d+(?:\.\d+)*))?", dep_text):
        d = (mm.group(1) + (" " + mm.group(2) if mm.group(2) else "")).strip()
        if d not in deps:
            deps.append(d)
    hardware = _join(sec.get("hardware", []))
    if not hardware:
        gm = re.search(r"[^.]*\b(?:GPU|CPU|RAM|NVIDIA|Tesla|V100|A100)\b[^.]*\.", dep_text)
        hardware = gm.group(0).strip() if gm else ""

    for fld, items in (("method", sec.get("method", [])), ("problem", sec.get("problem", [])), ("architecture", sec.get("architecture", [])),
                       ("training", sec.get("training", [])), ("hardware", sec.get("hardware", [])),
                       ("preprocessing", sec.get("preprocessing", []))):
        if items:
            evidence.append({"field": fld, "key": fld, "excerpt": items[0][1][:200], "page": items[0][0], "label": "explicit"})
    for mname in metrics:
        key_ = mname.rstrip(".").lower()
        src = next(((pg, ln) for pg, ln in lines if key_ in ln.lower().replace("\u2019", "'")), None)
        if src:
            evidence.append({"field": "metrics", "key": mname, "excerpt": src[1][:200], "page": src[0], "label": "explicit"})

    page_refs: dict[str, set[int]] = {k_: {p for p, _ in v} for k_, v in sec.items() if v and k_ not in ("other", "references")}
    for e in evidence:
        if e.get("page"):
            fld = "results" if e["field"] == "reported_results" else e["field"]
            page_refs.setdefault(fld, set()).add(e["page"])

    spec = ResearchSpec(
        title=title, authors=authors, abstract=_join(sec.get("abstract", [])),
        problem=_join(sec.get("problem", [])), method=_join(sec.get("method", [])),
        equations=_bullets(sec.get("equations", [])), architecture=_join(sec.get("architecture", [])),
        datasets=datasets, preprocessing=_bullets(sec.get("preprocessing", [])), hyperparameters=hp,
        training=_join(sec.get("training", [])), metrics=metrics, reported_results=results,
        hardware=hardware, limitations=_bullets(sec.get("limitations", [])),
        task_type=infer_task(metrics), page_refs={k_: sorted(v) for k_, v in page_refs.items()},
        extraction_method="rule-based (no LLM)", baselines=list(dict.fromkeys(baselines)), splits=splits,
        dependencies=deps, evidence=evidence, validation_issues=issues,
    )
    return finalize_spec(spec).validate()
