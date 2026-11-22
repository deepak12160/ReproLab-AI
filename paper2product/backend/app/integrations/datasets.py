"""Stage C - dataset discovery & validation with provenance."""
from __future__ import annotations
import csv
import hashlib
import os
import re
from dataclasses import dataclass, field, asdict


class ResourceError(Exception):
    """Resource unavailable / needs credentials / failed validation. Message is user-safe."""


@dataclass
class DatasetInfo:
    id: str
    name: str
    source: str
    version: str
    license: str
    url: str
    n_rows: int
    n_features: int
    n_classes: int | None
    sha256: str
    validated: bool
    provenance: str
    warnings: list[str] = field(default_factory=list)
    split: str = "full dataset; the train/test split is made by the experiment code"
    expected_files: list[str] = field(default_factory=list)
    access: str = "public"
    user_supplied: bool = False
    differs_from_paper: bool = False       # True when the paper names other dataset(s): needs explicit user approval

    def to_dict(self) -> dict:
        return asdict(self)


MAX_ROWS = 1_000_000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# Offline, bundled classification datasets (ship inside scikit-learn: no network, no downloads).
SKLEARN_DATASETS = {
    "iris": ("load_iris", (150, 4, 3)),
    "wine": ("load_wine", (178, 13, 3)),
    "breast_cancer": ("load_breast_cancer", (569, 30, 2)),
    "digits": ("load_digits", (1797, 64, 10)),
}
_ALIASES = {"breast cancer": "breast_cancer", "breast-cancer": "breast_cancer", "wisconsin": "breast_cancer"}


def resolve_sklearn(name: str) -> DatasetInfo:
    name = _ALIASES.get(name, name)
    if name not in SKLEARN_DATASETS:
        raise ResourceError(f"bundled dataset 'sklearn:{name}' is not available (choose one of {sorted(SKLEARN_DATASETS)})")
    try:
        import sklearn
        import sklearn.datasets as skd
    except ImportError as e:
        raise ResourceError("scikit-learn is required for the bundled datasets") from e
    loader, (rows, feats, classes) = SKLEARN_DATASETS[name]
    d = getattr(skd, loader)()
    if d.data.shape != (rows, feats) or len(set(d.target.tolist())) != classes:
        raise ResourceError(f"bundled {name} dataset failed schema validation")
    digest = hashlib.sha256(d.data.tobytes() + d.target.tobytes()).hexdigest()
    info = DatasetInfo(
        id=f"sklearn:{name}", name=name, source="scikit-learn bundled copy", version=f"scikit-learn {sklearn.__version__}",
        license="See scikit-learn documentation for this dataset's origin/licence - verify before redistribution",
        url="https://scikit-learn.org/stable/datasets/toy_dataset.html", n_rows=rows, n_features=feats, n_classes=classes,
        sha256=digest, validated=True, provenance=f"scikit-learn `{loader}()` (offline copy)", warnings=[],
        expected_files=["(bundled inside the scikit-learn package; nothing is downloaded)"],
        access="offline copy shipped with scikit-learn; no credentials needed")
    if name == "iris":
        info.license = "Per UCI ML Repository listing (CC BY 4.0) - verify at source before redistribution"
        info.url = "https://archive.ics.uci.edu/dataset/53/iris"
        info.provenance = "Fisher (1936), Annals of Eugenics 7(2):179-188; distributed via UCI and scikit-learn"
        info.warnings = ["UCI's Iris copy is documented to have errors in two rows relative to Fisher's paper; scikit-learn's "
                         "copy corrects them (since v0.20). Other copies may differ slightly."]
    return info


def resolve_sklearn_iris() -> DatasetInfo:
    return resolve_sklearn("iris")


def resolve_huggingface(dataset_id: str, token: str = "") -> DatasetInfo:
    if not dataset_id or len(dataset_id) > 120 or not all(c.isalnum() or c in "-_./" for c in dataset_id):
        raise ResourceError("invalid Hugging Face dataset id")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ResourceError("huggingface_hub is not installed: pip install huggingface_hub")
    try:
        info = HfApi(token=token or None).dataset_info(dataset_id)
    except Exception as e:
        name = type(e).__name__
        if "NotFound" in name:
            raise ResourceError(f"dataset '{dataset_id}' not found on Hugging Face")
        if "Gated" in name or "Unauthorized" in name or "401" in str(e):
            raise ResourceError("dataset requires authentication: set HF_TOKEN")
        raise ResourceError(f"Hugging Face lookup failed ({name})")
    card = getattr(info, "card_data", None)
    lic = (getattr(card, "license", None) if card else None) or "unknown"
    warnings = [] if lic != "unknown" else ["license not declared on dataset card"]
    return DatasetInfo(id=f"hf:{dataset_id}", name=dataset_id, source="Hugging Face Hub", version=info.sha or "unknown",
                       license=str(lic), url=f"https://huggingface.co/datasets/{dataset_id}", n_rows=-1, n_features=-1,
                       n_classes=None, sha256="", validated=False,
                       provenance="metadata only; size/schema must be validated before use", warnings=warnings,
                       access="public" if not token else "HF_TOKEN configured",
                       expected_files=["(not downloaded - this app only reads Hugging Face metadata)"])


def resolve_upload(path: str) -> DatasetInfo:
    if not os.path.isfile(path):
        raise ResourceError("uploaded dataset not found")
    size = os.path.getsize(path)
    if size > MAX_UPLOAD_BYTES:
        raise ResourceError("dataset larger than 50 MB limit")
    if not path.lower().endswith(".csv"):
        raise ResourceError("only .csv uploads are supported")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with open(path, newline="", encoding="utf-8", errors="strict") as f:
        rows = csv.reader(f)
        header = next(rows, None)
        if not header or len(header) < 2:
            raise ResourceError("CSV needs a header and at least 2 columns")
        n = 0
        for r in rows:
            if len(r) != len(header):
                raise ResourceError(f"row {n + 2} has {len(r)} fields, expected {len(header)}")
            n += 1
            if n > MAX_ROWS:
                raise ResourceError("dataset exceeds row limit")
    if n < 10:
        raise ResourceError("dataset has fewer than 10 rows")
    return DatasetInfo(id=f"upload:{os.path.basename(path)}", name=os.path.basename(path), source="user upload",
                       version=h.hexdigest()[:12], license="user-asserted", url="", n_rows=n,
                       n_features=len(header) - 1, n_classes=None, sha256=h.hexdigest(), validated=True,
                       provenance="uploaded by user; license/ownership not verified",
                       expected_files=[f"data/{os.path.basename(path)}"], access="user-provided file",
                       user_supplied=True, split="unspecified; the experiment code must split (last column = label)",
                       warnings=["User-supplied dataset: identity and version are asserted by the user and cannot be verified."])


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def _bundled_key_in(names: list[str]) -> str | None:
    low = " ".join(_norm(n) for n in names)
    for key in ("iris", "wine", "breast cancer", "digits"):
        if re.search(rf"(?<![a-z]){key}(?![a-z])", low):
            return key.replace(" ", "_")
    return None


def safe_upload_name(name: str) -> str:
    """Uploaded dataset file names are reduced to a safe basename; anything else is refused."""
    base = os.path.basename(name or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,80}\.csv", base) or ".." in base or base != name:
        raise ResourceError("dataset file name must be a plain name like data.csv (letters, digits, _ . - only)")
    return base


def resolve_for_spec(spec_datasets: list[str], hint: str = "", hf_token: str = "", upload_dir: str | None = None) -> DatasetInfo:
    """Pick a dataset source. An explicit user hint wins, but if it differs from the dataset(s) named in the paper the
    result is flagged `differs_from_paper` (the plan gate requires explicit approval for that substitution)."""
    if hint:
        if hint.startswith("sklearn:"):
            info = resolve_sklearn(hint[8:])
        elif hint.startswith("hf:"):
            info = resolve_huggingface(hint[3:], hf_token)
        elif hint.startswith("upload:"):
            if not upload_dir:
                raise ResourceError("dataset uploads are not available in this context")
            info = resolve_upload(os.path.join(upload_dir, safe_upload_name(hint[7:])))
        else:
            raise ResourceError(f"unsupported dataset source '{hint}' (use sklearn:iris, hf:<id> or upload:<name>)")
        if spec_datasets:
            paper = " ".join(_norm(n) for n in spec_datasets)
            ident = _norm(info.name.rsplit(".", 1)[0] if info.user_supplied else info.name)
            if ident not in paper and not any(_norm(n) in ident for n in spec_datasets):
                info.differs_from_paper = True
                info.warnings.append(f"Paper names {spec_datasets}; the run uses '{info.id}' instead (user-selected substitution).")
        return info
    key = _bundled_key_in(spec_datasets)
    if key:
        return resolve_sklearn(key)
    raise ResourceError("no supported dataset could be resolved from the paper ("
                        + (", ".join(spec_datasets) if spec_datasets else "no dataset named")
                        + "); supply a dataset source (sklearn:iris | hf:<id> | upload a CSV)")


def preview_resolution(spec_datasets: list[str], hint: str = "") -> dict:
    """Cheap, offline check used when planning: can the dataset be obtained, and does it differ from the paper?"""
    out = {"resolvable": False, "source": "", "reason": "", "differs_from_paper": False, "user_supplied": False,
           "needs": ""}
    if hint:
        out["source"] = hint
        if hint.startswith("sklearn:"):
            nm = _ALIASES.get(hint[8:], hint[8:])
            out["resolvable"] = nm in SKLEARN_DATASETS
            out["reason"] = "" if out["resolvable"] else f"bundled dataset '{nm}' does not exist (choose one of {sorted(SKLEARN_DATASETS)})"
            ident = nm.replace("_", " ")
        elif hint.startswith("upload:"):
            out["resolvable"], out["user_supplied"] = True, True
            ident = _norm(hint[7:].rsplit(".", 1)[0])
        elif hint.startswith("hf:"):
            out["reason"] = "Hugging Face datasets are metadata-only in this app (no download/execution); upload a CSV instead"
            ident = _norm(hint[3:])
        else:
            out["reason"] = f"unsupported dataset source '{hint}'"
            ident = ""
        if spec_datasets and ident:
            paper = " ".join(_norm(n) for n in spec_datasets)
            out["differs_from_paper"] = ident not in paper and not any(_norm(n) in ident for n in spec_datasets)
        return out
    key = _bundled_key_in(spec_datasets)
    if key:
        out.update(resolvable=True, source=f"sklearn:{key}")
        return out
    out["reason"] = ("the paper's dataset(s) " + (", ".join(spec_datasets) if spec_datasets else "(none named)")
                     + " cannot be fetched automatically: only bundled scikit-learn datasets (iris, wine, breast_cancer, digits) "
                       "and user-uploaded CSV files are supported")
    out["needs"] = "Upload the dataset as CSV (last column = label) or choose a bundled dataset; this will be recorded as a substitution if it differs from the paper."
    return out
