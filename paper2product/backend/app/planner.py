"""Stage B - research planner. Deterministic for the supported template, LLM-assisted otherwise (schema-validated).

The plan lists tasks with dependencies, compute needs, assumptions and blockers, and says whether the intended run is
an exact, approximate or partial reproduction.  Assumptions need explicit user approval; blockers stop the plan until
they are resolved (e.g. a dataset that cannot be fetched automatically)."""
from __future__ import annotations
import re

from .codegen import supports_template
from .evaluation import SUPPORTED_KEYS
from .integrations import datasets as ds_mod
from .llm import prompts
from .metrics_vocab import canonical_key
from .schemas import Plan, PlanTask, ResearchSpec, SchemaError
from .spec_quality import spec_sha256

DS_PREFIX = "Dataset: "


def classify_reproduction(spec: ResearchSpec, preview: dict, assumptions: list[str]) -> tuple[str, str]:
    """('exact'|'approximate'|'partial', reason).  Conservative: 'exact' needs the paper to pin the data split."""
    why: list[str] = []
    unsupported = [m for m in spec.metrics if canonical_key(m) not in SUPPORTED_KEYS]
    if unsupported:
        why.append(f"metric(s) {unsupported} cannot be computed independently from predictions")
    if len(spec.datasets) > 1 and preview.get("resolvable"):
        why.append(f"the paper uses {len(spec.datasets)} datasets but only one dataset is run")
    if not preview.get("resolvable") or not spec.reported_results:
        why.append("no valid comparison with reported numbers is possible yet")
    if why:
        return "partial", "; ".join(why)
    txt = (" ".join(spec.splits) + " " + spec.training + " " + " ".join(spec.preprocessing)).lower()
    pinned = bool(re.search(r"official|standard split|predefined|pre-defined|fixed split|provided split|benchmark split", txt))
    hp = spec.hyperparameters
    gaps = []
    if preview.get("differs_from_paper"):
        gaps.append("the dataset differs from the one in the paper")
    if not any(k in hp for k in ("random_seed", "seed", "random_state")):
        gaps.append("the random seed is not reported")
    if not pinned:
        gaps.append("the paper does not pin the exact train/test split, so it is re-created")
    if not hp:
        gaps.append("no hyperparameters are reported")
    if gaps:
        return "approximate", "; ".join(gaps)
    return "exact", "dataset, split protocol, seed and hyperparameters are all given by the paper"


def apply_dataset_choice(plan: Plan, spec: ResearchSpec, hint: str) -> Plan:
    """(Re)compute everything in the plan that depends on which dataset will be used."""
    plan.assumptions = [a for a in plan.assumptions if not a.startswith(DS_PREFIX)]
    gap_prefix = "Gap flagged by the planner (not a hard stop; accepting it means the run proceeds without it): "
    for blocker in plan.blockers:
        if blocker.startswith(DS_PREFIX):
            continue
        gap = gap_prefix + blocker
        if gap not in plan.assumptions:
            plan.assumptions.append(gap)
    plan.blockers = []
    prev = ds_mod.preview_resolution(spec.datasets, hint)
    if not prev["resolvable"]:
        plan.blockers.append(DS_PREFIX + prev["reason"] + (". " + prev["needs"] if prev["needs"] else ""))
    else:
        if prev["differs_from_paper"]:
            plan.assumptions.append(DS_PREFIX + f"SUBSTITUTION - the paper names {spec.datasets or 'no dataset'} but the run uses "
                                    f"'{prev['source']}'. Results on a different dataset cannot reproduce the paper's numbers.")
        if prev["user_supplied"]:
            plan.assumptions.append(DS_PREFIX + "a user-supplied CSV is used; its identity/version/licence are asserted by the user.")
        if not spec.datasets and hint:
            plan.assumptions.append(DS_PREFIX + f"the paper does not name a dataset; the user-selected source '{hint}' is used.")
    plan.dataset_requirements = [f"{prev['source'] or 'unresolved'}" + (" (differs from paper)" if prev["differs_from_paper"] else "")] \
        if prev["resolvable"] else [r for r in plan.dataset_requirements if "unresolved" not in r][:1] + ["unresolved - see blockers"]
    plan.reproduction_type, plan.reproduction_type_reason = classify_reproduction(spec, prev, plan.assumptions)
    return plan


def _base_assumptions(spec: ResearchSpec) -> list[str]:
    a = [f"Missing in paper: {m}" for m in spec.missing_info]
    hp = spec.hyperparameters
    if not any(k in hp for k in ("random_seed", "seed", "random_state")):
        a.append("Random seed not given in the paper; default 42 will be used and recorded.")
    return a


def make_plan(spec: ResearchSpec, provider=None, dataset_hint: str = "") -> Plan:
    assumptions = _base_assumptions(spec)
    hp = spec.hyperparameters
    open_q = list(spec.missing_info)
    sha = spec_sha256(spec.to_dict())
    if supports_template(spec):
        assumptions.append("Train/test split is re-created with a seeded stratified split; the paper's exact split is unknown, "
                           "so exact numerical agreement is not expected.")
        for k, default in (("test_size", "0.3"), ("regularization", "1e-9")):
            if k not in hp:
                assumptions.append(f"'{k}' is not given in the paper; default {default} is used.")
        files = ["README.md", "requirements.txt", "config.json", "experiment_metadata.json", "lda_repro/__init__.py",
                 "lda_repro/lda.py", "lda_repro/data.py", "run_experiment.py", "tests/__init__.py", "tests/test_lda.py"]
        tasks = [
            PlanTask("T1", "Load Iris and build deterministic stratified split", [], ["lda_repro/data.py"]),
            PlanTask("T2", "Implement Fisher/Gaussian LDA from the paper's equations", ["T1"], ["lda_repro/lda.py"]),
            PlanTask("T3", "Write unit tests incl. cross-check against scikit-learn", ["T2"], ["tests/test_lda.py"]),
            PlanTask("T4", "Write experiment runner producing predictions + metadata", ["T2"], ["run_experiment.py", "config.json"]),
            PlanTask("T5", "Evaluate independently and compare with reported results", ["T3", "T4"], []),
        ]
        plan = Plan(
            template="fisher_lda_iris",
            approach="Implement the paper's linear discriminant classifier from scratch in NumPy, run it on the Iris data "
                     "with the stated split and seed, and recompute metrics independently of the script.",
            tasks=tasks, files=files, packages=["numpy", "scikit-learn (dataset + reference cross-check only)"],
            dataset_requirements=["Iris (bundled with scikit-learn, no network required)"],
            test_strategy="Shape/prior invariants, separable-blob sanity test, agreement with scikit-learn reference, "
                          "split determinism/stratification, accuracy floor, invalid-input rejection.",
            metrics=spec.metrics or ["accuracy"], expected_outputs=["artifacts/predictions.json", "artifacts/run_meta.json"],
            compute="CPU only, seconds, < 100 MB RAM.", compute_needs="CPU only, seconds, < 100 MB RAM.",
            risks=["Original split unknown -> numeric differences expected", "Single small dataset -> high variance (45 test samples)",
                   "Document contained prompt-injection text (treated as data)" if spec.security_flags else "None identified in document text"],
            assumptions=assumptions, open_questions=open_q, spec_sha256=sha).validate()
        return apply_dataset_choice(plan, spec, dataset_hint)
    if provider is None:
        return Plan(template="unsupported", approach="", tasks=[], files=[], packages=[], dataset_requirements=[],
                    test_strategy="", metrics=spec.metrics, expected_outputs=[], compute="", risks=[],
                    assumptions=assumptions, open_questions=open_q, feasible=False, spec_sha256=sha,
                    infeasible_reason="DEMO MODE (no LLM configured) only implements the bundled Fisher-LDA/Iris template, "
                                      "which does not match this paper. Set GEMINI_API_KEY to let the LLM plan and generate "
                                      "code for other papers.")
    d = provider.generate_validated(prompts.plan_prompt(spec.to_dict(), dataset_hint), _validate_plan_dict) \
        if hasattr(provider, "generate_validated") else provider.generate_json(prompts.plan_prompt(spec.to_dict(), dataset_hint))
    try:
        tasks = [PlanTask(str(t["id"]), str(t["title"]), [str(x) for x in t.get("depends_on", [])], [str(x) for x in t.get("files", [])])
                 for t in d.get("tasks", [])]
        plan = Plan(template="llm_generic", approach=str(d.get("approach", "")), tasks=tasks,
                    files=[str(f) for f in d.get("files", [])], packages=[str(x) for x in d.get("packages", [])],
                    dataset_requirements=[str(x) for x in d.get("dataset_requirements", [])],
                    test_strategy=str(d.get("test_strategy", "")), metrics=[str(x) for x in d.get("metrics", [])],
                    expected_outputs=[str(x) for x in d.get("expected_outputs", [])], compute=str(d.get("compute", "")),
                    compute_needs=str(d.get("compute", "")),
                    risks=[str(x) for x in d.get("risks", [])], assumptions=assumptions + [str(x) for x in d.get("assumptions", [])],
                    open_questions=open_q + [str(x) for x in d.get("open_questions", [])],
                    blockers=[str(x) for x in d.get("blockers", [])], spec_sha256=sha)
        plan.validate()
    except (KeyError, TypeError, SchemaError) as e:
        raise SchemaError(f"LLM plan failed validation: {e}")
    return apply_dataset_choice(plan, spec, dataset_hint)


def _validate_plan_dict(d: dict) -> dict:
    """Structural validation used by the provider's validate-and-retry loop."""
    if not isinstance(d.get("tasks"), list) or not d["tasks"]:
        raise SchemaError("plan needs a non-empty 'tasks' list")
    for t in d["tasks"]:
        if not isinstance(t, dict) or "id" not in t or "title" not in t:
            raise SchemaError("each task needs 'id' and 'title'")
    if not isinstance(d.get("files"), list) or not d["files"]:
        raise SchemaError("plan needs a non-empty 'files' list")
    ids = {str(t["id"]) for t in d["tasks"]}
    for t in d["tasks"]:
        for dep in t.get("depends_on", []) or []:
            if str(dep) not in ids:
                raise SchemaError(f"task {t['id']} depends on unknown task {dep}")
    return d
