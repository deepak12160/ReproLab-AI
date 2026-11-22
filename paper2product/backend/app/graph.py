"""Explicit agent workflow: typed shared state, validated transitions, human gates, bounded repair loop.

Each node is a plain function over WorkflowState so it can be dropped into LangGraph unchanged
(LangGraph could not be installed in the offline build environment; the state machine here is self-contained).
"""
from __future__ import annotations
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import ingest, planner, codegen, repair, evaluation, report, spec_quality
from .config import Settings
from .integrations import datasets
from .llm.provider import LLMError, LLMProvider, LLMTimeoutError, CURRENT_PROJECT
from .sandbox import SandboxResult
from .schemas import Budget, Plan, PlanTask, ResearchSpec, SchemaError
from .security import SecurityError, redact, scan_python
from .storage import Storage


class S:
    CREATED = "created"; INGESTING = "ingesting"; AWAITING_SPEC = "awaiting_spec_approval"
    PLANNING = "planning"; AWAITING_PLAN = "awaiting_plan_approval"; RESOLVING = "resolving_data"
    GENERATING = "generating"; TESTING = "testing"; REPAIRING = "repairing"; EXECUTING = "executing"
    EVALUATING = "evaluating"; REPORTING = "reporting"; COMPLETED = "completed"
    FAILED = "failed"; BLOCKED = "blocked"; CANCELLED = "cancelled"


TERMINAL = {S.COMPLETED, S.FAILED, S.BLOCKED, S.CANCELLED}
_X = {S.CANCELLED}
TRANSITIONS: dict[str, set[str]] = {
    S.CREATED: {S.INGESTING} | _X,
    S.INGESTING: {S.AWAITING_SPEC, S.FAILED} | _X,
    S.AWAITING_SPEC: {S.PLANNING} | _X,
    S.PLANNING: {S.AWAITING_PLAN, S.BLOCKED, S.FAILED} | _X,
    S.AWAITING_PLAN: {S.RESOLVING} | _X,
    S.RESOLVING: {S.GENERATING, S.BLOCKED, S.FAILED} | _X,
    S.GENERATING: {S.TESTING, S.BLOCKED, S.FAILED} | _X,
    S.TESTING: {S.REPAIRING, S.EXECUTING, S.BLOCKED, S.FAILED} | _X,
    S.REPAIRING: {S.TESTING, S.BLOCKED, S.FAILED} | _X,
    S.EXECUTING: {S.EVALUATING, S.REPAIRING, S.BLOCKED, S.FAILED} | _X,
    S.EVALUATING: {S.REPORTING, S.FAILED} | _X,
    S.REPORTING: {S.COMPLETED, S.FAILED},
    S.COMPLETED: set(), S.FAILED: {S.RESOLVING, S.GENERATING}, S.BLOCKED: set(), S.CANCELLED: set(),
}


def _rmtree(path: str) -> None:
    """shutil.rmtree that tolerates short-lived file locks (antivirus/indexer on Windows)."""
    for i in range(5):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.2 * (i + 1))


class InvalidTransition(Exception):
    pass


class Cancelled(Exception):
    pass


class GateError(Exception):
    """An action was attempted in the wrong state / without required approval."""


def check_transition(cur: str, new: str) -> None:
    if new not in TRANSITIONS.get(cur, set()):
        raise InvalidTransition(f"{cur} -> {new} is not allowed")


@dataclass
class WorkflowState:
    project_id: str = ""
    mode: str = "demo"                     # demo | llm
    llm_info: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    dataset_hint: str = ""
    inject_fault: bool = False
    auto_run: bool = False
    codegen_partial: dict = field(default_factory=dict)
    codegen_missing: list = field(default_factory=list)
    codegen_completed: list = field(default_factory=list)
    codegen_resumed: bool = False
    spec: Optional[dict] = None
    spec_original: Optional[dict] = None   # parser/LLM output exactly as extracted (before user corrections)
    spec_edits: list = field(default_factory=list)   # [{field, before, after}] user corrections made at the review gate
    spec_sha256: str = ""                  # hash of the approved spec; the plan and the run must match it
    degraded: list = field(default_factory=list)     # honest notes when an LLM step fell back to rules / failed
    is_fixture: bool = False               # bundled synthetic demo paper - NOT a real paper
    narrative: str = ""                    # optional LLM-drafted summary (never used for status)
    approved_assumptions: list = field(default_factory=list)
    plan: Optional[dict] = None
    dataset: Optional[dict] = None
    files: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    final_tests: Optional[dict] = None
    run_meta: Optional[dict] = None
    metrics: Optional[dict] = None
    script_metric_check: str = ""
    comparisons: list = field(default_factory=list)
    reproduction_status: str = "unable_to_verify"
    reproduction_note: str = ""
    sandbox: Optional[dict] = None
    usage: dict = field(default_factory=dict)
    started_at: float = 0.0
    blocked_reason: str = ""
    report_md: str = ""
    manifest: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def summarize_unittest(res: SandboxResult) -> str:
    m = re.search(r"Ran (\d+) tests? in", res.stderr)
    ran = m.group(1) if m else "?"
    if res.timed_out:
        return f"timed out after {res.duration_s}s"
    if res.ok and "OK" in res.stderr:
        return f"{ran} tests passed"
    f = re.search(r"FAILED \(([^)]*)\)", res.stderr)
    return f"{ran} tests ran, FAILED ({f.group(1)})" if f else f"failed (exit {res.exit_code})"


class Engine:
    def __init__(self, settings: Settings, storage: Storage, sandbox, provider: Optional[LLMProvider]):
        self.s, self.db, self.sandbox, self.llm = settings, storage, sandbox, provider
        self._cancel: dict[str, threading.Event] = {}
        self._pinned: dict[str, object] = {}      # sandbox chosen for a project's run (stays fixed during the run)
        self._lock = threading.Lock()

    # ---- plumbing --------------------------------------------------------
    def _concrete_sandbox(self):
        return self.sandbox.current() if hasattr(self.sandbox, "current") else self.sandbox

    def project_dir(self, pid: str) -> str:
        return os.path.join(self.s.data_dir, "projects", pid)

    def work_dir(self, pid: str) -> str:
        return os.path.join(self.project_dir(pid), "work")

    def cancel_event(self, pid: str) -> threading.Event:
        with self._lock:
            return self._cancel.setdefault(pid, threading.Event())

    def request_cancel(self, pid: str) -> None:
        self.cancel_event(pid).set()
        p = self.db.get_project(pid)
        if p and p["status"] in (S.AWAITING_SPEC, S.AWAITING_PLAN, S.CREATED):
            self._set_status(pid, S.CANCELLED)  # not running: cancel immediately
            self.db.add_event(pid, "cancel", "Cancelled by user", "warn")
        self.db.audit("cancel", "user", pid, {})

    def load(self, pid: str) -> WorkflowState:
        p = self.db.get_project(pid)
        if not p:
            raise KeyError(pid)
        return WorkflowState.from_dict(p["data"].get("state", {"project_id": pid}))

    def _usage(self, pid: str):
        if not self.llm:
            return None
        return self.llm.usage_for(pid) if hasattr(self.llm, "usage_for") else self.llm.usage

    def save(self, st: WorkflowState) -> None:
        if self.llm:
            st.usage = self._usage(st.project_id).to_dict()
        self.db.update_data(st.project_id, state=st.to_dict())

    def status(self, pid: str) -> str:
        return self.db.get_project(pid)["status"]

    def _set_status(self, pid: str, new: str) -> None:
        cur = self.status(pid)
        check_transition(cur, new)
        self.db.set_status(pid, new)
        self.db.add_event(pid, "state", f"{cur} -> {new}", "info", {"from": cur, "to": new})

    def _log(self, st: WorkflowState, stage: str, msg: str, level: str = "info", data: dict | None = None) -> None:
        self.db.add_event(st.project_id, stage, msg, level, data)

    def _guard(self, st: WorkflowState) -> None:
        if self.cancel_event(st.project_id).is_set():
            raise Cancelled()
        b = st.budget
        if time.time() - st.started_at > b.get("max_wall_s", 900):
            raise TimeoutError("wall-clock budget exhausted")
        if self.llm and self._usage(st.project_id).total > b.get("max_tokens", 200_000):
            raise TimeoutError("token budget exhausted")

    # ---- creation ----------------------------------------------------------
    def create(self, name: str, pdf_bytes: bytes, budget: Budget, dataset_hint: str = "", inject_fault: bool = False,
               is_fixture: bool = False, auto_run: bool = False) -> str:
        ingest.extract_pages(pdf_bytes, self.s.max_upload_mb * 1024 * 1024, self.s.max_pdf_pages)  # validate early
        pid = self.db.create_project(name[:120], {})
        os.makedirs(self.project_dir(pid), exist_ok=True)
        with open(os.path.join(self.project_dir(pid), "input.pdf"), "wb") as f:
            f.write(pdf_bytes)
        st = WorkflowState(project_id=pid, mode="llm" if self.llm else "demo",
                           llm_info=self.llm.info() if self.llm else {"provider": None, "mock": False,
                                                                       "note": "no LLM configured - demo mode"},
                           budget=budget.to_dict(), dataset_hint=dataset_hint, inject_fault=inject_fault,
                           is_fixture=is_fixture, auto_run=auto_run, started_at=time.time())
        self.save(st)
        self.db.add_event(pid, "create", f"Project created in {st.mode.upper()} mode", "info",
                          {"inject_fault": inject_fault})
        return pid

    # ---- Stage A -----------------------------------------------------------
    def run_ingest(self, pid: str) -> None:
        st = self.load(pid)
        CURRENT_PROJECT.set(pid)
        try:
            self._set_status(pid, S.INGESTING)
            self._guard(st)
            with open(os.path.join(self.project_dir(pid), "input.pdf"), "rb") as fh:
                pdf = fh.read()
            pages = ingest.extract_pages(pdf, self.s.max_upload_mb * 1024 * 1024, self.s.max_pdf_pages)
            self._log(st, "ingest", f"Extracted text from {len(pages)} pages (tool: pdf extractor)")
            spec = ingest.build_spec(pages, self.llm)
            st.spec = spec.to_dict()
            st.spec_original = spec.to_dict()
            st.spec_sha256 = spec_quality.spec_sha256(st.spec)
            if "LLM extraction failed" in spec.extraction_method:
                st.degraded.append("LLM extraction failed; the specification came from rule-based extraction. " + spec.extraction_method)
                self._log(st, "ingest", "LLM extraction failed - fell back to rule-based extraction", "warn")
            for iss in spec.validation_issues:
                self._log(st, "validate", iss, "warn")
            self._log(st, "ingest", f"Specification extracted via {spec.extraction_method}", data={"missing": spec.missing_info})
            if spec.security_flags:
                self._log(st, "security", f"{len(spec.security_flags)} prompt-injection pattern(s) found in the document; "
                          "content is treated as data only", "warn", {"flags": spec.security_flags})
            self.save(st)
            self._set_status(pid, S.AWAITING_SPEC)
        except Cancelled:
            self._cancelled(st)
        except Exception as e:
            self._fail(st, "ingest", e)

    # ---- gate 1 -> Stage B ------------------------------------------------------
    def approve_spec(self, pid: str, edits: dict | None, actor: str = "user") -> None:
        if self.status(pid) != S.AWAITING_SPEC:
            raise GateError("project is not awaiting spec approval")
        st = self.load(pid)
        if edits:
            # strict validation (wrong-field values are rejected with a message) -> merge -> recompute missing info
            spec, log = spec_quality.apply_user_edits(st.spec, edits)
            st.spec = spec.to_dict()
            st.spec_edits = list(st.spec_edits) + log
            self.db.audit("spec_edited", actor, pid, {"fields": sorted(e["field"] for e in log), "changes": log})
            if log:
                self._log(st, "approval", f"User corrected spec fields: {sorted(e['field'] for e in log)}")
        st.spec_sha256 = spec_quality.spec_sha256(st.spec)
        self.db.audit("spec_approved", actor, pid, {"spec_sha256": st.spec_sha256})
        self._log(st, "approval",
                  "Specification AUTO-approved by automatic pipeline" if actor == "auto"
                  else "Specification approved by user",
                  "warn" if actor == "auto" else "info")
        self.save(st)

    def run_plan(self, pid: str) -> None:
        st = self.load(pid)
        CURRENT_PROJECT.set(pid)
        try:
            self._set_status(pid, S.PLANNING)
            self._guard(st)
            spec = ResearchSpec.from_dict(st.spec)       # the APPROVED (possibly user-corrected) spec, never re-parsed
            plan = planner.make_plan(spec, self.llm, st.dataset_hint)
            plan.spec_sha256 = spec_quality.spec_sha256(st.spec)
            st.plan = plan.to_dict()
            self.save(st)
            if not plan.feasible:
                st.blocked_reason = plan.infeasible_reason
                self.save(st)
                self._log(st, "plan", plan.infeasible_reason, "warn")
                self._set_status(pid, S.BLOCKED)
                return
            self._log(st, "plan", f"Plan with {len(plan.tasks)} tasks ({plan.template}); {len(plan.assumptions)} assumption(s) need approval")
            self._set_status(pid, S.AWAITING_PLAN)
        except Cancelled:
            self._cancelled(st)
        except Exception as e:
            self._fail(st, "plan", e)

    # ---- gate 2 -----------------------------------------------------------------------
    def approve_plan(self, pid: str, accept_assumptions: bool, actor: str = "user") -> None:
        if self.status(pid) != S.AWAITING_PLAN:
            raise GateError("project is not awaiting plan approval")
        st = self.load(pid)
        plan = st.plan or {}
        if plan.get("blockers"):
            raise GateError("the plan has unresolved blockers; resolve them first (e.g. choose or upload a dataset): "
                            + "; ".join(plan["blockers"])[:400])
        if plan.get("assumptions") and not accept_assumptions:
            raise GateError("the plan contains assumptions that must be explicitly accepted")
        st.approved_assumptions = list(plan.get("assumptions", []))
        self.db.audit("plan_approved", actor, pid, {"assumptions": st.approved_assumptions})
        self._log(st, "approval",
                  "Plan and assumptions AUTO-accepted by automatic pipeline; execution permission granted"
                  if actor == "auto"
                  else "Plan and assumptions approved; execution permission granted",
                  "warn" if actor == "auto" else "info")
        st.started_at = time.time()
        self.save(st)

    def set_dataset(self, pid: str, source: str, actor: str = "user") -> dict:
        """At the plan gate the user may choose / upload a dataset. Recomputes dataset-dependent plan parts only;
        any substitution it creates becomes an assumption that must be approved."""
        if self.status(pid) != S.AWAITING_PLAN:
            raise GateError("a dataset can only be chosen while the plan awaits approval")
        st = self.load(pid)
        from .integrations.datasets import resolve_for_spec, ResourceError
        try:
            resolve_for_spec(ResearchSpec.from_dict(st.spec).datasets, source, self.s.hf_token, self.upload_dir(pid))
        except ResourceError as e:
            raise GateError(f"dataset rejected: {e}")
        st.dataset_hint = source
        plan = Plan(**{**st.plan, "tasks": [PlanTask(**t) for t in st.plan["tasks"]]})
        plan = planner.apply_dataset_choice(plan, ResearchSpec.from_dict(st.spec), source)
        st.plan = plan.to_dict()
        self.db.audit("dataset_selected", actor, pid, {"source": source})
        self._log(st, "dataset", f"User selected dataset source {source}")
        self.save(st)
        return st.plan

    def resume(self, pid: str) -> None:
        if self.status(pid) != S.FAILED:
            raise GateError("only failed projects with partial code generation can be resumed")
        st = self.load(pid)
        if not st.codegen_partial or not st.codegen_missing:
            raise GateError("no resumable code-generation progress is available")
        st.codegen_resumed = True
        st.blocked_reason = ""
        st.degraded.append("Code generation was resumed; missing calls are rerun while preserved files remain.")
        self.save(st)
        self.db.audit("codegen_resumed", "user", pid, {"missing": st.codegen_missing,
                                                       "preserved_files": sorted(st.codegen_partial)})
        self._log(st, "resume", "Resume requested; rerunning only missing code-generation calls", "warn")
        self._set_status(pid, S.RESOLVING)

    def upload_dir(self, pid: str) -> str:
        return os.path.join(self.project_dir(pid), "uploads")

    # ---- Stages C-H ---------------------------------------------------------------------------
    def run_execution(self, pid: str) -> None:
        st = self.load(pid)
        CURRENT_PROJECT.set(pid)
        try:
            plan = Plan(**{**st.plan, "tasks": [PlanTask(**t) for t in st.plan["tasks"]]})
            spec = ResearchSpec.from_dict(st.spec)
            if plan.spec_sha256 and plan.spec_sha256 != spec_quality.spec_sha256(st.spec):
                self._set_status(pid, S.RESOLVING)
                return self._block(st, "The specification changed after the plan was made; the plan is stale. Re-plan before running.")
            # Stage C
            self._set_status(pid, S.RESOLVING)
            self._guard(st)
            try:
                ds = datasets.resolve_for_spec(spec.datasets, st.dataset_hint, self.s.hf_token, self.upload_dir(pid))
            except datasets.ResourceError as e:
                return self._block(st, f"Dataset unavailable: {e}")
            st.dataset = ds.to_dict()
            self._log(st, "dataset", f"Resolved dataset {ds.id} ({ds.n_rows} rows, license: {ds.license})", data={"warnings": ds.warnings})
            if not ds.validated:
                return self._block(st, f"Dataset {ds.id} is metadata-only and not yet validated for use in the sandbox.")
            # Stage D
            self._set_status(pid, S.GENERATING)
            self._guard(st)
            if plan.template == "fisher_lda_iris":
                files = codegen.generate_template_project(spec, plan, st.inject_fault)
                self._log(st, "codegen", f"Generated {len(files)} files from deterministic template (no LLM)",
                          data={"files": sorted(files)})
                if st.inject_fault:
                    self._log(st, "codegen", "DEMO: a deliberate bug was injected into lda_repro/lda.py to exercise the repair loop", "warn")
            else:
                if not self.llm:
                    return self._block(st, "Code generation for this paper requires an LLM (set GEMINI_API_KEY).")
                if hasattr(self.llm, "set_cancel_event"):
                    self.llm.set_cancel_event(self.cancel_event(pid))
                if hasattr(self.llm, "set_warn_callback"):
                    self.llm.set_warn_callback(lambda msg: self._log(st, "llm", msg, "warn"))
                timeout_for = getattr(self.llm, "timeout_for", None)
                codegen_timeout = timeout_for(240) if callable(timeout_for) else 240
                t0 = time.time()
                self._log(
                    st,
                    "codegen",
                    f"Asking the LLM to write the project... usually 30-90 seconds; abandoned after {codegen_timeout}s per attempt",
                    "info",
                )
                try:
                    current_label = {"value": ""}
                    def codegen_progress(message: str, chunk_files: dict | None = None) -> None:
                        self._log(st, "codegen", message)
                        if chunk_files:
                            st.codegen_partial.update(chunk_files)
                            st.files = dict(st.codegen_partial)
                            for label in ("model implementation", "dataset and experiment runner",
                                          "unit tests", "configuration and documentation"):
                                if message.startswith("Generated " + label) and label not in st.codegen_completed:
                                    st.codegen_completed.append(label)
                            self.save(st)
                    files = codegen.generate_llm_project(spec, plan, self.llm, st.dataset,
                                                         progress=codegen_progress,
                                                         only_missing=st.codegen_missing if st.codegen_resumed else None,
                                                         initial_files=st.codegen_partial if st.codegen_resumed else None)
                except LLMTimeoutError as e:
                    elapsed = round(time.time() - t0, 1)
                    msg = f"Code generation timed out after {codegen_timeout}s per attempt ({elapsed}s elapsed)"
                    st.degraded.append(msg)
                    all_labels = ["model implementation", "dataset and experiment runner", "unit tests", "configuration and documentation"]
                    st.codegen_missing = [x for x in all_labels if x not in st.codegen_completed]
                    self._log(st, "codegen", msg, "error")
                    return self._fail(st, "codegen", e)
                except LLMError as e:
                    all_labels = ["model implementation", "dataset and experiment runner", "unit tests", "configuration and documentation"]
                    st.codegen_missing = [x for x in all_labels if x not in st.codegen_completed]
                    msg = f"Code generation stopped at a missing step: {', '.join(st.codegen_missing)}"
                    st.degraded.append(msg)
                    self._log(st, "codegen", msg, "error")
                    return self._fail(st, "codegen", e)
                elapsed = round(time.time() - t0, 1)
                if st.codegen_resumed:
                    st.codegen_missing = []
                    st.codegen_resumed = False
                    st.degraded.append("Code generation was resumed; missing code-generation calls were rerun and existing partial files were preserved.")
                self._log(st, "codegen",
                          f"LLM generated {len(files)} files in {elapsed}s; passed path/size/static-security validation")
            wd = self.work_dir(pid)
            _rmtree(wd)
            codegen.write_project(wd, files)
            if ds.user_supplied:                      # make the user's CSV visible to the sandboxed code at data/<name>
                os.makedirs(os.path.join(wd, "data"), exist_ok=True)
                shutil.copyfile(os.path.join(self.upload_dir(pid), ds.name), os.path.join(wd, "data", ds.name))
            st.files = files
            sb = self._concrete_sandbox()
            self._pinned[pid] = sb
            st.sandbox = {"mode": sb.mode, "isolated": sb.isolated, "note": sb.description}
            self.save(st)
            ok = self._test_and_run(st)
            if not ok:
                return
            self._evaluate(st, spec)
        except Cancelled:
            self._cancelled(st)
        except TimeoutError as e:
            self._block(st, f"Budget exhausted: {e}")
        except Exception as e:
            self._fail(st, "execution", e)

    # ---- Stage E/F loop -----------------------------------------------------------
    def _run(self, st: WorkflowState, stage: str, args: list[str]) -> SandboxResult:
        b = st.budget
        sb = self._pinned.get(st.project_id) or self._concrete_sandbox()
        res = sb.run_python(self.work_dir(st.project_id), args, b["exec_timeout_s"], b["memory_mb"], b["max_output_kb"])
        n = len(st.runs) + 1
        summary = summarize_unittest(res) if stage == "tests" else ("completed" if res.ok else "failed")
        st.runs.append({"n": n, "stage": stage, "ok": res.ok, "exit_code": res.exit_code, "duration_s": res.duration_s,
                        "summary": summary, "stdout": redact(res.stdout)[-4000:], "stderr": redact(res.stderr)[-4000:],
                        "mode": res.mode, "isolated": res.isolated, "timed_out": res.timed_out})
        self._log(st, "exec", f"[{stage}] sandbox({res.mode}) exit={res.exit_code} {summary}", "info" if res.ok else "error",
                  {"run": n})
        return res

    def _test_and_run(self, st: WorkflowState) -> bool:
        pid, b = st.project_id, st.budget
        wd = self.work_dir(pid)
        while True:
            self._guard(st)
            self._set_status(pid, S.TESTING)
            problems = [p for path, src in st.files.items() if path.endswith(".py") for p in scan_python(src, path)]
            if problems:
                self._log(st, "static", f"Static policy check failed: {problems[0]}", "error")
                res = SandboxResult(["static"], 1, "", "SecurityError: " + problems[0], 0.0)
                stage = "static"
            else:
                self._log(st, "static", "Static checks passed (syntax + security policy)")
                res = self._run(st, "tests", ["-m", "unittest", "discover", "-s", "tests", "-t", "."])
                stage = "tests"
                if res.ok:
                    self._set_status(pid, S.EXECUTING)
                    res = self._run(st, "experiment", ["run_experiment.py"])
                    stage = "experiment"
                    if res.ok:
                        if st.attempts:
                            st.attempts[-1]["outcome"] = "fixed: tests and experiment pass"
                        st.final_tests = next(r for r in reversed(st.runs) if r["stage"] == "tests")
                        self.save(st)
                        return True
            if st.attempts:
                st.attempts[-1]["outcome"] = f"still failing at {stage}"
            # failure -> bounded repair
            if len(st.attempts) >= b["max_repairs"]:
                st.final_tests = next((r for r in reversed(st.runs) if r["stage"] == "tests"), None)
                self._block(st, f"Repair budget exhausted after {len(st.attempts)} attempt(s); last failure at {stage}.")
                return False
            t0 = time.time()
            diag = repair.diagnose(res.stdout, res.stderr, st.files, res.timed_out)
            self._log(st, "diagnose", f"{diag.category}: {diag.summary}", "warn", diag.to_dict())
            patch, tool_calls = None, []
            if diag.repairable_by_rule:
                patch = repair.rule_based_patch(diag, st.files)
                tool_calls.append("rule_based_patch")
            if patch is None and self.llm and diag.file and diag.category not in ("timeout", "memory"):
                try:
                    patch = repair.llm_patch(diag, st.files, res.stderr, res.stdout, self.llm)
                    tool_calls.append("llm_patch")
                except LLMError as e:
                    self._log(st, "repair", f"LLM repair call failed: {e}", "error")
            if patch is None:
                st.final_tests = next((r for r in reversed(st.runs) if r["stage"] == "tests"), None)
                self._block(st, f"No safe automated repair available for: {diag.summary}")
                return False
            try:
                diff = repair.validate_patch(patch, st.files)
            except repair.PatchRejected as e:
                self._block(st, f"Proposed patch rejected by policy: {e}")
                return False
            self._set_status(pid, S.REPAIRING)
            codegen.write_project(wd, {patch.file: patch.new_content})
            st.files[patch.file] = patch.new_content
            attempt = {"n": len(st.attempts) + 1, "stage": stage, "diagnosis": diag.to_dict(), "patch": patch.to_dict(),
                       "diff": diff, "changed_files": [patch.file], "tool_calls": tool_calls,
                       "elapsed_s": round(time.time() - t0, 3), "outcome": "pending"}
            st.attempts.append(attempt)
            self._log(st, "repair", f"Attempt {attempt['n']}/{b['max_repairs']}: patched {patch.file} ({patch.source}) - {patch.rationale}",
                      "info", {"diff": diff})
            self.save(st)

    # ---- Stage G ---------------------------------------------------------------------------
    def _evaluate(self, st: WorkflowState, spec: ResearchSpec) -> None:
        pid = st.project_id
        self._set_status(pid, S.EVALUATING)
        self._guard(st)
        wd = self.work_dir(pid)
        try:
            with open(os.path.join(wd, "artifacts", "predictions.json"), encoding="utf-8") as fh:
                preds = json.load(fh)
            with open(os.path.join(wd, "artifacts", "run_meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            if spec.task_type == "regression":
                m = evaluation.regression_metrics([float(x) for x in preds["y_true"]], [float(x) for x in preds["y_pred"]])
            elif spec.task_type == "clustering":
                raise ValueError("no independent evaluator exists for clustering tasks; cannot verify")
            else:
                m = evaluation.classification_metrics(preds["y_true"], preds["y_pred"], preds.get("labels"))
        except Exception as e:
            return self._fail(st, "evaluate", e)
        # Fill useful provenance when an LLM-generated runner omits optional metadata.
        # Runtime is measured by the orchestrator around the experiment subprocess.
        if not meta.get("runtime_s"):
            experiment_runs = [r for r in st.runs if r.get("stage") == "experiment"]
            if experiment_runs:
                meta["runtime_s"] = experiment_runs[-1].get("duration_s")
                meta["runtime_source"] = "orchestrator subprocess wall time"
        env_meta = meta.get("environment") or {}
        import platform
        try:
            import numpy as _np
            numpy_version = _np.__version__
        except Exception:
            numpy_version = None
        try:
            import sklearn as _sklearn
            sklearn_version = _sklearn.__version__
        except Exception:
            sklearn_version = None
        fallback_env = {
            "python": platform.python_version(), "numpy": numpy_version,
            "scikit_learn": sklearn_version, "platform": platform.platform(),
            "source": "application host fallback; not guaranteed to match Docker image",
        }
        for key, value in fallback_env.items():
            if not env_meta.get(key) and value is not None:
                env_meta[key] = value
        meta["environment"] = env_meta
        st.metrics, st.run_meta = m, meta
        reported_acc = meta.get("script_reported_accuracy")
        if reported_acc is not None and "accuracy" in m:
            ok = abs(reported_acc - m["accuracy"]) < 1e-9
            st.script_metric_check = ("agrees with independent recomputation" if ok else
                                      f"MISMATCH: script said {reported_acc:.6f}, evaluator computed {m['accuracy']:.6f}")
            if not ok:
                self._log(st, "evaluate", st.script_metric_check, "error")
        summary = (f"accuracy={m['accuracy']:.4f} macro-F1={m['f1']:.4f}" if "accuracy" in m
                   else f"rmse={m['rmse']:.4f} mae={m['mae']:.4f}")
        self._log(st, "evaluate", f"Computed metrics independently from predictions: {summary} (n={m['n']})")
        run = {"dataset_name": (st.dataset or {}).get("name", ""), "test_size": meta.get("test_size"),
               "user_dataset": bool((st.dataset or {}).get("user_supplied"))}
        measured = evaluation.measured_values(m)
        implemented = [spec.title] + re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", spec.method)[:6]
        comps = evaluation.compare_results(spec.reported_results, measured, run, implemented)
        st.comparisons = [c.to_dict() for c in comps]
        tests_ok = bool(st.final_tests and st.final_tests["ok"])
        st.reproduction_status, st.reproduction_note = evaluation.overall_status(comps, tests_ok, True)
        if st.script_metric_check.startswith("MISMATCH"):
            st.reproduction_status, st.reproduction_note = "unable_to_verify", "Script-reported and independently computed metrics disagree."
        self._log(st, "evaluate", f"Reproduction status: {st.reproduction_status}. {st.reproduction_note}")
        self.save(st)
        self._set_status(pid, S.REPORTING)
        self._make_report(st)
        self._set_status(pid, S.COMPLETED)

    # ---- Stage H ---------------------------------------------------------------------------
    def _draft_narrative(self, st: WorkflowState) -> None:
        """Optional LLM summary written ONLY from verified records. It is labelled in the report and never affects status."""
        if not self.llm or st.narrative:
            return
        from .llm import prompts
        verified = {"status": st.reproduction_status, "note": st.reproduction_note, "metrics": {k: v for k, v in (st.metrics or {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}, "comparisons": st.comparisons,
                    "tests": (st.final_tests or {}).get("summary"), "repairs": len(st.attempts), "blocked": st.blocked_reason}
        try:
            out = self.llm.generate_json(prompts.narrative_prompt(verified))
            if isinstance(out.get("summary"), str):
                st.narrative = out["summary"][:2000]
        except Exception as e:
            self._log(st, "report", f"LLM summary skipped: {type(e).__name__}", "warn")

    def _make_report(self, st: WorkflowState) -> None:
        self._draft_narrative(st)
        d = st.to_dict()
        st.manifest = report.build_manifest(d)
        st.report_md = report.build_report_md(d)
        self.save(st)
        self._log(st, "report", "Report and reproducibility manifest generated")

    # ---- terminal helpers -----------------------------------------------------------------------
    def _block(self, st: WorkflowState, reason: str) -> None:
        st.blocked_reason = reason
        st.reproduction_status, st.reproduction_note = "unable_to_verify", f"Run blocked: {reason}"
        self._log(st, "blocked", reason, "error")
        self.save(st)
        self._make_report(st)
        self._set_status(st.project_id, S.BLOCKED)

    def _fail(self, st: WorkflowState, stage: str, e: Exception) -> None:
        # full details go to server logs only; users get a safe message
        import logging
        logging.getLogger("paper2product").exception("stage %s failed", stage)
        msg = redact(f"{type(e).__name__}: {str(e)[:300]}", [self.s.llm_api_key, self.s.github_token])
        st.blocked_reason = msg
        self._log(st, stage, f"Stage failed: {msg}", "error")
        self.save(st)
        cur = self.status(st.project_id)
        if cur not in TERMINAL:
            self._set_status(st.project_id, S.FAILED)

    def _cancelled(self, st: WorkflowState) -> None:
        self._log(st, "cancel", "Run cancelled", "warn")
        cur = self.status(st.project_id)
        if cur not in TERMINAL:
            self._set_status(st.project_id, S.CANCELLED)
