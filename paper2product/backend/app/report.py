"""Stage H - evidence-backed report, reproducibility manifest and project ZIP export."""
from __future__ import annotations
import hashlib
import io
import json
import time
import zipfile

from .security import redact

TOOL_VERSION = "1.0.0"
STATUS_LABEL = {"reproduced": "Reproduced", "partially_reproduced": "Partially reproduced",
                "not_reproduced": "Not reproduced", "unable_to_verify": "Unable to verify"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(st: dict) -> dict:
    files = st.get("files", {})
    meta = st.get("run_meta") or {}
    return {
        "tool": "Paper2Product AI", "tool_version": TOOL_VERSION, "project_id": st.get("project_id"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "honesty": {
            "mode": st.get("mode"), "llm_used": st.get("mode") == "llm", "llm": st.get("llm_info"),
            "simulated_results": False,
            "note": ("DEMO MODE: no LLM reasoning was performed. Spec extraction used heuristic rules and code came from a "
                     "deterministic reviewed template. Test results and metrics are REAL outputs of code executed in the sandbox."
                     if st.get("mode") == "demo" else
                     ("LLM-assisted run; all model output was schema-validated and grounded in the paper text."
                      + (" DEGRADED: " + " ".join(st.get("degraded", [])) if st.get("degraded") else ""))),
        },
        "paper": {"title": (st.get("spec") or {}).get("title"), "extraction_method": (st.get("spec") or {}).get("extraction_method")},
        "is_synthetic_fixture": bool(st.get("is_fixture")), "degraded_steps": st.get("degraded", []),
        "spec_sha256": st.get("spec_sha256"), "user_spec_edits": st.get("spec_edits", []),
        "reproduction_type": (st.get("plan") or {}).get("reproduction_type"),
        "dataset": st.get("dataset"), "seed": meta.get("seed"), "test_size": meta.get("test_size"),
        "hyperparameters": (st.get("spec") or {}).get("hyperparameters"),
        "environment": meta.get("environment"),
        "sandbox": st.get("sandbox"), "budget": st.get("budget"),
        "tests": st.get("final_tests"), "runtime_s": meta.get("runtime_s"),
        "metrics": st.get("metrics"), "comparisons": st.get("comparisons"),
        "reproduction_status": st.get("reproduction_status"), "reproduction_note": st.get("reproduction_note"),
        "repair_attempts": len(st.get("attempts", [])), "usage": st.get("usage"),
        "approved_assumptions": st.get("approved_assumptions"),
        "auto_run": bool(st.get("auto_run")),
        "code_generation_resumed": bool(st.get("codegen_resumed")) or any("resumed" in str(d).lower() for d in st.get("degraded", [])),
        "file_sha256": {p: sha256(c) for p, c in sorted(files.items())},
        "rerun": ["pip install -r requirements.txt", "python -m unittest discover -s tests -t .", "python run_experiment.py"],
    }


def _fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _cell(x) -> str:
    return str(x if x not in (None, "") else "-").replace("|", "/").replace("\n", " ")[:160]


def build_report_md(st: dict) -> str:
    spec, plan, ds = st.get("spec") or {}, st.get("plan") or {}, st.get("dataset") or {}
    m, sb = st.get("metrics") or {}, st.get("sandbox") or {}
    meta = st.get("run_meta") or {}
    L: list[str] = []
    a = L.append
    status = st.get("reproduction_status", "unable_to_verify")
    prov_label = {"explicit": "Explicitly stated in the paper", "derived": "Derived from the paper", "user_provided": "User-provided",
                  "assumed": "Assumed", "not_reported": "Not reported"}
    a(f"# Reproduction Report: {spec.get('title', 'Untitled')}")
    a(f"\n**Reproduction status: {STATUS_LABEL.get(status, status)}**  \n{st.get('reproduction_note', '')}\n")
    a("> A passing unit test, a generated report or a successful launch does not prove that a paper has been reproduced. "
      "Status is derived only from independently measured metrics compared with reported results under matching dataset/split conditions.\n")
    if st.get("is_fixture"):
        a("> **SYNTHETIC DEMO FIXTURE** - the input is a PDF written for this project's demo, NOT a published paper. "
          "Nothing in this report is a reproduction of real research.\n")
    if st.get("auto_run"):
        a("> **AUTO-RUN** - the specification and plan approvals were performed automatically. No person reviewed these approvals.\n")
    if any("resumed" in str(d).lower() for d in st.get("degraded", [])):
        a("> **CODE GENERATION RESUMED** - missing code-generation calls were rerun; preserved partial files were not discarded.\n")
    if st.get("mode") == "demo":
        a("> **DEMO MODE** - no LLM was used. Extraction was rule-based and code came from a deterministic template. "
          "Metrics below are real outputs from code executed in the sandbox, not simulated.\n")
    for d_ in st.get("degraded", []):
        a(f"> **Degraded step:** {d_}\n")
    if sb and not sb.get("isolated"):
        a(f"> **Sandbox warning:** executed in `{sb.get('mode')}` mode - NOT isolated. {sb.get('note')}\n")
    a("## 1. Paper")
    a(f"- **Title:** {spec.get('title') or 'n/a'}\n- **Authors:** {', '.join(spec.get('authors', [])) or 'n/a'}\n"
      f"- **Source file:** {st.get('source_name') or 'uploaded PDF'}\n- **Abstract:** {spec.get('abstract') or 'n/a'}\n- **Problem:** {spec.get('problem') or 'n/a'}")
    a("\n## 2. Extracted specification (with how each part is known)")
    prov = spec.get("provenance", {})
    pages = spec.get("page_refs", {})
    rows = [("Method", spec.get("method"), "method"), ("Datasets", "; ".join(spec.get("datasets", [])), "datasets"),
            ("Preprocessing", "; ".join(spec.get("preprocessing", [])), "preprocessing"),
            ("Hyperparameters", json.dumps(spec.get("hyperparameters", {})) if spec.get("hyperparameters") else "", "hyperparameters"),
            ("Splits", "; ".join(spec.get("splits", [])), "splits"), ("Metrics", ", ".join(spec.get("metrics", [])), "metrics"),
            ("Baselines", ", ".join(spec.get("baselines", [])), "baselines"), ("Hardware", spec.get("hardware"), "hardware"),
            ("Dependencies", ", ".join(spec.get("dependencies", [])), "dependencies")]
    a("| Field | Value | Status | Pages |\n|---|---|---|---|")
    for name, val, key in rows:
        a(f"| {name} | {_cell(val) if val else '*not reported*'} | {prov_label.get(prov.get(key, 'not_reported'))} | "
          f"{', '.join(str(p) for p in pages.get(key, [])) or '-'} |")
    for e in spec.get("equations", []):
        a(f"\n- Equation: `{e}`")
    a(f"\n- **Extraction method:** {spec.get('extraction_method')}")
    if spec.get("missing_info"):
        a("- **Missing / unresolved in the paper:**")
        for x in spec["missing_info"]:
            a(f"  - {x}")
    if spec.get("validation_issues"):
        a("- **Automatic corrections made while extracting:**")
        for x in spec["validation_issues"]:
            a(f"  - {x}")
    if spec.get("security_flags"):
        a("- **Security flags (treated as data, never obeyed):**")
        for f in spec["security_flags"]:
            a(f"  - `{f}`")
    a("\n## 3. Evidence")
    ev = spec.get("evidence", [])
    if ev:
        a("| Field | Item | Page | Label | Source excerpt |\n|---|---|---|---|---|")
        for e in ev[:80]:
            a(f"| {e['field']} | {_cell(e['key'])} | {e.get('page') or '-'} | {prov_label.get(e['label'], e['label'])} | {_cell(e['excerpt'])} |")
    else:
        a("- No evidence excerpts were recorded.")
    a("\n## 4. User corrections and approvals")
    if st.get("spec_edits"):
        for e in st["spec_edits"]:
            a(f"- Edited **{e['field']}**: `{json.dumps(e['before'], default=str)[:200]}` -> `{json.dumps(e['after'], default=str)[:200]}`")
    else:
        a("- The user approved the extracted specification without changes.")
    a(f"- Approved spec hash: `{(st.get('spec_sha256') or '')[:16]}`")
    for x in st.get("approved_assumptions", []):
        a(f"- Approved assumption: {x}")
    a("\n## 5. Reproduction plan")
    a(f"{plan.get('approach', 'n/a')}\n")
    a(f"- **Reproduction type:** {plan.get('reproduction_type', 'n/a')} - {plan.get('reproduction_type_reason', '')}")
    a(f"- **Compute:** {plan.get('compute_needs') or plan.get('compute') or 'n/a'}")
    for t in plan.get("tasks", []):
        a(f"- **{t['id']}** {t['title']} (depends on: {', '.join(t['depends_on']) or '-'})")
    for x in plan.get("blockers", []):
        a(f"- Blocker: {x}")
    a("\n## 6. Dataset provenance")
    if ds:
        a(f"- **{ds.get('name')}** ({ds.get('id')}) from {ds.get('source')} - version `{ds.get('version')}`\n- License: {ds.get('license')}\n"
          f"- Access: {ds.get('access')}\n- Split: {ds.get('split')}\n- Expected files: {', '.join(ds.get('expected_files', [])) or '-'}\n"
          f"- Source: {ds.get('url') or '-'}\n- Rows/features/classes: {ds.get('n_rows')}/{ds.get('n_features')}/{ds.get('n_classes')}\n"
          f"- sha256: `{ds.get('sha256', '')[:16]}`\n- Provenance: {ds.get('provenance')}")
        if ds.get("differs_from_paper"):
            a("- **DIFFERS FROM THE PAPER** - user-approved substitution; paper numbers are not comparable.")
        for w in ds.get("warnings", []):
            a(f"- Warning: {w}")
    else:
        a("- No dataset was resolved.")
    a("\n## 7. Environment, seeds and hardware")
    env = meta.get("environment") or {}
    a(f"- **Seed:** {meta.get('seed', 'not recorded')}  |  **Test fraction:** {meta.get('test_size', 'not recorded')}")
    a(f"- **Experiment runtime:** {_fmt(meta.get('runtime_s'), 3)} s")
    a(f"- **Python:** {env.get('python') or 'not reported'}  |  **NumPy:** {env.get('numpy') or 'not reported'}  |  "
      f"**scikit-learn:** {env.get('scikit_learn') or 'not reported'}")
    if env.get("source"):
        a(f"- **Environment metadata source:** {env['source']}")
    a(f"- **Paper-stated hardware:** {spec.get('hardware') or 'not reported'}")
    a("\n## 8. Generated code and commands")
    for p, h in sorted((build_manifest(st)["file_sha256"]).items()):
        a(f"- `{p}` (sha256 `{h[:12]}`)")
    a("\n**Rerun instructions**")
    a("```bash\npip install -r requirements.txt\npython -m unittest discover -s tests -t .\npython run_experiment.py\n```")
    a("\n## 9. Execution history, logs and repairs")
    for r in st.get("runs", []):
        a(f"- Run {r['n']} [{r['stage']}] exit={r['exit_code']} {'PASS' if r['ok'] else 'FAIL'} in {r['duration_s']}s ({r.get('mode')}, "
          f"{'isolated' if r.get('isolated') else 'NOT isolated'}) - {r['summary']}")
        if not r["ok"] and r.get("stderr"):
            a(f"  ```\n  {r['stderr'][-600:]}\n  ```")
    a("\n### Repair attempts")
    if not st.get("attempts"):
        a("- None needed.")
    for at in st.get("attempts", []):
        a(f"- **Attempt {at['n']}** ({at['stage']} failure): {at['diagnosis']['summary']}\n  - Patch ({at['patch']['source']}): {at['patch']['rationale']}\n"
          f"  - Changed: `{at['patch']['file']}` | outcome: {at.get('outcome', 'pending')}\n\n```diff\n{at['diff']}\n```")
    ft = st.get("final_tests") or {}
    a(f"\n- Final unit tests: {ft.get('summary', 'not run')} ({'PASS' if ft.get('ok') else 'FAIL'})")
    a("\n## 10. Measured results (computed independently from predictions)")
    if m.get("task") == "classification":
        a(f"- n={m['n']} accuracy={_fmt(m['accuracy'])} macro-precision={_fmt(m['precision'])} macro-recall={_fmt(m['recall'])} macro-F1={_fmt(m['f1'])} "
          f"G-Mean={_fmt(m.get('g_mean'))} Youden-J={_fmt(m.get('youden_j'))} MCC={_fmt(m.get('mcc'))}")
        a(f"- Confusion matrix (labels {m['labels']}): `{m['confusion_matrix']}`")
    elif m.get("task") == "regression":
        a(f"- n={m['n']} MAE={_fmt(m['mae'])} RMSE={_fmt(m['rmse'])} R2={_fmt(m['r2'])}")
    else:
        a("- No metrics were measured.")
    if st.get("script_metric_check"):
        a(f"- Cross-check against the generated script's own accuracy: {st['script_metric_check']}")
    a("\n## 11. Comparison with reported results")
    a("Paper-reported values are shown beside independently computed values. Tolerance rule: "
      "relative difference <= 2% reproduced, <= 10% partially reproduced, otherwise not reproduced.")
    if not st.get("comparisons"):
        a("\n- No reported numerical results were available to compare.")
    else:
        a("\n| Metric | Dataset | Reported (page) | Measured | Abs diff | Rel diff | Direction | Comparable | Verdict |\n|---|---|---|---|---|---|---|---|---|")
        for c in st["comparisons"]:
            a(f"| {_cell(c['metric'])} | {_cell(c.get('dataset'))} | {_fmt(c['reported'])} (p. {c.get('page') or '?'}) | {_fmt(c['measured'])} | "
              f"{_fmt(c['abs_diff'])} | {_fmt(c['rel_diff'])} | {c.get('direction') or '-'} | {'yes' if c['comparable'] else 'NO'} | "
              f"**{STATUS_LABEL.get(c['status'])}** |")
        for c in st["comparisons"]:
            for w in c["warnings"]:
                a(f"- {c['metric']}: {w}")
    a("\n## 12. Reproduction status")
    a(f"**{STATUS_LABEL.get(status, status)}** - {st.get('reproduction_note', '')}")
    a("\n## 13. What is verified, claimed, assumed and unverified")
    a("**Verified by this system (executed / computed here):**")
    a(f"- Unit tests: {ft.get('summary', 'not run')}; experiment {'completed' if m else 'did not complete'}; metrics computed from predictions.")
    a("\n**Claimed by the paper (not verified by this system):**")
    for r in spec.get("reported_results", []):
        a(f"- {r['metric']} = {r['value']} (dataset: {r.get('dataset') or 'unstated'}, page {r.get('page') or '?'}, {prov_label.get(r.get('provenance', 'explicit'))})")
    if not spec.get("reported_results"):
        a("- (no numerical results extracted)")
    a("\n**Assumed:**")
    for x in plan.get("assumptions", []) or ["(none)"]:
        a(f"- {x}")
    a("\n**Unverified / limitations:**")
    for x in spec.get("limitations", []) + spec.get("missing_info", []) + plan.get("risks", []):
        a(f"- {x}")
    if st.get("narrative"):
        a("\n## 14. LLM-drafted summary (not a verified record; the status above does not depend on it)")
        a(st["narrative"])
    if st.get("usage"):
        a(f"\n_LLM usage for this project:_ `{json.dumps(st['usage'])}`")
    return redact("\n".join(L)) + "\n"


def build_zip(files: dict[str, str], report_md: str, manifest: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p, c in sorted(files.items()):
            z.writestr(f"project/{p}", c)
        z.writestr("REPORT.md", report_md)
        z.writestr("reproducibility_manifest.json", json.dumps(manifest, indent=2))
    return buf.getvalue()
