# Change set: faithful extraction, grounded LLM use, honest reproduction

Root causes fixed
1. Table header cells ("Dataset") were parsed as section headings, so metric names/numbers landed in Datasets (extract_rules.find_tables).
2. Unrecognised headings never closed a section, so "3. Methodology" / "4. Experimental Setup" content was lost or leaked (classify_heading, keyword map).
3. No field-level validation anywhere (spec_quality.py); LLM output was trusted verbatim (now grounded in the paper text).
4. Review-gate edits left `missing_info` stale and nothing bound the plan to the approved spec (apply_user_edits, spec_sha256 check).
5. LLM token budget used a process-wide counter shared by all projects (now per project).
6. report.py emitted a literal "\\n" so sections 8 and 9 never rendered as headings.
7. Dataset hint `upload:<path>` accepted arbitrary server paths; there was no upload endpoint (safe_upload_name + /dataset-upload).
8. Run reported "LLM-assisted" even when extraction had silently fallen back to rules (now `degraded` notes).

9. LLM-written plan blockers are now moved to approval assumptions instead of permanently disabling plan approval; only unresolved dataset blockers remain hard blockers.
10. Gemini quota exhaustion is detected as a specific, immediate plan/billing error instead of wasting retries across fallback models.
11. Partial Gemini code generation is retained and can be resumed; failed projects expose a Resume action for missing calls only.
12. Opt-in auto-run supports attached CSV datasets and records all automatic approvals in the audit log.
