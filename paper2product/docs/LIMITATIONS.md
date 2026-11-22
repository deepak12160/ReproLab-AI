# Mocked, stubbed, demo-only or incomplete parts (inventory)

| Item | Status | How it is labelled |
|---|---|---|
| `sample_papers/fisher_lda_iris_fixture.pdf` | Synthetic fixture written for this project, includes a deliberate prompt-injection sentence; its "reported" 0.98 is not a published number | `SYNTHETIC DEMO FIXTURE` pill, report banner, manifest `is_synthetic_fixture` |
| DEMO MODE code generation | Only a fixed Fisher-LDA/Iris template; any other paper is BLOCKED with an explanation | `DEMO MODE` pill / report banner |
| Fault injection (`inject_fault`) | Deliberate bug to exercise the repair loop | README of generated project, event log |
| Rule-based extraction | Heuristic; works on text PDFs; table parsing handles one-metric-per-column tables with one cell per extracted line; scanned PDFs (no OCR) are rejected | `extraction_method`, `validation_issues`, missing-info list |
| Local sandbox | Subprocess with limits; **no network/filesystem isolation** | "NOT isolated" in UI, report, manifest |
| Docker sandbox | Implemented, command policy unit-tested, never executed in the build environment (no daemon) | - |
| Hugging Face datasets | Metadata lookup only; cannot be executed | plan blocker |
| Datasets | Bundled sklearn (iris, wine, breast_cancer, digits) and uploaded CSV only | plan blocker + substitution assumption |
| Evaluation | classification + regression metrics from predictions; AUC, clustering, multi-dataset papers are not verified | "not measured" / `partial` reproduction type |
| Multi-class Youden's J / G-Mean | One-vs-rest mean / geometric mean of recalls - definition may differ from the paper | comparison warning |
| Job queue, rate limiter | In-memory, single process | - |
| LangGraph | Not used (self-contained state machine) | docs |
| GitHub export, Hugging Face lookup | Tested against fakes only | - |
| Real Gemini behaviour | Provider logic tested with a mocked SDK call; extraction/plan/code quality on real papers not measured | README |
