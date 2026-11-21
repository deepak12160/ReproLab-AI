# Paper2Product AI

> Upload a research paper. Get an implementation, a reproducible experiment, verified execution evidence,
> and an **honest** comparison with the reported results.

Hackathon track: **AI and Developer Tools**. Paper2Product is a tool-using agent: it reads a paper, asks you to approve
its understanding, plans, resolves data, generates a project, runs it in a sandbox, reads failures, repairs them within a
budget, measures metrics itself, and reports exactly what was and was not verified.

## Start it (no technical knowledge needed)

**Windows 11:** double-click **`START_WINDOWS.bat`**.
The first run installs everything (1-3 minutes, internet needed). Your browser opens by itself. Leave the black window open; close it to stop.
If it says Python is missing: install Python 3.13 from python.org and tick **"Add python.exe to PATH"** on the first screen.

**macOS / Linux:** `python3 run.py`

Other commands: `python run.py --test` (run all tests), `python run.py --demo` (demo in the terminal, no browser).

## Two modes

| Mode | When | What it can do |
|---|---|---|
| **DEMO MODE** (default) | No API key | Full workflow on the bundled demo paper (Fisher LDA on Iris). Rule-based extraction + a reviewed code template. Every execution and metric is real, and the UI/report say "DEMO MODE". |
| **LIVE Gemini MODE** | `GEMINI_API_KEY` set | Gemini extracts the spec, plans, writes code and proposes repairs for other papers (classification on bundled scikit-learn datasets: iris, wine, breast_cancer, digits). |

**To enable LIVE mode:** get a free key at https://aistudio.google.com/apikey, open the `.env` file in the project folder
(created on first start) in Notepad, write `GEMINI_API_KEY=your-key` (no quotes), save, restart. Click **Test Gemini** in the header to check it.
The default model is `gemini-3.8-flash`; if a model is unavailable for your key, older ones are tried automatically.

## Demo flow (about 4 minutes)
1. Dashboard -> **Start demo (with injected fault)**.
2. Spec gate: show the page-referenced extraction and the security banner (a prompt-injection sentence hidden in the PDF is treated as data). Approve.
3. Plan gate: tick the assumptions box (the Approve button is refused without it) -> **Approve & run**.
4. Watch the live log: tests fail -> diagnosis -> automatic repair -> tests pass -> experiment -> evaluation.
5. Code & diffs, Experiment (metrics, confusion matrix, reported vs measured), Report (download ZIP / manifest).
Typical result: accuracy 0.9556 vs fixture reference 0.98 -> **Partially reproduced** (<=2% relative difference = reproduced, <=10% = partial).

The demo PDF `sample_papers/fisher_lda_iris_fixture.pdf` is a **synthetic fixture written for this project, not a published paper**;
its "reported" 0.98 is a reference value, not a published benchmark.

## Honesty rules for the reproduction status
* **Reproduced:** every comparable metric within 2% relative difference, all unit tests pass, and no reported number was left unverified.
* **Partially reproduced:** within 10%, or some numbers matched but others could not be validly compared, or unit tests failed.
* **Not reproduced:** beyond 10%. **Unable to verify:** experiment failed or dataset/split/metric not comparable.
* Exports (ZIP, report, manifest, GitHub) are scrubbed: `.env`, key/pem files and API-key patterns can never be included.

## Sandbox and security
* **Docker installed and running:** code runs in a container with `--network none --read-only --cap-drop ALL --security-opt no-new-privileges`,
  non-root user, memory/CPU/PID limits, no secrets and no Docker socket. A sandbox image (numpy + scikit-learn) is built automatically in the background the first time (internet needed once); until it is ready the labelled local sandbox is used.
* **No Docker:** a **local development sandbox** is used (timeout, output cap, scrubbed environment; memory limit only on Linux/macOS). It is **not a security boundary** and the UI, report and manifest say "NOT isolated". Do not run untrusted papers with it.
* `P2P_SANDBOX=docker` forces Docker and refuses to start without it; it never silently downgrades.
* Other controls: PDF size/page limits, prompt-injection scan with `<untrusted_document>` wrapping, path allowlist, AST policy scan of generated code, only `python -m unittest` / `python <script>.py` can run, repairs can never edit tests, secrets redacted, rate limiting, optional API key, GitHub export only after explicit confirmation.

## Tests - honest status
Run `RUN_TESTS_WINDOWS.bat` or `python run.py --test`.

Last results (Linux, Python 3.12, fresh environment with latest packages): **134 passed, 3 skipped, 0 failed**. The 3 skipped tests need real Gemini/GitHub credentials and are never counted as passed.
The web page was also driven end-to-end against the real server in a simulated browser (jsdom): demo -> approvals -> repair -> metrics -> report.

| Area | Status |
|---|---|
| Engine (ingest -> plan -> data -> codegen -> sandbox -> repair -> evaluate -> report) | Verified by automated tests and an end-to-end browser-simulation (jsdom) test |
| Generic LLM pipeline (non-template paper, Wine dataset) | Verified with a **stand-in model**, not real Gemini |
| Gemini model fallback, bad-key and rate-limit handling | Verified with a stand-in for the SDK call |
| Real Gemini calls | **Not live-tested** (needs your key; use the Test Gemini button) |
| Windows 11 / Python 3.13 | Windows code paths written and partly simulated; **never executed on real Windows** |
| Docker sandbox | Command policy unit-tested; **never executed** in the build environment |
| GitHub export, Hugging Face lookup | Tested against fakes; no live calls |
| Real browser (Chrome/Edge) | Not used; the page was driven in jsdom |

To run the 3 live tests yourself: set `P2P_LIVE_TESTS=1` plus `GEMINI_API_KEY` / `GITHUB_TOKEN`.

## Known limitations
* Without a key only the Fisher-LDA/Iris template runs. Other papers need Gemini, and LLM-written code quality varies.
* Datasets: bundled scikit-learn ones and CSV uploads; Hugging Face is metadata only.
* Heuristic extraction needs conventional section headings; use the approval gate to correct it.
* In-memory job queue and rate limiter (single process); SQLite storage; no multi-user auth.
* Passing tests do not prove reproduction: status comes only from measured-vs-reported comparison with dataset/split checks.

## Settings (`.env`, all optional)
`GEMINI_API_KEY`, `P2P_LLM_PROVIDER` (auto|gemini|demo), `P2P_LLM_MODEL`, `P2P_SANDBOX` (auto|docker|local), `P2P_SANDBOX_IMAGE`,
`GITHUB_TOKEN`, `HF_TOKEN`, `P2P_DATA_DIR`, `P2P_API_KEY`, `P2P_MAX_UPLOAD_MB`, `P2P_RATE_LIMIT_PER_MIN`. See `.env.example`.

Docker for the app itself: `docker compose up --build` (uses the local sandbox on purpose; the host Docker socket is never mounted).
