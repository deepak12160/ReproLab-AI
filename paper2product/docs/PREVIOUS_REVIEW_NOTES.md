# Paper2Product AI – final review fixes

## Changes in this review
- Normalized common extracted metric labels (for example, `macro precision`, `macro recall`,
  and `macro F1-score`) before comparing reported values with independently measured metrics.
- Added regression coverage for those metric aliases.
- Added runtime and environment metadata fallbacks when an LLM-generated runner omits them.
  Fallback environment values are explicitly labelled as application-host values and may not
  describe a Docker container exactly.
- Added runtime/environment details to the Markdown reproduction report.
- Removed the bundled local SQLite runtime database from this source distribution. The app
  creates its database on first launch.

## Verification performed in this review
- Python compilation passed for the changed modules.
- `tests.test_evaluation`: 11 tests passed.
- `tests.test_export_security`: 3 tests passed.
- `tests.test_frontend_static`: 3 tests passed.
- `tests.test_sandbox`: 14 tests passed.

The full test suite was not completed in this review. A live Gemini API call and an end-to-end
Docker-isolated run were not performed here.

## Sandbox security – important
- `P2P_SANDBOX=docker` requires Docker Desktop/Engine to be running and the configured image
  to be available. In Docker mode, generated code is run with network disabled, read-only
  container root, dropped capabilities, non-root user, and resource limits.
- `P2P_SANDBOX=auto` or `local` can run with the local development subprocess fallback when
  Docker is unavailable. Local mode is **not isolated** and must only be used for trusted
  code on a development machine. Do not use it for untrusted generated code or public deployment.
- This package does not claim that local mode is a secure sandbox.
