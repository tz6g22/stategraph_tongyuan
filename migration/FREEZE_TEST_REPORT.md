# Freeze-only validation

Only static source/config checks were run. No model endpoint, provider,
dataset runner, ingestion path, evaluator, benchmark, or StateGraph method was
run during freeze validation.

| Check | Result |
|---|---|
| Python syntax compilation | PASS: `python3 -m compileall -q baselines baseline_adapters evaluation_protocol scripts stategraph` |
| Non-inference adapter unit tests | PASS: 14 tests; helper logic only, with a mocked local token-count endpoint |
| Python source lookup | PASS: `mem0`, `graphiti_core`, `agentic_memory`, `memgpt`, and `cup_mem` resolve from the new `baselines/` trees |
| JSON/TOML/YAML parsing | PASS: 474 JSON files, 8 baseline `pyproject.toml` files, and 1 YAML config; TypeScript `tsconfig` JSONC excluded from standard JSON parsing |
| Secret and staged-file audit | PASS: no high-confidence secrets; no actual `.env`, `apikey/`, venv, model, DB, log, cache, nested `.git`, external checkout, or >25 MiB file staged |
| Qwen llama-server | stopped; port 8080 has no listener |
| External provider calls during freeze | 0 |

The adapter test suite initially exposed a missing standard-library `os`
import in `baseline_adapters/prepare_smoke.py`; that import was added and all
14 tests then passed. The vendored upstream baseline sources preserve upstream
whitespace as-is; no mass formatting or algorithm changes were made.
