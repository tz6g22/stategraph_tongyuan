# Output artifact manifest

The 37 GB local `outputs/` tree is not added wholesale. Current compatibility
summaries are the only included output artifacts:

| Artifacts | Disposition | Reason |
|---|---|---|
| `outputs/qwen27b_baseline_adapter/{ADAPTER_CHANGES,BASELINE_INVENTORY,COMPATIBILITY_MATRIX,DATASET_INVENTORY,EXTERNAL_PROVIDER_AUDIT,HANDOFF,MODEL_BACKEND,PROJECT_STATE,SMOKE_TEST_REPORT,TODO}.md` | committed | concise setup, status, and failure provenance needed to resume |
| `outputs/qwen27b_baseline_adapter/model_config.json` | committed | small non-secret model/server configuration |
| prepared source/gold JSONL, predictions, seals, traces, provider journals, raw logs, runtime stores | local-only | generated/reproducible or contains data/predictions; not needed to restore source code |
| all other benchmark results and `outputs/` subtrees | local-only | large result corpus; not copied into GitHub |
| `external_baselines/*/.venv`, caches, Redis/Chroma/Qdrant/FalkorDB stores | local-only; recreate on IRIDIS | platform-specific runtime state |
| Gitignored content | no deletion requested | original local artifacts are retained as rollback/evidence |

No source predictions or datasets are deleted by this freeze.
