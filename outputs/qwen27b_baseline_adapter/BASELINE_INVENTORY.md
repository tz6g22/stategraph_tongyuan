# Baseline inventory

Inventory is based on executable source and environments present in this workspace, not the paper's method names.

| Baseline | Path | Venv | Current LLM backend | Embedding backend | Runner | Needs adapter |
|---|---|---|---|---|---|---|
| Mem0 | `baselines/mem0` (commit `001c2352`; old checkout retained) | `external_baselines/mem0/.venv` (kept in place) | OpenAI-compatible Mem0 LLM | FastEmbed `BAAI/bge-small-en-v1.5` in the project adapter | `baseline_adapters/adapters.py`; Table 3 runner | Localhost provider adapter; source import now resolves under `baselines/mem0` |
| Graphiti / Zep | `baselines/graphiti` (commit `96ef997` + `migration/patches/graphiti.patch`; old checkout retained) | `external_baselines/graphiti/.venv` (kept in place) | OpenAI Responses/Chat client | project adapter uses cached `sentence-transformers/all-MiniLM-L6-v2`; default upstream may use OpenAI embeddings | `baseline_adapters/adapters.py`; Table 3 runner | Local Chat completion; retained MiniLM embedding; local patch preserved |
| A-MEM | `baselines/amem` (commit `ceffb86`; old checkout retained) | `external_baselines/amem/.venv` (kept in place) | LiteLLM/OpenAI controller; project adapter replaces the call client | cached `sentence-transformers/all-MiniLM-L6-v2` | `baseline_adapters/adapters.py`; Table 3 runner | Localhost provider adapter; source import now resolves under `baselines/amem` |
| Letta / MemGPT | `baselines/letta` (commit `dca193b01`; old checkout retained) | `external_baselines/letta/.venv` (kept in place) | OpenAI-compatible endpoint; project adapter blocks before provider call | native default is OpenAI `text-embedding-ada-002`; no representation-compatible local weights verified | `baseline_adapters/adapters.py` | Blocked: no embedding substitution; no inference attempted |
| CUPMem (checked-in implementation) | `baselines/cupmem` (commit `ea7d391`; configured upstream is STALE, not CUPMem-specific) | No dedicated venv; uses existing Graphiti venv | OpenAI-compatible custom client routed to localhost | cached `all-MiniLM-L6-v2` | `baseline_adapters/adapters.py`; native `CupMemEngine` | Source import moved; native query-verification context issue remains |

LLM and embedding backends are listed separately. The source consolidation does not move the existing `.venv` directories, which remain at their old paths. Observed smoke status is in `COMPATIBILITY_MATRIX.md`; baseline presence does not mean every dataset pair is verified.
