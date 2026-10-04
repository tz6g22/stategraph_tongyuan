# Repository inventory

The StateGraph root is the only repository intended for this freeze push.
Third-party repositories remain separate local checkouts and are not nested
Git repositories in the StateGraph commit. Their source snapshots are vendored
under `baselines/`; their exact upstream identities are in
`baselines/source_manifest.json`.

| Component | Path | Git repo | Branch/state | HEAD | Dirty at freeze |
|---|---|---|---|---|---|
| StateGraph main | `/home/cody/agent` | Yes; origin `https://github.com/tz6g22/stategraph_tongyuan.git` | `freeze/iridis-migration-2026-10-05` (created for freeze) | see `GITHUB_FREEZE.md` | freeze changes committed |
| Mem0 upstream checkout | `external_baselines/mem0` | Yes; `https://github.com/mem0ai/mem0` | detached | `001c235229be8795e3834520467bd0d661ed8f34` | clean |
| Graphiti upstream checkout | `external_baselines/graphiti` | Yes; `https://github.com/getzep/graphiti` | `main` | `96ef997b265d05222233553bcf86a23e2d1dfd89` | one local file change; preserved in `migration/patches/graphiti.patch` |
| A-MEM upstream checkout | `external_baselines/amem` | Yes; `https://github.com/agiresearch/A-mem` | `main` | `ceffb860f0712bbae97b184d440df62bc910ca8d` | clean |
| Letta upstream checkout | `external_baselines/letta` | Yes; `https://github.com/letta-ai/letta` | detached | `dca193b01cb260a558493f7feb725a004e1df7ae` | clean |
| CUPMem-labelled checkout | `external_baselines/cupmem_official` | Yes; configured origin is `https://github.com/icedreamc/STALE.git` | `main` | `ea7d391103a151927cd29d2f01d87597a782bdcb` | clean; provenance is not CUPMem-specific |
| LongMemEval evaluator/source | `external_baselines/LongMemEval` | Yes; `https://github.com/xiaowu0162/LongMemEval.git` | `main` | `9e0b455f4ef0e2ab8f2e582289761153549043fc` | clean |
| LongMemEval-V2 evaluator | `external_baselines/LongMemEval-V2` | Yes; `https://github.com/xiaowu0162/LongMemEval-V2.git` | detached | `2cc8c540bdb87fe6761629b585e727e1c4704520` | clean |
| STALE evaluator | `external_baselines/stale_eval_official` | Yes; `https://github.com/icedreamc/STALE.git` | `main` | `ea7d391103a151927cd29d2f01d87597a782bdcb` | clean |
| llama.cpp | `external_baselines/llama.cpp` | Yes; `https://github.com/ggml-org/llama.cpp` | detached at tag `b11379` | `1537a0a8b2f8711d840878b0a0677ab2213c882c` | clean; build directory excluded |
| Memora dataset checkout | `/home/cody/data/memora` | Yes; `https://github.com/geniesinc/Memora.git` | local dataset checkout | `a6493188efc836d6511ed5e4163fe3ba87da30ff` | clean |

Freeze remote observation: local `origin/main` was refreshed from GitHub and is
`e59655fd2bf157ea7e9e5643444e4044b2d1b392`; the freeze branch is based on that
commit. No force push is planned.
