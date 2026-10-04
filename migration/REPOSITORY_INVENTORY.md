# Repository inventory

The StateGraph root is the only Git repository intended for the freeze push.
The five baseline source snapshots are under `baselines/` and are not nested
repositories. Their former checkout source files and `.git` metadata were
removed only after remote verification; local `.venv`, cache, build, and runtime
data remain ignored. Exact source identities are in
`baselines/source_manifest.json`. Evaluator-only third-party checkouts remain
separate and are excluded from the StateGraph commit.

| Component | Path | Git repo | Branch/state | HEAD / source commit | Dirty at freeze |
|---|---|---|---|---|---|
| StateGraph main | `/home/cody/agent` | Yes; origin `https://github.com/tz6g22/stategraph_tongyuan.git` | `freeze/iridis-migration-2026-10-05` | freeze source commit `3e1ed09244eff2be2420b41897f58d96b985c582`; branch tip recorded in `GITHUB_FREEZE.md` | clean after commit |
| Mem0 | `baselines/mem0`; venv remains under ignored `external_baselines/mem0/.venv` | No nested repo | vendored snapshot | `001c235229be8795e3834520467bd0d661ed8f34` | clean source snapshot |
| Graphiti / Zep | `baselines/graphiti`; venv remains under ignored `external_baselines/graphiti/.venv` | No nested repo | vendored snapshot | `96ef997b265d05222233553bcf86a23e2d1dfd89` | local change preserved in `migration/patches/graphiti.patch` |
| A-MEM | `baselines/amem`; venv remains under ignored `external_baselines/amem/.venv` | No nested repo | vendored snapshot | `ceffb860f0712bbae97b184d440df62bc910ca8d` | clean source snapshot |
| Letta | `baselines/letta`; venv remains under ignored `external_baselines/letta/.venv` | No nested repo | vendored snapshot | `dca193b01cb260a558493f7feb725a004e1df7ae` | clean source snapshot |
| CUPMem-labelled implementation | `baselines/cupmem` | No nested repo; former origin was `https://github.com/icedreamc/STALE.git` | vendored snapshot | `ea7d391103a151927cd29d2f01d87597a782bdcb` | source snapshot clean; provenance is not CUPMem-specific |
| LongMemEval evaluator/source | `external_baselines/LongMemEval` | Yes; `https://github.com/xiaowu0162/LongMemEval.git` | `main` | `9e0b455f4ef0e2ab8f2e582289761153549043fc` | clean |
| LongMemEval-V2 evaluator | `external_baselines/LongMemEval-V2` | Yes; `https://github.com/xiaowu0162/LongMemEval-V2.git` | detached | `2cc8c540bdb87fe6761629b585e727e1c4704520` | clean |
| STALE evaluator | `external_baselines/stale_eval_official` | Yes; `https://github.com/icedreamc/STALE.git` | `main` | `ea7d391103a151927cd29d2f01d87597a782bdcb` | clean |
| llama.cpp | `external_baselines/llama.cpp` | Yes; `https://github.com/ggml-org/llama.cpp` | detached at `b11379` | `1537a0a8b2f8711d840878b0a0677ab2213c882c` | clean; build excluded |
| Memora dataset checkout | `/home/cody/data/memora` | Yes; `https://github.com/geniesinc/Memora.git` | local dataset checkout | `a6493188efc836d6511ed5e4163fe3ba87da30ff` | clean |

The freeze branch was based on refreshed `origin/main` at
`e59655fd2bf157ea7e9e5643444e4044b2d1b392`; no force push was used.
