# Baseline source consolidation

Baseline source code is consolidated under `baselines/`; adapter glue is under
`baseline_adapters/`. Source imports now resolve through these locations. Old
source checkouts remain intact until the GitHub push succeeds. Existing venvs
remain at their original paths and are excluded from Git.

| Baseline | Old path | New path | Upstream commit | Local modifications | Static check |
|---|---|---|---|---|---|
| Mem0 | `external_baselines/mem0` | `baselines/mem0` | `001c235229be8795e3834520467bd0d661ed8f34` | none | PASS: module source lookup, Python syntax compilation |
| Graphiti / Zep | `external_baselines/graphiti` | `baselines/graphiti` | `96ef997b265d05222233553bcf86a23e2d1dfd89` | one source change preserved/applied as `migration/patches/graphiti.patch` | PASS: module source lookup, Python syntax compilation |
| A-MEM | `external_baselines/amem` | `baselines/amem` | `ceffb860f0712bbae97b184d440df62bc910ca8d` | none | PASS: module source lookup, Python syntax compilation |
| Letta | `external_baselines/letta` | `baselines/letta` | `dca193b01cb260a558493f7feb725a004e1df7ae` | none | PASS: `memgpt` source lookup, Python syntax compilation |
| CUPMem-labelled local implementation | `external_baselines/cupmem_official` | `baselines/cupmem` | `ea7d391103a151927cd29d2f01d87597a782bdcb` | checkout clean; configured upstream is STALE, not a CUPMem-specific repository | PASS: `cup_mem` source lookup, Python syntax compilation |

The source manifests record upstream URLs and branch state. CUPMem provenance
is intentionally qualified: this checkout contains the local `cup_mem`
implementation but its configured origin is `https://github.com/icedreamc/STALE.git`.

The five baseline module lookups, Python compilation, JSON/TOML/YAML parsing,
and 14 non-inference adapter tests passed. No runtime baseline class was
instantiated and no provider was contacted. The adapter suite exposed a missing
standard-library `os` import in `baseline_adapters/prepare_smoke.py`; adding
that import made the suite pass and did not alter baseline algorithms.

The former `external_baselines/e2e_validation/` project-owned adapter glue was
copied to `baseline_adapters/`. Paths still mentioning
`external_baselines/<name>/.venv` refer only to intentionally unmoved virtual
environments; executable baseline source imports use `baselines/`.
