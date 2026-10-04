# Dataset manifest

Third-party datasets remain outside GitHub. Restore them under an IRIDIS data
root (for example `$DATA_ROOT`) from their already-approved local/project
source or the upstream dataset release; do not infer that the smoke artifacts
are the datasets themselves. Only the small project-authored SCB v5 file is
included in this repository, byte-for-byte.

| Dataset | Local path | Version/file | SHA256 | Git tracked? | IRIDIS restore method |
|---|---|---|---|---|---|
| StateChangeBench | `/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v5.jsonl`; copy at `data/benchmarks/statechangebench_cases_001_050_v5.jsonl` | project-authored v5, 50 cases, 199,181 bytes | `77d9c6c90e01a5bc5c1838d64cf523bd828c9a88c6f13711586409b976f60725` | Yes, copy only | use repository copy or set `STATECHANGEBENCH_V5_PATH` |
| STALE | `/home/cody/data/stale/T1_T2_400_FULL.json` | T1/T2 400-case source | `5f3ec375179e20e2e94469e018189188f34e2e7e5f21cbecbd99fcfa648c1876` | No | restore approved source file under `$DATA_ROOT/stale/` |
| LongMemEval-S | `/home/cody/data/longmemeval/longmemeval_s_cleaned.json` | cleaned S | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` | No | restore dataset release under `$DATA_ROOT/longmemeval/` |
| LongMemEval Oracle | `/home/cody/data/longmemeval/longmemeval_oracle.json` | Oracle query subset | `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` | No | restore approved source under `$DATA_ROOT/longmemeval/` |
| LongMemEval-M | `/home/cody/data/longmemeval/longmemeval_m_cleaned.json` | cleaned M | `9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f` | No | restore dataset release under `$DATA_ROOT/longmemeval/` |
| LongMemEval-V2 | `/home/cody/data/longmemeval_v2/questions.jsonl`, `trajectories.jsonl`, `haystacks/lme_v2_small.json`, `haystacks/lme_v2_medium.json` | text/multimodal-capable V2 files | `0a3ae5ebea938c24d7800e1e0b0828e08ae1646f939a53853b2b8cdc08e292b7`; `363cec9a8e87aa8d9101ce4e600aadbf7031d674056ebe4f969e8424abc5f3c6`; `9b5301defb23a088a5f06e45ff8d5f35e569d78305a66d492046a9fff9b46593`; `4756d5126347f0d18f045bb6c47b08cb3b23e9db24386cc48a9b2879e7969b59` | No | restore the matching LongMemEval-V2 data release under `$DATA_ROOT/longmemeval_v2/` |
| MemoryAgentBench | `/home/cody/data/memoryagentbench/data/*.parquet` | Conflict Resolution, Accurate Retrieval, Long Range Understanding, Test Time Learning | Conflict: `24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45`; Accurate: `56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1`; Long Range: `5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539`; Test Time: `5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0` | No | restore matching parquet files and README under `$DATA_ROOT/memoryagentbench/` |
| Memora | `/home/cody/data/memora` | dataset checkout commit `a6493188efc836d6511ed5e4163fe3ba87da30ff` | use Git commit as source identity; no aggregate directory hash | No | clone `https://github.com/geniesinc/Memora.git`, checkout the recorded commit, set `$DATA_ROOT/memora/` |

No third-party dataset or gold file is included in the GitHub source freeze.
