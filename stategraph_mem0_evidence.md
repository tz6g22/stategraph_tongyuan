# StateGraph / Mem0 证据审查

本文件是只读审查报告。没有修改实现、benchmark、gold、prediction 或 seal；没有重新运行付费实验，也没有调用 API。文中标签含义：`[代码事实]` 来自当前源码，`[产物观察]` 来自已有 trace/manifest/evaluator，`[推断]` 是基于前两者的诊断，`[版本错配]` 表示不能作受控比较。

## 版本与实验清单

### StateGraph

- 仓库 `/home/cody/agent` 当前没有 `.git`，因此没有可报告的 StateGraph commit。当前代码版本锚点是最终 Graphiti decoupling freeze：`outputs/stategraph_decoupling_module6_cleanup_v1/FREEZE.json`，`status=PASS`，`source_digest=9c20fadf5823ebb74787e3ffb6649a84e48caca1b17fb0a7aa2d09b38eee75ee`，生成时间 `2026-09-13T12:57:09.557432+00:00`。[产物观察]
- freeze 还记录 `graphiti_required_for_native_stategraph=false`、`graphiti_runtime_imports_in_native_path=0`、`graphiti_calls_in_native_pipeline=0`、`semantic_equivalence=true`。Architecture validation 同样记录 `graphiti_modules_loaded=[]`、`graphiti_calls=0`，native classes 为 `StateGraphNativeStateExtractor`、`NativeStateGraphBackend`、`StateGraphNativeRetriever`。[产物观察]
- fresh 诊断 manifest：`outputs/professor_minimal_post_decoupling_gpt5nano_v1/RUN_MANIFEST.json`；run id `professor-minimal-post-decoupling-v1`，OpenAI / `gpt-5-nano` / `minimal`，answer budget 由 traces 显示为 512。固定输入为 SCB `SCB_012/013/017`、Memora `activity_todos_158`、LongMemEval `5d3d2817`、MAB `row0-question0`、STALE `7c0ae4e7-...`，另有 LongMemEval-V2 `0f970f01` official 510-observation scope。[产物观察]
- 当前 fresh runner 的持久源码没有找到 `professor-minimal-post-decoupling-v1` 字符串；因此“实际 orchestration runner 文件”本身为 `NOT_RECORDED`。class/path、请求 hash、原始响应和调用阶段由 trace 直接证明，但不能把某个历史脚本冒充为 fresh runner。[版本错配]

### Mem0

- external baseline 是 `/home/cody/agent/external_baselines/mem0`，Git commit `001c235229be8795e3834520467bd0d661ed8f34`，日期 `2026-08-14`，消息 `Fix broken links: remove duplicate reranking redirect (#6975)`；工作树无未提交修改。[代码事实]
- `external_baselines/mem0/pyproject.toml:7` 声明版本 `2.0.18`；Mem0 包的 `__version__` 从 installed distribution metadata 读取，因此实际 SDK 版本为 `mem0ai 2.0.18`。[代码事实]
- 实验 adapter 是 `external_baselines/e2e_validation/adapters.py:23-89` 的 `Mem0Adapter`。它以 `Memory.from_config` 创建 Qdrant 本地库（`collection_name=baseline_e2e`, 384 dims）、FastEmbed `BAAI/bge-small-en-v1.5`，LLM provider `openai`，fresh OpenAI 配置为 `gpt-5-nano`、`reasoning_effort=minimal`、`is_reasoning_model=true`。`add_memory` 调 `memory.add(text,user_id='baseline-e2e')`，`query` 调 `memory.search(text, filters={'user_id':...}, top_k=5)`。[代码事实]
- adapter 将 OpenAI client 包成兼容 wrapper：timeout 默认 180 秒、`max_retries=1`，移除 `max_tokens`，设置 `max_completion_tokens=2048` 和 `reasoning_effort=minimal`（`adapters.py:50-80`）。这是 provider 参数/错误恢复适配，不是 Mem0 memory/retrieval 算法改写；但它使 Mem0 实际内部 completion budget 不是 `Memory.from_config` 中声明的 512。[代码事实]
- adapter 文件仍有 DeepSeek fallback 分支（`adapters.py:30-47`），但 fresh manifest 明确 provider 为 OpenAI；本报告未将旧 DeepSeek 结果混入。[产物观察]

### 结果范围和完整性

`RUN_MANIFEST.json` 和各 seal 显示 fresh post-decoupling 完成 StateGraph 7/8、Mem0 7/8，共 14/16。已 sealed 的完整数据集是：SCB 3+3、Memora 1+1、LongMemEval 1+1、MAB 1+1、STALE 1+1。LongMemEval-V2 StateGraph 在 215/510 后 timeout，Mem0 被手动停止且没有 seal；partial 不评分。[产物观察]

| 数据集 | fresh execution | sealed prediction | fresh metric 状态 |
|---|---:|---:|---|
| StateChangeBench | SG 3/3, Mem0 3/3 | 两边均 SEALED | `NOT_COMPUTABLE`：canonical v2 gold 缺失，未以 v3 替代 |
| Memora | SG 1/1, Mem0 1/1 | 两边均 SEALED | `NOT_COMPUTABLE`：fresh rubric/judge 输出不存在 |
| LongMemEval | SG 1/1, Mem0 1/1 | 两边均 SEALED | 可计算：SG F1/EM 0/0，Mem0 F1 0.5/EM 0 |
| MAB Conflict | SG 1/1, Mem0 1/1 | 两边均 SEALED | 可计算：两边 F1/EM 0/0 |
| STALE | SG 1/1, Mem0 1/1 | 两边均 SEALED | `NOT_COMPUTABLE`：official judge 未运行 |
| LongMemEval-V2 | SG incomplete 215/510；Mem0 manually stopped | SG seal 非完成；Mem0 无 seal | `N/A`，不能把 partial 当 0 |

Fresh runtime 只对 StateGraph 暴露完整 semantic calls；Mem0 `api_calls` 主要记录 answer client call，**不包含 adapter 内部 add/extraction/embedding/vector-store 成本**。因此表中的“调用数/耗时”不是 compute-matched comparison。[产物观察]

## Runtime flow

### Native StateGraph path（当前 fresh post-decoupling）

```text
raw Observation
  -> ObservationRecord
  -> StateGraphNativeStateExtractor.extract
  -> strict parse + source grounding + validators
  -> EvidenceRecord[] + StateCandidate[]
  -> StateGraph._ingest_unprofiled
  -> StateLinker / StateRevision / lifecycle
  -> dependency discovery -> relation typing -> verification
  -> StateGraph DependencyGraph persistence
  -> STRICT-only invalidation propagation
  -> StateGraphNativeRetriever (repository candidate source)
  -> premise check / stale filtering / dependency expansion
  -> grounded context
  -> shared answer call
```

关键代码直接证据：

- `stategraph/system.py:80-117`：constructor 接收 `backend`，无 backend 时默认 `NativeStateGraphBackend(repository)`；retriever 固定为 `StateGraphNativeRetriever(self.repository)`，注释明确 backend search adapter 不是 fallback。
- `stategraph/system.py:198-237`：先把 raw `Observation` 转成 `ObservationRecord`，调用 `self.extractor.extract(native_observation)`，得到 `ExtractionResult`；之后才调用 `self.backend.persist_observation(...)`。这证明 native extraction 不以前置 Graphiti episode/fact 为条件。
- `stategraph/system.py:291-340`：用 pre-revision snapshot 做 linking，构造 `StateNode`；`state.evidence_refs/evidence_ids` 是通用证据字段，`graphiti_fact_ids=backend_ids` 只作为 backend 返回的兼容 metadata。
- `stategraph/system.py:433-470`：从 `new_states` 发现 dependency candidates；`system.py:568-626` 依次 relation typing 与 typed verification；`system.py:678-685` 持久化 verified relation，再调用 invalidation propagation。
- `stategraph/system.py:802-814`：retrieve 委托给 native retriever。`stategraph/retrieval/native.py:20-34,49-73` 的 candidate source 直接 `repository.list_states`，并将 `graph_search=None` 传给父类。
- `stategraph/retrieval/current_state_retriever.py:232-403`：列出 CURRENT/UNCERTAIN 与 STALE/HISTORICAL，执行 premise extraction、状态评分、dependency expansion、stale/history filtering，并从 repository 读取 evidence 做 grounding；`current_state_retriever.py:576-605` 说明先按 query intent/StateGraph score 选候选，再扩展 typed dependency。
- `stategraph/answer_generation.py:24-49` 只允许 CURRENT 与 unresolved-conflict UNCERTAIN 进入 answer context；不会把 STALE/HISTORICAL 直接传给答案层。

### Native extraction 的实际 schema、prompt、默认值和失败处理

`stategraph/state/native_extraction.py` 是当前 native semantic extractor：

```python
# native_extraction.py:33-53
STATE_EXTRACTION_OUTPUT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'states': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'entity': {'type': 'string'},
            'attribute': {'type': 'string'},
            'value': {'type': ['string', 'number', 'boolean', 'null']},
            'evidence_span': {'type': 'string'},
        },
        'required': ['entity', 'attribute', 'value', 'evidence_span'],
    }}},
    'required': ['states'],
}
```

- `native_extraction.py:67-94`：`native_observation_only=True`，默认 `max_output_tokens=8192`，要求 client 有 `generate_response`。
- `native_extraction.py:110-155`：system prompt 要求每个 explicit source-grounded proposition、split clauses、exact contiguous `evidence_span`、不使用 query/gold/ontology/graph/backend；user payload 只含 raw `observation`、`reference_time`、`state_schema`；实际调用参数是 `prompt_name='stategraph.state_extraction.v2'`, `max_tokens=8192`, `candidate_schema=STATE_EXTRACTION_OUTPUT_SCHEMA`。
- `native_extraction.py:215-247`：解析非 object/缺字段/找不到 exact evidence/subject 不在 evidence/meta greeting/attribute 不与 evidence token 相交的项目时拒绝；拒绝记录 reason，不做 fuzzy repair。
- `native_extraction.py:250-312`：只有通过 grounding 的候选才生成 EvidenceRecord/StateCandidate；exact tuple `(entity, attribute, value, evidence_span)` 去重。`native_extraction.py:366-376`：未给 time_scope 时默认 observation timestamp，坏时间也退回 timestamp；confidence 缺省为 1.0（`285`），conditions 只保留同时出现在 evidence 的 key/value（`379-388`）。

重要 schema 事实：native provider schema 只允许四个 state 字段（entity/attribute/value/evidence_span），但 parser 仍支持 `canonical_field_id`、`time_scope`、`condition_scope`、`confidence`、`invalidates`、`conflicts` 等 optional 字段（`269-299`）。由于 `additionalProperties=False`，这些 optional 字段不能由当前 native structured schema 正常发出；缺失时走上述默认值。这里是代码直接证明的表示能力限制，不等同于每个错误的唯一原因。[代码事实]

### Compatibility / legacy path

- `stategraph/graphiti_adapter/state_extraction.py:68-114` 的 `GraphitiLLMStateExtractor` 是 compatibility facade；`native_mode=False` 是默认值，内部仍构造 `AutomaticDependencyDiscovery`，并保留 `GraphitiFact` 两参数接口。
- `state_extraction.py:169-203`：没有 `graphiti_facts` 或输入为 `ObservationRecord` 时转 native；传入 Graphiti facts 时进入 legacy path。
- `state_extraction.py:214-261`：legacy path 使用 lossless 1800-character chunks 和 fact partition；`263-377` 构建更丰富的 Graphiti-backed prompt，user payload 含 `graphiti_facts` 和 `supporting_fact_ids` 等字段。
- `stategraph/evaluation/run_stategraph_10.py:25-29,193-202` 直接 import `create_graphiti`, `GraphitiBackend`, `GraphitiLLMStateExtractor`，创建 Graphiti 并用该 extractor；这是历史/compatibility evaluator，不是当前 fresh native trace 的来源。[代码事实]
- `scripts/run_minimal_matrix_method.py:94-112` 的历史 SCB runner 也显式构造 `GraphitiLLMStateExtractor`；`run_agent` 在 `:136-162` 调 `stategraph.evaluation.run_stategraph_10`。这些路径不能代表 post-decoupling fresh run。[版本错配]

因此，当前代码存在两种可执行接口，但 fresh trace 的 `ARCHITECTURE_VALIDATION.json` 证明本轮 native path 没有加载 Graphiti；历史 runner 源码则仍是 compatibility/Graphiti path。两者的中间状态数量不能直接作效果或速度归因。[产物观察][版本错配]

## 关键代码：linking、revision、dependency、propagation、retrieval

### Linking / revision

- `stategraph/state/linking.py:64-100`：identity 先看 slot-grounding/canonical slot；无 canonical slot 时要求 entity identity 相同且 `attributes_compatible`，否则最多 `POSSIBLE_SAME_SLOT`，不会单靠相似度直接 revision。
- `stategraph/state/linking.py:102-149`：`link` 对 same slot、explicit effect/conflict、time/condition overlap 计分；`stategraph/revision/state_revision.py:30-...` 的 `StateRevision.revise` 才应用 lifecycle/revision。当前 fresh MAB trace 的 state status、conflicts、relation 数量来自这一链路，不是 Mem0 状态映射。
- `system.py:291-299` 特别明确 linking 使用一次 observation 前的 current/uncertain snapshot，避免同一 observation 中前一个候选改变后一个候选的竞争 slot。[代码事实]

### Dependency typing / verification

`stategraph/relation_typing.py:72-99` 是高精度 gate：没有 `explicit_source_relation`、没有 causal marker、evidence 不点名 prerequisite entity、same subject、或同 observation 同 evidence 时直接 `None`；只有 derivation/action-precondition/grounded causal 才给 `DERIVED_FROM`、`AFFECTS_ACTION` 或 `DEPENDS_ON`。`type_relation_candidates`（`102-145`）再做 latest evidence-grounded source resolution。[代码事实]

`system.py:457-626` 的实际调用顺序是 discovery → deterministic typing → `verify_typed_dependency_candidates`；`stategraph/propagation/dependency.py:24-47` 只持久化 counterfactually verified STRICT/WEAK edges。故 candidate 没过 typing 的 pair 不会到 verifier。[代码事实]

### Propagation

`stategraph/propagation/invalidation.py:46-67` 读取 persisted relations，只把 `DependencyStrength.STRICT` 放入 outgoing；`81-130` 用 BFS 将 downstream CURRENT/UNCERTAIN 改为 STALE，保留 historical records，visited 防环。WEAK edge 被保存但不触发 cascade。[代码事实]

### Retrieval / answer boundary

`stategraph/retrieval/current_state_retriever.py:428-519` 的分数由 evidence intersection、lexical overlap、evidence text、field overlap、subject anchor 和 continuity direct assertion 组合；`576-605` 只保留最终 score>0 的候选。`stategraph/retrieval/premise_checker.py:70-102` 默认 `ConservativePremiseExtractor` 只识别 `because|since|given that|assuming that`（另有中文模式）；没有独立 semantic premise model。`stategraph/answer_generation.py:24-49` 又对状态 status 做硬边界。[代码事实]

## Mem0 实际 runtime flow

### Add / extraction / update

`external_baselines/mem0/mem0/memory/main.py:760-877` 的 `Memory.add`：字符串先包装成 `[{role:'user',content:...}]`（`839-850`），`infer=True` 默认，进入 `_add_to_vector_store`（`869`）。

实际 benchmark adapter 调 `add(text)`，所以走 `main.py:916-989` 的 V3 phased ADD-only path：

1. `:918-930` 取 last messages=10、解析新消息、向 vector store 搜 top 10 existing memories。
2. `:940-962` 使用 `ADDITIVE_EXTRACTION_PROMPT` + `generate_additive_extraction_prompt`，response format `json_object`，**单次 extraction call**。
3. `:971-985` 先去 code block，再 `json.loads`; 失败时尝试 `extract_json`；仍失败则记录 error 并把 `extracted_memories=[]`，随后 `:986-989` 只保存 messages 并返回空结果。与 StateGraph parser 的 fail-closed rejection 不同，这是对 malformed extraction 的宽容/空结果行为。[代码事实]
4. `:991-1043` batch embed、MD5 text hash dedup、生成 UUID memory records；`:1045-1206` batch vector persist、history、entity linking、entity store update/insert、save messages。

`ADDITIVE_EXTRACTION_PROMPT` 在 `mem0/configs/prompts.py:468-516` 明确要求从 user 和 assistant 两侧提取丰富、自包含、完整事实，并将 Existing Memories 仅用于 dedup/linking；prompt 文本还要求 shared documents/data 的关键事实也提取。`generate_additive_extraction_prompt`（`1016-1062`）组合 Summary、Last k、Recently Extracted、Existing、New Messages、Observation Date、Current Date；adapter 实际只传 `existing_memories`, `new_messages`, `last_k_messages`, `custom_instructions`，没有 summary/recently extracted/timestamp 参数。[代码事实]

更新/合并边界必须准确描述：

- `main.py:1815-1867` 的 `Memory.update(memory_id, text=...)` 是显式 public update；`main.py:2032-2075` 更新已有 vector payload/hash/history，并不由 adapter 的 `add_memory` 自动调用。
- `main.py` 顶部只导入 `ADDITIVE_EXTRACTION_PROMPT`（`19-24`）；`DEFAULT_UPDATE_MEMORY_PROMPT` 虽在 `configs/prompts.py:176` 定义，但 current adapter add path 不调用它。[代码事实]
- 因此本轮 Mem0 实验实际是“ADD-only extraction + vector/entity persistence + retrieval”；没有证据表明它执行了 StateGraph 式 revision/invalidation，也没有证据表明 add 时做了 LLM update/delete/consolidation。把 Mem0 描述成每轮主动合并冲突会超出代码证据。[推断]

### Retrieval / answer

- adapter `query` 的参数是 `top_k=5`, user filter（`adapters.py:85-86`）；Mem0 `search` 默认 threshold=.1、rerank=False（`main.py:1379-1387`），本轮 adapter 显式覆盖 top_k=5。
- `main.py:1628-1731` 的 `_search_vector_store`：query lemmatization/entity extraction → embedding → semantic over-fetch `max(limit*4,60)` → keyword search → BM25/entity boosts → `score_and_rank` → top-k formatted memories。它不执行 StateGraph lifecycle/stale/dependency traversal。[代码事实]
- Answer generation 不属于 Mem0 core。fresh traces 的 StateGraph 和 Mem0 都有相同 `shared_answer` system prompt（“using only supplied retrieved context… insufficient…”）、answer max budget 512、同一 `config_sha256`；这说明 answer 是实验层共享协议，而非 Mem0 方法代码。[产物观察]

## 结果对比

### 可直接评分的 fresh 结果

| Dataset / case | StateGraph | Mem0 | 可核查结论 |
|---|---|---|---|
| LongMemEval `5d3d2817` | prediction `The available information is insufficient.`；F1 0、EM 0 | `marketing specialist`；F1 0.5、EM 0 | 同 ID Mem0 较好；有完整 context/extraction trace |
| MAB `row0-question0` | insufficient-information；F1/EM 0 | insufficient-information；F1/EM 0 | 两边均未答出 reference `Belgium` |

SCB fresh 3/3 的 sealed predictions 存在，但 canonical v2 gold 文件缺失，不能恢复正式 metrics；Memora fresh 1/1 的 predictions 存在但 rubric/judge artifact 缺失；STALE fresh 两边各 3 个 query prediction 存在但 official judge 缺失。不能把这些 execution-complete 项写成 metric-complete。[产物观察]

### 调用与耗时（仅作 trace 描述，不是公平成本比较）

| Dataset | StateGraph calls / walltime | Mem0 trace calls / walltime | 解释 |
|---|---:|---:|---|
| SCB 012/013/017 | 6/5/5；11.49/5.97/5.43s | 各 1；20.65/1.67/1.45s | Mem0 add 内部调用未暴露 |
| Memora | 8；19.33s | 1；58.48s | 不同层级计数 |
| LongMemEval | 2；2.11s | 1；7.03s | 同上 |
| MAB | 5；51.02s | 1；38.43s | 同上 |
| STALE | 53；200.07s | 3；5.92s | SG 50 extraction + 3 answers；Mem0 内部 add 不在 `api_calls` |
| LongMemEval-V2 | 215；1799.58s，timeout | UNKNOWN，手动停止 | partial，不能评分 |

历史 StateGraph pre-decoupling profile 只能作为辅助：旧 STALE 兼容 extraction 7c0 accepted 2209 states/472 chunks，后来旧 native fresh path 只 accepted 10/50 observations；这是路径/版本行为差异的证据，不是受控 A/B。Mem0 没有经历 Graphiti decoupling，不应做“Mem0 pre/post decoupling”。[版本错配]

## 案例证据

### 案例 1：LongMemEval `5d3d2817` — StateGraph 错，Mem0 相对正确

**原始输入 → extraction**

- input 是一条约 16,362 字符 observation；原文含精确事实：`I've used Trello in my previous role as a marketing specialist at a small startup...`。[产物观察]
- StateGraph trace `longmemeval/StateGraph/5d3d2817/TRACE.json`：唯一 extraction call `prompt_name=stategraph.state_extraction.v2`，input 3922 tokens，output 30，raw response `{"states": []}`，finish_reason `completed`，accepted/rejected 均 0。[产物观察]
- 因此最早可定位失败是 extraction omission。没有 rejected candidate，不能把问题归因于后续 grounding、linking、dependency 或 retrieval；这条判断直接来自 response、accepted/rejected 和空 context。[代码事实][产物观察]

**downstream → answer**

- SG trace 的 state_count=0、relations=0；native retrieval `grounded_states=[]`、candidate list=[]、`final_state_ids=[]`，answer context 是 `(no retrieved context)`，最终回答 insufficient。[产物观察]
- Mem0 trace 的内部 raw extraction response 未保存，故不能声称看到了其 extraction JSON。能直接核验的是 retrieval 返回 5 条 memory，其中一条是：`User's prior experience includes using Trello in a marketing specialist role at a small startup`，score `0.551713...`；最终回答 `marketing specialist`，reference `Marketing specialist at a small startup`，token F1=.5、EM=0。[产物观察]
- 缺失项：Mem0 的 add-time extraction raw response、每个 memory 的生成请求、embedding/vector-store 中间写入轨迹。因此可以证明“相关记忆在 retrieval 中存在”，但不能从现有 artifact 严格判断它是由哪一次内部 LLM response 直接生成，也不能把 .5 F1 称为 exact correctness。[产物观察]

**最早失败判断**：StateGraph 是明确的 extraction-level omission；Mem0 在可见链上至少完成了 relevant memory retrieval，随后 shared answer 只输出 occupation 前半段，形成部分 token overlap。[推断]

### 案例 2：MAB `row0-question0` — 两边都错，揭示长列表压缩/链式检索问题

问题：`What is the country of citizenship of the spouse of the author of Our Mutual Friend?`，reference `Belgium`。[产物观察]

**StateGraph raw → states → retrieval**

- 当前 SG trace 有 4 个 8k/2.157k 字符 observation chunks 的 extraction，最终 220 states、13 relations。
- 原始 observation 同时含：`87. Catherine Dickens is a citizen of United Kingdom.`、`107. The author of Our Mutual Friend is Charles Dickens.`、`131. Charles Dickens is married to Catherine Dickens.`。native raw model response 包含前两类相关项，但 `Catherine Dickens / citizenship / United Kingdom` 两次均因 `attribute_not_grounded_in_evidence` 被拒；`Charles Dickens / author / Our Mutual Friend` 被 accepted。关键 spouse fact `Charles Dickens is married to Catherine Dickens` 在该 extraction response 中没有对应 accepted candidate。[产物观察]
- SG retrieval top-10 只包含 `Charles Dickens -> author -> Our Mutual Friend`（score 1.0317）及大量无关 author/citizen 状态；没有 Catherine spouse/citizenship 链，premise_check 为空，answer 是 insufficient。[产物观察]

**Mem0 raw → memory → retrieval → answer**

- Mem0 trace 只展示一条高度压缩 memory：`User provided a list of factual statements (0-454) ...`，其中只概括若干早期事实；没有 spouse/Belgium 链。[产物观察]
- 它检索到该单条 summary，answer `insufficient information`；同样没有输出 Belgium。[产物观察]

**最早失败判断**：两边都有 information-loss，但位置不同。SG 有结构化 state，却在 extraction validation/relation chain 中丢掉 spouse/citizenship；Mem0 在 ADD-only extraction/consolidation 形成的单个 summary 中没有保留答案所需的长链。当前 artifact 没有足够数据判断哪一个内部调用首先丢失 Mem0 的具体事实。[推断]

### 案例 3：STALE `7c0ae4e7-...` — StateGraph extraction/grounding 丢失，不能以 judge 分数外推

STALE fresh trace 完成 50 sessions、3 queries，但 official judge 缺失，因此没有正式 FAA/SPRR。下面只报告可见机制证据。[产物观察]

- `stale/StateGraph/.../TRACE.json`：最终仅 10 states、0 relations；3 个 query 的 context 全为空，答案均 insufficient，`graphiti_calls=0`。
- `extraction_trace.jsonl` 汇总：50 observations，原始输入字符 764,573，raw model states 266，accepted 10，rejected 256；44/50 observations accepted=0。拒绝原因为 `subject_not_grounded_in_evidence=121`、`attribute_not_grounded_in_evidence=97`、`evidence_not_grounded_in_observation=38`。[产物观察]
- 旧/当前 raw facts 直接覆盖目标主题，但被 native gate 拒绝：session-13 的 Seattle 输出 10 项全为 `subject_not_grounded_in_evidence`；session-33 Austin 输出 9 项，其中 attribute 6、subject 2、evidence 1。最终 state list 只有工作时间、morning routine、podcast/食物/gym bag 等无关状态，没有 location state。[产物观察]
- native validator 的对应代码是 `native_extraction.py:234-247`：exact evidence、subject-in-evidence、attribute-token overlap 三个 hard filters。特别是 `entity='Austin Energy'`、evidence 使用代词/上下文而不重复 entity 时，会被 subject/attribute gate 丢弃；这不是答案层猜测，而是 trace reason 与代码条件一致。[代码事实][产物观察]
- Mem0 fresh 3 queries 的 retrieval 均 `{"results": []}`，答案 insufficient；由于没有 official judge，不能声明 Mem0 在 STALE 上“正确”或比 SG 分数高。[产物观察]

**历史对照（非 fresh、非受控）**：`outputs/stategraph_stale_long_session_extraction_gpt5nano_v1/MODULE_ONLY_RESULTS.json` 的旧 compatibility run 对同一 7c0 处理 472 chunks，accepted 2209，source reconstruction exact/provenance 1.0；旧 Module4 产物显示 2207 states、15 relations、ready_for_retrieval。旧 run 的 `max_llm_characters=1800`、GraphitiFact-aware/richer prompt 与当前 native one-call-per-observation schema 不同，故只能证明版本/path drift，不能单独证明“chunking一定比 native更好”。[版本错配]

### 反向案例（StateGraph 好、Mem0 坏）

当前 fresh post-decoupling 8-case 集合中，没有一个 artifact 支持“StateGraph 正确而 Mem0 错误”的严格反向 case：LongMemEval 是 Mem0 较好，MAB 是 tie，SCB/Memora/STALE 正式 metric 缺失。历史 `outputs/stategraph_local_benchmark_5case_v2` 中 SCB_002/SCB_003 的 StateGraph token-F1 略高（约 .3095 vs .2874；约 .2989 vs .2949），但两边 accuracy/EM 都是 0，不能称为 SG correct/Mem0 wrong，也不是当前 post-decoupling 结果。[产物观察][版本错配]

## 诊断结论：当前最有证据的弱点

1. **Extraction recall / representation 是首要可见瓶颈。** LongMemEval 是严格的 `states=[]` omission；STALE 是 266 raw → 10 accepted，256 个 validator rejects；MAB spouse/citizenship chain 在输入中但没有形成可用 accepted path。[产物观察]
2. **Native schema 与 parser 支持字段不一致。** provider schema 只允许四字段，parser 支持更多 time/condition/conflict/effect 字段；这会让 lifecycle/dependency 所需信息依赖默认值或 metadata，而不是 structured output。[代码事实]
3. **Grounding gate 对长、指代多、assistant-heavy session 很脆弱。** `subject in evidence` 与属性 token overlap 是 hard rejection；STALE session-13/33 的直接 reason 分布证明信息是“模型有输出但 validator 不接收”，同时 LongMemEval 则是模型完全空输出，两类 failure 需分开诊断。[代码事实][产物观察]
4. **StateGraph retrieval 对已有 StateNode 的链式覆盖有限。** MAB 中有 `Charles Dickens → Our Mutual Friend` anchor，但 top-10 没有 spouse/citizenship path；`_select_dependency_states` 仍受 score>0、limit=10 和已有 typed relations 约束。不能仅凭“state 总数 220”推断 retrieval 能找到答案。[代码事实][产物观察]
5. **Mem0 的优势来自宽松 ADD/summary/vector retrieval，而非显式 lifecycle/dependency。** LongMemEval 中 Mem0 存下与问题直接相关的一条 self-contained memory；MAB 中它也压缩过度。其 add path没有 StateGraph 式 explicit revision/invalidation，不能把两者比较成相同语义机制。[代码事实][产物观察]
6. **STALE 的当前 fresh 结果不能写成整体 performance 数字。** 只有 execution complete；official judge 缺失。历史旧 audit 中的 one-case/old-run rejection 数字不能冒充当前正式评测。[版本错配]

## 待验证问题

以下是证据缺口，不是未经验证的实现结论：

- fresh post-decoupling orchestration runner 源文件未保存；需要补 provenance 才能逐行核对它如何构建 StateGraph、query 与 shared answer input。
- Mem0 adapter 未保存 add-time raw model responses、每次 internal extraction request/hash、embedding/vector-store writes；因此只能观察最终 memories，不能精确定位 Mem0 facts 在哪一步压缩或丢失。
- canonical SCB v2 gold、Memora fresh rubric/judge、STALE official judge 缺失；这三类不能在当前 evidence 上给出正式 metric。
- StateGraph fresh trace 的 native schema 允许字段与 parser/downstream 字段不一致，需单独 fixture 审计其对 time_scope/condition/conflict/effect 的实际覆盖；本报告没有重新采样模型。
- MAB relation trace 13 条关系是否包含题目所需 spouse path，需读取完整 relation/dependency artifact；当前 trace 直接证明 retrieval context 未选入该 path，但不能仅由 final answer 反推“relation discovery 一定失败”。
- Mem0 的 `extract_json` fallback 与 malformed response 的空列表处理可能把 provider/parse failure 和“无可抽取事实”合并；需要已有 raw response/log 才能区分，本报告不把缺失证据当成失败率。
- Runtime/compute fairness 尚未成立：StateGraph trace 记录 semantic calls/tokens，Mem0 fresh trace 的 `api_calls` 主要是 answer call。若要论文比较，应额外披露两边完整 provider calls、input/output tokens、ingestion/retrieval latency；不能直接用当前表宣称 StateGraph 更贵或更快。

## 证据索引

- Fresh manifest/results/runtime：`outputs/professor_minimal_post_decoupling_gpt5nano_v1/{RUN_MANIFEST.json,RESULTS.json,RUNTIME_PROFILE.json,FAILURES.json}`。
- LongMemEval evidence：`longmemeval/{StateGraph,Mem0}/5d3d2817/TRACE.json`。
- MAB evidence：`mab_conflict/{StateGraph,Mem0}/row0-question0/TRACE.json`，以及 StateGraph `extraction_trace.jsonl`。
- STALE evidence：`stale/StateGraph/7c0ae4e7-.../{TRACE.json,extraction_trace.jsonl}`、Mem0 `TRACE.json`。
- Historical STALE compatibility evidence：`outputs/stategraph_stale_long_session_extraction_gpt5nano_v1/MODULE_ONLY_RESULTS.json`、`outputs/stategraph_stale_production_structured_output_gpt5nano_v1/CASE_7C0_RESULTS.json`。
- StateGraph implementation：`stategraph/system.py`、`stategraph/state/native_extraction.py`、`stategraph/graphiti_adapter/state_extraction.py`、`stategraph/relation_typing.py`、`stategraph/propagation/{dependency,invalidation}.py`、`stategraph/retrieval/{native,current_state_retriever,premise_checker}.py`、`stategraph/answer_generation.py`。
- Mem0 implementation/adapter：`external_baselines/mem0/mem0/memory/main.py`、`external_baselines/mem0/mem0/configs/prompts.py`、`external_baselines/e2e_validation/adapters.py`。

