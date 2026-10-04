# StateFrame Phase 3 Failure Forensics

Date: 2026-09-19

## 1. Scope And Decision

本报告只分析 sealed Phase 3 v1 的现有结果。没有修改方法代码、测试、配置或 sealed artifacts；没有运行 provider、canary、benchmark、修复或新的方法测试。唯一新增文件为本报告。

- Forensics: COMPLETE，覆盖全部 15 个 S2 failed checkpoints，分布于 11 条 histories。
- `STATEFRAME_PHASE3_CANARY_READY = NO`，保持原判定。
- 主要根因: 证据授权器依赖狭窄句式、本地 cardinality registry 覆盖不足、把世界中的变化和本地必须存在旧版本混为一谈。
- 修正范围选择: **B. RESOLVER_ONLY**。这里包括 `CARDINALITY_RESOLUTION` 的本地 registry/policy 和 `CHANGE_INTENT_RESOLUTION`，不是声称只需改一个函数。
- 最小 schema change: **0**。现有 StateFrame/FrameCandidate、Proposed/ResolvedChangeIntent 足以表达本次缺失的语义；没有证据要求增加 ontology、participant 类型或 support groups。
- 本报告的三个修正设计均为 `CORE-PRESERVING REFINEMENT`，尚未实施，也没有修正后的效果数据。
- Phase 3 v1 从现在起只作为 **DEVELOPMENT / DIAGNOSTIC SET**；未来可以做 regression，但不得再次标作 unseen acceptance。只有新的 sealed Phase 3 v2 unseen canary 通过，才可考虑 Phase 4。

## 2. Evidence And Integrity

### 2.1 Authoritative Artifacts

以下所有 JSON 路径均相对于 `outputs/stateframe_mvp_phase03_20260919_v1/`。

| Evidence | 用途 |
| --- | --- |
| `docs/stateframe_mvp_phase03.md` | 当前结果、限制、NO-GO 和原始指标解释 |
| `CANARY_MANIFEST.json` | 30 histories / 75 writes 的来源、作者、选择过程 |
| `CANARY_INPUTS.json` | 每步完整 source text、固定 extraction response |
| `CANARY_LABELS.json` | 运行前封存的逐步 expected CURRENT/UNCERTAIN semantic sets、operation |
| `<case_id>.runtime.json` | 同一 candidate 的 S1/S2 before/after、operation reason、version、provenance |
| `S1_VS_S2.json` | 总体、类别和每步比较 |
| `FAILURE_ATTRIBUTION.json` | 15 个 S2 失败的封存分类与 semantic differences |
| `FAILURE_REVIEW.json` | 已有运行后审阅；不替代原始分类 |
| `CONFIG.json` | schema v2、完整 13 条 registry rules、评分和 gate |
| `SOURCE_HASHES.json`, `PRE_RUN_SEAL.json`, `INFERENCE_SEAL.json`, `VERIFICATION.json` | 源码、输入、运行和结果的完整性链 |

源码 digest: `8dea5bc2e56a1e57231ea58be31b0cc7aa6a4bd43a71c90b9de2c58087f7a3c0`，manifest 含 121 个文件。root Git 无有效 repository，因此以 hash manifest 为依据。

本轮使用只读 `jq` + `sha256sum -c --quiet` 核对 source manifest、pre-run seal、inference seal 和 verification 中的 result hashes，全部通过。没有重写 seal，没有通过重新运行 inference 获取任何结果。

### 2.2 Frozen Source Locations

| Source | 关键证据 |
| --- | --- |
| `stategraph/state/stateframe.py:193` | ProposedChangeIntent 保存 operation、target_hint、changed_facets、evidence_refs |
| `stategraph/state/stateframe.py:327` | CardinalityRegistry 仅按 kind/predicate/facet 精确查本地规则；未知不会默认 FUNCTIONAL |
| `stategraph/state/stateframe.py:430` | materialize_frame；缺规则产生 UNKNOWN_CARDINALITY；未解决 identity 时把 provenance 纳入 unresolved_source |
| `stategraph/state/stateframe.py:473` | same-value 比较包含 polarity、modality、temporal/condition scope |
| `stategraph/state/stateframe.py:479` | _destructive_evidence 的句首 subject + 有限 cue 正则 |
| `stategraph/state/stateframe.py:492` | resolve_change；先 identity，再 slot，再 operation/source gate，再 target 和 PATCH 检查 |
| `stategraph/state/stateframe.py:506` | uncertain 分支将 resolved operation 设为 UNKNOWN，清空 destructive targets |
| `stategraph/state/stateframe.py:546` | missing target 和 ambiguous target 共用拒绝分支 |
| `stategraph/state/stateframe_repository.py:84` | apply_candidate 按 resolution.stale_version_ids 提交，不自行推断失效 |
| `stategraph/tests/stateframe_fixtures.py:20` | 当前 13 条 fixture registry rules |
| `scripts/run_stateframe_phase03.py:133`, `:311` | canary 配置和执行均使用该冻结 fixture registry |
| `scripts/run_stateframe_phase03.py:377` | 评分单位、false keep/stale、operation exactness、自动 attribution |

当前源码与 hash seal 一致。原报告与实际结果没有发现冲突；但 `FRAME_REPRESENTATION_FAILURE` 和 `REVISION_FAILURE` 是粗粒度评分标签，不能直接解释成 schema 缺陷或 commit bug。

### 2.3 Evidence Boundary

这是 **固定 extraction response 条件下的 persistence/revision canary**，不是 live extraction 实验。同一 agent 在执行前写了 sources、responses 和 labels。parser 一次解析后，两条路径接受完全相同的 FrameCandidate/provenance。API calls/input tokens/output tokens 均为 0。

因此本报告能够定位本地规则如何处理已收到的信息，不能将失败归咎于模型抽取遗漏，也不能声称已经测出了真实 provider 的 proposal precision/recall。75 次 parser 接受不等于 75 次 provider acceptance。读取的是已封存 canary 的预期转换，不是 benchmark gold；它们只用于事后诊断，不进入拟议方法规则。

## 3. Failure Clusters And Stages

| Cluster | Primary failed checkpoints | 独立 histories | 首个失败阶段 | Systemic pattern |
| --- | ---: | ---: | --- | --- |
| P1: Source-authorization bottleneck | 7 | 7 | E. CHANGE_INTENT_RESOLUTION | 已有正确/合理的 semantic hint，但授权器要求 source 以特定 subject/verb 句式开头 |
| P2: Closed registry coverage | 7 | 3 | C. CARDINALITY_RESOLUTION | kind/predicate 已有明确内容，本地规则没有命中；所有后续操作提前变 UNCERTAIN |
| P3: First-known postcondition rejected | 1 | 1 | E. CHANGE_INTENT_RESOLUTION | REPLACE hint 对应世界状态变化，但新主体在本地没有旧版本，明确新事实也被拒绝建档 |

计数相加为 15，不是 15 个独立根因。P2 是三个 histories 的 2 + 3 + 2 次失败；早期未建成 CURRENT 的损失延续到后续 checkpoints。P3 还静态适用于 P1 中 `subject_01:1`，但在那里被更早的 evidence gate 遮蔽，不重复计入 primary count。

| Stage | 本轮结论 |
| --- | --- |
| A. FRAME_EXTRACTION | 未运行 live extractor；固定 response 没有 parse rejection，不支持判定为模型遗漏 |
| B. FRAME_NORMALIZATION | 15 个失败均未报字段、span、subject 或 scope normalization 错误 |
| C. CARDINALITY_RESOLUTION | 7 个 primary failures，都是 UNKNOWN_CARDINALITY |
| D. CHANGE_INTENT_PROPOSAL | 15 个失败均有非 UNKNOWN operation；7 个 intent failures 没有 missing hint；第三方变化的 REPLACE 是世界语义 hint，不应当作已授权的本地命令 |
| E. CHANGE_INTENT_RESOLUTION | 8 个 primary failures: 7 个 evidence gate + 1 个 no-target 分支 |
| F. FRAME_IDENTITY | PATCH 两例 identity 正确；未知 cardinality 引发的 source-dependent identity 是 P2 的下游结果，不是独立 schema gap |
| G. REVISION_COMMIT | 15 例都收到 UNCERTAIN/no-stale decision 并按此提交；没有发现 resolver 已正确授权而 commit 错误的证据 |

## 4. All 15 Failure Records

### Reading Convention

- `step` 为 sealed JSON 的零基索引，完整来源为同名 `<case_id>.runtime.json` 的 `steps[step]`，与 `CANARY_INPUTS.json` / `CANARY_LABELS.json` 同步。
- `C/S/U` 分别为 CURRENT/STALE/UNCERTAIN。下表以规范化语义展示值；原始大小写、ID 和 payload 保留在 sealed runtime 中。
- 所有下列失败的 candidate 都有 confidence=1.0、canonical OBSERVATION_ABSOLUTE provenance、无界 temporal scope、空 condition scope、modality=null。除明确列出的 event instance 外，key_bindings 为空；participants 均为空。这些字段没有缺失的 parser 错误。
- `EXPECTED_FRAME` 来自 sealed semantic labels。期望 cardinality 是已有类型能够承载的本地 policy，不是 labels 中凭空存在的 canonical IDs；labels 不规定 hash。
- `EXPECTED_LIFECYCLE` 中旧版本 STALE 根据相邻 expected sets 和 retirement 定义推导；labels 本身直接列出 CURRENT/UNCERTAIN，而不是完整旧版本快照。
- `ACTUAL_CHANGE_INTENT` 区分原始 proposed hint 与最终 decision。runtime 保存 proposed hint、status/reason/stale IDs，没有直接序列化 ResolvedChangeIntent.operation；下述 resolved UNKNOWN 是由匹配 hash 的 `uncertain()` 源码确定，不伪称它是 runtime 的独立记录字段。
- 全部 15 个实际 decision 都为 UNCERTAIN，resolved operation=UNKNOWN，`stale_version_ids=[]`。原 proposed hint 并未从 persisted candidate 中丢失。
- 同一 history 的前一步已失败时，EXPECTED 是完整预期轨迹，不表示仅修本步就能自动恢复此前的 UNCERTAIN。

### F01: functional_02:1

| Field | Evidence |
| --- | --- |
| CASE_ID | functional_02, step=1 |
| SOURCE_TEXT | `Owen's home city is now Busan; the move is complete.` |
| PREVIOUS_STATE | FACT Owen/current_city=Sapporo, FUNCTIONAL, C |
| NEW_OBSERVATION | 上述全文；同一主体的当前城市变为 Busan |
| EXPECTED_FRAME | FACT Owen/current_city=Busan, POSITIVE, FUNCTIONAL，与旧值 same slot |
| ACTUAL_FRAME | subject/predicate/value/cardinality/slot 都正确；新版本为 U，identity_issue=null |
| EXPECTED_CHANGE_INTENT | label REPLACE；本地 REPLACE 旧城市版本 |
| ACTUAL_CHANGE_INTENT | proposed REPLACE -> UNCERTAIN/UNKNOWN；UNSUPPORTED_DESTRUCTIVE_HINT |
| EXPECTED_LIFECYCLE | Sapporo S；Busan C |
| ACTUAL_LIFECYCLE | Sapporo C；Busan U |
| FAILURE_CLASS | CHANGE_INTENT_FAILURE；P1 / stage E；false_keep=1 |

证据门禁要求 `Owen` 后直接空格和允许的 cue；实际为 possessive，且 `now` 出现在后方。不是 missing operation、scope mismatch 或 slot mismatch。

### F02: removal_02:2

| Field | Evidence |
| --- | --- |
| CASE_ID | removal_02, step=2 |
| SOURCE_TEXT | `Thyme no longer appeals to Sana.` |
| PREVIOUS_STATE | FACT Sana/likes=basil POSITIVE C；Sana/likes=thyme POSITIVE C；SET_VALUED |
| NEW_OBSERVATION | 上述全文；撤销 thyme preference，不改变 basil |
| EXPECTED_FRAME | Sana/likes=thyme, NEGATED, SET_VALUED；同一个 thyme member slot |
| ACTUAL_FRAME | member、polarity、subject 和 cardinality 正确；新 negative version 为 U |
| EXPECTED_CHANGE_INTENT | REMOVE thyme member |
| ACTUAL_CHANGE_INTENT | proposed REMOVE -> UNCERTAIN/UNKNOWN；UNSUPPORTED_DESTRUCTIVE_HINT |
| EXPECTED_LIFECYCLE | basil POSITIVE C；thyme POSITIVE S；thyme NEGATED C |
| ACTUAL_LIFECYCLE | basil POSITIVE C；thyme POSITIVE C；thyme NEGATED U |
| FAILURE_CLASS | CHANGE_INTENT_FAILURE；P1 / E；false_keep=1 |

`no longer` 已在 evidence 内，但 semantic subject 是后置 experiencer，不是句首 grammatical subject。existing member 唯一，缺少显式 target_hint 并不妨碍用 slot 找到目标。

### F03: patch_01:3

| Field | Evidence |
| --- | --- |
| CASE_ID | patch_01, step=3 |
| SOURCE_TEXT | `The budget review will begin at 11:00 instead; nothing else changes.` |
| PREVIOUS_STATE | EVENT budget review/meeting，instance=budget review；time=09:30 C，location=Harbor Hall C，status=confirmed C |
| NEW_OBSERVATION | 上述全文；只变更 time facet |
| EXPECTED_FRAME | 同 event frame、同 time slot，time=11:00，SINGLE_EVENT_INSTANCE |
| ACTUAL_FRAME | frame、slot、facet、binding 正确；time=11:00 U |
| EXPECTED_CHANGE_INTENT | PATCH；changed_facets=[time] |
| ACTUAL_CHANGE_INTENT | proposed PATCH、changed_facets=[time] -> UNCERTAIN/UNKNOWN；UNSUPPORTED_DESTRUCTIVE_HINT |
| EXPECTED_LIFECYCLE | 09:30 S；11:00 C；location/status 的原版本继续 C |
| ACTUAL_LIFECYCLE | 09:30 C；11:00 U；location/status 的原版本继续 C |
| FAILURE_CLASS | CHANGE_INTENT_FAILURE；P1 / E；false_keep=1 |

source 以 `The` 开头，候选 subject 不含该冠词；`will begin ... instead` 也不在许可 cue 结构中。拒绝发生在 PATCH facet 检查之前，未执行任何 target stale。

### F04: patch_02:3

| Field | Evidence |
| --- | --- |
| CASE_ID | patch_02, step=3 |
| SOURCE_TEXT | `Briefing-8 has been pushed back to Wednesday; the room is unchanged.` |
| PREVIOUS_STATE | EVENT Briefing-8/meeting，instance=briefing-8；time=Tuesday C，location=Cedar Room C，status=confirmed C |
| NEW_OBSERVATION | 上述全文；只推迟 time facet |
| EXPECTED_FRAME | 同 event frame、同 time slot，time=Wednesday，SINGLE_EVENT_INSTANCE |
| ACTUAL_FRAME | frame、slot、facet、binding 正确；time=Wednesday U |
| EXPECTED_CHANGE_INTENT | PATCH；changed_facets=[time] |
| ACTUAL_CHANGE_INTENT | proposed PATCH、changed_facets=[time] -> UNCERTAIN/UNKNOWN；UNSUPPORTED_DESTRUCTIVE_HINT |
| EXPECTED_LIFECYCLE | Tuesday S；Wednesday C；location/status 原版本 C |
| ACTUAL_LIFECYCLE | Tuesday C；Wednesday U；location/status 原版本 C |
| FAILURE_CLASS | CHANGE_INTENT_FAILURE；P1 / E；false_keep=1 |

source subject 已在句首；但 `has been pushed back` 不符合 optional `has` 后紧跟有限动词列表的正则。不是 target resolution 缺失，也不是 PATCH 被转为 REPLACE。

### F05: relation_02:0

| Field | Evidence |
| --- | --- |
| CASE_ID | relation_02, step=0 |
| SOURCE_TEXT | `Uma collaborates with Keiko on acoustic research.` |
| PREVIOUS_STATE | 空 |
| NEW_OBSERVATION | 上述全文；一个二元 collaboration relation |
| EXPECTED_FRAME | RELATION Uma/collaborates_with=Keiko，POSITIVE；SET_VALUED member |
| ACTUAL_FRAME | kind/subject/predicate/value 正确；cardinality=null，identity_issue=UNKNOWN_CARDINALITY，U |
| EXPECTED_CHANGE_INTENT | ASSERT，创建第一个 member |
| ACTUAL_CHANGE_INTENT | proposed ASSERT -> UNCERTAIN/UNKNOWN；UNKNOWN_CARDINALITY |
| EXPECTED_LIFECYCLE | Keiko relation C |
| ACTUAL_LIFECYCLE | Keiko relation U |
| FAILURE_CLASS | FRAME_REPRESENTATION_FAILURE；P2 / C；IMPLEMENTATION GAP |

schema 能用 RELATION 的 value 承载另一端，registry 没有该 predicate。不能据此推断缺少 participant role 字段。

### F06: relation_02:1

| Field | Evidence |
| --- | --- |
| CASE_ID | relation_02, step=1 |
| SOURCE_TEXT | `A separate project has Uma collaborating with Tomas as well.` |
| PREVIOUS_STATE | 实际 Keiko relation U；预期前一步为 C |
| NEW_OBSERVATION | 上述全文；增加第二个 collaboration member |
| EXPECTED_FRAME | RELATION Uma/collaborates_with=Tomas，POSITIVE，SET_VALUED；与 Keiko 分属 member slots |
| ACTUAL_FRAME | Tomas cardinality=null，UNKNOWN_CARDINALITY，U；原 Keiko 仍 U |
| EXPECTED_CHANGE_INTENT | ADD，不 stale 第一位 collaborator |
| ACTUAL_CHANGE_INTENT | proposed ADD -> UNCERTAIN/UNKNOWN；UNKNOWN_CARDINALITY |
| EXPECTED_LIFECYCLE | Keiko C；Tomas C |
| ACTUAL_LIFECYCLE | Keiko U；Tomas U |
| FAILURE_CLASS | FRAME_REPRESENTATION_FAILURE；P2 / C；IMPLEMENTATION GAP；含前步累计损失 |

这不是已将 SET 误判为 FUNCTIONAL 后发生错误 replacement，而是根本未解析 cardinality。

### F07: preference_01:2

| Field | Evidence |
| --- | --- |
| CASE_ID | preference_01, step=2 |
| SOURCE_TEXT | `Black tea has lost its appeal for Yara, who still enjoys the other drink.` |
| PREVIOUS_STATE | FACT Yara/likes=oat milk POSITIVE C；Yara/likes=black tea POSITIVE C；SET_VALUED |
| NEW_OBSERVATION | 上述全文；撤销 black tea preference，保留其他饮品 |
| EXPECTED_FRAME | Yara/likes=black tea，NEGATED；同一 member slot |
| ACTUAL_FRAME | member/polarity/cardinality 正确；新 negative version U |
| EXPECTED_CHANGE_INTENT | REMOVE black tea member |
| ACTUAL_CHANGE_INTENT | proposed REMOVE -> UNCERTAIN/UNKNOWN；UNSUPPORTED_DESTRUCTIVE_HINT |
| EXPECTED_LIFECYCLE | oat milk C；black tea POSITIVE S；black tea NEGATED C |
| ACTUAL_LIFECYCLE | oat milk C；black tea POSITIVE C；black tea NEGATED U |
| FAILURE_CLASS | CHANGE_INTENT_FAILURE；P1 / E；false_keep=1 |

隐含撤销已经体现在 REMOVE + NEGATED 中；本地 gate 不支持这种后置 experiencer 和表达形式。不能归类为输入没有识别 implicit REMOVE。

### F08: ownership_01:0

| Field | Evidence |
| --- | --- |
| CASE_ID | ownership_01, step=0 |
| SOURCE_TEXT | `Asha owns a brass compass inherited from a relative.` |
| PREVIOUS_STATE | 空 |
| NEW_OBSERVATION | 上述全文；建立 ownership member |
| EXPECTED_FRAME | RELATION Asha/owns=brass compass，POSITIVE，SET_VALUED |
| ACTUAL_FRAME | kind/subject/predicate/value 正确；cardinality=null、UNKNOWN_CARDINALITY、U |
| EXPECTED_CHANGE_INTENT | ASSERT |
| ACTUAL_CHANGE_INTENT | proposed ASSERT -> UNCERTAIN/UNKNOWN；UNKNOWN_CARDINALITY |
| EXPECTED_LIFECYCLE | brass compass C |
| ACTUAL_LIFECYCLE | brass compass U |
| FAILURE_CLASS | FRAME_REPRESENTATION_FAILURE；P2 / C；IMPLEMENTATION GAP |

### F09: ownership_01:1

| Field | Evidence |
| --- | --- |
| CASE_ID | ownership_01, step=1 |
| SOURCE_TEXT | `Asha bought a field notebook and kept the compass.` |
| PREVIOUS_STATE | 实际 brass compass U；预期为 C |
| NEW_OBSERVATION | 上述全文；增加一个 ownership member，保留旧 member |
| EXPECTED_FRAME | RELATION Asha/owns=field notebook，POSITIVE，SET_VALUED |
| ACTUAL_FRAME | field notebook cardinality=null、UNKNOWN_CARDINALITY、U；compass 仍 U |
| EXPECTED_CHANGE_INTENT | ADD |
| ACTUAL_CHANGE_INTENT | proposed ADD -> UNCERTAIN/UNKNOWN；UNKNOWN_CARDINALITY |
| EXPECTED_LIFECYCLE | compass C；notebook C |
| ACTUAL_LIFECYCLE | compass U；notebook U |
| FAILURE_CLASS | FRAME_REPRESENTATION_FAILURE；P2 / C；IMPLEMENTATION GAP；含前步累计损失 |

### F10: ownership_01:2

| Field | Evidence |
| --- | --- |
| CASE_ID | ownership_01, step=2 |
| SOURCE_TEXT | `Asha no longer owns the field notebook, having given it away.` |
| PREVIOUS_STATE | 实际 compass POSITIVE U、notebook POSITIVE U；预期两者 C |
| NEW_OBSERVATION | 上述全文；撤销 notebook ownership |
| EXPECTED_FRAME | RELATION Asha/owns=field notebook，NEGATED，SET_VALUED；与 positive notebook 同 member slot |
| ACTUAL_FRAME | negative notebook cardinality=null、UNKNOWN_CARDINALITY、U；与旧 positive notebook 的 frame/slot 不同 |
| EXPECTED_CHANGE_INTENT | REMOVE notebook，不影响 compass |
| ACTUAL_CHANGE_INTENT | proposed REMOVE -> UNCERTAIN/UNKNOWN；UNKNOWN_CARDINALITY，尚未进入 destructive evidence/target 检查 |
| EXPECTED_LIFECYCLE | compass C；notebook POSITIVE S；notebook NEGATED C |
| ACTUAL_LIFECYCLE | compass POSITIVE U；notebook POSITIVE U；notebook NEGATED U |
| FAILURE_CLASS | FRAME_REPRESENTATION_FAILURE；P2 / C，附带 F 下游影响；IMPLEMENTATION GAP；false_keep=0 |

identity_issue 使每步 provenance 进入 unresolved_source，导致本应同 member 的正负断言没有稳定 slot。不能仅去掉该安全隔离，或把此前 U 原地提升为 C。此次未正确退休也不计 false keep，因为旧 positive 从未 CURRENT。

### F11: subject_01:1

| Field | Evidence |
| --- | --- |
| CASE_ID | subject_01, step=1 |
| SOURCE_TEXT | `Cora mentioned that Deni moved to Riga, not Cora.` |
| PREVIOUS_STATE | FACT Cora/current_city=Tallinn C；没有 Deni 的旧状态 |
| NEW_OBSERVATION | 上述全文；被报告的迁移主体为 Deni，不是说话内容中的 Cora |
| EXPECTED_FRAME | FACT Deni/current_city=Riga，POSITIVE，FUNCTIONAL；独立主体 slot |
| ACTUAL_FRAME | Deni subject 和新 slot 正确，identity_issue=null；Riga U |
| EXPECTED_CHANGE_INTENT | sealed label 为 REPLACE，表示世界中的迁移；本地所需结果是非破坏性创建 Deni 首个 CURRENT，而不是替换 Cora |
| ACTUAL_CHANGE_INTENT | proposed REPLACE -> UNCERTAIN/UNKNOWN；UNSUPPORTED_DESTRUCTIVE_HINT |
| EXPECTED_LIFECYCLE | Cora/Tallinn C；Deni/Riga C；没有旧 Deni endpoint 需要 stale |
| ACTUAL_LIFECYCLE | Cora/Tallinn C；Deni/Riga U |
| FAILURE_CLASS | CHANGE_INTENT_FAILURE；P1 / E，P3 被遮蔽；false_keep=0 |

semantic subject 位于报告从句内，未通过句首 gate。即便放行该句式，源码后续仍会因为 Deni 没有本地旧 target 而拒绝。这是静态分支推断，没有执行修改后的反事实实验。

### F12: subject_02:1

| Field | Evidence |
| --- | --- |
| CASE_ID | subject_02, step=1 |
| SOURCE_TEXT | `Faye relocated to Ghent; Eli is only helping with the boxes.` |
| PREVIOUS_STATE | FACT Eli/current_city=Porto C；没有 Faye 的旧状态 |
| NEW_OBSERVATION | 上述全文；Faye 的已完成迁移，不是 Eli 的迁移 |
| EXPECTED_FRAME | FACT Faye/current_city=Ghent，POSITIVE，FUNCTIONAL；独立主体 slot |
| ACTUAL_FRAME | Faye subject、slot、cardinality 正确，identity_issue=null；Ghent U |
| EXPECTED_CHANGE_INTENT | sealed label REPLACE；在这个空的主体 slot 上，应非破坏性创建已明确陈述的新状态 |
| ACTUAL_CHANGE_INTENT | proposed REPLACE -> UNCERTAIN/UNKNOWN；MISSING_OR_AMBIGUOUS_DESTRUCTIVE_TARGET |
| EXPECTED_LIFECYCLE | Eli/Porto C；Faye/Ghent C；没有目标需要 stale |
| ACTUAL_LIFECYCLE | Eli/Porto C；Faye/Ghent U |
| FAILURE_CLASS | 封存 REVISION_FAILURE；实际首个失败位置 P3 / E，而不是 G；false_keep=0 |

`Faye relocated` 已通过 source gate。原因是零个匹配 target，不是多个目标不确定。保护 Eli 正确，但因此拒绝 Faye 的完整新事实没有必要。标签并未要求 stale 另一个主体；不改动 sealed label，也不把它说成 label ASSERT。

### F13: polarity_02:1

| Field | Evidence |
| --- | --- |
| CASE_ID | polarity_02, step=1 |
| SOURCE_TEXT | `Hugo isn't available anymore; the shift needs someone else.` |
| PREVIOUS_STATE | FACT Hugo/available=available，POSITIVE，FUNCTIONAL，C |
| NEW_OBSERVATION | 上述全文；当前 available polarity 变为 NEGATED |
| EXPECTED_FRAME | 同 slot、同 value，polarity=NEGATED |
| ACTUAL_FRAME | value/polarity/slot 正确；新 negative version U |
| EXPECTED_CHANGE_INTENT | REPLACE positive version |
| ACTUAL_CHANGE_INTENT | proposed REPLACE -> UNCERTAIN/UNKNOWN；UNSUPPORTED_DESTRUCTIVE_HINT |
| EXPECTED_LIFECYCLE | available POSITIVE S；available NEGATED C |
| ACTUAL_LIFECYCLE | available POSITIVE C；available NEGATED U |
| FAILURE_CLASS | CHANGE_INTENT_FAILURE；P1 / E；false_keep=1 |

`isn't ... anymore` 不在 cue 规则内。same-value merge 已比较 polarity，因此这里不是正负态被重复检测错误合并；polarity 已到 resolver，但没有帮助 evidence gate 识别这个已证实的变更。

### F14: recall_02:0

| Field | Evidence |
| --- | --- |
| CASE_ID | recall_02, step=0 |
| SOURCE_TEXT | `Pavel's blood type is AB-positive.` |
| PREVIOUS_STATE | 空 |
| NEW_OBSERVATION | 上述全文；普通不含状态变化的事实 |
| EXPECTED_FRAME | FACT Pavel/blood_type=AB-positive，POSITIVE；FUNCTIONAL 可表达 |
| ACTUAL_FRAME | subject/predicate/value 正确；cardinality=null、UNKNOWN_CARDINALITY、U |
| EXPECTED_CHANGE_INTENT | ASSERT |
| ACTUAL_CHANGE_INTENT | proposed ASSERT -> UNCERTAIN/UNKNOWN；UNKNOWN_CARDINALITY |
| EXPECTED_LIFECYCLE | 一个 AB-positive C |
| ACTUAL_LIFECYCLE | 一个 AB-positive U |
| FAILURE_CLASS | FRAME_REPRESENTATION_FAILURE；P2 / C；IMPLEMENTATION GAP |

source 的 possessive 不是本例首因：ASSERT 不走 destructive cue gate，identity/cardinality 阶段已拒绝。

### F15: recall_02:1

| Field | Evidence |
| --- | --- |
| CASE_ID | recall_02, step=1 |
| SOURCE_TEXT | `A second record confirms AB-positive as Pavel's blood type.` |
| PREVIOUS_STATE | 实际第一条 AB-positive U；预期前步为一个 C |
| NEW_OBSERVATION | 上述全文；同值佐证，没有修改语义值 |
| EXPECTED_FRAME | 同一 FACT slot / semantic value；可沿用已成立的 version 并合并 provenance |
| ACTUAL_FRAME | 第二条 AB-positive U；cardinality=null、UNKNOWN_CARDINALITY；两个不同 frame/slot |
| EXPECTED_CHANGE_INTENT | label ASSERT；在完整正确轨迹上执行 SAME_VALUE/MERGE |
| ACTUAL_CHANGE_INTENT | proposed ASSERT -> UNCERTAIN/UNKNOWN；UNKNOWN_CARDINALITY，在 MERGE 判断之前退出 |
| EXPECTED_LIFECYCLE | 一个 C，佐证增加，不生成第二个 CURRENT |
| ACTUAL_LIFECYCLE | 两个 U；没有合并，旧 U 仍在 |
| FAILURE_CLASS | FRAME_REPRESENTATION_FAILURE；P2 / C，附带 F 下游影响；IMPLEMENTATION GAP |

不能据此归咎于 MERGE 本身。未知 cardinality 既阻止到达该分支，又通过 unresolved_source 使相同语义产生不同 provisional identities。

## 5. Seven Change-Intent Failures: Proposal vs Resolution

结论: **主因是 RESOLUTION，不是 PROPOSAL 缺失。** 七例均有 operation hint、evidence refs、完整原文和 canonical provenance。

| Case:step | Proposed operation | 原文变化表达/句法 | Resolver 为什么拒绝 | Target 是否已足够 |
| --- | --- | --- | --- | --- |
| functional_02:1 | REPLACE | possessive + 后置 now | subject 后不是空格+cue | 是，唯一同 slot CURRENT |
| removal_02:2 | REMOVE | object/fronted subject + no longer appeals | semantic subject 不在句首 | 是，唯一同 member CURRENT |
| patch_01:3 | PATCH | 冠词 + will begin ... instead | subject prefix 与 cue 均不符 | 是，唯一 time facet CURRENT |
| patch_02:3 | PATCH | has been pushed back | 不支持该助动词链/变化表达 | 是，唯一 time facet CURRENT |
| preference_01:2 | REMOVE | lost its appeal + 后置 experiencer | subject 位置和 cue 均不符 | 是，唯一同 member CURRENT |
| subject_01:1 | REPLACE | reported clause 内 moved | semantic subject 不在允许的句首位置 | 没有旧版本；应另行判断非破坏性建档 |
| polarity_02:1 | REPLACE | isn't ... anymore | contraction/negative-current-state 不匹配 | 是，唯一同 slot CURRENT |

问题逐项回答:

- Operation hint 错/缺失: 六个真正有旧版本的更新，其 hint 与 expected operation 一致。第三方例的 REPLACE 合理描述世界变化，但不能单独决定 repository mutation。
- Resolver 过于保守: 是，六个明确、唯一目标的状态更新被句式 gate 阻止。
- Target selector 不足: 不是这六例的首因，slot/member/facet 已唯一。不能用给每例补 target_hint 掩盖 source gate。
- Polarity 没被利用: NEGATED 已正确抵达，same-value 检查也使用它；但 evidence authorization 只看 quote/subject/operation，不能结合 polarity 解释已支持的负态变化。不能因此简单允许所有 NEGATED 都删除旧态。
- Lexical cue 未进入 resolver: 不是 evidence 丢失。完整句子在 evidence_quotes 内；问题是本地仅接受窄句式。
- ASSERT / REPLACE / PATCH / REMOVE 边界: 两个 PATCH 没有被错解析成 REPLACE，两个 preference REMOVE 也没丢。真正边界缺口是“世界发生 REPLACE”不等于“本地必有旧版本可 REPLACE”。
- UNKNOWN 导致 false keep: UNKNOWN 是拒绝后的 resolved 结果，不是七例的输入 operation。不能把 outcome 倒当 root cause。

## 6. Seven Representation Failures: Implementation Gap, Not Schema Gap

| Histories / checkpoints | 已具备的语义 | 缺什么 | 阶段判断 |
| --- | --- | --- | --- |
| relation_02:0,1 | RELATION、subject、predicate、object value、ASSERT/ADD | kind/predicate/facet 对应的本地 SET_VALUED policy | IMPLEMENTATION GAP，C |
| ownership_01:0,1,2 | RELATION、subject、object member、NEGATED、ASSERT/ADD/REMOVE | 同类本地 cardinality policy；因此 member identity 未获授权稳定化 | IMPLEMENTATION GAP，C -> F |
| recall_02:0,1 | FACT、subject、同一 value、重复 ASSERT、grounded provenance | 本地 FUNCTIONAL policy；因此无法走同值 MERGE | IMPLEMENTATION GAP，C -> F |

13 条 frozen rules 直接来自 `stategraph.tests.stateframe_fixtures.registry`。这三个 predicates 均不在表内，`CardinalityRegistry.rule()` 又是精确查找。执行没有把它们错设为某个不合适的 cardinality，而是保持 null 并返回 UNKNOWN_CARDINALITY。

未发现以下缺口: kind 无法表达、participant role 无处存放、event discriminator 丢失、temporal scope 丢失、polarity/modality 字段不存在。七例没有 EVENT/ROLE/ACTION 类型；简单二元 relation 的另一端已在 value，participants 为空并不导致这些错误。更复杂关系未来是否要额外 participant，是本轮未覆盖的问题，不能据此扩 schema。

`unresolved_source` 的行为是对不确定 identity 的安全隔离。它解释重复事实不能归并、同 member 的正负版本不能匹配，但不能直接删除以“提高 identity 稳定性”。先解决 policy 未解析，才可讨论稳定身份；对既有 U 的迁移仍应有显式重放/迁移验证，不能偷偷提升 lifecycle。

## 7. PATCH Regression: Exact Causal Trace

S1 PATCH=4/4，S2=2/4；S2 PATCH isolation=4/4。后者只说明其它 expected states 未受伤，不说明目标 facet 被成功更新。

| Check | patch_01:3 | patch_02:3 |
| --- | --- | --- |
| identity binding | instance=budget review，前后相同 | instance=briefing-8，前后相同 |
| kind / predicate / facet | EVENT / meeting / time | EVENT / meeting / time |
| cardinality | SINGLE_EVENT_INSTANCE | SINGLE_EVENT_INSTANCE |
| changed_facets | [time]，不宽于目标 | [time]，不宽于目标 |
| target uniqueness | 一个 CURRENT time version | 一个 CURRENT time version |
| forward order | sequence 3 晚于原 time observation | sequence 3 晚于原 time observation |
| observed operation | PATCH -> UNCERTAIN | PATCH -> UNCERTAIN |
| observed reason | UNSUPPORTED_DESTRUCTIVE_HINT | UNSUPPORTED_DESTRUCTIVE_HINT |
| stale IDs | [] | [] |
| collateral location/status | 同版本、保持 CURRENT | 同版本、保持 CURRENT |

Identity 直接核对值:

| Case | 前后相同 frame_id | 前后相同 time slot_id |
| --- | --- | --- |
| patch_01 | `frame:36865e98843c1eb745cf311a2f61117ab787964891f4c159c22e882a06b995d2` | `slot:6f9a4c13260618df6efe4b3d87dac66c2561b21b567dd937760570fed984cdfd` |
| patch_02 | `frame:101c4a563c9d8788bd521574e16ec6aa4819644007aee501bb4d6f0dbdbb0ee9` | `slot:04200f027125be18fc03448a83d7fb1d1eea29b062d847253a0b98a0e0806ad7` |

旧/新 version 前缀分别为 `50a702dcfd7c` / `ac6e5cfdc3a5` 和 `1e54b78c9950` / `a1d456211021`，完整值在对应 runtime 的 before/after/returned_id 中。新版本身份生成正常，旧版本没有被 stale。

因果链为: 正确 FrameCandidate -> 正确 frame/slot -> 唯一旧 target -> source authorization 拒绝 -> UNCERTAIN/no stale resolution -> repository 正常提交该保守决定。精确 target 检查和 PATCH_MUST_NAME_EXACTLY_ONE_FACET 位于 gate 后面，未成为这两次失败的触发条件。

因此两例都是 **BAD FALSE KEEP**，不是 event identity、facet、changed_fields、PATCH/REPLACE 混淆或 commit 隔离失败。无需以这两例为理由改 persistent schema、event identity、dependency endpoint 类型或 propagation。

## 8. False Keep And The Actual Trade-Off

### 8.1 All Six False Keeps

| Case:step | 应退休的版本 | 未完成的转换 | 判断 |
| --- | --- | --- | --- |
| functional_02:1 | 旧城市 Sapporo CURRENT | FUNCTIONAL REPLACE | BAD FALSE KEEP；确定来源、同 slot |
| removal_02:2 | thyme POSITIVE CURRENT | SET REMOVE | BAD FALSE KEEP；精确 member 和 NEGATED 已有 |
| patch_01:3 | time=09:30 CURRENT | time PATCH | BAD FALSE KEEP；唯一 event facet |
| patch_02:3 | time=Tuesday CURRENT | time PATCH | BAD FALSE KEEP；唯一 event facet |
| preference_01:2 | black tea POSITIVE CURRENT | SET REMOVE | BAD FALSE KEEP；精确 member 和 NEGATED 已有 |
| polarity_02:1 | available POSITIVE CURRENT | polarity REPLACE | BAD FALSE KEEP；negative current state 已表达 |

分布: 2 REMOVE + 2 REPLACE + 2 PATCH。六例都由 UNSUPPORTED_DESTRUCTIVE_HINT 拒绝，没有一例由 scope mismatch、多目标歧义或缺失 member selector 触发。它们不是“为了保护其它 state 必须放弃的含糊更新”。

### 8.2 What Is Good Fail-Safe

`ambiguous_01:1` 和 `ambiguous_02:1` 的 proposed operation 与 polarity 都是 UNKNOWN。实际 reason 为 LOW_CONFIDENCE_OR_UNKNOWN_POLARITY；旧 CURRENT 保留，新 observation 为 UNCERTAIN，与 sealed expected set 一致。这两例 **GOOD FAIL-SAFE** 必须保留，不能为了降低 false keep 将未知输入强行 destructive commit。

unknown-cardinality 七例在当前缺少可信 policy 时拒绝任意 destructive update 是合理安全边界，但整体 recall 失败仍需归因于 policy coverage；不能把拒绝产生的“没有 false stale”当作完整成功。第三方首次建档两例根本不要求退休任何旧版本，不属于六个 false keep。

### 8.3 Quantitative Judgment

| Metric | S1 | S2 | 解释 |
| --- | ---: | ---: | --- |
| CURRENT state precision | 91/100 = 91.00% | 105/111 = 94.59% | 仅 CURRENT 计入，不能靠多写 U 提高 recall |
| CURRENT state recall | 91/123 = 73.98% | 105/123 = 85.37% | typed member retention 存在真实收益 |
| False stale | 15/52 | 0/52 | 52 个 expected keep opportunities，不是所有更新 |
| False keep | 7/16 | 6/16 | S2 仍漏退休 37.50% 的 expected transitions |
| Correct required endpoint retirement | 9/16 | 9/16 | 没有因为 wiring 稳定而改善应退休 endpoint 的总成功数 |
| ADD / REMOVE / PATCH | 0/14, 0/7, 4/4 | 12/14, 4/7, 2/4 | 安全性改善同时存在 update completeness 回归 |
| Ambiguous safety | 0/2 | 2/2 | 这部分保守性正确 |

S2 的 16 个 retirement opportunities 可按 sealed checkpoints 对齐为: **9 个正确 STALE + 6 个仍 CURRENT + 1 个从未成功成为 CURRENT 的 ownership 旧态**。最后一个为 UNKNOWN_CARDINALITY 历史损失，不算 false keep，却同样不能拿到 endpoint retirement credit。

结论是 **存在局部、系统性的 over-conservative revision**，但不能简化成“所有提升只是从不更新”。S2 实际执行了正确退休，并保留了 ADD/REMOVE 的成员隔离收益；同时，六个明确更新被 gate 拒绝，0 false stale 并不代表更新完整性达标。

Aggregate precision 增加掩盖了类别回归: functional replacement precision 1.00 -> 0.75，partial patch 1.00 -> 0.8889。S2 precision 的六个错误 CURRENT 对应上述六个 false keep；schema 扩容或多建 U 不能消除这些错误。小样本和相关 checkpoints 不支持推断总体真实世界错误率，也不能把 0/52 解读为普遍安全证明。

## 9. Minimal Generic Changes: Design Only

以下最多三个 change，均不引用实体、benchmark 模板或 gold 作为方法输入。病例编号仅用于说明覆盖证据，不构成规则。收益均为待验证假设，不给修正后分数承诺。

### G1. Operation-Specific, Source-Grounded Revision Authorization

| Field | Design |
| --- | --- |
| ROOT CAUSE | 正确的 subject/member/facet/polarity/hint 已到本地，但 source authority 被近似为句首 subject + 有限 change cue，忽略合法句法变化 |
| GENERIC FIX | 将 source-supported postcondition、明确的目标、scope/order 和操作要求分别验证；使用已有 grounded semantic fields 与原文证据判断当前肯定/否定状态及 change scope，而非把 subject 的表面位置和 cue 白名单作为唯一授权依据。LLM hint 仍只是 proposal，不能直接授权 stale；不扩大为任意 NEGATED/REPLACE 自动通过 |
| AFFECTED FAILURES | F01/F02/F03/F04/F07/F11/F13 的首个 gate；其中 F11 还需要 G3 才可能完成建档 |
| EXPECTED BENEFIT | 减少明确更新的六个 false keep，恢复 PATCH 实际更新而非只保留 isolation；跨 possessive、从句、语态和 negative-current-state 的稳定性假设 |
| REGRESSION RISK | 将否定“发生过变化”、疑问、引述中的不确定性、future/conditional 内容误当已成立状态；主客体绑定不准造成 false stale |
| BOUNDARY | 必须继续区分 negated change event 与 negated present state、目标不唯一与证据不足；不通过单纯追加 7 例对应 regex 或放宽为 substring cue 实现 |
| CORE CHECK | CORE-PRESERVING REFINEMENT；增强 write-time revision 的语义授权，不取消 evidence gate、lifecycle 或 stale-premise 机制 |

这是本地 revision authorization 的有限语义处理，不是重新设计 dependency verifier。哪些局部规则足以实现并保持 precision 需要后续 generic fixtures 证明；本轮未实现可运行的新授权器。

### G2. Deterministic Cardinality Policy Resolution Beyond Fixture Vocabulary

| Field | Design |
| --- | --- |
| ROOT CAUSE | runtime 使用 13 条开发 fixture registry，开放 predicate 名称只有精确命中才能获得 cardinality；简单已可表达的 state 也因此全程 UNCERTAIN |
| GENERIC FIX | 将运行时本地 predicate/cardinality policy 与测试 fixture 表分离，采用独立维护、版本化且明确约束的 predicate/facet policy 与受控语义别名；让已支持的 functional 或 member relation 按通用 policy 落到现有 enum/identity。未知且无法确定的 policy 仍保持不确定，不使用 kind=RELATION 就一律 SET 或未知一律 FUNCTIONAL 的兜底 |
| AFFECTED FAILURES | F05/F06/F08/F09/F10/F14/F15；含 repeated assertion 和 removal identity 的下游影响 |
| EXPECTED BENEFIT | 在 schema 不变的前提下提高 CURRENT recall，使 member identity 与 same-value merge 有机会工作；避免每个新表面谓词都只能产生 source-isolated U |
| REGRESSION RISK | policy/alias 误合并多义 predicates，错误 FUNCTIONAL 导致 destructive stale，错误 SET 导致 false keep；policy 更新导致 identity 漂移 |
| BOUNDARY | policy 需由源任务语义、可审阅约束和独立测试确定，不能只把本次三个失败 predicate 加进表就宣称泛化；不能由 provider 自报 cardinality 决定破坏性更新；旧 U 的升级不得静默发生 |
| CORE CHECK | CORE-PRESERVING REFINEMENT；仍由本地规则决定 cardinality/identity，保留 member isolation、versioning、revision 和 dependency endpoint 单位 |

现有 schema 已有全部需要的字段。预定义 policy 也不可能从任意新 predicate 的一次观察中可靠推断全局 cardinality；覆盖边界必须显式记录。该 change 不承诺解决所有开放 ontology 问题，也不要求实现 ontology 新层级。

### G3. Distinguish First-Known State From Destructive Target Resolution

| Field | Design |
| --- | --- |
| ROOT CAUSE | REPLACE 世界语义 hint 被当作必需存在本地 old target 的命令；零 target 与多/不确定 target 合并拒绝，丢失完整新事实 |
| GENERIC FIX | 本地 resolver 区分 zero / unique / ambiguous target。对 identity/cardinality/provenance 已确认、来源明确陈述当前完整 postcondition、且目标 slot 真正未建档的情况，允许将世界变化 hint 解析为非破坏性 ASSERT/CREATE；有唯一旧版本时仍执行经授权的 REPLACE，存在歧义时仍 UNCERTAIN |
| AFFECTED FAILURES | F12 直接，F11 为 G1 之后静态可见的第二层问题；跨两个第三方 histories |
| EXPECTED BENEFIT | 建立首次听到的新主体状态，同时不修改其它主体；区分 recall admission 与 retirement authority |
| REGRESSION RISK | 将缺失 target 的 REMOVE、缺 event discriminator 的 PATCH、指代不清或尚未生效的计划错误降级为 ASSERT，造成伪 CURRENT |
| BOUNDARY | 不是“destructive 失败就 fallback ASSERT”；不适用于证据不支持、缺 scope/identity、存在冲突或歧义目标，也不凭空合成 prior versions |
| CORE CHECK | CORE-PRESERVING REFINEMENT；仅在没有旧版本可修订时无损建档，有旧版本的 write-time revision 仍保留 |

G1/G2/G3 有先后遮蔽关系，不是三个独立百分比增益。尤其 G1 单独放行第三方从句，仍不能越过零 target 分支；G2 补足 policy 后也不能未经验证就迁移旧 unresolved identities。不得将本报告的归因覆盖数写成未来修复成功数。

## 10. Schema vs Resolver Decision

**选择 B. RESOLVER_ONLY。**

这里的最小修改面是本地 cardinality policy resolution 和 intent resolution，而非 extraction wire schema、persistent StateFrame schema 或 graph semantics。没有证据支持新增 target selector 字段: 六个 false keep 已有唯一 slot/member/facet；两个 third-party 例恰好没有可选旧目标。没有证据支持新增 event/participant/scope 字段来修补本次结果。

七个 FRAME_REPRESENTATION_FAILURE 应解释为 **IMPLEMENTATION GAP**，不是 REPRESENTATION GAP；原 sealed 类别保留不改。一个 REVISION_FAILURE 实際首因属于 resolver，不是 repository commit。七个 CHANGE_INTENT_FAILURE 的主要责任也在本地 resolution。

最小 schema change=0，identity hash 公式也不需要因这些失败直接重写。P2 的 unresolved_source 变化须在 cardinality policy 被可靠解决后自然消失，而不是通过移除安全隔离来掩盖未知语义。若后续独立 fixtures 证明当前字段确实不足，需单独提出证据和授权，不能将本报告当作任意扩 schema 的许可。

## 11. Future Validation, Not Executed

本节只复述并细化用户要求的后续步骤，不授权执行。

1. **Generic deterministic fixtures**: 先冻结语义矩阵，覆盖相同操作的不同句法、正负 current state 与 negated change event、唯一/缺失/多个 target、已知/未知 policy、同值佐证、exact-facet 更新。使用通用实体和独立输入，不复制 v1 句式作为规则，不新增 benchmark 特例。
2. **最多三个 generic fixes**: 对 G1/G2/G3 分别实现和留开关/隔离对比，使每个改动对安全性与覆盖的影响可归因；维持无 dependency/propagation/retrieval/planner/answer 修改的边界。
3. **Local regression**: 旧 deterministic suite、新增 generic fixtures、protected hashes、canonical provenance、member isolation、ambiguous safety、legacy compatibility、endpoint mapping 全部检查。既要求无 false destructive stale，也要求确定更新完成，不以更多 UNCERTAIN 代替正确 revision。
4. **Phase 3 v1 dev regression**: 从原始 sources/responses 在新的输出目录运行，原 v1 artifacts 永不覆盖；报告相同完整指标和全部 histories，不筛失败、不改标签。该结果只能证明已知诊断分布回归，不能恢复 unseen 身份。
5. **全新 sealed Phase 3 v2 unseen canary**: 修正方法、policy、配置和 thresholds 先冻结，再使用未参与 fixes/fixture 设计的新 inputs；sources/expected transitions 在执行前 seal，不依据 S2 成败挑选，不在 cases 之间 patch。保留 unknown policy 和歧义负例，不能只测试新 registry 词表。固定同一 response 比较 S1/S2；如果另行授权真实 extraction，明确区分它与 conditional persistence acceptance。

v2 至少仍报告 CURRENT P/R、false stale/keep、各操作 exactness、PATCH isolation、subject/scope、ambiguity、endpoint wiring 与 retirement，并分别展示类别退步。特别要求观察 false keep 与 state recall，避免通过全部 UNCERTAIN 换取 false stale=0。

只有 v2 按事前 gates 通过才允许申请进入 Phase 4；v1 改后通过、平均 precision 提高、endpoint 能寻址或 deterministic tests 全绿，都不能单独替代该 gate。若 v2 失败，仅分类记录，不能边看 v2 边修再把同一集合叫 unseen。

## 12. Verification And Stop State

| Item | 本轮状态 |
| --- | --- |
| 全部 S2 failures 逐例分析 | 15/15，11 histories；不是新执行的测试计数 |
| Sealed source hashes | PASS，121 个 manifest entries |
| Pre-run / inference / result hash integrity | PASS |
| Method/code/config/test changes | 0 |
| Sealed artifact changes | 0 |
| Provider/API/network calls | 0 |
| Canary/benchmark/replay runs | 0 |
| New method test executions | 0 |
| Production/dependency/propagation/retrieval changes | 0 |
| 本轮新增文件 | `docs/stateframe_phase03_failure_forensics.md` |
| Phase 3 readiness | NO，未更改 |
| Phase 4 | NOT STARTED |

明确证据支持三个本地根因、两个 PATCH 的拒绝分支、六个 false keep 的来源、现有 schema 对这些语义的表达能力。尚无证据证明任何拟议修正已经正确、真实 provider 会给出同样充分的 candidates，或新 unseen 分布一定改善。未来仍需新验证；本轮在报告完成处停止。
