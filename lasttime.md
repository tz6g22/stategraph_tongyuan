# 最近一次实验结果

## Memora 10-question diagnostic

数据范围：`academic_researcher / weekly`，完整 158-session history，10 questions，3 methods。

评测协议：`EVALUATOR_PROTOCOL = MEMORA_OPENAI_SINGLE_JUDGE_DIAGNOSTIC`。保留 Memora 官方 judge prompt、yes/no rubric 与 FAMA 公式，但使用单一 OpenAI `gpt-4.1` judge，不是官方 OpenRouter 三模型 ensemble。

| Method | Questions | FAMA | Presence | Forgetting/Absence | Task Accuracy |
|---|---:|---:|---:|---:|---:|
| StateGraph | 10 | 15.571 | 10/43 (0.233) | 17/19 (0.895) | 0.435 |
| Graphiti | 10 | 13.452 | 8/43 (0.186) | 15/19 (0.789) | 0.371 |
| Mem0 | 10 | 39.571 | 19/43 (0.442) | 15/19 (0.789) | 0.548 |

## 前 3 题与新增 7 题

| Partition | Method | FAMA | Presence | Forgetting/Absence | Task Accuracy |
|---|---|---:|---:|---:|---:|
| 前 3 题 | StateGraph | 16.667 | 0.125 | 1.000 | 0.632 |
| 前 3 题 | Graphiti | 16.667 | 0.125 | 0.818 | 0.526 |
| 前 3 题 | Mem0 | 33.333 | 0.250 | 0.909 | 0.632 |
| 新增 7 题 | StateGraph | 15.102 | 0.257 | 0.750 | 0.349 |
| 新增 7 题 | Graphiti | 12.075 | 0.200 | 0.750 | 0.302 |
| 新增 7 题 | Mem0 | 42.245 | 0.486 | 0.625 | 0.512 |

## StateGraph 机制

- CURRENT: 150
- STALE: 54
- HISTORICAL: 11
- UNCERTAIN: 5
- UPDATES: 7
- INVALIDATES: 53
- DEPENDS_ON: 0
- DERIVED_FROM: 0
- AFFECTS_ACTION: 4
- zero-result queries: 0
- stale/historical states in final context: 0
- premise rejections: 0

## 结论

StateGraph 的 Forgetting/Absence（0.895）高于 Graphiti 和 Mem0（均为 0.789），但 Presence（0.233）明显低于 Mem0（0.442）。结果呈现出“较强 obsolete-memory rejection、较弱 memory recall”的模式。Mem0 的主要优势来自 Presence/recall，而不是 Forgetting。

没有在检索 trace 中发现 STALE/HISTORICAL 状态进入最终 context 的 over-invalidation 证据；但单 persona、10 questions 的结果不足以支持广泛泛化结论。可继续做多 persona diagnostic，但不能视为官方 Memora multi-judge 分数。

## 成本与封存

- 未重新 ingestion/extraction。
- 新答案生成：21 次 `gpt-5-nano` 调用。
- 新 judge：129 次有效 `gpt-4.1` 调用；前 57 条 judge 结果复用。
- 另一次脚本变量错误导致的 129 次未持久化 judge 调用已在 `API_USAGE.json` 中披露。
- 30/30 predictions 已 seal，gold 只在 seal 后进入 evaluator。

详细产物：[`outputs/stategraph_memora_weekly_1persona_10q_openai_judge/`](outputs/stategraph_memora_weekly_1persona_10q_openai_judge/)
