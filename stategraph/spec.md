# Graphiti → StateGraph Implementation Plan (historical)

> This document records the original Graphiti-based implementation plan.  The
> current code has completed the decoupling migration: StateGraph semantics use
> backend-neutral records and native retrieval, while Graphiti is optional
> backend/legacy compatibility only.  See `README.md` and the final architecture
> artifact for the current implementation contract.

## 目标

基于当前 Graphiti baseline 实现 StateGraph idea。

原则：

* 保留 Graphiti 已有 temporal graph storage、entity extraction、episode ingestion、retrieval 基础能力。
* 不重写 graph memory 系统。
* 在 Graphiti graph layer 上增加 state revision、state validity management、invalidation propagation 和 premise-aware retrieval。
* 最终系统行为需要符合 StateGraph 方法定义。

---

# 1. 当前 Graphiti 与目标 StateGraph 的主要差异

## Graphiti 当前流程

```
episode
  ↓
entity/fact extraction
  ↓
temporal knowledge graph
  ↓
graph retrieval
  ↓
LLM response
```

## StateGraph 目标流程

```
observation
  ↓
state extraction
  ↓
state linking
  ↓
conflict detection
  ↓
state revision
  ↓
invalidation propagation
  ↓
current state retrieval
  ↓
premise checking
  ↓
answer generation
```

---

# 2. 保留 Graphiti 部分

以下模块保持不变或仅做 adapter：

## Graph storage

继续使用 Graphiti graph backend。

新增 state-related node/edge 类型，不替换原 graph。

## Entity resolution

继续使用 Graphiti entity linking。

## Episode ingestion

继续支持 observation 写入。

## Retrieval infrastructure

保留 Graphiti graph traversal 能力。

但是 retrieval 入口需要从：

```
retrieve relevant facts
```

修改为：

```
retrieve valid current states + evidence
```

---

# 3. 新增 StateGraph 数据层

## 新增 State Node

在 Graphiti fact representation 上增加 state abstraction。

需要支持：

```
state_id

entity

attribute

value

time_scope

condition_scope

status

evidence_id

confidence
```

status:

```
current
stale
historical
uncertain
```

目的：

替代 Graphiti 中 fact 永久有效的假设。

---

## 新增 Evidence Node

保存：

```
source observation

timestamp

original text/span

origin
```

要求：

每个 current state 必须可追溯 evidence。

---

# 4. 新增 State Linking Module

新增：

```
state_linking.py
```

功能：

输入：

```
new_state
existing_states
```

输出：

```
candidate related states
```

匹配条件：

* same entity
* same attribute
* temporal overlap
* semantic relation

不要直接修改 state。

只负责建立 candidate relation。

---

# 5. 新增 Conflict Detection Module

新增：

```
conflict_detection.py
```

功能：

判断 new state 与 old state 的关系。

输出：

```
consistent

duplicate

update

explicit_conflict

implicit_invalidation

temporary_exception

uncertain
```

要求：

支持 implicit invalidation。

例如：

旧：

```
availability = free Friday
```

新：

```
flight Friday
```

需要识别：

```
availability should become invalid
```

---

# 6. 新增 State Revision Module

新增：

```
state_revision.py
```

负责：

* 更新 current state
* 标记 stale state
* 创建 revision edge

规则：

new state:

```
status=current
```

old state:

```
status=stale
```

新增 edge：

```
updates

invalidates
```

禁止：

删除旧 state。

必须保留历史。

---

# 7. 新增 Invalidation Propagation

新增：

```
invalidation_propagation.py
```

实现 cascading invalidation。

新增 edge 类型：

```
depends-on

derived-from

affects-action
```

传播逻辑：

```
invalidated state

↓

dependent states

↓

derived states

↓

actions
```

输出：

```
invalidated_state_ids
```

要求：

支持 precision / recall evaluation。

---

# 8. 修改 Retrieval Pipeline

修改 Graphiti retrieval。

当前：

```
graph search
 ↓
facts
```

改为：

```
query
 ↓
premise extraction

current state graph filtering

 ↓

valid states + evidence
```

限制：

禁止返回：

```
stale state
historical state
```

除非显式用于分析。

---

# 9. 新增 Premise Checking

新增：

```
premise_checker.py
```

位置：

answer generation 前。

输入：

```
query

current states
```

输出：

```
premises

conflicting_state_ids

response_policy
```

功能：

检测 query 中基于旧状态的假设。

例如：

query:

```
Because I am free Friday...
```

graph:

```
Friday unavailable
```

需要：

```
reject stale premise
```

---

# 10. 修改 Answer Generation

当前：

```
retrieved context → answer
```

修改：

```
validated current states
+
supporting evidence
+
premise correction
→ answer
```

要求：

generation 不允许直接访问 stale memory。

---

# 11. 推荐实现顺序

## Phase 1

Graphiti baseline 固化。

完成：

* 原始 Graphiti runner
* baseline evaluation

不要修改。

---

## Phase 2

增加：

```
state schema

state extraction

state node storage
```

目标：

observation → StateGraph

---

## Phase 3

增加：

```
state linking

conflict detection

state revision
```

目标：

支持状态更新。

---

## Phase 4

增加：

```
dependency edges

invalidation propagation
```

目标：

支持 cascading invalidation。

---

## Phase 5

修改：

```
retrieval

answer generation

```

增加：

```
premise checking
```

目标：

完整 StateGraph inference pipeline。

---

# 12. 不允许的实现方向

## 不允许退化成 Graph RAG

错误：

```
Graph retrieval + LLM
```

这仍然是 Graphiti。

---

## 不允许使用 summary memory 替代 state revision

错误：

```
history summary
```

无法表达：

* current
* stale
* dependency

---

## 不允许删除历史状态

必须保留：

```
current

stale

historical
```

---

## 不允许只做 retrieval filtering

StateGraph 核心不是：

```
remove old facts
```

而是：

```
state revision + invalidation propagation
```

---

# 13. 最终目标代码结构

```
stategraph/

├── graphiti_adapter/

├── state/

│   ├── schema.py

│   ├── extraction.py

│   ├── linking.py


├── revision/

│   ├── conflict_detection.py

│   ├── state_revision.py


├── propagation/

│   ├── dependency.py

│   ├── invalidation.py


├── retrieval/

│   ├── premise_checker.py

│   ├── current_state_retriever.py


├── evaluation/

└── baselines/
```

最终实现：

Graphiti:

```
temporal graph memory
```

↓

StateGraph:

```
temporal graph +
state revision +
typed invalidation propagation +
premise-aware retrieval
```
