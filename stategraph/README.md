# StateGraph

This package implements a lifecycle-aware, state-centric memory method.  The
semantic path owns extraction, evidence, revision, dependency propagation, and
retrieval.  Persistence is selected explicitly through a backend contract; the
native backend has no Graphiti runtime dependency.

The optional Graphiti backend and legacy readers live behind compatibility
boundaries.  They are not part of the StateGraph semantic contract, and no file
under `baselines/` is imported for modification or patched.

## Native construction

Production construction is explicit:

```python
from stategraph import NativeStateGraphBackend, Observation, StateGraph

state_graph = StateGraph(backend=NativeStateGraphBackend())
result = await state_graph.ingest(observation)
retrieval = await state_graph.retrieve(question, group_id=observation.group_id)
answer_input = retrieval.grounded_context()
```

`StateGraph.from_graphiti(...)` remains a legacy compatibility factory for
existing artifacts.  New semantic code uses `ObservationRecord`,
`EvidenceRecord`, and StateGraph-owned IDs; backend-specific identifiers remain
opaque metadata.

## Lifecycle semantics

- `current`: the latest valid state and eligible for answer retrieval.
- `stale`: invalidated by revision or propagation; retained for audit only.
- `historical`: validity ended normally; retained for audit only.
- `uncertain`: unresolved low-confidence or equal-confidence conflict; excluded from
  answer retrieval.
- `updates` and `invalidates` preserve revision history.
- Dependency candidates are discovered after all direct revisions in one observation,
  then counterfactually classified as `strict_dependency`, `weak_dependency`, or
  `no_dependency`. Verified strict and weak edges are retained; only strict edges can
  propagate invalidation.
- `depends-on`, `derived-from`, and `affects-action` use one direction:
  prerequisite/source state to downstream/dependent state or action.
- Ingestion builds and persists the verified dependency graph before running one
  cycle-safe, multi-seed cascade pass. Propagation records its root seed, traversed
  edge, reason, and depth.

`CurrentStateRetriever` cannot return stale/historical state. It raises if a selected
state lacks evidence. Historical access is a separate analytical method and is never
accepted by `build_answer_context`.

## Tests

The method-level tests use only synthetic state records and do not load benchmark data:

```bash
python3 -m unittest discover -s stategraph/tests -v
```

Benchmark evaluation remains outside this package. Formal dataset runs must use only
LongMemEval, LongMemEval-V2, Memora, and the Conflict Resolution split of
MemoryAgentBench, as required by the repository evaluation protocol.
