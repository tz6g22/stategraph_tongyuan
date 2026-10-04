# StateFrame MVP Phase 2

Date: 2026-09-19
Status: PASS
STATEFRAME_PHASE2_SHADOW_READY: YES
API_CALLS: 0
FORMAL_STATEGRAPH_WRITES: 0
PRODUCTION_SWITCH: NO
PHASE3_STARTED: NO

## Scope

Phase 2 implements and validates the isolated comparison paths:

```text
S1: FrameCandidate -> compatibility StateCandidate/StateNode -> existing StateRevision
S2: FrameCandidate -> StateFrame -> typed shadow persistence/revision
```

The typed path is an in-memory shadow repository only. The production StateNode
repository, dependency verifier, STRICT/WEAK semantics, propagation, retrieval,
planner, and answer generation were not switched or semantically modified.

## Implementation

`stategraph/state/stateframe_repository.py` contains the Phase 2 boundary:

- `TypedStateFrameShadowRepository` commits only the pure `resolve_change`
  proposal. It persists current, stale, uncertain, and historical-compatible
  frame versions in an isolated map and records a deterministic revision log.
- `LegacyStateCandidateShadowRepository` projects through
  `FrameCandidate.project_state_candidate()` and invokes the existing
  `StateLinker`, `InMemoryStateRepository`, and `StateRevision` path. This is
  the S1 control path, not a production fallback.
- `StateFrameEndpoint` and `relation_from_frame_endpoints` map a typed
  `version_id` to the unchanged `StateRelation.source_state_id`/
  `target_state_id` fields. No verifier or propagation code is called or
  changed.
- `legacy_state_node_to_frame` is an explicit read-only adapter for old
  snapshots. Unknown legacy cardinality/identity is readable but remains
  `UNCERTAIN`; it is never promoted to `CURRENT` automatically.

`stategraph/tests/test_stateframe_mvp_phase02.py` provides the local revision
fixtures. `scripts/verify_stateframe_phase02.py` seals the run with network
blocking, protected-module hashes, compileall, Phase 1 regression modules, and
the S1/S2 artifact.

## S2 semantics verified

- Functional replacement: London is `STALE`; Paris is `CURRENT`; the slot ID
  remains stable while the version ID changes.
- Set ADD: tea and coffee are independent member slots; adding coffee does not
  stale tea.
- Set REMOVE: coffee's positive version becomes `STALE`; tea remains `CURRENT`;
  the removal is recorded as a grounded negative current version.
- Event PATCH: Friday -> Monday changes only the time facet. Location and
  status remain current; all facets retain one event frame identity.
- Employment: Google and Microsoft are distinct membership frames and can both
  be current. Removing Google stales only Google's role/status facets.
- Same-value merge: the existing version ID is retained and corroborating
  provenance is added.
- Ambiguous destructive update: result is `UNCERTAIN`, with zero destructive
  stale transitions.
- Provenance: persisted evidence remains canonical
  `OBSERVATION_ABSOLUTE` with exact evidence quotes/spans.
- Dependency endpoint compatibility: StateRelation endpoints resolve to typed
  version IDs in the shadow repository, while relation type and verification
  behavior remain outside this phase.
- Legacy snapshot read: serialized StateNode data round-trips through the
  adapter without any production fallback.

## S1 versus S2 evidence

The detailed machine-readable comparison is:

`outputs/stateframe_mvp_phase02_local_20260919_r2/s1_vs_s2.json`

| Fixture | S1 current/stale | S2 current/stale | S2 revision statuses |
| --- | ---: | ---: | --- |
| functional replacement | 1 / 1 | 1 / 1 | CREATE, REPLACE |
| set ADD | 1 / 1 | 2 / 0 | CREATE, ADD |
| set REMOVE | 1 / 1 | 2 / 1 | CREATE, ADD, REMOVE |
| event PATCH | 3 / 1 | 3 / 1 | CREATE, CREATE, CREATE, PATCH |
| employment membership | 2 / 2 | 3 / 2 | CREATE, CREATE, ADD, ADD, REMOVE |

S1's set and employment rows demonstrate projection loss: the old
entity/attribute slot cannot preserve member cardinality. S2's slot identity
includes the set member or role organization binding, so the typed path keeps
independent memberships. The S1 result is retained as an expected control
observation, not patched in this phase.

## Verification

Authoritative run:

`outputs/stateframe_mvp_phase02_local_20260919_r2/VERIFICATION.json`

- 183/183 tests passed; 0 failures; 0 errors.
- `compileall` passed.
- `API_CALLS=0`; network attempts: 0.
- Formal StateGraph writes: 0.
- Protected production-path hashes are identical before and after the run.
- Typed gates all passed: functional replacement, set ADD, set REMOVE, event
  PATCH, employment isolation, ambiguous fail-safe, legacy compatibility,
  dependency endpoint mapping, and production untouched.
- Existing Phase 1 suite and the relevant local StateGraph regression modules
  were included. No provider, benchmark, dataset, or gold artifact was read.

## Files

Implementation:

- `stategraph/state/stateframe_repository.py`
- `stategraph/state/__init__.py`
- `stategraph/__init__.py`
- `stategraph/tests/test_stateframe_mvp_phase02.py`
- `scripts/verify_stateframe_phase02.py`

Artifacts:

- `outputs/stateframe_mvp_phase02_local_20260919_r2/VERIFICATION.json`
- `outputs/stateframe_mvp_phase02_local_20260919_r2/tests.log`
- `outputs/stateframe_mvp_phase02_local_20260919_r2/compileall.log`
- `outputs/stateframe_mvp_phase02_local_20260919_r2/s1_vs_s2.json`

## Phase gate

Phase 2 is shadow-ready and remains isolated. Phase 3 is not authorized or
started. The typed path has not become the production persistence path.
