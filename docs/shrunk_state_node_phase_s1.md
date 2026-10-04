# Shrunk StateNode Phase S1 Implementation

Status: PASS. **SHRUNK_STATE_NODE_S1_READY = YES**.

Authority: `docs/shrunk_stateframe_and_conditioned_eval_spec.md`, plus the explicit
isolated S1 implementation request. This is a local implementation gate only.
No production switch, provider implementation, benchmark, unseen canary,
conditioned evaluation, dependency redesign, or subsequent phase was executed.

Final evidence directory:
`outputs/shrunk_state_node_phase_s1_20260920_v3/`.

## Implemented Files

| File | Actual change |
| --- | --- |
| `stategraph/state/schema.py` | Four optional typed fields on StateNode/StateCandidate; typed slot identity and version endpoint alias; backward-compatible serialization |
| `stategraph/state/shrunk.py` | Independent local registry, single revision resolver, narrow verifier protocol, isolated StateNode repository |
| `stategraph/__init__.py` | Lazy compatibility exports; importing the new entrypoint no longer loads old typed pipelines |
| `stategraph/state/__init__.py` | Lazy historical exports while keeping the schema immediately available |
| `stategraph/tests/test_shrunk_state_node.py` | 44 offline contract/safety/wiring tests |
| `scripts/verify_shrunk_state_node_s1.py` | Network-disabled test runner, import boundary, transition audit, protected hashes and reports |

The new runtime entrypoint is
`stategraph.state.shrunk.ShrunkStateRepository`. It owns a private
`InMemoryStateRepository`; callers cannot pass a production repository into its
constructor. Its persisted semantic objects are StateNode, EvidenceRecord and
StateRelation, not another frame class. Existing production ingestion does not
select or call it. The S1 persistence mechanism is the existing in-memory store
and StateNode serialization, not a new disk/backend implementation.

## Runtime Contract Implemented

Input is an existing StateCandidate plus exactly one EvidenceRecord. Candidate
polarity and assertion mode are semantic content; registry cardinality and member
identity are locally derived. Provider-supplied cardinality/member keys and opaque
metadata do not override local policy or canonical slot IDs.

The four fields are `cardinality`, `member_key`, `polarity`, `assertion_mode`.
Existing canonical subject and full canonical field paths identify the object
and facet. There is no additional semantic-kind taxonomy. Registry entries are
immutable field policies supplied locally; an unregistered field is UNKNOWN,
never implicitly functional. S1 does not add a general-language predicate
classifier. Each policy binds a field to its cardinality and literal source
surface, independently of entity names and values.

Slot identity includes group, exact normalized subject, full field path,
time/condition scope, assertion mode, cardinality and member key. Functional
slots omit value; member slots use a JSON-typed normalized scalar member key.
Polarity is a version distinction, not a separate member slot. Version identity
hashes the slot, normalized value, polarity and birth observation/span anchor.
It excludes lifecycle, later merged evidence and wall-clock insertion time.
For typed nodes, `canonical_version_id == state_id`.

The single resolver implements:

- First locally supported assertion/member/facet creates CURRENT without an old
  target. An asserted negative member can be recorded without retiring anything.
- A supported functional value change retires only the exact CURRENT slot
  version, creates a new CURRENT version and returns its old version as an
  invalidation seed. Arrival order alone does not authorize a replacement.
- Positive member additions use independent slots. A supported exact negative
  member assertion retires that positive member only and persists a negative
  version. A later positive reassertion creates a new positive version.
- Same-value/same-polarity assertions merge provenance and retain the old ID.
- Facet updates revise only the full field path of the same object.
- Ambiguous identity, unknown cardinality, multiple targets, unsupported
  semantics, incompatible chronology or conditional/hypothetical updates cannot
  destructively retire a CURRENT state. New unresolved assertions are UNCERTAIN.
- Explicit bounded past scopes are retained as HISTORICAL. Future assertions
  are not promoted to current validity.

The local source checks are deliberately bounded literal assertion/negation
checks, not a second extraction engine. They require the complete grounded
clause to match the subject, field surface and value. Unsupported paraphrases
abstain; the tests do not claim general natural-language coverage. Exact subject
canonicalization is required in S1; unresolved aliases fail closed.

All decisions, evidence writes, old-version retirement, new-version insertion
and revision edges use the existing observation transaction boundary. Replay is
idempotent, including replay of an already retired version. Reused evidence with
different content is rejected. Only UPDATES edges are generated by this resolver;
dependency discovery and invalidation propagation are not executed here.

## Narrow Verifier Interface

`NarrowSemanticVerifier` is a Protocol only. There is no provider adapter or
default LLM call. The default repository has no verifier configured.

The interface is eligible only for an UNKNOWN local result on a possible
destructive transition with one exact CURRENT target, known cardinality,
compatible operation/scope and forward chronology. Ordinary first assertions,
ADD, same-value merge, deterministic SUPPORTED and CONTRADICTED results never
call it. Multiple targets, wrong members and incompatible scopes cannot be
resolved by asking the verifier to choose a target.

Its immutable request exposes semantic subject/field/old and new scalar values,
polarity, scope, cardinality, derived transition and grounded evidence. It carries
no state ID, case ID, expected answer, lifecycle target or repository handle.
Replies are tri-state, require exact operation and target confirmation, and must
echo the complete supplied evidence span. Malformed replies, exceptions and
unsupported quotes fail closed. The local resolver rechecks the target snapshot,
identity/scope compatibility and provenance before committing.

Injected probes in `VerifierBoundaryTests` test wiring only. In particular, the
SUPPORTED wiring probe is not evidence that its deliberately ambiguous sentence
supports a real-world update. Semantic correctness and false-stale measurements
use the default repository with no injected judge or verifier.

## Provenance And Endpoint Compatibility

The boundary accepts only `OBSERVATION_ABSOLUTE` EvidenceRecord spans whose ID
matches the original observation substring/coordinates. Subject and value must
be present in that span, and the candidate must reference exactly that evidence.
The existing EvidenceRecord serialization is unchanged. A fixture with a
nonsemantic prefix verifies that persisted absolute coordinates exclude it.
This module does not guess local coordinates or import the old source adapters.

Dependency endpoints remain StateRelation `source_state_id`/`target_state_id`
version references. Endpoint tests attach an existing STRICT relation directly
as a compatibility fixture, revise its premise, then verify that its original
endpoint remains resolvable and is STALE. No edge is automatically rebound to a
new version; the replacement ID differs. Facet/member siblings remain CURRENT.
This demonstrates endpoint compatibility, not dependency discovery accuracy or
cascade effectiveness. Dependency evaluation was not run.

Legacy StateNode default serialization retains its previous shape and identity
behavior. Explicit typed nodes serialize the extensions. Candidate roundtrips
also retain NEGATIVE polarity before cardinality is locally resolved. Historical
Full StateFrame modules remain available through explicit lazy exports for their
existing regressions. They are never a runtime fallback for the new path.

## Verification

Final run used Python 3.12 from the existing Graphiti virtual environment with
the existing system packages appended through a fixed PYTHONPATH. No package was
installed and no baseline repository was modified. The exact executable/version
are recorded in VERIFICATION.json.

```bash
PYTHONPATH="$PROJECT_ROOT/baselines/graphiti:$PROJECT_ROOT/external_baselines/graphiti/.venv/lib/python3.12/site-packages:/usr/lib/python3/dist-packages" \
  external_baselines/graphiti/.venv/bin/python scripts/verify_shrunk_state_node_s1.py \
  --output outputs/shrunk_state_node_phase_s1_<fresh_run_id>
```

A future fresh verification directory must contain a BEFORE_HASHES.json copied
from the approved pre-edit baseline (or an independently captured pre-edit hash
manifest); without that baseline the protected-source gate fails closed. Do not
reuse or overwrite any sealed verification directory.

| Gate | Final actual result |
| --- | --- |
| New shrunk tests | 44/44 PASS; zero skips |
| Existing Phase1 runner | 171/171 PASS |
| Existing Phase2 runner | 183/183 PASS |
| Relevant StateGraph suite | 507/507 PASS; zero skips |
| compileall | PASS: stategraph plus the new verification script |
| Deterministic transition audit | 16 writes; 74 should-keep opportunities |
| False destructive stale | 0/74 |
| False keep in audit | 0; four expected direct retirements completed |
| Member/facet isolation | PASS |
| Dependency endpoint compatibility | PASS |
| New-path Full StateFrame runtime imports | 0 |
| Default-path semantic verifier calls | 0 |
| Provider/API calls | 0; zero blocked network attempts in final suites |
| Source stable during validation | PASS |
| Protected existing files | Unchanged, except explicitly authorized schema and two import facades |

Counts above are separate overlapping suites, not a sum of independent tests.
The 507-test suite includes legacy StateGraph extraction, identity, revision,
storage/snapshot, dependency, propagation, retrieval/planning and answer-related
regressions. Five historical Full StateFrame authorization/refinement development
modules are explicitly excluded from that suite because they belong to the old
separately configured semantic-verifier gate; the exact list is in VERIFICATION.
Both required frozen Phase1/Phase2 runners were run unmodified and passed in full.

The new tests and transition audit run before legacy imports, with a meta-path
guard that raises on any Full StateFrame/source/shadow/repository, standalone
authorization or old semantic-verifier import. The post-run sys.modules audit
also finds none. Networking is denied at socket connect/connect_ex and
create_connection throughout local suites. No extraction provider is created
for the new path.

`BEFORE_HASHES.json` records the root's pre-edit source state. Root Git is absent;
`SOURCE_HASHES.json` is the final source revision authority. Comparing every
pre-existing StateGraph Python file identifies only the two import facades and
schema as changed. Dependency discovery/verifier, STRICT/WEAK policy,
propagation, ranking, planner, answer generation, system entrypoints and old
Full StateFrame implementation were not changed.

## Run History And Artifacts

- `..._v1`: new 36-test suite passed; legacy verification failed because the
  virtual environment lacked jsonschema. Result preserved as FAIL, not relabeled.
- `..._v2`: 42 new tests and all required regressions passed using already
  installed dependencies. This remains an intermediate source revision.
- `..._v3`: final 44 tests plus all required regressions passed after the
  unresolved-candidate serialization guard and verifier quote-boundary test.

The runner saves VERIFICATION.json, SOURCE_HASHES.json,
IMPLEMENTATION_SUMMARY.json, tests.log, per-suite logs/results, compileall.log,
TRANSITION_AUDIT.json and both original phase runner artifact directories.
No old sealed canary, benchmark artifact, dataset or exposure ledger was changed.

New runtime implementation is 304 lines; its 44-test module is 422 lines and the
verification harness is 209 lines. Full StateFrame is bypassed, not deleted from
historical storage. Thus repository line count grows during this isolated phase,
but the new runtime has no parallel frame/intent/authorization state machine.

## Scope Limits And Handoff

S1 is not proof of unseen generalization or production readiness. Unsupported
natural language stays UNCERTAIN; the verifier has no provider implementation.
Inputs are locally constructed StateCandidates, not a new live extractor. Complex
multi-participant objects and unresolved subject aliases are not accepted as
atomic members. Negative CURRENT nodes carry explicit polarity, but legacy
production consumers have not been switched or audited for that new contract.
Do not feed this isolated store into the legacy production path.

There was no production persistence switch, runtime old-pipeline fallback,
support-group implementation, benchmark invocation, or automatic next phase.
The next action requires explicit authorization for the next isolated stage.

**SHRUNK_STATE_NODE_S1_READY = YES**, limited to the verified offline S1 scope.
