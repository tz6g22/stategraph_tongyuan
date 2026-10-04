# StateFrame MVP Phase 0 + Phase 1

Date: 2026-09-19
Status: PASS
STATEFRAME_PHASE1_SHADOW_READY: YES
API_CALLS: 0
FORMAL_STATEGRAPH_WRITES: 0
PHASE2_STARTED: NO

## Scope

This report records the local implementation of the StateFrame MVP and its
offline shadow path. The existing production StateNode extraction, revision,
dependency, propagation, retrieval, planner, answer generation, and repository
write paths were not switched or semantically changed.

## Phase 0 implementation

`stategraph/state/stateframe.py` contains the isolated MVP contracts:

- `FrameCandidate` is the unified provider-facing semantic candidate.
- `StateFrame` is the typed, versioned persistent target representation used by
  the shadow resolver; it is not yet the production repository representation.
- `FrameKind` has six values: FACT, RELATION, ROLE, EVENT, ACTION, DERIVED.
- `CardinalityRegistry` is deterministic and explicit. Unknown rules produce
  `UNKNOWN_CARDINALITY`; the resolver does not default an unknown candidate to a
  destructive functional update.
- `ProposedChangeIntent` cannot contain provider-selected IDs. `ResolvedChangeIntent`
  is produced only by the local resolver.
- `frame_id`, `slot_id`, and `version_id` are deterministic SHA-256 IDs. Values
  are excluded from functional slot identity and included for set-valued member
  identity. Version identity includes value, polarity, modality, provenance and
  confidence.
- `FrameProvenance` accepts only `OBSERVATION_ABSOLUTE` coordinates and validates
  exact evidence/value bounds and source identity.
- `resolve_change` is pure and fail-closed. It supports ASSERT, REPLACE, ADD,
  REMOVE, PATCH, and UNKNOWN without mutating a repository.

The current implementation deliberately does not implement support groups,
dependency verification changes, propagation changes, or production persistence.

## Phase 1 implementation

`stategraph/state/stateframe_source.py` owns the one source-local to canonical
absolute coordinate boundary. It uses the existing semantic-source segmentation
rules, excludes session/serialization metadata, checks exact local spans, and
constructs canonical evidence references.

`stategraph/state/stateframe_shadow.py` contains:

- one strict `FRAME_EXTRACTION_OUTPUT_SCHEMA` for first-pass and singleton
  recovery responses;
- `parse_frame_response`, which rejects ungrounded values, off-target recovery,
  invalid scopes, unresolved first-person subjects, provider IDs, provider
  cardinality, and unsupported coordinate spaces;
- `RecordedFrameTransport`, which only replays pre-recorded source/pass/target
  matched responses and has no SDK, network, credentials, retry, or live fallback;
- `StateFrameShadowExtractor`, which produces shadow candidates and artifacts
  without a repository or formal StateGraph write;
- separate semantic equivalence, full StateCandidate contract equivalence,
  coverage, subject attribution, grounding, identity delta, recovery and replay
  request metrics.

The old extraction result is supplied by the caller. The new path does not call
the old extractor. First-pass and singleton recovery use exactly the same
FrameCandidate schema and the same parser.

## Tests and verification

The authoritative local run is:

`outputs/stateframe_mvp_phase01_local_20260919_r3/VERIFICATION.json`

It records 171/171 tests passing, compileall passing, API_CALLS=0, no formal
StateGraph writes, no network attempts, and no Phase 2 start. The detailed test
log is in `outputs/stateframe_mvp_phase01_local_20260919_r3/tests.log`; compile
output is in `compileall.log`; the replayed shadow artifact is in
`shadow_example.json`.

The fixture matrix covers functional replacement, functional ASSERT fail-safe,
set add/remove, role/employment membership, event facet patch and cancellation,
third-party subject attribution, polarity, temporal scope, condition/action,
ambiguous destructive identity, unknown cardinality, unknown operation, low
confidence, target-value mismatch, exact provenance, metadata exclusion,
provider-ID rejection, and off-target singleton recovery.

Existing local regression modules were included in the same run, including
revision, canonical slot identity, extraction subdivision, factual relation and
proposition coverage, cascade propagation, dependency relational separation,
semantic isolation, retrieval premise, planner attribution, final answer, and
structured JSON retry tests.

## Known Phase 0/1 boundaries

- `StateFrame` is currently an isolated typed model; production storage still
  uses the existing StateNode path.
- Provider calls/tokens are `0` for this run. Recorded usage fields are accepted
  in the shadow request artifact but no live cost claim is made.
- The shadow path measures coverage against the existing deterministic
  proposition planner and reports precision as unmeasured unless independent
  offline fixture references are supplied.
- Existing semantic output prompts were not changed. Phase 1 validates a
  unified recorded-response contract without claiming provider acceptance.
- No dependency verifier, STRICT/WEAK semantics, propagation algorithm,
  retrieval scoring/top-k, planner, answer generation, benchmark, or dataset was
  changed or run.

## Next gate

Do not begin Phase 2 automatically. Phase 2 requires explicit authorization and
must first review this report plus the sealed local verification artifact. The
next implementation gate is typed persistence/revision shadowing, not a
production switch or a provider run.
