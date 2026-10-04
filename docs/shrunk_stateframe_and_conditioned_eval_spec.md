# Shrunk StateFrame Architecture And Conditioned Evaluation Preregistration

Specification ID: SG-SHRINK-CME-1.0

Status: DESIGN FROZEN; NOT IMPLEMENTED; NO CASES SELECTED OR EXECUTED.

SHRINK_IMPLEMENTATION_READY = YES, for isolated implementation against this
specification. This is not production readiness, benchmark admission, an unseen
PASS, or authorization to reopen the failed autonomous run. All prior seals,
gates and NO-GO results remain unchanged.

This document is the only file added in this task. No code, dataset, output,
provider configuration or previous maintenance document is changed.

## 1. Final Architecture Choice

**A. StateNode + minimal extensions.** StateNode remains the only persistent
semantic record. There is no new StateFrameLite class and no Full StateFrame
on the new inference path. Existing StateCandidate is the single transient
extraction record; StateRelation continues to reference StateNode versions.

The refinement retains explicit cardinality and exact member identity, but
removes a separately materialized frame and the independent proposed/resolved
ChangeIntent protocol. Provider output describes an assertion, not a command
to mutate memory. One resolver derives a possible transition from assertion,
slot policy and actual current versions, authorizes it, and prepares an atomic
write. A narrowly eligible semantic verifier is an internal fallback, not a
second resolver.

The observed 0/35 false-stale result is an invariant target to preserve, not a
guarantee about this unimplemented design. No destructive action may affect a
different member, field, scope or object merely because a new value arrived.

### Existing Interfaces Reused

- `stategraph/state/schema.py`: StateNode, StateCandidate, StateStatus,
  canonical slot/version properties and StateRelation endpoint fields.
- `stategraph/revision/state_revision.py`: RevisionResult with state,
  changed_states, revision_edges, invalidated_state_ids and duplicate_of.
- `stategraph/storage/base.py`: get_state/list_states/apply/list_relations;
  `stategraph/storage/memory.py`: observation transaction boundary.
- `stategraph/state/stateframe_source.py`: source-local to
  OBSERVATION_ABSOLUTE evidence boundary, extracted from its full-frame coupling
  when implemented; its grounding invariants remain.
- `stategraph/propagation/invalidation.py`: existing propagation consumer of
  version IDs and verified dependency edges; no new propagation semantics.
- `evaluation_protocol/README.md`: RetrievalRecord -> shared answer generation
  -> PredictionRecord seal -> evaluator. No answer from a repaired graph.

These are verified interface locations, not claims that the new behavior already
exists. Existing canonical-slot implementation cannot simply be reused unchanged
for member identity; the minimal identity extension below is required.

## 2. Component Disposition

| Component | Decision | Exact disposition |
| --- | --- | --- |
| FrameCandidate | REMOVE | No new-path full-frame DTO; use one StateCandidate assertion contract for first pass and singleton recovery. Full-frame decoding remains read-only for historical replay |
| StateFrame | REMOVE | Persist StateNode only; no second graph, shadow canonical store, or full-frame materialization before revision |
| frame_id | REMOVE | Object identity is canonical_subject_id; event facets share that subject without an extra frame identifier |
| slot_id | KEEP | Existing canonical_slot_id property gains cardinality/member-aware identity; it is locally derived, never model generated |
| member_key | KEEP | A local canonical member key distinguishes set members and is compared identically at lookup, merge and retirement |
| cardinality registry | SIMPLIFY | A local versioned field policy chooses FUNCTIONAL, SET_VALUED or UNKNOWN plus the member-key codec; no provider cardinality decision or kind/facet cross-product ontology |
| ChangeIntent | REMOVE | Delete both ProposedChangeIntent and ResolvedChangeIntent from the new path; audit outcomes are not executable commands |
| target_hint | REMOVE | Provider never supplies old value or target IDs; the resolver selects an actual uniquely eligible old version or declines |
| local ChangeAuthorizationJudge | SIMPLIFY | Fold deterministic evidence authorization into the single resolver; remove the separate injectable semantic policy layer |
| SemanticChangeVerifier | KEEP | UNKNOWN-only fallback after all nonsemantic eligibility checks; it returns evidence support, never write authority |
| typed revision resolver | SIMPLIFY | One slot-delta resolver over StateCandidate and StateNode returns the existing RevisionResult; no parallel frame and legacy decision engines |
| legacy adapters | SIMPLIFY | Explicit snapshot read/replay and migration only; never a runtime fallback on new-path failure |

Two complete transition abstractions disappear: full-frame materialization and
the ProposedChangeIntent -> target_hint -> ResolvedChangeIntent command chain.
The standalone local judge is also merged into the resolver. Keeping audit reason
strings or a private local branch variable does not recreate a public intent DTO.

## 3. Minimal Persistent Schema

### Exactly Four New StateNode Fields

The same assertion-relevant fields have counterparts on StateCandidate. Local
policy fields are populated after extraction, not exposed as model authority.

| Field | Type / default | Source | Identity / mutability |
| --- | --- | --- | --- |
| cardinality | enum FUNCTIONAL, SET_VALUED, UNKNOWN; default UNKNOWN on unmigrated data | Local field-policy registry | Included in slot identity; immutable within a version |
| member_key | canonical typed scalar or canonical tuple, nullable | Local codec from a grounded member value/anchor; null for FUNCTIONAL | Included only for SET_VALUED; immutable |
| polarity | enum POSITIVE, NEGATIVE, UNKNOWN | Source-grounded semantic assertion, validated locally | Version semantics, not slot identity; immutable |
| assertion_mode | enum ASSERTED, PLANNED, OBLIGATORY, HYPOTHETICAL, UNKNOWN | Source-grounded semantics; omitted ordinary assertion normalizes to ASSERTED | Mode is a slot qualifier so plans/hypotheses cannot replace actual facts; immutable |

No persistent semantic_kind, frame_id, facet field, arbitrary participants map,
key_bindings map, ChangeIntent, target_hint, or second version_id field is added.
Snapshot schema version is an envelope integer, `state_node_schema_version=2`,
not a second per-node identity field. Registry/codec revisions are sealed in the
run/snapshot manifest; changing them requires explicit migration, never lazy
identity recomputation against a different registry.

Existing StateNode fields remain authoritative:

- entity/attribute/value preserve source-facing content; canonical_subject_id
  and canonical_field_id supply the locally resolved identity.
- time_scope and condition_scope retain validity qualifiers; ingestion time
  is observation order, not an inferred semantic time scope.
- state_id is the immutable **version endpoint ID**. canonical_version_id is
  its new-schema alias, not an independently mutable identifier.
- status remains CURRENT, STALE, HISTORICAL or UNCERTAIN; confidence is not
  evidence authorization and cannot authorize retirement by itself.
- evidence_refs resolve to canonical evidence records. Evidence spans/quotes,
  observation ID, original-source hash and sequence retain their current roles.
- metadata may contain diagnostics, policy version and trace outcomes, but no
  competing authoritative slot, operation, polarity or target selector.

Evidence merge can append corroboration to an existing version without changing
its state_id. A semantic change creates another version. Existing versions and
their value/member/identity fields are never rewritten in place.

### Subject, Field And Facet Representation

`canonical_subject_id` identifies the referent: person, named event, named plan
or object. It must not include a mutable time/location/status value. Two event
instances with the same display name need an evidence-supported discriminator;
without one, identity is unresolved and cannot authorize a destructive update.

`canonical_field_id` is the complete field path, such as `meeting.time` or
`meeting.status`, not a bare `time` shared across unrelated objects. A facet
is simply such a field on an identified subject. This removes separate facet
and frame-kind taxonomies without discarding facet isolation.

Do not assert universal scalarity for natural-language predicates. For example,
ordinary employer memberships and ordinary preferences are set-valued; a stated
single primary employer can use a distinct, registered functional field. A role
does not become single-valued per employer unless that exclusivity is explicit
in the field contract. Unknown fields are UNKNOWN, not default FUNCTIONAL.

The minimum employment contract is an independent employer member per person.
Role claims may use additional set-valued relation fields, keyed by grounded
organization/role tuples, without overwriting another role. Withdrawing employer
membership does not blindly retire every role fact: any downstream current-role
invalidation must use an explicit withdrawn claim or the existing verified
dependency pipeline. Historical role facts are not membership tombstones.

## 4. Identity Rules

Canonical encoding is a versioned, type-preserving canonical JSON encoding.
Text entity/member normalization is shared across lookup, comparison and hashing;
raw display strings are never a second target-equality rule. Numeric/string/bool
values remain distinct unless the field has an explicitly registered codec.

Let G=group_id, S=resolved subject, F=full canonical field, T=normalized semantic
time scope, C=normalized condition scope, M=assertion_mode, K=cardinality.

| Slot | Slot key before hashing |
| --- | --- |
| FUNCTIONAL | [schema=2, G, S, F, T, C, M, FUNCTIONAL] |
| SET_VALUED | [schema=2, G, S, F, T, C, M, SET_VALUED, member_key] |
| Facet | The FUNCTIONAL or SET_VALUED rule above, with the same object S and its own full field F |
| UNKNOWN / unresolved identity | No destructive canonical slot is assigned; retain an UNCERTAIN observation version in an explicitly unresolved namespace |

`canonical_slot_id = hash(slot_key)`. Functional slot identity excludes value;
set slot identity includes member_key but excludes polarity. Thus positive and
negative assertions about the same member share a slot. A spelling/case variant
normalized to the same member cannot fail a separate raw target-hint check,
because no such check exists in this design.

For a newly committed semantic version:
`state_id = version_hash(slot_id, canonical_value, polarity, birth_observation_id,
birth_sequence, assertion_anchor)`; the anchor distinguishes different assertions
in one observation. Status, later corroborating evidence and wall-clock retry time
are excluded. Exact ingestion replay is idempotent; a return to a previously
held value after a real intervening revision creates a new version.

An unresolved version instead hashes its grounded observation/anchor and
unresolved identity descriptor. It is UNCERTAIN and cannot be a CURRENT premise
or a retirement target. Failed grounding is rejected with a trace, not assigned
fabricated evidence so it can be stored as a state.

## 5. One Assertion Contract, No ChangeIntent

First pass and singleton recovery emit the same semantic fields:

| Provider field | Contract |
| --- | --- |
| subject surface | Literal subject/object referent plus source-local anchor; no canonical ID |
| field description | Source-grounded relation/property; locally mapped to canonical_field_id only when semantics match a known policy |
| value / value surface | New assertion content plus its literal anchor; not an old-target selector |
| member surface, optional | Only when the member identity is not the value, a grounded member/qualified-relation anchor under the field policy; no arbitrary free binding map |
| polarity | POSITIVE / NEGATIVE / UNKNOWN |
| assertion_mode | Existing minimal mode vocabulary above |
| time / condition scope | Source-supported qualifiers and evidence; not inferred from ingestion metadata |
| confidence and evidence spans | Confidence bounded to [0,1]; exact source anchors required |

Local ingestion converts offsets once to OBSERVATION_ABSOLUTE, resolves evidence
records, subject, field and member identity, and creates StateCandidate. The
provider cannot output cardinality, slot/version IDs, an old value, a target ID,
ASSERT/ADD/REMOVE/PATCH/REPLACE, or expected lifecycle. Extraction metadata cannot
smuggle these fields back in. The value/member may be unknown, but then it cannot
select a destructive target.

There is **no independent ChangeIntent object**. ASSERT/ADD/REMOVE/PATCH/REPLACE
may appear in human-readable revision logs for comparison with historical runs,
but they are locally derived outcomes, not provider proposals or separately
normalized semantic commands. The old target comes only from current repository
versions matching the locally computed slot, never from a model hint.

## 6. Minimal Revision Contract

### Decision Table

All rows assume valid grounded evidence; invalid evidence is rejected first.

| Situation | Resolver behavior |
| --- | --- |
| Known slot, no existing version, ordinary positive assertion | Create CURRENT without demanding an old target |
| Same slot, same semantic value/polarity/mode | Merge evidence into the existing version; no verifier and no retirement |
| SET_VALUED, positive new member | Create independent member CURRENT; never retire a different member |
| SET_VALUED, same positive member reiterated | Merge, not another version and not another membership |
| SET_VALUED, grounded negative assertion of an exact existing positive member | Derive a withdrawal candidate for that one slot; authorize locally, or use the narrowly eligible verifier |
| Negative member with no prior positive version | May store a CURRENT negative assertion when unambiguous; retire nothing and never call verifier solely to invent a target |
| FUNCTIONAL, same slot, different positive value | Derive a replacement candidate, not an automatic replacement; establish actual current replacement rather than coexistence/hypothesis/discussion |
| Facet, same subject and same full field, changed value | Same replacement contract, scoped only to this facet; no separate PATCH command |
| Explicit denial of the existing functional value | A withdrawal/negative version can replace that exact positive value if supported; a negative assertion of another value does not remove the current value |
| Positive assertion after a negative version of the same member | A potential destructive reversal; require evidence of the new truth, not automatic latest-write wins |
| Unknown identity/cardinality, multiple eligible targets, unresolved member | UNCERTAIN with no retirement; semantic verifier must not choose a target |
| Historical or incompatible temporal scope | Separate scoped/historical record; no retirement of the unqualified CURRENT slot |
| HYPOTHETICAL / UNKNOWN mode or polarity | UNCERTAIN, not an actual-world change; no semantic verifier to turn a hypothesis into fact |
| Distinct explicit condition scope | Separate conditional slot; not a replacement of an unconditional assertion |

CURRENT means the assertion is currently valid **under its declared scope and
mode**. A CURRENT plan is a current plan, not an executed action. Conditional
states are not usable as unconditional premises; existing premise validation
continues to enforce that distinction.

### Authority And Commit

The resolver owns one consistent relation between:
the canonical slot/member, the actual old CURRENT version, the new assertion,
source evidence, and any proposed retirement. Chronological order and confidence
alone never authorize replacement. The deterministic evidence check accepts
only a sound local semantic case; it does not grow into a phrase dictionary.

For implementation, the existing evidence-authorizer's supported/contradicted
checks are ported into a private resolver function over the assertion and actual
old version, not reinterpreted from an intent hint. No newly added phrase rule is
authorized by this specification. An opposite polarity or different value alone
is insufficient evidence authorization: if the existing grounded check cannot
establish support or contradiction, its result is UNKNOWN and Section 7 applies.
The verifier's locally derived operation no longer depends on a model-supplied
ASSERT/REMOVE distinction, closing that duplicate contract rather than relaxing
the identity/evidence boundary.

When a destructive candidate is supported, return the existing RevisionResult:
new version, changed old version marked STALE, appropriate revision edge, and
the exact old state_id in invalidated_state_ids. The observation transaction
rechecks target version/status and atomically commits both versions and edges.
If the target changed since verification, do not apply the old decision; retain
an unresolved outcome for a fresh resolution, without silently retrying a model.

The replacement version can never be among its own invalidation seeds. A member
operation never generates sibling-member seeds; a facet operation never generates
unmentioned sibling-facet seeds. Existing downstream propagation may still
invalidate other states through independently verified STRICT dependencies.

If multiple contradictory candidates for the same slot occur in one observation,
an unambiguous source ordering may be resolved locally with evidence; otherwise
the whole conflicting slot update fails closed. Separate members/facets can still
be committed, with the conflict explicitly logged rather than last-item wins.

### Required Examples, As Contracts Not Fixtures Run Here

- A person's current city changing from one explicit residence to another:
  same functional slot, supported old STALE/new CURRENT; a holiday visit does not
  satisfy the replacement evidence contract.
- Two positive beverage preferences: two member slots. Negating one retires only
  that positive member and records its negative assertion.
- Two employer memberships: different organization member keys. Ending one does
  not retire the other; distinct employers are never competing scalar values.
- A named event time update: same subject, same `meeting.time` slot; location
  and status versions remain unchanged unless independently invalidated by a
  verified dependency. No new unrelated event identity is generated.

## 7. Narrow Semantic Verifier

All five eligibility conditions are mandatory:

1. The resolver has identified exactly one existing CURRENT version in the same
   semantic slot/member. It supplies that version; the verifier does not select it.
2. Cardinality and member/value codecs are locally known and permit the candidate
   transition; a set addition cannot become a replacement of another member.
3. Temporal, condition and assertion-mode scopes are compatible and grounded.
4. Deterministic source-evidence authorization is UNKNOWN, not CONTRADICTED.
5. The actual candidate would retire that old version if supported.

ASSERT, independent ADD, same-value merge, absent targets, ambiguous identity,
unknown cardinality and explicit hypotheticals never call it. UNKNOWN identity
is not the same as UNKNOWN semantic evidence.

The resolver supplies only normalized old/new assertion semantics, cardinality,
scope, the locally derived question REPLACE or REMOVE, and exact evidence. The
question is ephemeral call input, not another persisted intent or provider hint.
No case IDs, query, benchmark labels, expected lifecycle, graph-write operations,
or answer/evaluator result is permitted.

Response remains SUPPORTED / CONTRADICTED / UNKNOWN plus operation confirmation,
target-match confirmation, short rationale and an exact evidence quote/span.
The local boundary verifies grounding and then rechecks all identity/cardinality/
scope/uniqueness/provenance constraints. Only SUPPORTED can permit retirement;
the resolver and transaction retain final authority. No output creates IDs,
changes lifecycle or writes the graph.

Maximum one real request per exact old-version/new-assertion/evidence/config key;
cache reuse is invalid if the target version or evidence changes. Provider failure
returns no authorization and a visible engineering error. Calls, abstentions,
tokens, latency and calls per observation are always reported. The previous
7/7 positive unseen decisions do not establish unseen negative safety.

## 8. Stage Count And Data Flow

The count is of separately represented semantic decision boundaries, not Python
functions. The optional semantic call counts once in both old and new paths.
Provenance validation is included in extraction; graph propagation/retrieval are
outside this write-path count and are unchanged.

| Old full path: 7 | New shrunk path: 4 |
| --- | --- |
| 1. Extract FrameCandidate | 1. Extract and ground one StateCandidate assertion |
| 2. Materialize typed StateFrame/policy identity | 2. One local slot/member delta resolver, including deterministic authorization |
| 3. Interpret independent ProposedChangeIntent | 3. Optional narrow semantic verifier for eligible UNKNOWN only |
| 4. Resolve/check provider target_hint | 4. Local final recheck and atomic StateNode version commit |
| 5. Separate local ChangeAuthorizationJudge | Removed as a separate policy layer; included in 2 |
| 6. Optional SemanticChangeVerifier | Preserved as 3 |
| 7. Resolved intent / revision commit | Simplified to existing RevisionResult in 4 |

OLD_STAGE_COUNT = 7. NEW_STAGE_COUNT = 4, or three traversed stages when the
optional verifier is unnecessary. There is no new component allowed between
stages 2 and 4 that independently guesses an operation, old value or lifecycle.

```text
Raw source
  -> canonical grounded StateCandidate assertion                 [REFINED]
  -> one cardinality/member-aware local revision resolver         [REFINED]
       -> eligible UNKNOWN-only semantic evidence check            [NARROW]
  -> atomic StateNode version update + direct invalidation seeds  [CORE]
  -> existing dependency discovery / semantic verification        [CORE]
  -> existing verified validity propagation / cascade             [CORE]
  -> premise-aware retrieval / stale rejection / planning          [CORE]
  -> shared answer generation                                     [UNCHANGED]
```

## 9. Endpoint And Migration Compatibility

New dependency endpoints remain `StateRelation.source_state_id` and
`target_state_id`, referring to StateNode **versions**, never a slot or whole
multi-facet object. Replacement leaves the old endpoint addressable as STALE;
there is no automatic redirection of its edges to the new CURRENT version.
New supporting dependencies require existing verification before persistence.
Revision edges and ordinary factual relations never become invalidation edges.

The existing storage protocol and RevisionResult shape remain. Evidence rendering
to verifier/retrieval must expose polarity and assertion_mode faithfully; a
consumer that renders a negative member as a positive fact fails compatibility.
This is a serialization/validity requirement, not a ranking/top-k change.

Implementation must adapt slot properties, revision and serialization explicitly;
simply adding fields while legacy conflict logic ignores member_key is prohibited.
The dependency verifier, STRICT/WEAK definitions, propagation algorithm, planner,
answer generator and ranking function are not redesigned by this specification.

Snapshot loading dispatches by schema version before inference. Legacy IDs and
sealed results remain readable; migration preserves existing relation references
through an explicit checked ID map or leaves them in read-only replay. Unknown
legacy cardinality is not guessed from whatever new value arrives. A group cannot
mix old and new write semantics; replay into an isolated new namespace is allowed
only as a separately declared migration, not an error fallback.

Production switch is a later decision. Initial implementation is isolated and
must test member/facet isolation, exact targets, first writes, negative assertions,
scope/identity ambiguity, duplicate evidence, idempotence, version edges and
negative consumer rendering before any live evaluation. No test is run here.

## 10. Conditioned Mechanism Protocol: Frozen Question

Protocol ID: CME-1.0. Purpose:

**Given correct upstream construction by the actual production path, does
StateGraph revise, discover/verify dependencies, propagate invalidity, reject
stale premises and adapt actions correctly?**

This is a diagnostic estimand, not a new acceptance gate, a replacement for E2E,
or a way to bypass failed upstream behavior. Runtime always uses its own states
and edges, never gold nodes, manual repairs, gold dependencies, or gold-selected
targets. Historical v1-v4 are development-only and excluded from new sampling.

## 11. Minimal Scope And Selection

Fixed maximum first cohort: **14 evaluation cases**.

| Dataset | Cases | Unit and intended information |
| --- | ---: | --- |
| StateChangeBench | 8 | Full source history + update + query; aim for observed coverage of all four reference-depth strata, without gold-based preselection |
| STALE | 3 | Three independent complete source cases; retain their native stale/action probes, aggregate at parent-case level |
| MAB Conflict/State | 3 | Three previously unused independent memory groups, one question each; conflict resolution, no presumption of causal cascade |
| LongMemEval | 0 | Not a core cascade-conditioned dataset in this protocol |

Selection is performed only after method/config freezing, using source metadata
and IDs, not reference answers/dependencies. Remove the union of all previously
used development/test/canary/debug IDs and source hashes, then order remaining
units by SHA256(`CME-1.0|dataset_id|source_hash|case_id`) and take the fixed counts.
For MAB, any previously consumed memory group is excluded, not merely previously
asked questions; choose its first hash-ordered unconsumed question.

Public split/task metadata can identify conflict/state subsets; answer labels,
gold depths and gold dependency graphs cannot guide sample selection. Do not
filter by which registry predicates the method supports. Full raw context must
be preserved; no oracle session reduction or favorable utterance extraction.

No sources/cases are selected in this document. If the repository cannot supply
the fixed independent units with auditable exposure history, preregistered outcome
is DATA_SCOPE_BLOCKED; do not silently substitute seen cases or generated fixtures.
If some reference-depth strata later have no eligible cases, report them as
INSUFFICIENT_EVIDENCE; do not top up or replace cases after seeing predictions.

## 12. Frozen Execution And Blinding

Before any request, seal protocol, code hash manifest, registry/codecs, prompts,
source-only input IDs/hashes, full context scope, answer config and evaluator
version. With no root Git, content hashes are the revision authority. A missing
artifact or unresolved version pin blocks execution; it is not a discretionary
parameter to choose after results.

For this small diagnostic, fix OpenAI `gpt-5-mini`, reasoning `low` for semantic
model calls; extraction output cap 4096, change/dependency verification 2048,
answer output cap 512, answer context budget 8192 tokens. Shared answer prompt
and evaluators are the repository versions whose hashes are sealed before
selection. No baseline comparison is part of this first cohort. Any later
baseline admission requires a separate same-input/same-answer-protocol seal.

Per case cap: 256 memory-stage requests total, including recovery, dependency and
change verification; at most 16 semantic-change verifier requests within that
cap. At most two singleton recovery requests per semantic observation; no batch
recovery. Answer calls are one per native query, separately counted. No semantic
retry after a well-formed answer, no model switch, no runtime prompt edits.
Transport retries are disabled for this preregistered diagnostic; a failure is
recorded rather than reclassified as method evidence. Exhausted budgets cause an
explicit incomplete result, never source truncation or a repaired answer.

Run the whole fixed cohort, record source/candidate/pre-revision/post-revision/
candidate-edge/verified-edge/propagation/retrieval/action/answer traces, and seal
all inference outputs. **Only then** may evaluators load reference answers or
dependency/depth annotations. No inference rerun follows eligibility selection.
No method patches are allowed within the cohort. Consumed cases never return
to unseen acceptance, including infrastructure-incomplete ones.

## 13. Eligibility: Avoid Conditioning Away Revision Failure

Two predeclared sets separate extraction from memory mutation. Otherwise requiring
post-update lifecycle correctness would trivially exclude the revision failures
we intend to measure.

### U: Upstream Assertion Correctness

An independent evaluator sees raw source, chronological candidate/evidence trace
and neutral semantic atom definitions, but not runtime dependencies, cascade,
retrieval results, final predictions or reference answers. It checks:

- Required subjects, values, polarity, identity qualifiers, time/condition scope
  and assertion mode are faithful to the source.
- All source-explicit entities/relations needed to interpret the update exist;
  no manually supplied state can satisfy a missing item.
- Claims used as initial valid facts are current assertions, not historical,
  hypothetical, negated-positive, or merely mentioned facts.
- All relevant history and update assertions are actually produced by production
  extraction, with valid source evidence and no contradiction that makes the
  target ambiguous. Parser success alone is insufficient.

Required content is annotated from the entire case source before inspecting
system output; it cannot be restricted afterward to whichever states succeeded.
Where the source permits multiple equivalent representations, the evaluator
records source-equivalence, not byte equality with a preferred frame ontology.
Truth-conditional differences are not normalized away.

Two independent source auditors annotate correctness; disagreements go to a third
blinded adjudicator. Unresolved judgments are UNKNOWN, not quietly eligible.
Eligibility rows and rationales are sealed before downstream scores are exposed.
This is evaluator labeling, never a production input or an injected graph.

### H: Correct Actual Initial State Construction

H is the subset of U whose actual pre-change repository snapshot contains every
required initial state with correct subject/value/scope/member identity and
initial lifecycle. Positive valid premises must be CURRENT; historical/negative/
hypothetical facts must not masquerade as positive CURRENT premises. Duplicated
ambiguous targets or already-inadvertently-stale required states make H false.

The new observation's candidate must also satisfy U, but **its resulting revision,
new lifecycle, discovered edges and cascade success are not H criteria**. Thus a
case can be H-eligible and fail revision; that failure stays in the conditioned
mechanism denominator. No eligibility rule depends on an expected dependency
being discovered, an action being correct or a final answer matching gold.

Report U coverage, H coverage, all excluded/unknown reasons and the source-only
requirement inventory. Main conditioned mechanism metrics use H. Revision given
U is also reported to expose failures in initial construction; never silently
rename H as an unconditional extraction-success rate.

If implementation interleaves historical dependency handling with initial writes,
the actual pre-change snapshot is still used. H then represents correctness of
the whole initial construction path, not a causal claim about extraction alone;
U remains separately visible so initial dependency mistakes cannot disappear.

## 14. Reference Graph And Depth Definitions

Gold/reference dependency annotations are evaluator-only and are opened after
inference seal. They are never injected as runtime edges or fed to extraction.
If a dataset lacks dependency gold, two independent auditors construct and seal
a source-grounded reference annotation while blinded to runtime edges/success;
it is a derived diagnostic annotation, not claimed to be an official benchmark
label. Non-adjudicable edges are marked UNKNOWN and their coverage is reported.

Reference edges must answer whether losing the prerequisite invalidates the
dependent assertion/action, not whether two entities have a factual relation.
Missing formal graph annotations do not authorize calling MAB factual hops
dependency hops. Existing typed/strength semantics remain the runtime policy.

`REFERENCE_CASCADE_DEPTH` is the maximum required invalidation path length from
a directly revised root to a required downstream state in the adjudicated graph:

| Depth | Frozen meaning |
| --- | --- |
| 0 | No required dependency propagation; direct revision may still be required |
| 1 | One prerequisite-to-dependent edge; a direct dependent effect |
| 2 | Two successive dependency edges / dependent layers |
| 3+ | At least three such layers |

Direct root revision is depth zero, not the first propagation edge. For multiple
roots/targets, also report each target's shortest required root-to-target length;
case stratification uses their maximum. Condense reference strongly connected
components before measuring layers and flag cyclic cases separately; do not
inflate depth by walking cycles. Pure factual traversal receives no cascade depth.
Cases without a defensible dependency annotation have depth UNKNOWN, not zero.

`RUNTIME_DISCOVERED_DEPTH` is separately computed on persisted verified STRICT
runtime dependency edges, using the same SCC convention and actual seeds.
No seed gives NO_SEED, not zero. Missing discovery must not move a reference
Depth3 case into the Depth0 results. All primary stratification uses reference
depth; report the reference/runtime cross-table, not one interchangeable depth.

## 15. Metric Definitions

An evaluator aligns actual state versions to source semantic units one-to-one,
with polarity/scope/version distinctions preserved. Ambiguous mappings cannot
receive a true positive. The method never receives the alignment. Duplicate
candidate pairs are deduplicated for recall; unsupported persisted parallel edges
remain visible in precision and duplication diagnostics.

Let G be adjudicated, required invalidation-supporting directed reference edges;
C the proposed runtime directed pairs before verification; V the persisted verified
STRICT dependency edges. Weak edges and ordinary factual relations are not V.
Let P be expected downstream invalidations, excluding directly revised roots;
I the observed propagation invalidations, also excluding those roots; K the
reference should-keep CURRENT states. Discovery/verification outcomes never
change U or H membership.

| Metric | Definition and scope |
| --- | --- |
| DIRECT_REVISION_ACCURACY | Correct exact old-root retirement/new version and sibling isolation / required direct revision opportunities; report all cases and U/H separately |
| DEPENDENCY_CANDIDATE_RECALL | Matched directed G pairs present in C / all G; absent runtime endpoint counts as a miss, not a denominator exclusion |
| VERIFIED_DEPENDENCY_PRECISION | Correct directed invalidation-supporting V edges / all V; wrong direction, unsupported dependence, factual-as-dependency and unmapped endpoints are false positives |
| VERIFIED_DEPENDENCY_RECALL | G represented by correct V / all G; no conditioning on candidate discovery |
| PROPAGATION_PRECISION | Correct I intersect P / all I; direct root invalidations excluded |
| PROPAGATION_RECALL | Correct I intersect P / all P; a missing seed or edge is a miss, not an excluded case |
| SHOULD_KEEP_RETENTION | K still CURRENT and usable under unchanged scope after the update / all K; all observed retirement causes count against it |
| STALE_PREMISE_REJECTION | Required stale-premise probes with neither stale-positive context leakage nor reasoning/action based on the retired premise / all such probes; require a completed assessed response, not empty context as automatic success |
| ACTION_ADAPTATION_ACCURACY | Queries with correct keep/revise/invalidate/replan outcome and no contradicted action / all adjudicable action queries; valid revised actions are scored semantically against source and reference protocol |
| FINAL_ANSWER_ACCURACY | Dataset-native evaluator correctness among completed native query predictions; case-weighted primary, query-micro secondary |
| E2E_ACCURACY | Mean case score over all selected cases in each dataset, where case score is correct completed native query predictions / all that case's preregistered native queries; missing predictions contribute zero |
| CONDITIONED_ANSWER_ACCURACY | Mean of those same case scores over H cases in that dataset; same evaluator and answer budget, not a new answer generated after selection |

For STALE, preserve native stale-specific and action dimensions, and report the
stricter internal stale-context metric separately rather than relabeling it an
official score. For MAB use the official conflict task substring-EM metric;
factual multi-hop correctness is not evidence of cascading invalidation.
StateChangeBench final answer uses its frozen repository metric implementation;
mechanism annotations and answer metrics remain separate.

Zero numerator/denominator is N/A, never 100%. If no true downstream change is
required, propagation recall is N/A but should-keep retention still applies.
When verified edges contain no correct dependencies, precision cannot be rescued
by counting factual edges as support. Edge and invalidation micro counts are
reported alongside per-case rates; queries sharing history are not independent
replicates. No aggregate accuracy combines heterogeneous official metrics.

For infrastructural/budget incompleteness, show execution coverage and completed-
only diagnostic scores, plus conservative all-selected E2E accuracy with missing
outputs incorrect. Do not attribute these missing outputs to method failures or
silently remove them. Main conditional denominators retain all adjudicable H
cases; unknown upstream eligibility is separately enumerated.

## 16. Analysis Tables And Sufficiency Rules

Required tables, all fixed before execution:

1. Dataset totals: selected/completed/U/H/UNKNOWN counts and exclusion reasons.
2. Each metric on all cases and H, with numerator/denominator; direct revision
   also on U. No single column mixes E2E and conditioned scores.
3. The same tables by REFERENCE_CASCADE_DEPTH 0/1/2/3+/UNKNOWN, plus stale/action
   requirements and source length. Show eligibility rates within each stratum.
4. Reference-depth versus runtime-depth cross-table, absent seeds, lost endpoints,
   candidate misses, rejected genuine edges and false propagation separately.
5. Costs by extraction/recovery/local revision/semantic verifier/dependency/answer:
   calls, tokens, latency, structured-output failures and bounded abstentions.
6. Trace-level failure attribution and shared-context evidence for answer errors;
   correct upstream does not imply that retrieval or answer generation worked.

This 14-case design is a mechanism diagnostic, not a statistical superiority
test. A stratum with fewer than two independent H cases is marked
INSUFFICIENT_EVIDENCE for a depth trend; report its raw observations anyway.
No monotonic depth advantage is claimed from one case, and no broad generalization
claim is made even if all small-sample metrics equal one. A correctly constructed
case with a core failure is a diagnostic counterexample, not excluded data.

No new overall PASS threshold supersedes the old canary gates. The protocol's
administrative result is VALID_DIAGNOSTIC, INSUFFICIENT_COVERAGE, INCOMPLETE, or
PROTOCOL_INVALID. False-stale or factual/dependency contamination is an explicit
safety finding and prevents recommending production promotion, regardless of
conditioned answer accuracy. There is no automatic follow-up cohort or repair
loop in this preregistration.

## 17. Required Future Artifacts And Isolation

Future run directory (not created in this task):
`outputs/shrunk_stateframe_cme_<run_id>/`.

Before inference: PROTOCOL_HASH.json, SOURCE_HASHES.json, REGISTRY.json,
CONFIG.json, EXPOSURE_LEDGER.json, SOURCE_ONLY_INPUT_MANIFEST.json,
PRE_RUN_SEAL.json. The private evaluator reference inputs must never reside in
the inference process's loaded configuration or prompt payloads.

During inference: requests/responses and token/latency logs; observation and
candidate trace; initial snapshot; revision trace; candidate and verified edges;
propagation trace; RetrievalRecord and PredictionRecord files. At completion,
INFERENCE_SEAL.json binds every model/runtime output before evaluation begins.

After inference: blinded source requirement annotation, U/H eligibility with
auditor agreement, evaluator-only reference graph/depth/alignment files, official
answer scores, all/H/depth metric tables, cost and failure reports, FINAL_SEAL.json.
Neither eligibility nor post-hoc annotations may mutate any runtime artifact.

## 18. Implementation Readiness Checklist

| Requirement | Design status |
| --- | --- |
| Unique persistent representation | YES: StateNode + exactly four minimal fields; no full-frame production store |
| Fewer semantic boundaries | YES: 7 -> 4; remove frame materialization, independent intent and old-target hint; merge local judge |
| Member/cardinality isolation retained | YES: member-qualified slots and exact-version retirement; unknown policy fails closed |
| Verifier narrowed | YES: five mandatory guards, destructive candidate only, one unique local target, local final authority |
| Revision contract explicit | YES: decision table, no latest-wins default, first-write separation, atomic version commit and ambiguity policy |
| Dependency/propagation compatibility | YES at interface design level: StateNode version endpoints and RevisionResult retained; negative rendering must pass implementation tests |
| Conditioned protocol frozen | YES: U/H eligibility, blinding, 14-case selection rule, reference/runtime depth split, metric formulas and reporting rules |

SHRINK_IMPLEMENTATION_READY = YES.

The remaining runtime content hashes and case IDs are deterministic pre-run
bindings, not undecided architectural alternatives or thresholds to tune later.
No assertion is made that the implementation or future eligibility coverage
already passes. Those are separate evidence gates.

**One next action:** implement the isolated StateNode minimal-extension and
single-resolver path with offline contract tests against this specification.
Do not switch production or run the conditioned cohort as part of that action.
