# CME v3 State Lifecycle Forensics

Date: 2026-09-21

Status: forensic analysis only. No provider call, code change, benchmark scoring,
or sealed-artifact mutation was performed.

## Executive result

The 50 persisted states are not `CURRENT` because the frozen shrunk resolver
rejects the initial writes before any destructive transition is considered:

| Direct reason | Count | Dataset split | Meaning |
|---|---:|---|---|
| `B_RESOLVER_DEFAULT_UNCERTAIN` | 10 | SCB 10 | Cardinality is known, identity and scope are valid, but the resolver's bounded literal assertion matcher returns `UNKNOWN` for natural-language evidence such as `Eva was free on Wednesday.` |
| `F_CARDINALITY_UNKNOWN` | 40 | SCB 13, STALE 27 | The extracted field is absent from the frozen CME registry, so the resolver's certainty gate fails closed. |
| **Total** | **50** | **SCB 23, STALE 27** | Every persisted state is `UNCERTAIN`. |

This is not `REPLAY_DEFAULT_UNCERTAIN`, verifier unavailability, provenance
rejection, or destructive authorization failure. All 50 writes are classified
as `INITIAL_ASSERTION`; they have no `CURRENT` target and therefore do not reach
the narrow semantic verifier.

The replay is not semantically equivalent to the original production sequence
for downstream lifecycle evaluation. It synthesizes only the first observation
for each replayable group and does not restore later observations or their
between-observation repository snapshots. That limitation prevents a revision
or cascade conclusion from this replay. It does not explain the initial-write
`UNCERTAIN` result: the initial assertion contract requires no old state, and
the resolver still returns `UNCERTAIN` with a valid identity, effective scope,
and grounded evidence.

## Sealed inputs and counts

- Replay output: `outputs/canonical_evidence_bridge_recovery_20260921_r1/`
- Sealed production source: `outputs/conditioned_mechanism_eval_14case_v3/execution_recovery_20260921_r1/`
- Raw valid provider responses: 239
- Non-replayable provider outputs: 2
- Parsed candidates: 189
- Canonical evidence accepted: 186
- Persisted states: 50
- Final lifecycle counts: `CURRENT=0`, `STALE=0`, `UNCERTAIN=50`
- State commit rejections before persistence: 139, all due to the pre-existing
  strict `subject and value must be anchored in source evidence` check
- API calls in this forensic pass: 0

The 139 rejected candidates are separate from the 50 persisted states analyzed
below. Canonical evidence rejection was zero for the replayed evidence records.

## Lifecycle assignment path

For each of the 50 persisted records, the effective path was:

```text
sealed StateCandidate
  -> bridge_candidate_evidence
  -> StateGraph candidate/state projection
  -> ShrunkRevisionResolver
  -> ShrunkStateRepository._ground
  -> ShrunkStateRepository._node
  -> current-target lookup
  -> initial assertion certainty/local assertion gate
  -> repository commit as UNCERTAIN
```

The resolver behavior is visible in
[shrunk.py](/home/cody/agent/stategraph/state/shrunk.py:141): canonical
evidence is required and subject/value grounding is checked. The local matcher
at [shrunk.py](/home/cody/agent/state/shrunk.py:161) only accepts a bounded
literal clause matching the configured surface form. The certainty and
lifecycle branch at [shrunk.py](/home/cody/agent/state/shrunk.py:259) requires
known cardinality and `SUPPORTED` local evidence even when there is no existing
state.

For every persisted state:

- `change_classification = INITIAL_ASSERTION`
- `existing CURRENT states before the write = 0`
- `authorization = NOT_REQUIRED`
- `semantic_verifier = NOT_CALLED`
- `final lifecycle = UNCERTAIN`
- subject identity was known: 50/50
- scope was effective at the evidence timestamp: 50/50
- typed polarity was `POSITIVE`: 50/50
- typed assertion mode was `ASSERTED`: 50/50
- candidate cardinality was explicitly absent: 50/50; local registry derived
  `FUNCTIONAL` for 10 and `UNKNOWN` for 40
- candidate member key was explicitly absent: 50/50
- local assertion verdict was `UNKNOWN` for 49 and `SUPPORTED` for 1

The one `SUPPORTED` local assertion was still uncertain because its field had
unknown cardinality. The one non-empty condition scope was also retained; it
was not the primary reason because that candidate already had unknown
cardinality.

## Uncertain reason taxonomy

### Direct assignment

| Taxonomy | Count | Finding |
|---|---:|---|
| A `REPLAY_DEFAULT_UNCERTAIN` | 0 | The replay runner did not assign lifecycle status itself. |
| B `RESOLVER_DEFAULT_UNCERTAIN` | 10 | The resolver explicitly called `with_status(UNCERTAIN)` after local evidence returned `UNKNOWN`. |
| C `MISSING_CHANGE_CONTEXT` | 0 direct | No destructive transition was attempted. |
| D `MISSING_EXISTING_STATE_CONTEXT` | 0 direct | Initial assertion does not require an old target. Full later-state context is nevertheless missing from this replay. |
| E `AUTHORIZATION_UNKNOWN` | 0 | No candidate reached a destructive authorization check. |
| F `CARDINALITY_UNKNOWN` | 40 | The frozen registry did not recognize the extracted field. |
| G `SCOPE_UNKNOWN` | 0 primary | All persisted scopes were effective; one conditional candidate was already in F. |
| H `PROVENANCE_INSUFFICIENT` | 0 | Canonical bridge accepted all evidence used by persisted states. |
| I `OFFLINE_REPLAY_ARTIFACT` | global limitation | Replay is incomplete for sequence/revision evaluation, but it is not the direct cause of the 50 initial statuses. |
| J `OTHER` | 0 | No additional direct cause was needed. |

### Resolver evidence

The 10 known-cardinality states are all `availability` assertions. Their
evidence spans use natural source language (`was free on ...` or `was free`),
while the frozen local matcher constructs a literal form based on the policy
surface (`subject availability value`). This is a resolver contract mismatch,
not a missing canonical coordinate.

The 40 unknown-cardinality states include fields such as `scheduled_for`,
`arrangement_date`, `often_have`, `filled_with`, `intended_contents`, and
`periodic_check`. The frozen registry at
[cme_shrunk_runtime.py](/home/cody/agent/stategraph/evaluation/cme_shrunk_runtime.py:34)
contains neither these fields nor a generic initial-assertion policy, so the
resolver intentionally abstains.

## Replay semantic equivalence

`REPLAY_SEMANTIC_EQUIVALENCE = NO` for the full production lifecycle.

| Context item | Restored? | Evidence |
|---|---|---|
| Original observation sequence order | No | [replay script](/home/cody/agent/scripts/replay_canonical_evidence_bridge.py:141) uses `case['observations'][0]` and creates `observation_index=0`. |
| Repository state between observations | No | Later observations are not replayed, so no per-observation snapshot exists. |
| Existing `CURRENT` states before every production write | No for the full sequence | Only the synthesized first observation is evaluated. For the 50 writes, the actual target set is empty. |
| Revision context and chronological transitions | No | No later replacement/removal/patch is replayed. |
| Cardinality registry | Yes | The replay uses the same `cme_shrunk_registry()` binding. |
| Subject and slot normalization | Yes | The same shrunk runtime constructs the `StateNode` and slot/version IDs. |
| Canonical provenance | Yes | Persisted evidence has `coordinate_space=OBSERVATION_ABSOLUTE`, observation index 0, and valid evidence identity. |
| Semantic verifier decisions | Not applicable | No destructive candidate has a unique `CURRENT` target; verifier calls are zero. |

The replay is therefore suitable for diagnosing the initial state-construction
write gate, but not for revision, stale transition, dependency, propagation, or
answer evaluation.

## Initial assertion contract

For the stated preconditions:

```text
grounded evidence
AND positive assertion
AND unique identity
AND no conflicting CURRENT state
AND effective scope
-> CURRENT
```

Therefore:

```text
EXPECTED_INITIAL_ASSERTION_LIFECYCLE = CURRENT
ACTUAL_INITIAL_ASSERTION_LIFECYCLE   = 0 CURRENT, 50 UNCERTAIN
```

The existing offline S1 contract confirms this expectation: a first registered
functional assertion is expected to be `CURRENT`, while unknown cardinality and
unsupported evidence remain fail-closed. The CME replay inputs satisfy the
identity/scope/provenance portions, but the natural-language assertion gate and
registry coverage prevent the expected initial lifecycle.

## Bridge and semantic-field audit

The bridge only copies canonical provenance. The bridge tests and replay show
that it did not alter candidate semantics:

- `canonical_subject_id`: present 50/50
- `canonical_field_id`: present 50/50
- value and entity: preserved and grounded
- time scope: present/effective 50/50
- condition scope: preserved; one non-empty conditional scope
- typed polarity: `POSITIVE` 50/50
- typed assertion mode: `ASSERTED` 50/50
- candidate cardinality: `None` 50/50 before local registry resolution
- candidate member key: `None` 50/50 before local registry resolution
- canonical evidence coordinate space: `OBSERVATION_ABSOLUTE` 50/50
- evidence observation index: 0/50, because this is a first-observation-only replay

There is an upstream typed-semantics detail worth preserving in the limitation
record: opaque candidate metadata contains `value_polarity=positive` for 30
records and `value_polarity=historical` for 20 records, while the typed
`StateCandidate.polarity` is `POSITIVE` for all 50. This was already present in
the sealed candidate artifacts; the provenance bridge did not delete it. It is
not the direct reason for the 50 statuses, but it means a future lifecycle run
must define whether that metadata is promoted before scoring historical facts.

## Semantic verifier audit

`VERIFIER_REQUIRED_COUNT = 0` for the 50 persisted writes.

Every write had no `CURRENT` target. The resolver therefore never entered its
destructive transition branch and never invoked the narrow verifier. The
verifier's absence from offline replay cannot explain the initial `UNCERTAIN`
states, and no verifier authorization or false authorization occurred.

## Representative decision traces

The following traces are representative, not patches or selected by outcome.

### SCB trace 1: SCB_015 / candidate 0

- Candidate: `Eva`, `availability`, `free on Wednesday`, confidence `0.90`
- Polarity/cardinality/member: `POSITIVE` / `FUNCTIONAL` / none
- Scope: starts `2025-01-01T00:00:00Z`, effective; condition empty
- Evidence: `evidence:a333be227e28758403cbb0b16c60afcbaf596aee72c5a1c2f0bf47024b0aaecd`
- Evidence span: `Eva was free on Wednesday.`
- Slot/version: `slot:7882a4d4a16f1034ccf1f9ac2295bfab5ffeb93514c63e616253bce8a630f174` /
  `version:eb970ba4be24928ebc0b9696bfe3b36d732b30f6348e04dfb03ae8ac807bbc38`
- Resolver: identity known, cardinality known, scope effective, no target;
  `_local_assertion=UNKNOWN` because the natural clause does not match the
  literal policy grammar.
- Authorization/verifier: not required / not called.
- Actual lifecycle: `UNCERTAIN`; direct reason `B_RESOLVER_DEFAULT_UNCERTAIN`.

### SCB trace 2: SCB_016 / candidate 1

- Candidate: `the class`, `scheduled_day`, `arranged for Thursday`, confidence `0.50`
- Polarity/cardinality/member: `POSITIVE` / `UNKNOWN` / none
- Scope: effective at `2025-01-01T00:00:00Z`; condition empty
- Evidence span: `The class was arranged for Thursday`
- Slot/version: `slot:0fcebdacd58430541590d8ec1cf9a049a890641c60306b79399bf0145241cd09` /
  `version:cc7bf74b8cdb0e3fd687988571d34a70d80e5f2a4dbc81ffd45fe91f1f7529bf`
- Resolver: identity known and scope effective, but registry policy is
  `UNKNOWN`; certainty gate fails before any change authorization.
- Authorization/verifier: not required / not called.
- Actual lifecycle: `UNCERTAIN`; direct reason `F_CARDINALITY_UNKNOWN`.

### SCB trace 3: SCB_037 / candidate 0

- Candidate: `Grace`, `availability`, `free on Sunday`, confidence `0.90`
- Polarity/cardinality/member: `POSITIVE` / `FUNCTIONAL` / none
- Scope: effective; condition empty
- Evidence span: `Grace was free on Sunday.`
- Slot/version: `slot:e3b0f2a7f9409e6a48c70787caecb158eba7b39a9d68b4efbc61ec00f999a3cf2` /
  `version:af5e67e0b2598dfa46078b9e33c01e183d56050f446bc2e6ecaa17bbfdfd6e92`
- Resolver: same known-cardinality/local-assertion gate as SCB_015.
- Authorization/verifier: not required / not called.
- Actual lifecycle: `UNCERTAIN`; direct reason `B_RESOLVER_DEFAULT_UNCERTAIN`.

### STALE trace 1: 14897e47... / candidate 5

- Candidate: `Salvation Army`, `is_a_thrift_store_(example_of_thrift_stores)`,
  `Salvation Army`, confidence `0.86`
- Polarity/cardinality/member: `POSITIVE` / `UNKNOWN` / none
- Scope: effective at `2022-12-04T02:30:00Z`; condition empty
- Evidence span: `Salvation Army`
- Slot/version: `slot:f8c57d8147161ea4b52a0cea931f210f9d2d2da12cdc913fecbf9540f9bdd12a` /
  `version:46f7073970f7a03ab67ef0e038ad079cd1b3040a955d795017d32bbfcbbbe3d5`
- Resolver: registry policy `UNKNOWN`; no target and no verifier.
- Actual lifecycle: `UNCERTAIN`; direct reason `F_CARDINALITY_UNKNOWN`.

### STALE trace 2: 14897e47... / candidate 6

- Candidate: `Online Retailers`, `often_have`, `a wide selection of vases`, confidence `0.88`
- Polarity/cardinality/member: `POSITIVE` / `UNKNOWN` / none
- Scope: effective; condition empty
- Evidence span: `**Online Retailers**: Websites like Amazon, Walmart, or Overstock often have a wide selection of vases`
- Slot/version: `slot:ea9f4a9d71a8281cd9fa370da67aa17e2c8635adfbe5c83a746ca74d0b3a5201` /
  `version:03c4243aae4c7ee94873f08ab19bc2b2108e862a9a6f002a74c4f3861a3df1f2`
- Resolver: registry policy `UNKNOWN`; no target and no verifier.
- Actual lifecycle: `UNCERTAIN`; direct reason `F_CARDINALITY_UNKNOWN`.

### STALE trace 3: 14897e47... / candidate 14

- Candidate: `your vase`, `filled_with`, `Decorative stones or pebbles`, confidence `0.77`
- Polarity/cardinality/member: `POSITIVE` / `UNKNOWN` / none
- Scope: effective; condition empty
- Evidence span: `**Decorative stones or pebbles**: Fill your vase with decorative stones or pebbles in a color that matches your home decor.`
- Slot/version: `slot:e31c784265dcb4ccbf2fbe854ad250e32c9d37f11a0b8ec62b9be50f2d8531c5` /
  `version:488be2711fdb03393cbabbe8119fccdd39e553e64910e3bcdc07c242f86062cd`
- Resolver: registry policy `UNKNOWN`; no target and no verifier.
- Actual lifecycle: `UNCERTAIN`; direct reason `F_CARDINALITY_UNKNOWN`.

## Full persisted-state ledger

The following is the complete 50-state ledger. `F` means
`F_CARDINALITY_UNKNOWN`; `B` means `B_RESOLVER_DEFAULT_UNCERTAIN`. All rows
have `INITIAL_ASSERTION`, `NONE`, `NOT_REQUIRED`, `NOT_CALLED`, and
`UNCERTAIN` in the corresponding transition, resolver, authorization,
verifier, and lifecycle columns. Scope is effective for every row.

| Dataset/case/index | Subject.attribute = value | Conf | Pol | Card | Member | Local | Reason | Slot ID | Version ID |
|---|---|---:|---|---|---|---|---|---|---|
| SCB_015/0 | Eva.availability = free on Wednesday | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:7882a4d4a16f1034ccf1f9ac2295bfab5ffeb93514c63e616253bce8a630f174` | `version:eb970ba4be24928ebc0b9696bfe3b36d732b30f6348e04dfb03ae8ac807bbc38` |
| SCB_015/1 | the interview.scheduled_for = Wednesday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:6fef75451f57fd71e95a6e7d02b7c66901649e3b96f686ea95aa3a0ab0e20507` | `version:c8e303be52a230ca6efdde4aa27a3c50bccbbf5c01001f5a1a9488e7aa9526fd` |
| SCB_016/0 | Farah.availability = free on Thursday | .50 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:4752e97db9ca42a124c99fbd9beba9f1354580a67dad5e3cce6ca1ed772c3795` | `version:106ca71240f70c4b20ba0f69be39adbeea94f4b208fecc239aa8582ca9f2cec6` |
| SCB_016/1 | the class.scheduled_day = arranged for Thursday | .50 | P | UNKNOWN | - | UNKNOWN | F | `slot:0fcebdacd58430541590d8ec1cf9a049a890641c60306b79399bf0145241cd09` | `version:cc7bf74b8cdb0e3fd687988571d34a70d80e5f2a4dbc81ffd45fe91f1f7529bf` |
| SCB_016/2 | that availability.made_it_feasible = made it feasible. | .80 | P | UNKNOWN | - | UNKNOWN | F | `slot:47a4f96c25100d7564475ffd209d68cf1c3891087ceb3a1b1ec90dbf592cb16f` | `version:49fe75263bb6205e503929192cc96a87cefb1aa60cc33adf9bc214a0ac33a86d` |
| SCB_025/0 | Eva.availability = free | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:19b0f26649cc9ae808e23d72391227b8ba1565190738d6d54039c170a6d23751` | `version:26b416a31717636273aeebfaddb3704a228892ea76a4dab1e3c34a27ea430dc4` |
| SCB_025/1 | interview.arranged_for = Wednesday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:ead07de5ac7a4db61edfe4636ac2e29e972d0e009ef3b8337788be211a5e4d72` | `version:8a5bb954301cd1ff6a12756bddbfd3e830710f2cca3b5deb94fb44bb55387607` |
| SCB_026/0 | Farah.availability = free on Thursday | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:c4f0a88650fb6f691295ce1c3f620f0a97f31ad4d2f73a4666351c7cdac58785` | `version:da78f4842df6637e571f67a65bbc37a68de73578fcaed856ca34f992d7328d8d` |
| SCB_026/2 | The class.scheduled_for = Thursday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:350c3231afc6d4f289a0d01e419a5fee0a55170123cbaf4dc821f5dfbfe64d31` | `version:9099234b7c84d81909068057610d16acb509ed9b79eb39bb736b99c3c54a9981` |
| SCB_030/0 | Jon.availability = free | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:6c96104b0759dfe46bf0835c16396c9b0e6ede62ebd6aa9311f017ecc00a1b14` | `version:7b8d3568d531f2316af4ae87fcc3c6a33272a36ef49c33edcce79611cbc548ef` |
| SCB_030/1 | the pickup.arrangement_date = Monday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:304cb5f35547cf7facef0e04787eda72f6779a06df54198b0ed93293caf01565` | `version:b3676edc14f79187ca0fc55d1b545df7c9abe19f29e1550cdd97eba0f2c404a8` |
| SCB_035/0 | Eva.availability = free on Wednesday | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:cd3ffc89ca2056e980bf1b0a73ffd42f8f6610d198b26db731fdf800599af811` | `version:5e0584a71bc1089a2ddd5dd2b5388126aebc93462320ec28ca1fae9ecad4bc2f` |
| SCB_035/1 | The interview.scheduled_day = Wednesday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:c8d9b2c797ae154acbd293db020363ba00358f3b5e946fb84b55342858084414` | `version:58e708362de90a79c7291c017cca5149ec2dd26c9fb8bd5692dad5e79dbd7c55` |
| SCB_037/0 | Grace.availability = free on Sunday | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:e3b0f2a7f9409e6a48c70787caecb158eba7b39a9d68b4efbc61ec00f999a3cf2` | `version:af5e67e0b2598dfa46078b9e33c01e183d56050f446bc2e6ecaa17bbfdfd6e92` |
| SCB_037/1 | The trip.scheduled_date = Sunday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:c1e56f630ffa06f59086984e681815e677e690f064645454c3a38ecbf4657e35` | `version:4081f8ab206000f621e0a73da2875dbbe8d1938fff28271a9ea9790109653332` |
| SCB_037/2 | that availability.made_it_feasible = feasible | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:05c26b41b3ee677b2094a1948009f02da48d2d30af16c92eaa2ce4358f76c4f2` | `version:c7ca0cc387ea0797ea046ca5e166e7c368aaadaefc4434ea69687143bc7e8f48` |
| SCB_040/0 | Jon.availability = free | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:32d3b3f12500dcd5f092f66ec112d55973baa6be9e2f51a00921d4b75024eaf1` | `version:c44b6803cad35b3fb4925c33e17199f6ba706a8827dc8c9a7bba81e20a52255b` |
| SCB_040/1 | the pickup.arrangement_date = Monday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:5196b35c3329cf9b1628793001c106db0c26080578bec26865f942a568670649` | `version:58d95e2210a931b2f5f350d6a204c1a105f871bd14a67118da0d418b85ce3669` |
| SCB_042/0 | Ben.availability = free on Tuesday | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:c2bf0314b1dbe9395eeadfa60769689bf0d96ebd2064bc1cc1ecf376d53418d0` | `version:6f386d860d68c32096c487ecae196c87f43e077c1bb590b67bf768851c249bdb` |
| SCB_042/1 | the doctor visit.scheduled_date = arranged for Tuesday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:d12c0c29dd76bd418af43009a997fcc91b3067d820ca1587bc3686a47dec4d1a` | `version:093850bc68f0f612b9055011d75f26026395a17fbf49a93bcfc4b4a841e7bf86` |
| SCB_048/0 | Hugo.availability = free | .90 | P | FUNCTIONAL | - | UNKNOWN | B | `slot:6a16d3c91d43b7ffbca6b6610c776de96780b6863dd20e234de35541c746d6d9` | `version:0365b33aaeb58fa17299519f6a4ff47b60f8cd6d38104e7ec8d44564e` |
| SCB_048/1 | the deployment.scheduled_for = Tuesday | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:daf1de9138a32109acfb6fb58cb7cb87e4c2a0a29f591d7e7cae4cc73686fab9` | `version:b57a821048617c82e00caf42a319d593e58d0acde93502599284c897a7946ccc` |
| SCB_048/2 | that availability.effect_on_deployment_feasibility = made it feasible | .85 | P | UNKNOWN | - | UNKNOWN | F | `slot:88e16ff01791fe60fd101b7273f67f3be58fe1541312cf9d4d45a5a020581d1e` | `version:b5d4671d7334856d6955a93ce1a6059bd0f1c97ecb7b84a9033658b869ff7aad` |
| STALE/14897.../5 | Salvation Army.is_a_thrift_store_(example_of_thrift_stores) = Salvation Army | .86 | P | UNKNOWN | - | UNKNOWN | F | `slot:f8c57d8147161ea4b52a0cea931f210f9d2d2da12cdc913fecbf9540f9bdd12a` | `version:46f7073970f7a03ab67ef0e038ad079cd1b3040a955d795017d32bbfcbbbe3d5` |
| STALE/14897.../6 | Online Retailers.often_have = a wide selection of vases | .88 | P | UNKNOWN | - | UNKNOWN | F | `slot:ea9f4a9d71a8281cd9fa370da67aa17e2c8635adfbe5c83a746ca74d0b3a5201` | `version:03c4243aae4c7ee94873f08ab19bc2b2108e862a9a6f002a74c4f3861a3df1f2` |
| STALE/14897.../14 | your vase.filled_with = Decorative stones or pebbles | .77 | P | UNKNOWN | - | UNKNOWN | F | `slot:e31c784265dcb4ccbf2fbe854ad250e32c9d37f11a0b8ec62b9be50f2d8531c5` | `version:488be2711fdb03393cbabbe8119fccdd39e553e64910e3bcdc07c242f86062cd` |
| STALE/14897.../17 | multipurpose centerpieces or decorative items.can_be_used_for = birthdays, holidays | .80 | P | UNKNOWN | - | UNKNOWN | F | `slot:83580268efde51b40ed586c67c74d6aab6123e5d7083472d7c1ef6eea37096f1` | `version:28b580e5042f9d80c2ce1d0232e4d0085a666a854a9ddcd5031b4a4aead540a2` |
| STALE/14897.../18 | candle centerpiece.versatility = can be used for birthdays, holidays, and even everyday decor | .50 | P | UNKNOWN | - | UNKNOWN | F | `slot:3f103ba7e20c1265346169e37b99afe7914326dd12a51c56d677d841f5fec50e` | `version:bb217c437311644d890cad68963e90bcd12886c13901ae602f9b9fac9815a96d` |
| STALE/14897.../20 | floral garland.usable_occasions = can be used for birthdays, weddings, and holidays | .50 | P | UNKNOWN | - | UNKNOWN | F | `slot:3bb1d6da11b89acb2b3977071a74ac0dcd7de97ddbb2fddc08dda91fafd7642b` | `version:abf498c7af9ad07c858635be9ea3ce518d7e58d1f5cb30564acc482d46e0c261` |
| STALE/14897.../22 | vase filler.components = decorative items like pebbles, marbles, or small ornaments | .50 | P | UNKNOWN | - | UNKNOWN | F | `slot:fe9a25fd3b55af56f0d9db1fb45d3c9f05b751c4b385ba28327b3ee80c939e4b` | `version:9a3454343107c27663890fe0106b33f799749d75e57369ea2cab4bfb296a0e2e` |
| STALE/14897.../27 | monogrammed or personalized item.usable_occasions = can be used for birthdays, holidays, and other celebrations | .50 | P | UNKNOWN | - | UNKNOWN | F | `slot:c69f65c4d9258d5c6b0bd2c18aefe03b7fcabb244b85729c57a5eb984c4528b3` | `version:5004320ed05ce27026a8b2ed0b9ec15e2212b6cc756b9e9013fcd8e3e3688b3b` |
| STALE/14897.../30 | items.that_are = neutral in color and design | .78 | P | UNKNOWN | - | UNKNOWN | F | `slot:c242b117b7c2de8e015748f269a961dcb7bd0d62ece34232962e0580732ffad5` | `version:70df9745ace8e157eb83fb1a43966d5f4e127f550b4e9e65b14e9c6929ead04a` |
| STALE/14897.../32 | A candle centerpiece.is = a great idea | .69 | P | UNKNOWN | - | UNKNOWN | F | `slot:14748faf876100292f4a9bd336d2542a1e6aad9a948415bb9dec3312f1f82ecc` | `version:cd3f4427780d7596f83e0b39f058cab27476eff3a4701f14ce28487dd3747454` |
| STALE/14897.../41 | 5 oz candles.priced_around = $20-$25 | .80 | P | UNKNOWN | - | UNKNOWN | F | `slot:fc0e290f6f84b62b7a2ad80216b3e110499d58ceaeba2945f0f6dea11ecb10fe` | `version:b2afd8be597a38da3ee88721166e88322b335b5f5ebbfbe6da3b7de6294a9551` |
| STALE/14897.../42 | 12 oz candles.price = around $20-$25. | .63 | P | UNKNOWN | - | UNKNOWN | F | `slot:931a2c3d94510dd23de5d5ea74324c5eed43e9ad64333f658df44e429bd190bf` | `version:698838013c810b345c688cae8357b4a2e7db926e00a04bd0b4fc712d71fb7008` |
| STALE/14897.../50 | sizes.can_fit = a standard candle | .80 | P | UNKNOWN | - | UNKNOWN | F | `slot:8beefe513829ac703587f7948f1a7b8ce821f37908da8838c8ad3119273dcdfb` | `version:436e699cda4e437566f5fde0aca327eebf44b1f6f900f6aec7cbc98586d59511` |
| STALE/14897.../60 | a mercury glass holder or decorative vase.can_add = a touch of elegance and sophistication to your space | .82 | P | UNKNOWN | - | UNKNOWN | F | `slot:afa5f3d6315d6f9db07a148848987030d673b01e66e2d1fbab34c9fb9a583ac9` | `version:52087696717407990de89aa8988900b2659b0f3ce38ae68fce6436f3ea163225` |
| STALE/a372.../26 | "5-minute rule".are_great_ways_to_maintain_a_tidy_home = great ways to maintain a tidy home | .78 | P | UNKNOWN | - | UNKNOWN | F | `slot:994780aba467824a6596b006e04871cb23a6437a6c0d233d5fdeaefe89dcae4c` | `version:d8dbc42baf7a21be56a896f52a629204d0922c70f6c1584dec6cdad09891ae06` |
| STALE/a372.../43 | This visual cue.will_help_you_remember_to = do a separate load for delicates every other Wednesday. | .93 | P | UNKNOWN | - | UNKNOWN | F | `slot:ceb85f2a0bca5d41c4aec7af2b92c6c79d3b1e7ae6fe869ab13f4761d66624b5` | `version:a82ad3d272f2ae8de8be97a0a690145678ba39eb9c863824c341c4340c091ced` |
| STALE/a372.../50 | reminders.will_definitely_help_you_stay_on_track = will definitely help you stay on track | .88 | P | UNKNOWN | - | UNKNOWN | F | `slot:a0e15776bb0071e6ae340627c615cbb2a562572ce10df3ee61ef8443b9028bc8` | `version:97531b4e38083fd7578447c45d39bf5bac70aed5cd1efe03c5be59cba0e0925d` |
| STALE/a372.../51 | the laundry basket system.is = a great way to visualize | .84 | P | UNKNOWN | - | SUPPORTED | F | `slot:679f1d923b4c90a3d577d4052eab2b4e9a915f5690639dee3a52e1abdd4203bf` | `version:70adca3388fd51aa255eb66c95e0a43ef4cf50e61b48f4cbc6382743e1a76077` |
| STALE/a372.../61 | your cupboards.purge_and_declutter = Remove everything from your cupboards and sort items into three categories: keep, donate/sell, and discard | .80 | P | UNKNOWN | - | UNKNOWN | F | `slot:62cd4fdac882833eed76ebbcfa28ea84542107c4782eaefa6ad32945bcc21681` | `version:bc9acdf2850ae1542b2b67e11e810973942ad3e4e2839ae1127bcc5ee200598e` |
| STALE/a372.../63 | your cupboards.categorize_and_group = group similar items together (e.g., baking supplies, cooking utensils, dinnerware) | .80 | P | UNKNOWN | - | UNKNOWN | F | `slot:283beda254f76ffecf180a7d2abf8d84a659af91e6743f8d1a77cbea5d28847b` | `version:6e6f908aae027941fe32a2d7eec030ed246586b1dc31ac53ff1e4e0d37c16b60` |
| STALE/a372.../64 | your cupboards.assign_zones = Divide your cupboards into zones based on the types of items you'll store in each | .85 | P | UNKNOWN | - | UNKNOWN | F | `slot:ab53edfc54773af8f2664aeb24dd8104af5e99884e8cf064216cd23face4174f` | `version:5daa0e61e822e46fc640cdcc86c10cead504ca0d3e628be2d1b450388705bc33` |
| STALE/a372.../65 | top shelf.intended_contents = infrequently used items (special occasion dishes, cookbooks) | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:1c3b43314519e49545a9058248e2e600034bd946052a9fc62ae2b0a0ad8e9cc9` | `version:b6d7c6161889a32fe8b0af2a944878e688540b690ebeb521a49c52bf6bb5d867` |
| STALE/a372.../66 | middle shelves.intended_contents = frequently used items (dinnerware, glasses, utensils) | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:17efdb1f9f60c5c5e36c29147b5e1208e1f814784c88c1d37b7b6d14e6961a` | `version:b017c129484cf927bd5bb8770ca4cffb0afe024a17fa9349f5c9c95665cb3108` |
| STALE/a372.../67 | bottom shelf.intended_contents = heavy items (pots, pans, appliances) | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:90d7b2cee553e6d2dec0ec459d1314867b1a856abc8fab6607809f4be852f89b` | `version:fc32422d92b17715ea8bda3dd1f74f7d8c919cb23c85cee763dba59f9025ce14` |
| STALE/a372.../68 | drawers.intended_contents = kitchen tools, gadgets, and utensils | .90 | P | UNKNOWN | - | UNKNOWN | F | `slot:62a8d05492f5493214ac86ddb280a76c87c4838dfe0d4b8e4373a9e577186b74` | `version:b47fa05209ac2fc6532f1f359b8cd8280e60ad78dbf09de31b42eaeffca670b5` |
| STALE/a372.../73 | frequently used items.stored_in = easy-to-reach locations | .87 | P | UNKNOWN | - | UNKNOWN | F | `slot:c6c18830a6b2a87db0ba0bbed7c7028bd5785b270a5b6578cbe26fe4e90e6e6c` | `version:6c27198f6991ed18c9236aeb317cc98936231be11c9274cbb32f853b74176219` |
| STALE/a372.../76 | each zone.periodic_check = Go through each zone and ensure everything is still organized and in its assigned home | .82 | P | UNKNOWN | - | UNKNOWN | F | `slot:1a7f32a8e5a6193744c105db2d912837e1a7cb13ef693c69bb2d1be8970f11e5` | `version:5ee22f42c9a462fd6f52370aa4e395008193fef738cda955146cc9296e014f52` |

The ledger lists the persisted replay-event indices only; the 139 rejected
candidates and the two malformed provider outputs are intentionally absent.

## Final classification

`REPLAY_SEMANTIC_EQUIVALENCE = NO` means this artifact cannot support a full
conditioned mechanism score. It does not convert the initial-write finding into
a replay-only explanation. The direct initial lifecycle failure occurs inside
the frozen resolver with grounded evidence and no old-state requirement.

`ROOT_CAUSE_CLASS = METHOD_LIFECYCLE_FAILURE` for the initial assertion
contract, specifically a resolver/policy contract failure. This is not a claim
that dependency, propagation, or the broader StateGraph methodology failed;
those stages were not evaluated here.

`RECOVERY_CLASSIFICATION = METHOD_LIFECYCLE_FAILURE` rather than
`RECOVERABLE_REPLAY_OR_INTEGRATION_BUG` because the two direct causes remain
after canonical evidence, identity, effective scope, and initial-write context
are available. A later replay can repair evaluation completeness, but it cannot
make these already-observed resolver decisions a valid `CURRENT` initial-state
result without changing resolver behavior.

No code or sealed artifact was changed in this forensic pass.
