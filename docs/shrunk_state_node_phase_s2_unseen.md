# Shrunk StateNode Phase S2 Unseen Validation

Status: BLOCKED before unseen inference. `SHRUNK_STATE_NODE_S2_UNSEEN_READY =
BLOCKED`.

Final run directory: `outputs/shrunk_state_node_phase_s2_20260920_v1/`.

## Scope

S2 is an isolated comparison between:

- **S0**: the existing legacy `StateRevision`/`StateLinker` path; and
- **S2**: `ShrunkStateRepository` with StateNode minimal extensions and its
  single local revision resolver.

Both paths were prepared to receive the exact same provider-extracted
`StateCandidate` objects, canonical `OBSERVATION_ABSOLUTE` EvidenceRecord and
sequence order. There is no production switch, no dependency discovery,
verification/STRICT-WEAK change, propagation, retrieval, planner or answer
generation execution in this phase. The S2 runner imports no Full StateFrame,
old standalone change authorization or old semantic-verification module.

## Sealed Unseen Set

The run prepared 18 newly authored natural-language sequences with 42 source
observations and 18 distinct coverage categories. The source set includes
functional replacement, confirmation, member add/remove and coexistence,
employment, preference/ownership/relation membership, polarity, temporal scope,
event facets, third-party subject, ambiguous update, ordinary no-change,
conditional action and nonliteral update wording.

Each text passed the frozen exact-source novelty audit against the old Phase 3
source files, StateFrame autonomous C-v2/v3/v4 inputs/authoring artifacts and
S1/StateFrame fixture source strings. No previous StateFrame canary, benchmark
case, historical debugging source or S1 fixture source is present in the set.
The definition is deliberately narrow: it establishes no exact-source reuse in
the audited developer corpus. These are synthetic authored sources, rather than
randomly selected external benchmark records, and must not be presented as a
benchmark result.

The evaluator labels were authored and sealed before inference but the run
command is structured to read `CANARY_LABELS.json` only after writing
`INFERENCE_SEAL.json`. No label is loaded during source extraction or either
revision path. The set has never crossed `RUN_STARTED`; it therefore remains
unconsumed under the project rule that only executed unseen cases lose unseen
status.

## Frozen Configuration

The manifest froze all StateGraph Python source hashes, the S2 runner and
authoring script, plus protected hashes for schema, shrunk resolver, legacy
revision/conflict/linking and in-memory storage. It also froze the generic local
registry:

- functional: residence, availability, event time/location/status and action
  status;
- set-valued: preference, membership, employment, affiliation, relationship and
  ownership.

The registry uses generic field/cardinality semantics only. It contains no case
ID, entity name, expected value, benchmark identifier or answer rule. Unknown
fields remain fail-closed. The planned extraction is one shared structured
provider request per observation, no recovery and no retry. S0 and S2 would be
fed the same parsed candidate list; provider variance cannot explain a
representation difference.

The narrow verifier adapter is run-scoped and only implements the existing
`NarrowSemanticVerifier` protocol. It has no Frame, ChangeIntent, target ID,
lifecycle, dependency or answer access. It can be invoked only by the S2
resolver after the resolver has already established a single target, known
cardinality, compatible scope and an otherwise UNKNOWN destructive transition.
The adapter requires the provider to echo the whole canonical source evidence,
then the local resolver rechecks all conditions before a stale write. It cannot
choose a target or write a state itself.

## Preflight Block

One harmless provider preflight was attempted with the sealed `gpt-5-mini`
configuration and SDK retries disabled. It had no canary source in its request.
The request remained unresponsive past the configured 90 second timeout and was
still running after approximately 135 seconds. It produced no response and no
`PREFLIGHT.json` before being terminated to prevent an unbounded provider wait.

This is recorded as `PROVIDER_INFRASTRUCTURE_TIMEOUT`. The run did not create
`RUN_STARTED.json`, runtime files, `INFERENCE_SEAL.json`, S0/S2 metrics or a
provider response artifact. Consequently no result can be attributed to
identity, cardinality, polarity, scope, local revision, verifier, extraction or
provenance. `FAILURE_ATTRIBUTION.json` is intentionally empty for method cases;
the only blocker is preflight infrastructure.

The attempt count is one requested preflight call, zero completed provider calls,
zero canary extraction calls and zero semantic-verifier calls. The process was
terminated only after the preflight timeout breach; no data or sealed artifact
was deleted or overwritten.

## Preflight Retry (No Canary Input)

The same sealed preflight was retried after the source, protected-module hashes,
pre-run seal, manifest, configuration and canary-consumption markers were
rechecked. No proxy or custom base-URL environment override was present; the
credential key was available through the existing loader. The retry was run with
the unchanged preflight command and a process-level 120-second limit. It again
returned `PROVIDER_INFRASTRUCTURE_TIMEOUT` without a response.

No canary source was sent: `RUN_STARTED.json` and `INFERENCE_SEAL.json` remain
absent, and the 18-case manifest remains unconsumed. This is a provider
preflight gate failure, not a StateNode, resolver, extraction, or S0/S2
comparison result. The retry evidence is recorded in
`outputs/shrunk_state_node_phase_s2_20260920_v1/PREFLIGHT_RETRY_20260919_r2.json`
and `preflight_retry.log`.

## Local Checks After the Block

The S2 runner and authoring code compile. The S1 shrunk suite remains 44/44
PASS. The final pre-run audit verifies:

| Check | Result |
| --- | --- |
| Frozen source hashes unchanged | PASS |
| Protected module hashes unchanged | PASS |
| PRE_RUN_SEAL hashes valid | PASS |
| `RUN_STARTED.json` absent | PASS |
| `INFERENCE_SEAL.json` absent | PASS |
| Full StateFrame runtime import in S2 runner | None (static import audit) |
| API/provider inference calls | 0 |

The preflight command used the existing project Graphiti Python environment with
its current site packages plus the system `jsonschema` location. No dependency,
credential, provider setting, proxy or code was changed after the sealed source
set was prepared.

## Required Artifacts

`CANARY_MANIFEST.json`, `SOURCE_HASHES.json`, `CONFIG.json` and
`PRE_RUN_SEAL.json` are pre-run sealed. `S0_VS_S2.json`,
`VERIFIER_STATS.json`, `FAILURE_ATTRIBUTION.json` and `VERIFICATION.json` record
the infrastructure block without inventing scores. `tests.log`, `compile.log`
and `PRE_RUN_AUDIT.json` give the full local evidence. `FINAL_SEAL.json` binds
the directory and this report after the block.

## Result And Next Boundary

No S2 unseen readiness claim is possible. This is not a StateNode method failure,
not a failed S0/S2 comparison, and not a canary failure: inference never began.
The sealed sources are still eligible for one later execution only after a fresh
provider connectivity preflight succeeds without changing method code, registry,
prompts, config, source inputs or labels. A new method patch or source rewrite
would invalidate this prepared set rather than turn it into a development case.

Do not start conditioned mechanism evaluation from this blocked S2 result.

## Formal Execution Outcome

The sealed S2 runner was subsequently started with the recovered process-only
proxy path. It completed shared extraction and S0/S2 runtime capture for all
18 sequences, then wrote `INFERENCE_SEAL.json`. That seal records the 18
runtime artifacts and confirms that evaluator labels were not loaded before
inference. Consequently all 18 cases are permanently consumed.

Evaluation then stopped before producing metrics. The runner's local evaluator
initializes `previous_current` as a path-to-set mapping, but subtracts
`expected_current` directly from that mapping rather than from the selected
path's set. The resulting `TypeError` occurs at evaluator bookkeeping before
any S0/S2 metrics, verifier statistics, acceptance gate, or per-case method
attribution can be calculated. This is an `EVALUATION_PROTOCOL` blocker, not a
classification of any canary as identity, cardinality, polarity, scope, local
revision, verifier, extraction, or provenance failure.

No code or configuration was changed after the first canary source was sent.
No rerun, repair, new canary generation, threshold change, or manual metric
recalculation was performed. The prior `FINAL_SEAL.json` remains an immutable
pre-inference-block artifact; the post-inference outcome is separately recorded
in `POST_INFERENCE_EVALUATION_BLOCKER_20260920_r1.json`.
