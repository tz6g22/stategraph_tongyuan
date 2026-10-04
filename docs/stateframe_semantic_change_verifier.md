# Semantic Change Verifier: Autonomous A2

## Scope

This is a read-only semantic decision behind the default deterministic judge,
not an extractor, graph writer or dependency verifier. Persistent StateFrame
schema, cardinality semantics, propagation, retrieval, planner and answer
generation are unchanged. Production is not switched.

The previous deterministic-only stop is
`LOCAL_DETERMINISTIC_AUTHORIZATION_NO_GO`, not a StateGraph method failure.

## Runtime Boundary

`resolve_change` calls the semantic branch only when its default deterministic
judge returns UNKNOWN. An explicitly supplied judge (including wiring-test
mocks) never enables that branch. A run must explicitly establish
`semantic_verification_context`; absent this, behavior remains deterministic.

Eligibility requires an explicit proposed REPLACE, REMOVE or PATCH, exactly one
CURRENT target, known matching cardinality, exact subject/predicate/kind/binding/
facet/scope/modality/namespace match, forward observation order, consistent
target-value hint, sufficient confidence and grounded source evidence. Unknown
identity, ambiguous targets, absent evidence refs and unsupported operation
cardinality fail locally before any request. ASSERT, ADD, same-value merge and
deterministically SUPPORTED/CONTRADICTED paths do not use the verifier.

The provider receives an explicit semantic allowlist: candidate fields, one
target's semantic fields, operation, changed facet, cardinality and evidence
quotes. No state IDs, observation IDs, case IDs, lifecycle, arbitrary metadata,
proposal rationale, query, expected answer, evaluator output or gold is included.

Response is tri-state plus operation, target-match confirmation, short rationale,
evidence index and exact quote. Local code derives the unique quote's absolute
coordinates from canonical provenance. Provider character counts are not trusted.
Missing/ambiguous/nonliteral quotes, extra fields, incorrect operation and a
SUPPORTED response without target confirmation fail closed. The local boundary
and payload immutability are checked again after verification. Only the resolver
can return stale-version targets; only the unchanged repository commits them.

The semantic branch currently handles one target version, not multiple ROLE
facets. Scope support remains conservative. This limitation is not silently
filled by a broader target-selection LLM.

## Cost And Failure Handling

`SemanticChangeProvider` persists each request before sending, then the response,
usage, latency and sanitized failure category. SDK automatic retries are disabled.
Requests are bounded by an explicit per-run maximum. Identical semantic requests
within a run reuse exact successful provider payloads; no expected decisions are
cached. Malformed output fails local validation. Transport exceptions fail closed
at the resolver and are additionally visible in the run-level provider-failure
gate; they cannot count as successful acceptance.

`RecordedSemanticChangeProvider` is offline replay of sealed actual response
files keyed by exact request hash. Missing recordings fail closed. It is not a
fixture decision table or an acceptance oracle. Mock transports are used only
in the explicitly labeled interface boundary tests, never as semantic evidence.

## Development History

| Run | Outcome |
| --- | --- |
| A2 r1 | 13 boundary tests pass. Preflight imports fail before HTTP because system PYTHONPATH shadows the SDK's typing_extensions. |
| A2 r2 | 13 boundary tests pass. With corrected package precedence, one direct API attempt times out. Not a method failure. |
| A2 r3 | Existing Windows local proxy restored for process only; preflight passes. 7 real verifier calls. 57 development tests: 2 failures (model span counts; mock override routing). |
| A2 r4 | Exact quote grounding and explicit-judge isolation fixed generically. 14 boundary checks pass. 6 real verifier calls; 56/57 development tests pass. One unsupported discussion was incorrectly SUPPORTED and retired the target. |
| A2 r5 | Same prompt/model/schema, reasoning low, output cap 2048. One unsafe discussion authorization remains; one added fixture had a literal-value capitalization error before inference. 9 real verifier calls. |
| A2 r6 | Fixed gpt-5-mini/low, cap 2048. Live development 65/65; 10 calls, 4429 input / 2633 output tokens. Decisions: 1 SUPPORTED, 4 CONTRADICTED, 5 UNKNOWN. Local replay requirements 17/17, relevant suite 258/258; Phase1/2 171/171 and 183/183; interface 14/14. |

No unseen set was used in these rounds. The original 17 requirements are
unchanged development tests. The minimal-effort false-stale result is preserved;
it is not discarded or promoted to a pass. No phrase rule was added to repair it.
The nano-to-mini verifier model change is an explicit capability/cost change,
not attributed to better deterministic semantics. Main extraction/answer model
configuration remains untouched; later controlled comparisons must freeze and
disclose verifier model/budget separately alongside shared extraction/answer budgets.

## Environment

Use the existing Graphiti virtualenv's OpenAI SDK without modifying its installed
packages. Append `/usr/lib/python3/dist-packages` after venv packages to expose
the system jsonschema package; prepending it breaks typing_extensions. Run the
script by absolute path when invoking through runpy. Output directories are
explicit CLI arguments; there is no mutable global output-path override.

The existing Windows Internet Settings specifies local proxy
`http://127.0.0.1:7897`. Only the evaluation process receives HTTP(S)_PROXY;
no OS configuration was changed. Credentials are loaded from the existing
`apikey/openai.env`; values and HTTP headers are not logged.

## API Reference

OpenAI Docs was used to verify the Responses structured-output request format
and the reasoning-effort setting. JSON schema conformity is not evidence of
semantic truth; the local and live tests remain required.

- [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [GPT-5-family reasoning settings](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2)

## Current Readiness

A2 passed its development gate; Phase B old-v1 DEV regression is next. No benchmark gold was read, no
unseen input was consumed, Mem0 is unchanged and no production switch occurred.
The authoritative current controller is `STATEFRAME_AUTONOMOUS_GOAL_STATUS.md`.
