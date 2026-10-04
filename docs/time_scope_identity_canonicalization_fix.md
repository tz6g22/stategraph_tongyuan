# Time-scope identity canonicalization fix

The shared `canonical_semantic_scope()` helper removes a time-scope start only
when extraction materialized it at the observation timestamp. Explicit bounds
remain semantic identity. Observation timestamps, sequence indexes, and version
creation timestamps remain unchanged for chronology and audit.

The helper is used by:

- `canonical_state_slot_key()` and the `StateNode` extension slot key;
- native candidate scope consolidation;
- the active duplicate invariant's normalized scope check.

Thus linker compatibility reaches the same representation through
`StateNode.canonical_slot_key`, instead of maintaining a second scope rule.

## Offline SCB_016 replay

Source artifacts: `production_integration_20260921_r2/runtime/cme-scb-SCB_016`.
The replay used the sealed old state and the saved observation-1 candidate. It
made no provider calls.

- old and new canonical slot keys are equal after removing their implicit starts;
- `StateLinker` returns `SAME_SLOT` and selects the old state;
- the duplicate invariant does not raise at the identity/linker checkpoint.

The replay intentionally stops before applying the frozen resolver lifecycle
decision; its existing revision warning remains open for the next integration
run. Result artifact: `outputs/conditioned_mechanism_eval_14case_v3/identity_canonicalization_replay_20260922/SCB_016_REPLAY.json`.

