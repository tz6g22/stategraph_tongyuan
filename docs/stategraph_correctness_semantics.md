# State lifecycle and weak dependency semantics

- A revision is committed only for the same canonical slot and compatible time/condition scope, with grounded evidence and a supported assertion. Open-ended scopes that overlap at the new observation time may revise one another; bounded or conditional scopes must remain compatible. Otherwise the candidate remains `UNCERTAIN` with an `uncertainty_reason`.
- Explicit negative polarity is carried from grounded extraction into the CME shrunk writer. A negated value paraphrase is reduced to its lexical head only when that head is an exact source substring and the evidence has explicit negative polarity. Copular negation without a field name is accepted only when the registered field and value share a clear lexical root; other forms abstain.
- `STRICT` dependency invalidates downstream state. `WEAK` dependency never cascades invalidation; it records pending revalidation metadata.
- Propagation merges strict invalidation and weak revalidation metadata independent of seed order. A query-selected current state with a stale weak prerequisite is passed to the answer layer as unresolved (`CLARIFY`); unrelated pending states do not block a query.
- A directly observed replacement is protected from its predecessor's cascade. Other strict prerequisites are necessary conditions, not proof of independent sufficient support.
