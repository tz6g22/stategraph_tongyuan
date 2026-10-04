# Secret audit

- Audited the current StateGraph commit history (all 5 reachable commits after
  refreshing `origin/main`) for credential-shaped values, private-key headers,
  GitHub token forms, OpenAI-style token forms, bearer credentials, and common
  provider-token assignments. No high-confidence secret match was found in
  reachable Git history.
- Audited the staged 3,092-file tree: no high-confidence key, bearer,
  private-key, or provider-token pattern; no credential-named secret file,
  actual `.env`, API-key directory, model, database, log, cache, venv, nested
  `.git`, `external_baselines/`, or file larger than 25 MiB is staged.
- `apikey/openai.env` and `apikey/第三方api.txt` exist locally and are excluded
  by `.gitignore`; their contents are not reproduced here. All `.env.example`
  files are templates. The upstream `Mem0Api.credentials.ts` file defines a
  credential field in source code; it contains no credential value.
- The locally modified Graphiti patch was inspected for credential material;
  none was found.
- No secret value is recorded in this report.
