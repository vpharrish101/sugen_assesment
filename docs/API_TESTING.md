# API integration checks without paid calls

The new socket suite uses Python's standard `http.server`, the actual pinned Anthropic SDK and actual HTTP/TLS transport. It follows the fake-server approach of [fake-anthropic](https://github.com/tscircuit/fake-anthropic), but implements this application's forced `rank_icd_codes` tool contract. The linked project's default Hello world response cannot validate that contract. No Node/Bun dependency or production endpoint override was added.

Seven new tests passed in the production CPU image with networking disabled (21.739 seconds). Only **three clinical notes** were processed: English, Spanish and Chinese. Real encoders/retrieval and the actual SDK/socket transport executed; two simulated API successes and one injected 429 produced two API rankings and one retrieval fallback, with all three IDs present and codes valid. The other tests use tiny protocol fixtures rather than running the clinical dataset. The newly attempted 600-note test was stopped at the user's request and removed from this suite; it is not counted as completed validation.

Checks include actual current SDK Bearer authentication and API version headers; tool/schema/model serialization; HTTP 400/401/403/404/413/429/500/529 without retries; malformed JSON; missing/wrong/duplicate tool blocks; truncated or invalid selections; timeouts; dropped connections; and disabling the failing provider after three failures. The HTTPS test uses an actual CONNECT proxy and TLS connection with an ephemeral trusted test certificate and hostname verification enabled. The proxy never forwards outside the test process. This tests the proxy request path, not the grader's specific infrastructure.

The subsequent complete smoke pass found and fixed optional-component failure boundaries in the runner/router: provider initialization/cleanup, invalid Qwen configuration, damaged metadata and incomplete/invalid semantic batches. Model choice, shared prompt and ranking algorithms are unchanged. These changes are included on the `final-prototype` branch. Historical aggregate results remain in README; note-level reports were removed for privacy.

With the declared API dependencies installed:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_anthropic_wire.py -v
```

The three-note real-encoder check requires `CODER_TEST_ASSETS` pointing to the built semantic assets. Otherwise it is explicitly skipped. The TLS test uses `openssl` to generate a temporary test certificate, or `CODER_TEST_TLS_DIR` for pre-generated `cert.pem`/`key.pem`. Neither certificate is trusted by production. No new dependencies are required for the server.

## Once a real key is available

Export the key securely as `ANTHROPIC_API_KEY`, then explicitly run:

```sh
PYTHONPATH=src python tests/smoke_live_api.py --input /path/to/private/notes.jsonl
```

This opt-in command makes one real, potentially billable request using the same adapter and freshly retrieved candidates from the external input. It tests authentication/model access and output parsing, not fresh-note retrieval accuracy. It never uses fallback to report success. No key means exit 2 and zero requests; failure means exit 1; PASS means the API actually returned a valid ranking. The helper checks only one externally supplied note. It was checked without a key and correctly refused to claim a live PASS.

No fake server can establish real account permissions, provider-side model/schema acceptance, clinical accuracy or latency. Those remain explicitly unverified until this opt-in smoke check is run with a real key. No live request was made during this work.

## Complete app and Docker pass

The full suite passed **51 tests in 60.757 seconds** in the CPU image, including the three clinical inputs reused across scenarios. `tests/test_app_smoke.py` additionally starts the real CLI in subprocesses, without patching the SDK or production base URL, through a local non-forwarding HTTPS CONNECT proxy. The real SDK reads `HTTPS_PROXY`/`SSL_CERT_FILE`, verifies the ephemeral test certificate and sends authentication/tool requests. One scenario returns successful rankings; another injects 401/404/429 while Qwen is unreachable. No-key/offline, missing assets, broken CA/Qwen configuration, and invalid JSON config/input checks cover startup and output behavior. Invalid configuration/input fails clearly before overwriting input; optional service failures still yield valid predictions for every supplied ID.

The resilience tests first reproduced actual gaps: initialization/cleanup exceptions could escape, and a partial semantic batch could silently lose IDs. The runner now clears rejected batches before lexical fallback. It validates input before initializing optional clients and catches the expected damaged-metadata failures. The router isolates provider setup and cleanup errors, logs actionable redacted diagnostics and preserves the existing provider order.

See the local cleanup verification record for the newly cleaned images. The image is built from a clean copy of the current source with no local model binaries or secrets included. Only three unique clinical inputs are used, with small protocol fixtures for other tests. No 600-note run is part of this smoke pass. Image source hashes, dependency checks, no-key Compose invocation and environment forwarding are checked separately.

Successful simulation establishes tested application behavior, not real account authorization or provider-side model availability. Missing/corrupt mandatory catalogue data, invalid input/configuration, an unwritable output volume, or an OS/container kill are operational failures that cannot be repaired by an API fallback. The app reports ordinary file/configuration errors; deploy with the tested image, valid input and a writable output mount. Clinical accuracy is independent of these reliability checks.

Final Docker outcome: build passed; `pip check` passed; CPU Torch 2.9.1+cpu and Anthropic SDK 2.6.0 verified; image source equals the tested working tree; no dev gold/archived experiments included. Compose forwards key/model and both proxy-variable cases, enforces four CPUs/6 GiB, and the exact entrypoint produced all three valid predictions offline in **7.1874 seconds**, at **1,475.9 MiB** peak RSS. Four app CLI smoke methods passed again using the rebuilt image's own code, including the unpatched SDK HTTPS success/failure paths and clear invalid-input handling. The live-key helper correctly exited 2 with no key and no requests. No live provider access or model quality is claimed. The smoke-tested changes are included on the `final-prototype` branch.

## 2026-11-07 — Review of main, with no live key

Reviewed main at `0602417` and ran the current working tree in the cached CPU
runtime. The user had already verified the Docker build and Qwen path; neither
was changed or rebuilt for this review. Current source and tests were mounted
into the runtime, which had external networking disabled. Only the same three
existing clinical inputs were reused across real-encoder checks; other tests
use small protocol fixtures, not a new clinical dataset.

The final discovery run took **93.533 seconds**: **79 passed, one skipped**.
The skipped source-checkout privacy test needs Git, absent from the runtime; it
passed separately on the host. There were no failures. `pip check` passed with
Torch 2.9.1+cpu, Transformers 4.57.6, NumPy 2.3.5, Anthropic 2.6.0, httpx2 2.16.0,
langid 1.1.6 and abbreviations 0.2.5 available. Existing cached dependencies were
used; no models or packages were downloaded during the review.

The added checks cover:

- API-first configuration, explicit Ollama bypass, custom model/price settings,
  missing credentials and invalid budgets or prices.
- Actual SDK authentication, API version, system/user separation, forced-tool
  schema, candidate limits of 3/50/100, deadlines and independent requests.
- HTTP 400/401/402/403/404/408/409/413/422/429/500/502/503/529; timeouts, dropped
  connections, no hidden retries, budget accounting and consecutive failures.
- Invalid JSON/tool inputs, floats, booleans, strings, duplicates, out-of-range
  choices, refused/unfinished turns and output truncation.
- Successful HTTPS through a local CONNECT proxy and rejection of an untrusted
  certificate before the HTTP request and credentials are sent.
- Real CLI runs through API success, Qwen fallback and retrieval fallback,
  preserving IDs and valid catalogue codes. These CLI checks use the actual SDK
  and HTTP transport without replacing the SDK or production base URL.
- Ignoring gold labels in API input, rejecting invalid input before truncating
  an existing output file, and evaluating protocol predictions with score.py.

Two gaps were reproduced and corrected in the Anthropic adapter: request exactly
one tool call with `disable_parallel_tool_use=True`, and require `tool_use` as
the completed stop reason. A refusal or paused/incomplete turn must fall back,
even if a malformed response also contains a tool-shaped block. The shared
prompt, retrieval, Qwen settings and Docker/Compose files are unchanged.

Additional checks validated the emitted tool schema with JSON Schema, resolved
both Compose configurations to verify key/model forwarding and data mounts,
and confirmed the opt-in live helper exits 2 with no key instead of reporting
fallback as API success. The configured default model is listed in the pinned
SDK's model type; that does not establish account access to it.

To repeat the tests after installing the declared dependencies:

```sh
CODER_TEST_ASSETS="$PWD/assets/semantic" \
CODER_TEST_NOTES=/path/to/private/three-notes.jsonl \
PYTHONPATH=src:tests python -m unittest discover -s tests -v
```

The local API/proxy never forwards to Anthropic. No paid request was made.
Real authentication, credit, model permissions, provider-side acceptance,
latency and clinical accuracy remain unverified until the one-note live smoke
is run with a real key.
