# Brief audit

The newly attached brief matches `BRIEF.md` byte for byte. This document separates submission compliance from model accuracy.

| Requirement | Implementation / verification |
|---|---|
| Fresh `id,text` input; one output per ID | Main runner ignores gold/style, generates candidates from each note, validates IDs and catalogue membership, flushes per note. |
| Exact Docker Compose invocation | Passed from a fresh clone using the API-enabled image; no key/local model, 60 valid outputs. |
| CPU-only, 16 GB, 600 notes / 60 minutes | Offline test uses stricter four-CPU/6 GB limits; repeated texts validate throughput, not generalisation. Local GPU Qwen is optional. |
| Runtime network only model API through proxy | Anthropic SDK official endpoint, inherited proxy and TLS trust. No runtime asset downloads. Local Qwen client is restricted to loopback and not enabled on the grader by default. |
| $6 shared API cap | Conservative $5.50 reservation, zero automatic retries, retained reservation on errors, explicit prices for model changes. Grading key cap remains the billing limit. |
| No-key and API failure fallback | API → configured local Qwen → measured retrieval fallback. Skip each provider after three consecutive failures; complete the rest. |
| Titles-only valid catalogue | 18,375 supplied categories; immutable English/multilingual title matrices checked against code/title hash. No invented hierarchy, definitions or codes. |
| Specificity / unspecified coding | Existing shared clinical prompt and exact titles retained. This is a model decision, not a guarantee of correct specificity. |
| Multilingual / broader chapters | Full catalogue searched; multilingual encoder with Spanish/Chinese extraction evaluated. Weak translated accuracy is disclosed. No held-out clinical dataset or hidden-note score is available. |
| One-page README | Concise score, uncertain hidden estimate, data provenance, beyond-dev evaluation and caught AI error. |
| Real git history and DECISIONS | Preserve existing commits; append contemporaneous integration entries. No squash. |
| Hours and bundle/email deliverable | User must enter their actual hours. The brief ultimately requests a bundle/email, but the user explicitly deferred both: do not create a bundle or send any email without a new instruction. GitHub push alone is not assessment submission. |

## Model selection

Best *measured* model remains Qwen3-4B-Instruct-2507 Q4_K_M (38/49, 52.08/75 on the laptop). MedGemma and the single-pass hierarchy run scored worse. Two-pass has no complete measured result here. The user requested Haiku 5.5 as the primary API, so it is configured as `claude-haiku-5-5` with Qwen as an optional fallback; **Haiku has no live quality result yet**.

The model ID was verified in the [official Anthropic SDK model list](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/model.py). The pinned SDK is 2.6.0. Its current Messages interface does not accept `temperature`; the adapter uses supported forced-tool output and disables thinking. SDK tests exercise the actual installed version through an in-memory HTTP transport. They do not verify server-side model access or live inference. Neither API key is configured in this cloud environment; the current policy also does not list `api.anthropic.com` for live egress. A real API smoke test needs both a configured secret and the supported network configuration applied. No bypass was attempted.

Pricing was cross-checked in [LiteLLM's maintained registry](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), 2026-11-06. Official pricing documentation redirected to an unavailable domain under this environment's policy. Conservative default input/output rates are five times the listed normal-context price, matching its higher-context tier. Check current provider pricing before submission, especially if changing models.

## Final verification results

All implementation checks completed; the API quality/access check remains pending credentials.

- **35/35 tests passed**, no skips, including real encoders + actual Anthropic SDK + simulated HTTP: API success, Qwen fallback, final retrieval fallback on mixed English/Spanish/Chinese notes. Also verified proxy configuration, 429 without retries, strict output validation, no-key/outage coverage and inference deadline handling.
- **Clean clone build and exact Compose command passed**. The implementation commit was `9692537`; subsequent changes only add tests, reports and documentation. CPU image `sha256:533c94441e07fa78962ca5afb1495a423253c2d952d1420f409cc07f2f00c29a` includes pinned model assets and the actual API SDK. The exact run wrote all 60 predictions in 13.64 seconds inside the runner, about 1,655 MiB RSS. The first rebuild briefly exhausted this cloud host's VFS disk; only task-created obsolete cache layers and the completed temporary model download were removed, then the clean build succeeded.
- **English functionality preserved**: all 60 fresh candidate pools and fallback lists exactly match the frozen Qwen experiment inputs. Shared Qwen prompt is byte-identical. The unmodified official scorer reports 27/60 top-one, 44/60 top-three, **40.83/75** for offline retrieval, with zero missing/invalid codes.
- **600-note CPU load passed** under four CPUs/6 GB, with network disabled: **65.03 seconds** inside the runner, **1,719 MiB** peak RSS, no semantic fallback errors, zero missing/invalid predictions. Inputs repeat the dev texts with unique IDs; this establishes throughput, not clinical generalisation or API latency.
- **Translated evaluation reproduced**: English 2/6 top-one, 4/6 top-three; Spanish 2/6, 3/6; Chinese 0/6, 2/6. Same visible paired fixtures, not independent clinical validation.
- **Three real Qwen requests passed** through the final llama.cpp transport on CPU. Their top-three outputs match the earlier local predictions. Cache disabled, one slot, server RSS 5,417,944 → 5,418,664 KiB across the three calls. The server was shut down afterward. This short test does not establish 60-note memory stability or remeasure the full GPU score.
- **No live Haiku calls**: no configured API key; official API host also needs an allowed egress configuration here. SDK/MockTransport tests cannot establish server-side access, accuracy or real latency. Evaluate this before calling the API-primary system submission-ready. The configured API reservation would total about $2.29 for 600 repetitions of the dev prompt distribution, at the conservative rates; that is not measured invoice usage.

Evidence: `reports/final_validation.json`, `reports/final_qwen_smoke.json`. The optional offline acceptance harness is `tests/validate_submission.py` and is excluded from the production image along with dev labels/tests. Existing prototype branches remain intact. Nothing has been pushed to GitHub/main for this integration; final user approval is required. No bundle has been created and no email has been sent. Enter actual hours before assessment submission.
