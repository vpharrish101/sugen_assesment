# Brief audit

The newly attached brief matches `BRIEF.md` byte for byte. This document separates submission compliance from model accuracy.

| Requirement | Implementation / verification |
|---|---|
| Fresh `id,text` input; one output per ID | Main runner ignores gold/style, generates candidates from each note, validates IDs and catalogue membership, flushes per note. |
| Exact Docker Compose invocation | Final clean-clone result recorded below after testing. |
| CPU-only, 16 GB, 600 notes / 60 minutes | Offline test uses stricter four-CPU/6 GB limits; repeated texts validate throughput, not generalisation. Local GPU Qwen is optional. |
| Runtime network only model API through proxy | Anthropic SDK official endpoint, inherited proxy and TLS trust. No runtime asset downloads. Local Qwen client is restricted to loopback and not enabled on the grader by default. |
| $6 shared API cap | Conservative $5.50 reservation, zero automatic retries, retained reservation on errors, explicit prices for model changes. Grading key cap remains the billing limit. |
| No-key and API failure fallback | API → configured local Qwen → measured retrieval fallback. Skip each provider after three consecutive failures; complete the rest. |
| Titles-only valid catalogue | 18,375 supplied categories; immutable English/multilingual title matrices checked against code/title hash. No invented hierarchy, definitions or codes. |
| Specificity / unspecified coding | Existing shared clinical prompt and exact titles retained. This is a model decision, not a guarantee of correct specificity. |
| Multilingual / broader chapters | Full catalogue searched; multilingual encoder with Spanish/Chinese extraction evaluated. Weak translated accuracy is disclosed. No held-out clinical dataset or hidden-note score is available. |
| One-page README | Concise score, uncertain hidden estimate, data provenance, beyond-dev evaluation and caught AI error. |
| Real git history and DECISIONS | Preserve existing commits; append contemporaneous integration entries. No squash. |
| Hours and bundle/email deliverable | User must enter their actual hours. Create `git bundle create yourname.bundle --all` and manually email before the deadline; GitHub push alone is not submission. |

## Model selection

Best *measured* model remains Qwen3-4B-Instruct-2507 Q4_K_M (38/49, 52.08/75 on the laptop). MedGemma and the single-pass hierarchy run scored worse. Two-pass has no complete measured result here. The user requested Haiku 5.5 as the primary API, so it is configured as `claude-haiku-5-5` with Qwen as an optional fallback; **Haiku has no live quality result yet**.

The model ID was verified in the [official Anthropic SDK model list](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/model.py). The pinned SDK is 2.6.0. Its current Messages interface does not accept `temperature`; the adapter uses supported forced-tool output and disables thinking. SDK tests exercise the actual installed version through an in-memory HTTP transport. They do not verify server-side model access or live inference. Neither API key is configured in this cloud environment; the current policy also does not list `api.anthropic.com` for live egress. A real API smoke test needs both a configured secret and the supported network configuration applied. No bypass was attempted.

Pricing was cross-checked in [LiteLLM's maintained registry](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), 2026-11-06. Official pricing documentation redirected to an unavailable domain under this environment's policy. Conservative default input/output rates are five times the listed normal-context price, matching its higher-context tier. Check current provider pricing before submission, especially if changing models.

## Final verification results

Pending completion of the build and clean-clone checks. No push to GitHub/main has been performed for this integration. User approval is required at the final push step.
