# Decisions

## Branch → architecture

| Branch | Architecture / purpose |
|---|---|
| `main` | Cleaned final API-first application |
| `final-prototype` | Final BM25/SapBERT → Claude → optional Qwen → retrieval fallback |
| `bm25` | Full-note BM25 baseline |
| `bm25-enhanced` | Diagnosis-focused BM25 and bounded family candidates |
| `sapbert` | English/multilingual SapBERT hybrid retrieval |
| `qwen3b-prototype` | Basic single-call Qwen3 **4B** reranking; branch name is historical |
| `qwen-hierarchy-prototype` | One-call hierarchy-aware Qwen reranking |
| `qwen-two-pass-prototype` | Initial Qwen ranking followed by hierarchy verification |
| `tmp/ollama-rerank-download` | Archived delivery branch; no active Docker runtime |

## Unified log

Consolidated from the existing branch logs; original entries remain in history.
Dates below identify the associated implementation commits in IST. Laptop results
reported afterward are identified as such. Counts are top-1/top-3 out of 60 unless
another denominator is given. Commit IDs refer to the locally sanitized history.

### 2026-11-05 18:58 IST — BM25 baseline

Commit: `2366f83`. Use full-note title matching as the control, rather than start with an LLM. It scored 5/60 top-1 and 9/60 top-3 (7.92/75). Keep catalogue validation and one output per ID; improve retrieval next.

### 2026-11-05 19:18 IST — Enhanced BM25

Commit: `1a4c4ed`. Search the clinician assessment before the full note and add bounded catalogue-family candidates. This improved results to 14/60 and 23/60 (21.25/75); the recorded family pool reached 40/60 recall@50. Prefix families are retrieval hints, not official WHO edges.

### 2026-11-06 11:43 IST — Hybrid SapBERT

Commit: `9a197e9`. Choose biomedical synonym embeddings over a general encoder. English SapBERT reached 26/42 versus MiniLM 16/28; the English/multilingual mixture reached 27/44 (40.83/75). SapBERT + BM25 candidates covered 59/60 at 50. Validate these choices on independent notes later.

### 2026-11-06 16:11 IST — Qwen versus MedGemma

Commit: `f16b670`. Keep Qwen3 4B after laptop results of 38/49 (52.08/75), versus MedGemma 31/52 (47.50/75). Medical pretraining did not win under this scorer. These are development results, not evidence that Qwen is universally better.

### 2026-11-06 16:24 IST — Bound local inference memory

Commit: `74b30e8`. Disable llama-server RAM prompt caching and idle-slot caching instead of relying on Python garbage collection. A short CPU probe showed stable memory. This bounds prompt caching, not total process RAM; Ollama manages its own server memory.

### 2026-11-06 17:30 IST — Single-pass hierarchy experiment

Commit: `a51a096`. Add bounded catalogue relatives and use opaque selection IDs instead of mixed codes/numbers. The later laptop run scored 38/43 (49.58/75), below basic Qwen. Keep this as an experimental branch; expanded-pool recall@50 was not separately recorded.

### 2026-11-06 17:45 IST — Two-pass verifier

Commit: `fee1612`. Check the first ranking against a 24-candidate family shortlist, retaining the first pass on invalid verification. Later laptop results: 39/50 before verification, 37/54 after; both scored 53.33/75. Better top-3 did not justify a score gain or twice the calls.

### 2026-11-06 22:05 IST — Multilingual addon

Commit: `032ba4d`. Preserve the measured English mixture; use multilingual retrieval with Spanish/Chinese assessment cues and note-local abbreviation expansion. On six paired cases per language, Spanish reached 2/3 and Chinese 0/2. These were limited diagnostics, not independent clinical validation.

### 2026-11-06 22:05 IST — CPU grading and API-first routing

Commit: `032ba4d`. Use Claude first, configured local Qwen second, retrieval last. Keep a $5.50 conservative API reservation and 3,300-second inference budget. The grader has no GPU; no key must still yield predictions. Real Claude access and accuracy remain untested.

### 2026-11-06 23:17 IST — Failure handling and small smoke tests

Commit: `3349164`. Keep predictions when providers time out, reject requests or return invalid rankings. Three consecutive provider failures disable that provider. SDK/socket/TLS and three-note CLI checks passed without paid calls. Later cleanup checks also cover setup/cleanup errors and incomplete retrieval batches.

### 2026-11-06 23:39 IST — Optional Docker host networking

Commit: `50ae4ca`. Add compose.qwen.yaml for reaching host Ollama through loopback. Keep it optional for API-only grading and keep model/key settings in .env. With both providers configured, automatic routing remains Claude first; --backend ollama explicitly bypasses Claude.

### 2026-11-07 08:09 IST — Private data and branch cleanup

Commit: `11dcc83`. Keep only the public catalogue tracked; remove clinical inputs, derivatives and note-level reports from prepared history. Mount private input externally. Remove frozen/duplicate launchers, retain branch-specific Docker runners, and archive the temporary download branch. Seven distinct builds and three-note smokes passed; publication still needs approval.

### 2026-11-07 — Ship public catalogue vectors

Keep the pinned English and multilingual title matrices in assets/catalogue/ (about 108 MiB total). Native and Docker setup validate and reuse them, while query encoder weights remain downloaded and Git-ignored. Catalogue/model metadata and matrix checksums prevent stale or damaged assets from being used. No clinical data is included.

### 2026-11-07 — Reuse complete semantic assets in Docker

Copy assets/semantic/ into the image instead of downloading encoders and encoding
the catalogue during each fresh build. Keep the weights Git-ignored and provide
the Drive download in instructions.md. Copied multilingual tokenizer metadata
contained a list for extra_special_tokens, while Transformers 4.57.6 expects a
mapping; passing an empty mapping at load time resolved the laptop smoke test.
Use the same compatibility option for both SapBERT encoders; no ranking changes.

### 2026-11-07 — Keep prepared assets in Drive

Remove the redundant assets/catalogue/ matrices from the current tree. Distribute
the complete semantic/ folder through the documented Drive link, keeping encoder
weights and vectors Git-ignored. Docker copies that folder; native setup can also
regenerate it. This supersedes the earlier decision to ship matrices in Git.
