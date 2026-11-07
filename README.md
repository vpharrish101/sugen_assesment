# ICD-11 clinical note coder

Fresh note → diagnosis-focused BM25/SapBERT retrieval → Claude reranking → optional local Qwen → retrieval fallback. Only supplied catalogue codes can be returned. English keeps the measured embedding mixture; other languages use multilingual SapBERT.

## Run

Put your private `notes.jsonl` and evaluation labels in the Git-ignored `private-data/` directory, or set CODER_INPUT_DIR. No clinical dataset is shipped. Only the public ICD-11 catalogue is tracked.

```sh
mkdir -p private-data out
docker compose build coder
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold private-data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

Set ANTHROPIC_API_KEY in .env for Claude; HTTPS_PROXY is forwarded. No-key fallback produces predictions for every valid input. For local Ollama use compose.qwen.yaml; see docs/RUNNING.md. The image builds catalogue title indexes ahead of time and never downloads models at runtime. Defaults: four CPUs/6 GiB, 3,300s inference budget and $5.50 conservative API reservation, configurable up to $6. Predictions flush per note. Optional service failures fall back with redacted diagnostics.

## Recorded development results

| Approach | Top-1 | Top-3 | Score /75 |
|---|---:|---:|---:|
| BM25 | 5/60 | 9/60 | 7.92 |
| Enhanced BM25 | 14/60 | 23/60 | 21.25 |
| Hybrid SapBERT | 27/60 | 44/60 | 40.83 |
| Original local Qwen 4B | 38/60 | 49/60 | 52.08 |
| Single-pass hierarchy v2 | 38/60 | 43/60 | 49.58 |
| Two-pass hierarchy v1 | 37/60 | 54/60 | 53.33 |

The two-pass run's fresh first pass also scored 53.33/75 (39/60, 50/60). Claude has no live accuracy result. These are historical development measurements, not new smoke scores. Hidden estimates are uncertain: top-1 55–60%, top-3 about 65%, candidate recall@50 about 85%. Recall@50 is not directly scored.

Added resources: pinned public English/multilingual SapBERT encoders from https://github.com/cambridgeltl/SapBERT and build-generated catalogue embeddings. Earlier Spanish/Chinese translations of six supplied cases were limited diagnostics, not independent clinical validation or training data. All clinical originals/derivatives and note-level reports are removed from Git history in the cleanup draft. Aggregate scores remain.

Software validation uses small protocol fixtures and at most three externally supplied notes, including SDK/HTTP/TLS, proxy, timeout, malformed-output, fallback and Docker checks. No paid API calls or large clinical benchmark is needed for smoke testing. AI tooling initially suggested a larger Qwen would necessarily improve accuracy; the author reports limited benefit in local comparisons, with extra latency. No quantitative 8B result is claimed here.

Hours spent: approximately 23 (4 + 7 + 12), per the author's accounting. Architecture decisions/history remain in DECISIONS.md. Privacy cleanup rewrites historical commit IDs; do not push from an old unsanitized clone after the approved replacement.
