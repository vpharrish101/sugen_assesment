# ICD-11 clinical note coder

Fresh note → diagnosis-focused retrieval → Anthropic Haiku 5.5 → local Qwen if configured → validated retrieval fallback. English keeps the measured SapBERT mixture and BM25 candidate pool. Non-English notes use multilingual SapBERT; Spanish/Chinese have a small structural extraction addon. Providers see the full note and numbered catalogue titles. Only supplied catalogue codes can be output; there is no training on dev labels.

```sh
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

Inputs mount from `./data`, outputs to `./out`. Set `ANTHROPIC_API_KEY`; Compose forwards the grader's `HTTPS_PROXY`. See [.env.example](.env.example) and [running instructions](docs/RUNNING.md). Without a key/local server every note still gets an offline prediction. Encoders and title indexes are built into the CPU image. Runtime never downloads assets. Default limits: 4 CPUs/6 GB, 3,300 seconds of inference, $5.50 conservative API reservation, at most $6 configured. Predictions flush per note.

| Measured dev system | Top-1 | Top-3 | Official-score equivalent |
|---|---:|---:|---:|
| Focused BM25 | 14/60 | 23/60 | 21.25/75 |
| Offline SapBERT mixture | 27/60 | 44/60 | 40.83/75 |
| Local Qwen3 **4B** GPU prototype | 38/60 | 49/60 | 52.08/75 |

The historical branch name `qwen3b-prototype` refers to that 4B model. Haiku API accuracy has not yet been measured; the Qwen result cannot be attributed to Haiku. Provisional hidden estimate: **40–55/100**, low confidence given narrow dev coverage and weak non-English results.

Added data: pinned public English/multilingual [SapBERT](https://github.com/cambridgeltl/SapBERT) weights (UMLS entity linking), build-generated title vectors, and reviewed assistant translations of six existing cases into Spanish/Chinese. No new clinical cases or labelled training set were invented. No WHO definitions were added. Models/provenance and all controlled results are documented in [multilingual evaluation](docs/MULTILINGUAL_RETRIEVAL.md).

Beyond dev: 18 paired-language fixtures, offline/no-key and outage coverage, provider SDK contract/proxy/budget tests, catalogue validation and a 600-note repeated-text throughput check. Translations are not independent clinical validation; their extraction path gives Spanish 2/6 top-1, 3/6 top-3 and Chinese 0/6, 2/6. Full results and remaining submission checks: [brief audit](docs/BRIEF_CHECK.md).

AI error caught: early research called `1F51.Z` chickenpox; the supplied catalogue identifies it as African trypanosomiasis. Catalogue validation and title-constrained selection replaced free code guessing. A final SDK test also caught an unsupported `temperature` argument before deployment.

Hours spent: **TODO: enter your actual total before submitting**. History is preserved; decisions are in [DECISIONS.md](DECISIONS.md). Deliver a bundle with `git bundle create yourname.bundle --all`; send it yourself on the assessment email thread.
