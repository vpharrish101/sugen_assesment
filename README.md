# ICD-11 title-only baseline

CPU-only BM25 retrieval over all **18,375 supplied category titles**. Uses Python's standard library; no model, API key, external data, or runtime network. This is the control for later clinical extraction, semantic retrieval, and family expansion, not the final assessment submission.

```sh
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

The included `data/notes.jsonl` is the dev input with **only id/text**. Default mounts are `./data:/in:ro` and `./out:/out`. For another input, set `CODER_INPUT_DIR` to a directory containing `notes.jsonl`; set `CODER_OUTPUT_DIR` to an existing writable directory if desired. Defaults are 1 CPU/1 GB, configurable via `CODER_CPUS`/`CODER_MEMORY`. Output has one flushed JSONL record per unique input ID. Duplicate input IDs or malformed records fail with a line-numbered error. Successful summaries go to stderr, not the predictions file.

Local execution and verification (Python 3.12):

```sh
python -m coder data/notes.jsonl out/predictions.jsonl
python -m coder.evaluate --gold data/dev_notes.jsonl
python -m unittest discover -s tests -v
```

**Measured dev result:** top-1 **5/60 (8.33%)**, top-3 **9/60 (15%)**, official score **7.9/75**. Candidate recall@25 **26/60 (43.33%)**, recall@50 **31/60 (51.67%)**. The dev set has no B slice; its maximum is 75. No missing, invalid, or duplicate predictions. Hidden-score estimate is deferred: this baseline has not established cross-distribution clinical accuracy and is not submission-ready.

**Method:** Unicode case folding/tokenization; BM25 k1=1.5, b=0.75; each query term counts once. Equal scores use code order. Zero-overlap notes receive deterministic valid codes with zero relevance, reported in stderr. These are coverage fallbacks, not reliable diagnoses. Full notes include history, negation, and hypothetical advice that plain lexical matching cannot interpret; the baseline deliberately adds no diagnosis-specific rules.

**Data/provenance:** original assessment files are preserved, including unmodified `score.py`; `data/notes.jsonl` is derived from the supplied dev input. No external medical data added. Dev gold/scorer are excluded from the image; only catalogue and runner are copied. The Python base image is digest-pinned.

**Beyond dev:** five unit/CLI tests cover unseen/Unicode IDs, empty and unmatched notes, ignored gold fields, duplicate rejection, and input preservation. A fresh clone passed a no-cache image build, the exact Compose command, the official scorer, and all five tests. Native and Docker dev predictions were identical. With Docker networking disabled and a 1 CPU/1 GB limit, 600 repeated-dev notes with new IDs took **2.24 seconds** inside the runner, with all 600 IDs covered and all codes valid. This checks throughput only, not generalisation. Metrics and source hashes are in `reports/baseline.json`.

**AI error caught:** preliminary AI research mapped chickenpox to `1F51.Z`; the supplied catalogue identifies that as African trypanosomiasis. Uncomplicated varicella is `1E90.0`. Outputs are therefore always checked against the catalogue; this still does not guarantee clinical correctness.

Hours spent: approximately 0.15 hours for this baseline implementation/verification; total assessment accounting remains pending. See `DECISIONS.md` for decisions and evidence.
