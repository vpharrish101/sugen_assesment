# ICD-11 candidate retriever

CPU-only retrieval over all **18,375 supplied ICD-11 MMS 2026-01 categories**. The current strategy extracts generic diagnosis/assessment text, runs BM25 over exact titles, preserves some full-note results, and expands a few real dot-prefix code families for a future reranker. It uses Python's standard library, no API key, no external medical data, and no runtime network.

```sh
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

Default mounts are `./data:/in:ro` and `./out:/out`; `data/notes.jsonl` contains only dev `id`/`text`. Set `CODER_INPUT_DIR`, `CODER_OUTPUT_DIR`, `CODER_CPUS`, or `CODER_MEMORY` as needed. Output is one flushed JSONL record per unique ID. Malformed or duplicate inputs fail clearly. The original control remains reproducible with `--strategy baseline`.

## Measured development results

| Strategy | Top-1 | Top-3 | Recall@25 | Recall@50 | Score |
|---|---:|---:|---:|---:|---:|
| Full-note title BM25 control | 5/60 | 9/60 | 26/60 | 31/60 | 7.9/75 |
| Diagnosis-focused + family pool | **14/60** | **23/60** | **38/60** | **40/60** | **21.2/75** |

The focused strategy gained 14 top-3 hits and lost none relative to the control. It gained 10 top-1 hits and lost one. The improvement shows that encounter focus was a larger problem than BM25 execution. Recall@50 is still only 66.7%, so a reranker alone cannot solve the remaining cases; semantic terminology coverage is the next retrieval problem. No hidden-score estimate is justified yet. The dev set has no B slice, so its maximum is 75.

## Method

Generic cues such as `Assessment:`, `Imp:`, `Dx`, and “this is” select diagnosis-focused text. If no cue exists—or the focused text has no title overlap—the original note is used. The 50-candidate pool contains the best 25 focused matches, up to 10 new full-note matches, then round-robin members of at most five real catalogue families. A family is the exact string before the decimal; membership aids recall but is not treated as an official clinical hierarchy. The runner never constructs codes or suffixes.

BM25 uses Unicode case folding/tokenization, k1=1.5, b=0.75, and unique query terms. The current no-key output is the top three focused lexical results. Every output is checked against the supplied catalogue. Structural validity does not guarantee clinical correctness.

## Reproduction and limits

```sh
python -m coder data/notes.jsonl out/predictions.jsonl
python -m coder.evaluate --gold data/dev_notes.jsonl
python -m coder.evaluate --gold data/dev_notes.jsonl --strategy baseline
python -m unittest discover -s tests -v
```

Tests cover focus extraction, family membership, Unicode/unseen IDs, empty/unmatched notes, ignored gold fields, duplicate rejection, and input preservation. Docker networking is disabled and the Python base image is digest-pinned. Dev gold and scorer are excluded from the runtime image. Metrics are stored under `reports/`.

Added data: none. Supplied files are preserved; `data/notes.jsonl` is derived by removing dev labels/styles. AI error caught: preliminary research mapped chickenpox to `1F51.Z`, which the supplied catalogue defines as African trypanosomiasis; uncomplicated varicella is `1E90.0`.

Hours spent: approximately 0.6 hours through this retrieval iteration; final accounting remains pending. See `DECISIONS.md`.
