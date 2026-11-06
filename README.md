# ICD-11 hybrid candidate retriever

The runner maps consultation notes to the **18,375 supplied ICD-11 MMS 2026-01 categories** using diagnosis-focused extraction plus two local biomedical entity-linking encoders. English SapBERT is primary; multilingual SapBERT-XLMR receives a smaller vote for English notes and becomes primary when offline `langid` detects another language. Reciprocal-rank fusion produces the final three codes. No API key or runtime network is required.

```sh
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

The first build downloads pinned model weights. Default mounts are `./data:/in:ro` and `./out:/out`. The container has runtime networking disabled and defaults to 4 CPUs/6 GB, configurable with `CODER_CPUS` and `CODER_MEMORY`. It validates every output code and flushes predictions after each 32-note batch. If semantic assets fail to load or infer, focused BM25 still covers every note.

## Measured development results

| Method | Top-1 | Top-3 | Recall@50 | Score |
|---|---:|---:|---:|---:|
| Full-note title BM25 control | 5/60 | 9/60 | 31/60 | 7.9/75 |
| Diagnosis-focused BM25 | 14/60 | 23/60 | 41/60 | 21.2/75 |
| General MiniLM control | 16/60 | 28/60 | 39/60 | 25.0/75 |
| English SapBERT CLS | 26/60 | 42/60 | 54/60 | 39.2/75 |
| Final English + multilingual ensemble | **27/60** | **44/60** | — | **40.8/75** |

For a later verifier, equal fusion of English SapBERT and focused BM25 retrieves a gold code for **52/60 at 25** and **59/60 at 50**. This is a candidate-pool metric; final top-three accuracy is scored separately. The network-disabled image processed 600 repeated-dev texts with unique IDs in **61.2 seconds inside the runner** and 83.3 seconds including container startup. This validates throughput and coverage, not clinical generalisation.

## Models and method

- `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` at `090663c…`, CLS pooling: trained on UMLS synonym pairs for biomedical entity linking.
- `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` at `47b6bd0…`, CLS pooling: cross-lingual entity linking for non-English notes.
- Both title matrices are computed during image build and catalogue-hash checked at runtime. Models never generate codes; rankings contain only supplied catalogue entries.
- Final fusion uses English:multilingual weights 3:1 for English and 1:3 otherwise, with RRF constant 10. The mean-token SapBERT checkpoint and MiniLM were measured and rejected.

The models are public SapBERT project artifacts trained from UMLS terminology; no labelled notes, WHO definitions, or hand-built disease aliases were added. The official SapBERT project is MIT licensed, the English model card reports Apache-2.0, and the multilingual model card does not declare a separate license field. Runtime API cost is $0.

## Reproduction and limits

```sh
python -m unittest discover -s tests -v
python -m coder data/notes.jsonl out/fallback.jsonl --strategy focused
docker compose run --rm --entrypoint python coder -m coder.evaluate --gold /in/dev_notes.jsonl
```

Eleven tests cover extraction, language routing, fusion, catalogue integrity, Unicode/unseen IDs, malformed/duplicate inputs, and fallback coverage. Beyond the 60 labels, testing covers all catalogue codes structurally, offline/no-key execution, multilingual routing, clean builds, and 600-note load; there is no independent labelled ICD set, so the provisional hidden estimate is **48–56/100**. The dev set covers only 11 chapters and fusion choices were measured on it.

AI error caught: preliminary research mapped chickenpox to `1F51.Z`, which the supplied catalogue defines as African trypanosomiasis; uncomplicated varicella is `1E90.0`. Catalogue validation now prevents such nonexistent or mismatched generated-code paths.

Hours spent: approximately 1.8 hours through this retrieval iteration. Decision evidence is in `DECISIONS.md`; machine-readable metrics are under `reports/`.
