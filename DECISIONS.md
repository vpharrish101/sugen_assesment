# Decisions

Commit references use unique commit subjects to avoid circular self-hashes; resolve them with `git log --oneline`. Times are UTC (+05:30 for India).

## 2026-11-05 13:22 UTC — Title-only control

Commit: `Implement title-only BM25 baseline`.

Decision: implement standard BM25 (k1=1.5, b=0.75) using an inverted index and the Python standard library. Alternatives: a third-party BM25 package, TF-IDF, or starting with embeddings/LLM ranking. Evidence: only 18,375 short documents; an inverted index avoids scanning every title for each note and requires no dependency downloads. Query terms count once; tokenization is Unicode-aware case folding without aliases, stemming, or hand-built medical rules. Ties use code order. These are defaults, not choices tuned against gold labels. Revisit retrieval only after recording this control's actual results.

## 2026-11-05 13:22 UTC — Grading contract and failure visibility

Commit: `Implement title-only BM25 baseline`.

Decision: the runner consumes id/text only, writes one final line per unique input ID, and flushes each record. It validates code membership/uniqueness and rejects duplicate input IDs or input/output path collisions. Zero-overlap queries return deterministic zero-score catalogue entries, explicitly tracked in stderr; these provide coverage but no meaningful diagnosis. Alternatives: omit uncertain notes or emit provisional corrections. Evidence: missing notes score zero and the official scorer keeps the first duplicate prediction. Revisit the weak zero-overlap path when semantic retrieval is added.

## 2026-11-05 13:22 UTC — Offline container

Commit: `Implement title-only BM25 baseline`.

Decision: ship only the runner and supplied code catalogue in a digest-pinned Python 3.12 container; exclude dev gold and scorer from the image. Disable container networking and default to 1 CPU/1 GB (overridable with CODER_CPUS/CODER_MEMORY). Alternatives: install libraries during startup or bake dev labels into the image. Evidence: runtime permits only model APIs and requires CPU-only no-key operation. Initial image build passed, but Docker rejected an explicit 8-CPU limit because this managed daemon exposes only 5 CPUs. The single-process baseline needs no 8-CPU allocation; lower limits also provide a stricter performance check than grading. The baseline uses no model calls or secrets. Revisit network configuration when model-API ranking is implemented.

## 2026-11-05 13:27 UTC — Freeze the first measured control

Commit: `Implement title-only BM25 baseline`.

Decision: record the raw-title baseline without tuning against dev gold. Alternatives: immediately add stopwords, favour leaf/residual categories, or adjust query sections to improve the score. Evidence: official scorer reports 5/60 top-1, 9/60 top-3, 7.9/75, zero missing/invalid/duplicate predictions; recall@25 is 26/60 and recall@50 is 31/60. Five unit/CLI tests passed. Full-note matching can rank hypothetical fever or background symptoms ahead of the encounter diagnosis, confirming a limitation rather than a runtime failure. The network-disabled Docker run matched the dev contract and processed 600 repeated-dev notes with new IDs in 2.24 seconds under 1 CPU/1 GB; this is a throughput check, not generalisation evidence. Revisit search coverage and clinical interpretation as separate measured changes.

## 2026-11-05 13:29 UTC — Include the official evaluation artifact

Commit: `Include official scorer for reproducible evaluation`.

Decision: track the supplied scorer byte-for-byte, outside the runtime image. Evidence: the first explicit staging list omitted score.py even though local evaluation used it; clean-clone review caught the omission. Alternative: document downloading the scorer separately, which would weaken reproduction. Revisit only if the assessment provides a new scorer.

## 2026-11-05 13:29 UTC — Clean-clone reproduction passed

Commit: `Record baseline validation and clean-clone results`.

Decision: accept source commit f3de75a as the first working control. Evidence: cloned with `git clone --no-local` into /tmp/su-baseline-check; a no-cache Docker build and the exact `docker compose run coder /in/notes.jsonl /out/predictions.jsonl` command passed with network disabled, 1 CPU/1 GB, and no model keys required. The clean-clone scorer reproduced 7.9/75 with no missing/invalid/duplicate predictions, candidate recall reproduced 26/60 at 25 and 31/60 at 50, and all five tests passed from the clone. Native and Docker predictions were byte-identical. The 600-note repeated-dev workload had 600 unique expected IDs and valid distinct codes; it establishes throughput, not clinical generalisation. Original scorer, catalogue, and gold files were byte-compared against the uploads. Alternative: proceed using only current-directory checks, rejected because the brief explicitly requires fresh-clone reproducibility. Revisit these checks when dependencies or the pipeline change.

## 2026-11-05 14:05 UTC — Focus lexical search on documented assessment

Commit: `Add diagnosis-focused family retrieval`.

Decision: detect generic assessment/diagnosis cues and search their trailing text; fall back to the full note when no cue or lexical overlap exists. Alternatives: keep full-note BM25, add disease-specific aliases, or call an LLM immediately. Evidence: the frozen control often ranked symptoms, precautions, and background conditions. The focused strategy improved top-1 from 5/60 to 14/60 and top-3 from 9/60 to 23/60, raising the official dev score from 7.9 to 21.2/75. It added 14 top-3 hits without losing any, while top-1 gained 10 and lost one. Rules contain no disease names, code prefixes, note IDs, or gold labels. Revisit cue precision on independent note styles.

## 2026-11-05 14:05 UTC — Create a bounded catalogue-family candidate pool

Commit: `Add diagnosis-focused family retrieval`.

Decision: preserve the top 25 focused results, add up to 10 new full-note results, then round-robin actual members of at most five dot-prefix families until 50 candidates. Alternatives: infer a WHO graph, expand all prefix matches, or use focused results alone. Evidence: every dotted code in the supplied catalogue has its root present, while official parent edges are absent. The bounded pool raises exact gold recall@50 from the control's 31/60 to 40/60 and recall@25 to 38/60. Family expansion never fabricates codes and begins after the top 25, so clinical ranking remains a separate future step. These pool sizes were selected after a small dev comparison and therefore require validation beyond dev. Revisit once a semantic retriever and constrained reranker exist.

## 2026-11-05 14:06 UTC — Prepare ontology-encoder evaluation before choosing a checkpoint

Commit: `Prepare biomedical dense retrieval benchmark`.

Decision: build a frozen local dense-index path around a pinned encoder, precomputed normalized title vectors, catalogue hashes, and reciprocal-rank fusion, while leaving the measured focused strategy as the runtime default until checkpoints are compared. Shortlist English SapBERT and multilingual SapBERT-XLMR; use a character TF-IDF control, and reject BioBERT alone because masked-language pretraining does not directly train synonym retrieval. Evidence: the 20 recall@50 misses are dominated by aliases and abbreviations such as “jiggers”/“tungiasis”, “vaginal thrush”/“vulvovaginal candidosis”, and “GORD”/“gastro-oesophageal reflux disease”. The official SapBERT repository documents UMLS synonym-pair entity-linking training, CLS pooling, cross-lingual variants, and an MIT license. A local character 3–5 gram control reached 42/60 recall@50 but reduced top-1 from 14 to 13 when fused into final ranking, so it is insufficient as the semantic stage. Also exclude patient turns and negated cue tails from assessment extraction; this preserved 14/60 top-1 and 23/60 top-3 while raising the existing family pool to 41/60 recall@50. Model artifact access is pending activation of the saved Hugging Face CDN policy. Revisit model and fusion defaults only after real checkpoint measurements under the CPU deployment constraint.

## 2026-11-05 15:21 UTC — Select SapBERT CLS after checkpoint controls

Commit: `Add hybrid SapBERT retrieval`.

Decision: use pinned `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` revision `090663c…` with the documented CLS-before-pooler representation. Alternatives: focused BM25, general `all-MiniLM-L6-v2`, official mean-token SapBERT, BioLORD, or a generative LLM. Evidence on the frozen 60 notes: English SapBERT reached 26/60 top-1, 42/60 top-3, and 54/60 recall@50; MiniLM reached 16/28/39 and mean-token SapBERT 24/38/53. BioLORD was not downloaded because its model card declares a nonstandard `other` license and the purpose-built SapBERT result already met the entity-linking need. Pure English dense ranking beat lexical+dense fusion for final top-three, while equal English-dense/BM25 fusion reached 59/60 recall@50 for a future verifier. Revisit with an independent labelled ICD set and a constrained verifier.

## 2026-11-05 15:21 UTC — Keep a lower-weight multilingual encoder

Commit: `Add hybrid SapBERT retrieval`.

Decision: ship pinned multilingual SapBERT-XLMR revision `47b6bd0…`, route with offline `langid`, and fuse English:multilingual at 3:1 for English notes and 1:3 otherwise using RRF constant 10. Alternatives: English only, multilingual only, equal fusion, or an API translation step. Evidence: multilingual-only English performance was 19/60 top-1, 28/60 top-3, and 45/60 recall@50, so it cannot replace English SapBERT. The 3:1 ensemble improved the measured final result to 27/60 top-1 and 44/60 top-3 (40.8/75); all 60 dev notes were classified as English. The brief explicitly warns that hidden notes are not always English, while the cross-lingual model is trained for biomedical entity linking in multiple languages. This adds roughly 1.1 GB of weights and is therefore an accuracy-over-image-size choice. Revisit language routing on labelled multilingual consultation notes.

## 2026-11-05 15:21 UTC — Bake semantic assets and retain lexical fallback

Commit: `Add hybrid SapBERT retrieval`.

Decision: download immutable model revisions and precompute normalized embeddings for all supplied titles during Docker build; run fully offline and fall back by batch to focused BM25. Alternatives: download at runtime, call an embedding API, commit binary weights, or remove fallback. Evidence: build-time title encoding took 254 seconds for English and 411 seconds for multilingual on the managed 5-CPU host. The resulting image was 2.87 GB. Runtime processed 60 notes in 16.3 seconds and 600 repeated texts with unique IDs in 61.2 seconds inside the runner, with no missing or invalid output, under the grading limit of 60 minutes and 16 GB. No API calls or keys are used. Revisit ONNX/int8 export if image transfer or cold-start time becomes material.

## 2026-11-06 10:40 UTC — Keep Qwen as the measured local reranker

Commit: `Restore Qwen prototype and bound llama-server RAM`.

Decision: restore the exact Qwen3 4B Ollama prompt and settings used in the local measured run. Alternative: MedGemma 1.5 4B with constrained catalogue selection. Evidence supplied from the laptop: Qwen reached 38/60 top-one, 49/60 top-three, 52.08/75 in 38.2 seconds; MedGemma reached 31/60, 52/60, 47.50/75 in 136.2 seconds. These are development-set GPU experiments, not Docker grading results. Revisit model choice only after independent notes or a compliant CPU/API runtime is tested.

## 2026-11-06 10:40 UTC — Bound llama-server prompt-cache RAM

Commit: `Restore Qwen prototype and bound llama-server RAM`.

Decision: launch llama.cpp build b12144 with `--cache-ram 0`, `--no-cache-idle-slots`, one slot, a 4,096-token context, and bounded batch sizes; disable per-request prompt caching in the llama-server client. Alternative: leave the server's 8,192 MiB default RAM prompt cache enabled or restart the process periodically. Evidence: `llama-server --help` documents that default; with caching disabled, ten distinct roughly 440-token prompts held resident memory at 4,715 MiB after each request on the cloud CPU host. This is a limited memory probe and does not prove laptop behavior over 60 notes. Revisit if the user's laptop still shows growth with the same flags.
