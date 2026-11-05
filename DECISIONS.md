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
