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
