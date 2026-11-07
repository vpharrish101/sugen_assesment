## Architecture Decision and AI Usage

**Note:** `DECISIONS.md` has the per-commit/per-branch decisions, but this should provide the overview of core architecture designed.

### 1. Architecture Decisions

Following sufficient research and analysis on the task, I landed on 4 architectures, each sequentially building upon each other. These were:

**a. BM25 baseline:** A basic lexical matching approach that compared the consultation note with the ICD-11 catalogue titles. This gave me a starting score to compare the later architectures against.

**b. Enhanced BM25:** The next version focused retrieval on the clinician’s assessment or diagnosis, since matching the entire note brought in unrelated symptoms, history and treatment details. It also included candidates from relevant code families.

**c. Hybrid SapBERT retrieval:** I added English and multilingual SapBERT embeddings alongside BM25. This helped retrieve diagnoses that were medically similar even when the note and catalogue used different wording. Their rankings were combined to produce a candidate list.

**d. Retrieval with LLM reranking:** The final architecture added an LLM over the existing retrieval pipeline. It received the full note and the retrieved codes with their titles, then selected the three best-supported codes. I tested the reranking with local Qwen. The final version has Claude as the primary API path, optional local Qwen as a fallback, and retrieval-only output if neither is available.

**e. Single-pass hierarchy-aware reranking:** I added code-family relationships from the supplied catalogue to the retrieved candidates. Qwen received the full note, candidate titles and related code variants in one call, and selected codes with supporting evidence. This tested whether hierarchy context helped distinguish specific, parent and unspecified categories. It reached 38/60 top-1 and 43/60 top-3, scoring 49.58/75, below the basic Qwen reranker.

**f. Two-pass hierarchy verification:** I separated the initial ranking from verification. The first Qwen call ranked the retrieved candidates; the second checked that ranking against the clinical note and relevant code-family context. This prototype was implemented, but we don’t have a completed 60-note evaluation result for it.

#### BM25 Baseline

<img width="1058" height="299" alt="image" src="https://github.com/user-attachments/assets/b2858c4e-22a5-4f7e-828c-5f20e08e2f49" />

#### Enhanced BM25

<img width="1125" height="351" alt="image" src="https://github.com/user-attachments/assets/1de01937-9928-4af6-b987-ed08b63ec957" />

#### Hybrid SapBERT retrieval

<img width="845" height="873" alt="image" src="https://github.com/user-attachments/assets/a40f44c0-5dac-4f90-9d9d-64751b0c737a" />

#### Retrieval with LLM reranking

<img width="845" height="873" alt="image" src="https://github.com/user-attachments/assets/1a22f972-2e53-4259-ad66-2b3792e96169" />

#### Single-pass hierarchy-aware reranking

<img width="845" height="873" alt="image" src="https://github.com/user-attachments/assets/437cc4f4-7fc0-4a18-930a-c21f73f18879" />

#### Two-pass hierarchy verification

<img width="845" height="873" alt="image" src="https://github.com/user-attachments/assets/046f534f-1131-4bac-b910-7664e7dd15f2" />

