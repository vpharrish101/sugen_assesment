## Architecture Decision and AI Usage

**Note:** This part was completely written by me (no AI), as I wanted to walk through individual architecture designs crafted. `DECISIONS.md` has the per-commit/per-branch decisions, but this should provide the overview of core architecture designed.

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

### 2. Workflow decisions, and other points

**a.** Testing was done between Qwen3 and MedGamma, since the latter is explicitly trained on medical corpus.

**a1.** However, the results were contradictory, and my initial reasoning is that Qwen is excellent as short range classification tasks, unlike MedGemma.

### 3. Future improvements

**a. Further comparison of general and medical models:** Qwen gave 38/60 top-1 and 49/60 top-3, while MedGemma gave 31/60 and 52/60. Qwen performed better under the scorer, but this does not establish that general models are better for medical coding. There were results published in Qwen's favour (https://papers.miccai.org/miccai-2026/paper/5540_paper.pdf), so I'd do more research here.

**b. SapBERT vs PLM-ICD / LAAT:** These are supervised coding architectures, mostly evaluated on ICD-9/ICD-10 datasets, rather than direct replacements for SapBERT embeddings. Adapting them to the ICD-11 label set would require suitable training data. With only 60 labelled notes, I kept them outside the assessment scope.

**c. GNN / graph-based representation:** ICD-11’s Foundation supports multiple parents, which makes graph-based retrieval worth investigating. However, the supplied MMS catalogue contains titles, chapter information and leaf/residual flags, without explicit relationships between nodes. I tried limited code-family context, but did not build a complete hierarchy or train a GNN. A further implementation should use official WHO relationships, since code prefixes alone can misrepresent the hierarchy.

**d. Extended-thinking models:** The correct code was present in the top-50 candidates for 59/60 development notes, leaving room to improve the final selection. I would test whether additional reasoning improves top-1 and top-3 accuracy, particularly when choosing between specific and unspecified variants. This would need to justify its extra latency, memory or API cost.

### 4. AI Usage

**a.** ChatGPT Codex was used as primary prototyping agent.

**b.** Architecture sketchings and design was done completely by me, as I wanted to get a deeper grasp of the problem statement and avoid superficial designs with AI.

**c.** AI was used in finiding relevant research papers and GitHub implementations, reducing time required to sieve through unrelated papers.

**d.** Coding was done mainly through AI, from the initial BM25 till the final architecture. I verified and reviewed the codebase of only the final draft (main branch), since allocating effort to understnad the intricate details of the prototypes seemed to be a diverting task. But to ensure that we understand what we ship, I've spent time analyzing and understanding the submitted architecture.
