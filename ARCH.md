## Architecture Decision and AI Usage.

Note: This part was completely written by me (no AI), as I wanted to walk through individual architecture designs crafted. DECISIONS.md has the per-commit/per-branch decisions, but this should provide the overview of core architecture designed. 

1. Architecture Decisions: -

Following sufficient research and analysis on the task, I landed on 4 architectures, each sequentially building upon each other. These were: -
a. BM25 baseline: A basic lexical matching approach that compared the consultation note with the ICD-11 catalogue titles. This gave me a starting score to compare the later architectures against.
b. Enhanced BM25: The next version focused retrieval on the clinician’s assessment or diagnosis, since matching the entire note brought in unrelated symptoms, history and treatment details. It also included candidates from relevant code families.
c. Hybrid SapBERT retrieval: I added English and multilingual SapBERT embeddings alongside BM25. This helped retrieve diagnoses that were medically similar even when the note and catalogue used different wording. Their rankings were combined to produce a candidate list.
d. Retrieval with LLM reranking: The final architecture added an LLM over the existing retrieval pipeline. It received the full note and the retrieved codes with their titles, then selected the three best-supported codes. I tested the reranking with local Qwen. The final version has Claude as the primary API path, optional local Qwen as a fallback, and retrieval-only output if neither is available.
e. Single-pass hierarchy-aware reranking: I added code-family relationships from the supplied catalogue to the retrieved candidates. Qwen received the full note, candidate titles and related code variants in one call, and selected codes with supporting evidence. This tested whether hierarchy context helped distinguish specific, parent and unspecified categories. It reached 38/60 top-1 and 43/60 top-3, scoring 49.58/75, below the basic Qwen reranker.
f. Two-pass hierarchy verification: I separated the initial ranking from verification. The first Qwen call ranked the retrieved candidates; the second checked that ranking against the clinical note and relevant code-family context. This prototype was implemented, but we don’t have a completed 60-note evaluation result for it.
BM25 Baseline: -

<img width="1058" height="299" alt="image" src="https://github.com/user-attachments/assets/b2858c4e-22a5-4f7e-828c-5f20e08e2f49" />

Enhanced BM25: -
<img width="1125" height="351" alt="image" src="https://github.com/user-attachments/assets/1de01937-9928-4af6-b987-ed08b63ec957" />

Hybrid SapBERT retrieval: -
<img width="845" height="873" alt="image" src="https://github.com/user-attachments/assets/a40f44c0-5dac-4f90-9d9d-64751b0c737a" />

Retrieval with LLM reranking: -


2. Workflow decisions, and other points: -
  a. Testing was done between Qwen3 and MedGamma, since the latter is explicitly trained on medical corpus.
    a1. However, the results were contradictory, and my initial reasoning is that Qwen is excellent as short range classification tasks, unlike MedGemma.

3. Future improvements: -
    a. SapBERT vs PLM-ICD / LAAT : Did not choose them because they were primarily trained for ICD-10. ICD 11 is polyhierarchial among other things, making adaptation to ICD-11 difficult, especially with less data and scope deviation.
    b. GNN/Graph based representation, since the given ICD-11 is polyhierarchial in design, naturally fitting in a graph based retrieval or GNN. It was not implemented in this assesment, since the details given only contains if a code is a parent or leaf, but not any relationship with other nodes. Constructing a graph from this could potentially misrepesent diagnosis.
    c. If extended thinking models could improve the Top 1/3 recalls, since recall@50 mostly had the correct term. Could reasoning over the given architectures produce a meaningful improvement over models like Qwen, would be my next implementation.
    d. Extended multilingual work.
   

4. AI Usage: -
   a. ChatGPT Codex was used as primary prototyping agent.
   b. Architecture sketchings and design was done completely by me, as I wanted to get a deeper grasp of the problem statement and avoid superficial designs with AI.
   c. AI was used in finiding relevant research papers and GitHub implementations, reducing time required to sieve through unrelated papers.
   d. Coding was done mainly through AI, from the initial BM25 till the final architecture. I verified and reviewed the codebase of only the final draft (main branch), since allocating effort to understnad the intricate details of the prototypes seemed to be a diverting task. But to ensure that we understand what we ship, I've spent time analyzing and understanding the submitted architecture.
   
