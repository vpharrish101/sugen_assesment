# ICD-11 candidate retriever

BM25 + English and multilingual SapBERT retrieval, followed by LLM reranking. The final version uses Claude as the primary API path, optional local Qwen as a fallback, and retrieval-only output if neither is available.

Note: Please refer to ARCH.md, for architecture details.

## Run and evaluate

```sh
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

Default mounts are `./data:/in:ro` and `./out:/out`. Set `ANTHROPIC_API_KEY` in the environment or `.env` to enable Claude. Compose forwards `HTTPS_PROXY`. Without a key, the system uses configured Qwen or offline retrieval. Embedding models and catalogue indexes are built into the image; runtime downloads are not required. Predictions are validated against the supplied catalogue and flushed per note.

For host Ollama, use `compose.qwen.yaml`; setup instructions are in `docs/RUNNING.md`.

## Development results

All results below use the 60 provided notes. Top-1 and top-3 are accuracy; recall@50 measures whether a gold code is present in the retrieval pool.

| Architecture | Top-1 | Top-3 | Retrieval recall@50 | Score /75 |
|---|---:|---:|---:|---:|
| a. BM25 baseline | 5/60 (8.33%) | 9/60 (15.00%) | 31/60 (51.67%) | 7.92 |
| b. Enhanced BM25 | 14/60 (23.33%) | 23/60 (38.33%) | 40/60 (66.67%) | 21.25 |
| c. Hybrid SapBERT retrieval | 27/60 (45.00%) | 44/60 (73.33%) | 59/60 (98.33%) | 40.83 |
| d. Local Qwen 4B reranking | 38/60 (63.33%) | 49/60 (81.67%) | 59/60 (98.33%) | 52.08 |
| e. Single-pass hierarchy  | 38/60 (63.33%) | 43/60 (71.67%) | 59/60 (98.33%) | 49.58 |
| f. Two-pass hierarchy  | 37/60 (61.67%) | 54/60 (90.00%) | 59/60 (98.33%) | 53.33 |

The fresh first pass within experiment f scored 39/60 top-1 (65.00%), 50/60 top-3 (83.33%) and 53.33/75. Its verifier improved top-3 but reduced top-1, leaving the score unchanged.

The MedGemma 1.5 4B comparison scored 31/60 top-1 (51.67%), 52/60 top-3 (86.67%) and 47.50/75, using the same retrieval pool as Qwen. Claude has no live accuracy result. Hierarchy experiments used catalogue-derived code families, not a complete official WHO relationship graph.

## Added resources and software testing

Pretrained encoders came from [SapBERT](https://github.com/cambridgeltl/SapBERT): `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` and `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`. Catalogue title embeddings are generated during build. Spanish/Chinese evaluation translations were derived from supplied notes; no new clinical cases or labelled training set were generated.

51 software checks passed, plus rebuilt-image CLI and Docker Compose smoke checks using three clinical inputs. These cover proxy/TLS handling, provider errors, timeouts, malformed responses and fallbacks. Simulated API responses test integration, not Claude's clinical accuracy. Details are in `docs/API_TESTING.md`; recorded results are under `reports/`. Decisions and experiment history are in `DECISIONS.md`.

## Future improvements

**a. Further comparison of general and medical models:** Qwen gave 38/60 top-1 and 49/60 top-3, while MedGemma gave 31/60 and 52/60. Qwen performed better under the scorer, but this does not establish that general models are better for medical coding. There were results published in Qwen's favour (https://papers.miccai.org/miccai-2026/paper/5540_paper.pdf), so I'd do more research here.

**b. SapBERT vs PLM-ICD / LAAT:** These are supervised coding architectures, mostly evaluated on ICD-9/ICD-10 datasets, rather than direct replacements for SapBERT embeddings. Adapting them to the ICD-11 label set would require suitable training data. With only 60 labelled notes, I kept them outside the assessment scope.

**c. GNN / graph-based representation:** ICD-11’s Foundation supports multiple parents, which makes graph-based retrieval worth investigating. However, the supplied MMS catalogue contains titles, chapter information and leaf/residual flags, without explicit relationships between nodes. I tried limited code-family context, but did not build a complete hierarchy or train a GNN. A further implementation should use official WHO relationships, since code prefixes alone can misrepresent the hierarchy.

**d. Extended-thinking models:** The correct code was present in the top-50 candidates for 59/60 development notes, leaving room to improve the final selection. I would test whether additional reasoning improves top-1 and top-3 accuracy, particularly when choosing between specific and unspecified variants. This would need to justify its extra latency, memory or API cost.

## AI Usage

**a.** ChatGPT Codex was used as primary prototyping agent.

**b.** Architecture sketchings and design was done completely by me, as I wanted to get a deeper grasp of the problem statement and avoid superficial designs with AI.

**c.** AI was used in finiding relevant research papers and GitHub implementations, reducing time required to sieve through unrelated papers.

**d.** Coding was done mainly through AI, from the initial BM25 till the final architecture. I verified and reviewed the codebase of only the final draft (main branch), since allocating effort to understnad the intricate details of the prototypes seemed to be a diverting task. But to ensure that we understand what we ship, I've spent time analyzing and understanding the submitted architecture.

## Notes

1. One place my ai tooling was wrong, is that: increasing parameter model (while testing, qwen 3-4b to qwen 3-8b without thinking) would increase the accuracy materially. But since my idea was that, it wouldn't produce meaningful results as this task is a constrained code classification, a 4B param model handles most of it well and an 8b would do little to accuracy but at the cost of 2x the latency. the results proved my side when we tested with both qwen 3-4b and 8b. I delibrately left thinking out, since it added a heavy constraint on my GPU, taking upwards of 140s for a single query in my RTX4060.

2. Evaluation beyond the 60 provided notes weren't done, since
     1. I wanted the focus to be more on experiments and not on dataset generation.
     2. Generating synthetic datasets would carry hallucination risks and divert away from the goal of the task in hand.
     3. There was a brief evaluation conducted on translating english diagnosis to Chinese and Spanish, to test multi-lingual retrieval accuracy. However, this was very limited in scope and only served as diagnostic tool for how the model would behave under multi-lingual text.

3. I spent around 4 + 7 + 12 hours , wherin
     1. (Day 1) First 4 hours was more dedicated to knowing the task in hand and other materials/logistics.
     2. (Day 2) 7 hours was more towards initial implementation and prototyping, and architecture design and some research paper / github implementation lookups. I initially did some prototyping on my own pc, and later moved to GitHub.
     3. (Day 3) 12 hours was towards implementing the majority of the architectures, testing them under circumstances, and seeing what improved vs what didn't. Additionally, docker setup and other logistics were also handled and a whole initalization to final testing was done to ensure no hiccups occur.

4. a. Top-1 accuracy: Around 55-60%.
b. Top-3 accuracy: Around 65%.
c. Retrieval recall@50: Around 85%, meaning the correct code appears somewhere in the 50 retrieved candidates. While multilingual issues may bring down the model's accuracy, I believe Haiku model would do a much better work on categorization than Qwen 3-4b.

Hours spent: approximately 23 hours.

The figures in note 4 are hidden-set estimates, not measured results. Recall@50 does not directly contribute to the assessment score.
