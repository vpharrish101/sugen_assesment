# Single-pass Qwen hierarchy prototype

Local branch: `qwen-hierarchy-prototype`. The original measured experiment remains on `qwen3b-prototype`: 38/60 top-one, 49/60 top-three, 52.08/75. Run this addition separately; no accuracy improvement is claimed before the full laptop evaluation.

Use the same installed Qwen model and Ollama service. From `ollama-experiment`:

```sh
python run_hierarchy_experiment.py --limit 4 --output-prefix hierarchy-v2-check
python run_hierarchy_experiment.py --output-prefix hierarchy-v2-full
```

If the old Qwen prediction file is available, add `--compare-predictions /path/to/ollama_predictions.jsonl` to the full command. It is used only after inference, to report top-one/top-three gains and losses by note. Continue an interrupted run with the same prefix and `--resume`. The runner rejects a different model or input hash under an existing prefix. Fallback predictions count as completed results and are included in the reported score.

The implementation is two small Python files and uses the standard library. `hierarchy_reranker.py` builds parent/child dictionaries from the supplied catalogue: each lower-level code must have a shorter existing code prefix at the preceding depth. These are relationships between the supplied coded MMS categories, not a reconstruction of the entire WHO Foundation ontology.

All 50 frozen SapBERT/BM25 candidates stay in their original order. The first ten seed catalogue expansion: coded ancestors, residual variants of those families, and immediate children. Round-robin expansion adds at most 30 codes, so a large family cannot consume the prompt. Every title, residual flag and parent comes from the supplied catalogue. No disease-specific rule, synthetic example, model training, WHO API service or graph database is introduced.

Each candidate is presented with an opaque selection ID (`C001`, `C002`, etc.), title, parent ID and category type. ICD code strings stay in Python, preventing the model from mixing codes with candidate numbers. One Qwen request selects three quoted selection IDs and a short exact note excerpt. The prompt distinguishes unspecified from other specified, asks for documented qualifiers before selecting specific children, and treats diagnosis-focus text only as a retrieval hint. Temperature 0, seed 42, the same Q4 model, and 8,192-token context are retained; the output budget is 384 tokens to accommodate the excerpt. This is one inference call per note, with no second pass or retry.

Selection IDs must be distinct and belong to the expanded candidate list; Python maps them to exact ICD codes. Invalid model output uses the saved SapBERT ranking and continues, with an explicit fallback count. The evidence excerpt is limited to one contiguous phrase of at most 120 characters, checked literally against the note and flagged separately if invalid; valid predictions are retained. A matching excerpt establishes traceability, not clinical correctness or proof that absent qualifiers are absent.

The runner writes `*_predictions.jsonl`, `*_trace.jsonl`, `*_run.json` and, after all 60 notes, `*_results.json`. Traces include the expanded code list, selection-ID mapping, added codes, raw model response and evidence flag. The original candidate file and catalogue are hash recorded. Gold labels never enter candidate expansion or prompts and are read only in the final scoring phase. Version 2 rejects resuming version 1 outputs; use a fresh prefix.

This remains a laptop prototype. The assessment's CPU-only Docker entry point and US$6 model-API allowance still need a separately measured deployment path; GPU throughput cannot establish grading feasibility.

Version 1 verification: all seven experiment tests and eleven existing retrieval tests passed. A real Qwen Q4 GGUF smoke check through llama.cpp b12144 on CPU produced three valid ranked responses with literal note excerpts, then one repeated-choice response. The laptop's complete version 1 trace subsequently showed 19/60 fallbacks caused by mixing candidate numbers with bare ICD codes, and scored 36/60 top-one, 47/60 top-three (49.58/75) after retrieval fallbacks. Version 2 changes the selection vocabulary instead of adding permissive code/number parsing. A new full laptop run is required before judging the hierarchy addition's accuracy.

Version 2 verification: seven experiment tests pass. Two targeted real Qwen CPU checks (dev-001 and dev-004, both previously malformed) now return distinct listed selection IDs and literal excerpts within the length budget. These checks establish response-contract behavior on those inputs, not an accuracy gain or clean operation across all 60 notes. The test server was stopped afterward.
