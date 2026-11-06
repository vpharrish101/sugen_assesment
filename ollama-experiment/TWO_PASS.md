# Two-pass Qwen prototype

Local branch: `qwen-two-pass-prototype`. This is an addition alongside the original Qwen and single-pass hierarchy scripts. It is not integrated into the CPU-only Docker submission path and has no new dependencies.

The original Qwen baseline reached 38/60 top-one, 49/60 top-three, 52.08/75. The completed single-pass hierarchy v2 run reached 38/60, 43/60, 49.58/75, with zero code-format fallbacks and 55/60 valid excerpts in 86.2 seconds. The new experiment therefore preserves the original first pass and measures whether a smaller second comparison helps.

## Run

Use the same Qwen model already installed in Ollama. To generate both passes for each note:

```sh
python run_two_pass_experiment.py --limit 4 --output-prefix two-pass-check
python run_two_pass_experiment.py --output-prefix two-pass-full
```

To test the verifier against your existing **original Qwen** 38/49 predictions instead of repeating pass one:

```sh
python run_two_pass_experiment.py --first-pass-predictions /path/to/ollama_predictions.jsonl --limit 4 --output-prefix verifier-check
python run_two_pass_experiment.py --first-pass-predictions /path/to/ollama_predictions.jsonl --output-prefix verifier-full
```

Do not use the single-pass hierarchy predictions as that file. Saved first-pass predictions must contain all 60 IDs exactly once, with three distinct codes from each original retrieval pool. They contain no evidence or gold labels. Resume an interrupted run with the same options, prefix and `--resume`; changing the saved file, model or inputs is rejected.

## Flow

1. The first pass uses exactly the original Qwen prompt and generation parameters on the same frozen 50-candidate pool. A protocol test compares the complete request with the original runner. If pass one fails, its saved retrieval ranking is used and the failure is counted.
2. Python keeps that pass's three codes and the first ten retrieval alternatives, deduplicates them, and adds nearby coded ancestors, residual categories and children. The verifier sees at most 24 candidates, with expansion seeded by the three first-pass choices.
3. One second Qwen request reviews the provisional ranking against the full note. Candidates use opaque IDs, titles and parent links; Python retains the ICD-code mapping. The verifier must return three distinct listed IDs and a literal supporting excerpt of at most 120 characters.
4. If the verifier returns invalid choices or an invalid excerpt, the output remains the **first-pass ranking**, rather than dropping back to SapBERT. A literal excerpt provides traceability; it does not independently establish clinical correctness. Valid verification may change all three codes, so top-three accuracy can still fall and must be measured.

There are no verifier retries. Fresh mode uses two inference calls per note; saved-first-pass mode uses only the one verifier call. The latter's timing excludes the earlier Qwen run and must not be reported as total two-pass latency.

## Evaluation

Files with the selected prefix hold final predictions, traces, run configuration and results. Traces include first-pass codes/errors, verifier errors, both raw responses, candidate codes, evidence and phase timings. Results compare retrieval, actual first pass and final top-one/top-three scores; they list note-level gains/losses, fallback counts and inference calls. Gold is read only after inference completes. This lets us detect a verifier that fixes some notes while damaging others, rather than assume two passes improve accuracy.

Local checks: eleven experiment tests pass, including equivalence with the original first request and a simulated 60-note two-pass run with resume. Two real Qwen GGUF verifier checks through llama.cpp b12144 on CPU (using the supplied original rankings for dev-050 and dev-057) returned valid distinct IDs and literal excerpts. In both, the previously missing unspecified variant entered the top three, while the first choice remained the parent category. These two targeted cases are not a full accuracy estimate. The CPU smoke server was stopped afterward; the Ollama/GPU full run is still outstanding.
