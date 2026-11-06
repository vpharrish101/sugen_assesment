# Ollama ICD reranking experiment

Install [Ollama](https://ollama.com/) and Python 3, then pull the model:

```sh
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

Run from this directory with Ollama running:

```sh
python run_ollama_experiment.py --limit 3 --output-prefix check
python run_ollama_experiment.py --output-prefix full
```

Run the full experiment only if the first three finish without a fallback. The script uses frozen SapBERT candidates and the supplied 60-note dev set. It writes `full_predictions.jsonl`, `full_trace.jsonl`, and `full_results.json` here. Check `ollama ps` during inference to verify GPU use. No Python packages or API keys are needed. This Ollama adapter has not yet been tested on a GPU.

To continue an interrupted run, use `python run_ollama_experiment.py --output-prefix full --resume`. The script stops after an invalid response and preserves the error in the trace. Do not resume that failed run until the error is fixed, since the fallback prediction is already saved. On Windows, use `py` if `python` is unavailable.

Gold labels are used only for scoring after inference. A model error or invalid choice falls back to the original SapBERT ranking, and the result report counts fallbacks. The 50-candidate pool contains a gold code for 59 of the 60 notes. The SapBERT baseline is 27/60 top-one and 44/60 top-three (40.8/75).
