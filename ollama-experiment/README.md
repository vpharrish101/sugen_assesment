# Ollama ICD reranking experiment

Install [Ollama](https://ollama.com/) and Python 3, then pull the model:

```sh
ollama pull medgemma1.5:4b
```

Run from this directory with Ollama running:

```sh
python run_ollama_experiment.py --limit 4 --output-prefix medgemma-check
python run_ollama_experiment.py --output-prefix medgemma-full
```

This MedGemma run keeps the model loaded for 30 minutes and uses the original 8,192-token context, matching the settings in the script used for your first six notes. MedGemma naturally returned ICD codes on dev-006, so the prompt now asks for exact listed codes. The runner accepts either listed codes or numbered positions and rejects unknown or repeated codes. If a response is invalid, it retries once. It stops without saving a fallback prediction if the retry also fails, so `--resume` can retry the same note later.

The script uses the frozen SapBERT candidates and supplied 60-note dev set. It writes predictions, trace, and results files with your chosen prefix. `--resume` keeps valid completed predictions and retries interrupted or invalid rows, including fallback rows saved by the old runner. Use a fresh prefix for a clean MedGemma comparison. No Python packages or API keys are needed. The first four MedGemma responses you supplied were replayed successfully in a local smoke test; no MedGemma model is available in this cloud environment for live GPU inference.

If the interrupted command was just `python run_ollama_experiment.py`, resume its default `ollama_*` files with `python run_ollama_experiment.py --resume`. The output prefix must match the interrupted run. Previously successful notes will be kept; the failed dev-006 row will be retried. Its second response, `1E90.0`, `1E90`, `1E90.2`, is now recognized as three valid catalogue candidates.

Gold labels are used only for scoring after inference. A model error or invalid choice stops the run and leaves the note unfinished; the script will retry that note on resume. The 50-candidate pool contains a gold code for 59 of the 60 notes. The SapBERT baseline is 27/60 top-one and 44/60 top-three (40.8/75).
