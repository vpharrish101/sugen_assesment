# Ollama ICD reranking experiment

Install [Ollama](https://ollama.com/) and Python 3, then pull the model:

```sh
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

Run from this directory with Ollama running:

```sh
python run_ollama_experiment.py --limit 3 --output-prefix check2
python run_ollama_experiment.py --output-prefix full --resume
```

The previous runner kept the model loaded for 30 minutes and used an 8,192-token context. This version uses a 3,072-token context, a 64-token processing batch, and `keep_alive: 0`, which asks Ollama to release the model after *each* note. Loading it every time is slower but limits persistent GPU use. Python garbage collection cannot release memory owned by the separate Ollama process. If the desktop still freezes, run `python run_ollama_experiment.py --output-prefix full --resume --cpu` to continue on CPU; this is much slower. Close other GPU-heavy applications during the GPU run.

The script uses the frozen SapBERT candidates and the supplied 60-note dev set. It writes `full_predictions.jsonl`, `full_trace.jsonl`, and `full_results.json` here. `--resume` keeps completed predictions and retries interrupted or invalid rows, including fallback rows saved by the old runner. If you resume an old run, its final score combines outputs from two prompt versions; `runner_versions` in the report will show `[1, 2]`. Use a fresh prefix and all 60 notes for a clean comparison. `ollama ps` shows active model use while a note is running; the model should unload after each note. No Python packages or API keys are needed. This Ollama adapter has been tested with a mocked Ollama API, not on a GPU.

If the interrupted command was just `python run_ollama_experiment.py`, resume its default `ollama_*` files with `python run_ollama_experiment.py --resume`. The output prefix must match the interrupted run.

Gold labels are used only for scoring after inference. A model error or invalid choice stops the run and leaves the note unfinished; the script will retry that note on resume. The 50-candidate pool contains a gold code for 59 of the 60 notes. The SapBERT baseline is 27/60 top-one and 44/60 top-three (40.8/75).
