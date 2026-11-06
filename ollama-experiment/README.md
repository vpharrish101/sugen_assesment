# Qwen ICD-11 reranking prototype

This folder contains the frozen 60-note, 50-candidate experiment. The Qwen3 4B Q4 model reranked the supplied catalogue candidates on the user's RTX 4060 laptop with **38/60 top-one, 49/60 top-three, 52.08/75**, zero fallbacks, and 38.2 seconds total inference time. The same retrieval baseline scored 27/60, 44/60, 40.83/75. MedGemma 1.5 4B scored 31/60, 52/60, 47.50/75 in a separate local run. These are development results, not hidden-set estimates.

## Reproduce the measured Ollama run

```sh
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
python run_ollama_experiment.py --limit 4 --output-prefix qwen-check
python run_ollama_experiment.py --output-prefix qwen-full
```

The Ollama script restores the exact Qwen prompt, `format: json`, seed 42, 8,192-token context and 256-token output limit from the measured run. It uses no API key and writes `*_predictions.jsonl`, `*_trace.jsonl`, and, after all 60 notes, `*_results.json`. Choose a fresh prefix; do not combine Qwen and MedGemma outputs. Gold labels are read only for the final score. The frozen input SHA256 is `524d4aa5ea7d55ea76d2a42b2b9fed6472c8dda01824fbc30e91d5b723824045`.

## llama-server with bounded RAM

The separate llama.cpp path uses the same Qwen GGUF but is **not** the measured 38/60 Ollama run. On llama.cpp build `b12144`, `llama-server --help` reports `--cache-ram` defaulting to **8192 MiB**. This stores prompt checkpoints from distinct requests and can look like a memory leak. Start it with the bundled launcher, which disables that cache and idle-slot caching, limits context to 4,096 tokens, uses one slot, and bounds batch sizes:

```sh
LLAMA_SERVER_BIN=/path/to/llama-server ./start_llama_server.sh /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
python test_local_reranker.py --input candidates.jsonl --output qwen-llama-predictions.jsonl
```

The launcher defaults to GPU offload; set `LLAMA_GPU_LAYERS=0` for CPU. The client also sends `cache_prompt: false`. On the cloud CPU host, ten distinct roughly 440-token prompts produced **4,715 MiB RSS after every request**, with `--cache-ram 0`; this checks for growth across ten requests, not a guarantee on the laptop or over all 60 notes. Use `ps -C llama-server -o pid,rss,cmd` to watch RSS during a longer run. If a different build does not recognize `--cache-ram`, update llama.cpp rather than silently running without a bound.

## Assessment boundary

The [brief](../BRIEF.md) grades `docker compose run coder /in/notes.jsonl /out/predictions.jsonl` on **600 unseen notes, 8 CPUs, 16 GiB RAM, no GPU, and 60 minutes**. Runtime networking is limited to OpenAI/Anthropic model APIs, with a shared **US$6** cap. The local Qwen/Ollama and llama-server experiments are not part of that Docker entry point and their GPU timing cannot establish CPU grading feasibility. The current Docker submission path remains the offline SapBERT/BM25 retriever with a no-key fallback. The hidden set also spans more ICD-11 chapters and languages than the 60 English development notes.

Before final submission, the brief separately requires a clean-clone run of the exact Docker command, one output for every ID, an official `score.py` result, evaluation beyond the 60 development labels, a one-page root README with a hidden-score estimate, source data and AI-error disclosure, a contemporaneous `DECISIONS.md`, hours spent, and a `git bundle --all` delivered by email. The Qwen score is evidence for the prototype; it must not replace the deployable runner's score in that README until a compliant runtime path is built and measured.
