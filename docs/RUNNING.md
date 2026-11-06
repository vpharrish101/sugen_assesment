# Running and configuration

The main pipeline reads fresh `id,text` notes, extracts a diagnosis-focused retrieval query, detects language, retrieves catalogue candidates and ranks them. English keeps the measured 3:1 English/multilingual SapBERT fallback and the original English SapBERT + BM25 candidate ordering. Other languages use multilingual SapBERT, with Spanish/Chinese structural extraction and only positive BM25 votes. Each provider sees the **whole original note**, the focus hint and 50 numbered code/title pairs. Providers cannot return codes outside those candidates.

`src/coder/` contains the pipeline, retrieval and shared prompt. `src/coder/providers/` contains the Anthropic and local inference clients. Settings are in `config/default.json`, provider settings in environment variables. `assets/semantic/` is generated during image build; binaries are not committed. Supplied and translated evaluation data stay in `data/`, historical experiments in `experiments/`, and measurements in `reports/`.

## Assessment / API first

```sh
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

Compose passes `ANTHROPIC_API_KEY` and `HTTPS_PROXY` from the environment. `.env.example` lists supported settings; copy to `.env` for Compose. Native Python reads exported environment variables, not `.env`. The API client uses the official SDK, the official HTTPS endpoint, normal certificate verification, zero SDK retries, and a forced ranking tool whose numbers are validated. No API response is interpreted as a newly generated ICD code.

Automatic routing is **Anthropic → configured local Qwen → retrieval**. With no API key and no Qwen configured, retrieval remains fully offline. Only API calls leave the machine; models/indexes are downloaded during build. An OpenAI-only key is not used by this implementation. API accuracy has not yet been measured; the local Qwen score is not an API score.

`ANTHROPIC_MODEL=claude-haiku-5-5` is the requested primary model. `CODER_API_BUDGET_USD` defaults to 5.5 and cannot exceed 6. Conservative reservation rates default to $0.50/M input and $2.50/M output, above the normal-context $0.10/$0.50 values in [LiteLLM's registry](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), checked 2026-11-06. The provider pricing page was inaccessible through this environment's policy, so check current provider pricing before submission. A different model requires explicit rate settings. The client reserves UTF-8 input bytes plus overhead as an overestimate of input tokens, plus all 256 possible output tokens; failed calls keep their reservation. This is deliberately conservative accounting, not actual invoice usage. The grading key's $6 cap remains the external billing limit.

Inference stops consuming model time after 3,300 seconds, leaving time for retrieval fallbacks. Each provider is skipped after three consecutive failures. Errors never stop the remaining notes. Predictions flush after each note. The default container limit is four CPUs and 6 GB; the grader's eight CPUs/16 GB are sufficient for the tested offline path. Qwen is optional and is not downloaded/launched in the grading image.

## Local Qwen on your laptop

Keep the existing Python environment with CPU/GPU PyTorch, install `requirements.txt` and `requirements-api.txt`, and make the image-built semantic assets available locally. For example, after building:

```sh
docker compose create coder
docker cp "$(docker compose ps -aq coder)":/app/assets/semantic ./assets/semantic
```

Run Ollama and pull the measured model once:

```sh
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

For API first with local fallback, export `ANTHROPIC_API_KEY`, `QWEN_BACKEND=ollama`, `QWEN_MODEL=hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M`, and `QWEN_ENDPOINT=http://127.0.0.1:11434`, then:

```sh
python run.py data/notes.jsonl out/predictions.jsonl --trace out/trace.jsonl
```

For Qwen only, use `--backend ollama`. For offline retrieval only, use `--backend none`. These use the same clinical prompt/candidate retrieval; no hierarchy or second reasoning pass is added. The unchanged frozen GPU experiment remains runnable from `experiments/ollama/`.

The default Qwen endpoint is loopback. A Docker container's loopback is its own namespace, so use native Python for your laptop's Ollama service. Do not expect a grader container to reach your laptop's GPU.

## llama.cpp memory

For the optional llama.cpp transport, start the server with `--cache-ram 0 --no-cache-idle-slots -np 1 -c 8192 -b 256 -ub 128`; use `--backend llama --endpoint http://127.0.0.1:8097`. The client also sends `cache_prompt:false`. These settings disable the growing RAM prompt cache; they are not a hard limit on total process RAM. Model weights and the bounded KV cache still occupy memory. The existing Python-managed launcher in `experiments/ollama/test_local_reranker.py` starts/stops a server with bounded caching, as before. Ollama controls its own server memory; clearing Python responses cannot free another process's model/KV allocations.

## Verification

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Provider tests use the **real Anthropic SDK with MockTransport**, not paid inference: success, 429/no retry, forced-tool serialization, proxy configuration, budget exhaustion and complete fallback coverage. A live API check still requires a configured key and allowed API egress. Translation fixtures are evaluation data, not training material; see `data/translation_eval/README.md`. Detailed final results and unmet submission items are in `docs/BRIEF_CHECK.md`.

## Docker with host Ollama

Set `QWEN_BACKEND=ollama`, `QWEN_MODEL` and `QWEN_ENDPOINT=http://127.0.0.1:11434` in `.env`. Run the existing host Ollama service with the model installed. Use the optional override:

```sh
docker compose -f compose.yaml -f compose.qwen.yaml run --rm coder /in/notes.jsonl /out/predictions.jsonl
```

The default automatic route remains Claude → configured Qwen → retrieval; with no Claude key it starts with Qwen. Add `--backend ollama` for Qwen-only testing. The override changes networking only and keeps model/key settings in `.env`. Linux Docker supports host networking directly; Docker Desktop requires a supported version with host networking enabled. Without this override, the container's localhost does not reach host Ollama. The ordinary grading/API-only Compose invocation stays unchanged.
