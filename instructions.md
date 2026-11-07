# Running the final prototype

Run commands from the repository root. Use the Anthropic path first. Ollama is
optional; no separate llama-server installation or launch is needed.

Keep input notes and gold labels in `private-data/`, and predictions in `out/`.
Neither directory should be committed. The examples below assume your input is
`private-data/notes.jsonl`. Replace that filename with your own.

## 1. Anthropic API — Docker

Copy the environment template:

```bash
cp .env.example .env
mkdir -p private-data out
```

Edit `.env`:

```dotenv
ANTHROPIC_API_KEY=your_actual_key
ANTHROPIC_MODEL=claude-haiku-5-5
QWEN_BACKEND=
```

Use a model enabled for your Anthropic account. If you change `ANTHROPIC_MODEL`,
also set `ANTHROPIC_INPUT_USD_PER_MILLION` and
`ANTHROPIC_OUTPUT_USD_PER_MILLION` to its current prices. The application requires
explicit prices for a different model. Do not commit your key or `.env`.

Build and run:

```bash
docker compose build coder
docker compose run --rm coder /in/notes.jsonl /out/predictions.jsonl
```

The build downloads the pinned English and multilingual query encoders. It
reuses the bundled public catalogue embeddings, so it does not encode the
catalogue again. No GPU is required for this path.

Automatic routing is Anthropic first, configured local Qwen second, retrieval
last. With `QWEN_BACKEND` empty, API failures fall back to retrieval. A completed
output file alone does not prove Claude was used: check the summary printed to
stderr. `routes`, `api_calls` and `provider_errors` show which path ran; check
`semantic_enabled` too. Genuine API access still depends on a valid key, account
credit and access to the configured model. Previous simulated API tests did not
establish live account access or Claude accuracy.

## 2. Optional Ollama

Install Ollama if needed and keep its service running. Pull the measured model:

```bash
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

### Apply the Ollama memory fix before the experiment

On the tested Linux system, Ollama 0.33.3 launched its own `llama-server` runner.
Saved prompt states accumulated in host RAM and swap across requests. Disabling
the cache in the runner's environment resolved the observed growth. Python
garbage collection cannot release memory owned by this separate process.

For a Linux systemd installation, stop any active experiment and run:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d

printf '[Service]\nEnvironment="LLAMA_ARG_CACHE_RAM=0"\nEnvironment="LLAMA_ARG_CACHE_IDLE_SLOTS=0"\nEnvironment="LLAMA_ARG_CTX_CHECKPOINTS=0"\n' \
  | sudo tee /etc/systemd/system/ollama.service.d/cache.conf

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

These variables must be set on **the Ollama service**, not just the application's
`.env`. Our first attempt had no effect because they never reached the runner.

Once a request has loaded the model, verify propagation:

```bash
sudo sh -c 'for p in $(pgrep -u ollama -x llama-server); do
  tr "\0" "\n" < /proc/$p/environ | grep "^LLAMA_ARG_"
done'
```

Expect all three variables with value `0`. This command assumes the service
runs as user `ollama`. For other installations, set the same variables in the
environment that starts `ollama serve`, then restart it. Support depends on the
bundled runner version; the workaround is described in
[Ollama issue #18264](https://github.com/ollama/ollama/issues/18264).

This disables saved prompt caching and checkpoints; it is not a total RAM cap.
Model weights, the active KV cache and inference buffers still require memory.
The app container's memory limit does not constrain Ollama running on the host.

### Run through Ollama

Set these fields in `.env`:

```dotenv
QWEN_BACKEND=ollama
QWEN_MODEL=hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
QWEN_ENDPOINT=http://127.0.0.1:11434
```

Force Ollama for a local test, even if an Anthropic key is present:

```bash
docker compose -f compose.yaml -f compose.qwen.yaml run --rm coder \
  /in/notes.jsonl /out/predictions.jsonl --backend ollama
```

Omit `--backend ollama` to retain Anthropic-first routing with Ollama as fallback.
The Compose override uses host networking to reach host Ollama. Linux supports
this directly; Docker Desktop requires host networking to be enabled.

## 3. Native Python, without Docker

Use Python 3.12. Create a virtual environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
python -m pip install -r requirements.txt -r requirements-api.txt
mkdir -p private-data out
```

Prepare the query encoders once. These commands reuse the bundled catalogue
matrices; encoder/tokenizer downloads are still needed:

```bash
python -m scripts.build_semantic_assets \
  --catalogue data/mms-2026-01.json \
  --output assets/semantic/english \
  --model cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --revision 090663c3ae57bf35ffe4d0d468a2a88d03051a4d

python -m scripts.build_semantic_assets \
  --catalogue data/mms-2026-01.json \
  --output assets/semantic/multilingual \
  --model cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR \
  --revision 47b6bd041ba61311584bb2494edfda5c7d9b719f

python scripts/finalize_tokenizers.py
```

Native Python does not automatically load `.env`. For Anthropic, export the
settings in your terminal (and the two token prices above if changing models).
Enter the key without placing it in shell history:

```bash
read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY; echo
export ANTHROPIC_API_KEY
export ANTHROPIC_MODEL=claude-haiku-5-5
python run.py private-data/notes.jsonl out/predictions.jsonl --backend anthropic
```

For an Ollama-only run, apply the service fix above first, then:

```bash
python run.py private-data/notes.jsonl out/predictions.jsonl \
  --backend ollama \
  --endpoint http://127.0.0.1:11434 \
  --model hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

## 4. Evaluate the output

If gold labels are available locally:

```bash
python score.py \
  --gold private-data/dev_notes.jsonl \
  --pred out/predictions.jsonl \
  --mms data/mms-2026-01.json
```

Use the gold file corresponding to the input notes. Each prediction contains the
note ID and up to three ranked catalogue codes. Gold labels are used only for
evaluation, not retrieval or reranking.
