# Running the final prototype

Run commands from the repository root in Bash. Use the Anthropic path first. Ollama is
optional; no separate llama-server installation or launch is needed.

Keep input notes and gold labels in `private-data/`, and predictions in `out/`.
Neither directory should be committed. The examples below assume your input is
`private-data/notes.jsonl`. Replace that filename with your own.
Input is JSONL: one object per line with a string `id` and the consultation note
in a string `text` field. IDs must be unique within the file.

Docker mounts `private-data/` as `/in` (read-only) and `out/` as `/out`.
For example, `private-data/notes.jsonl` on your machine becomes `/in/notes.jsonl`
inside the container. To use other host directories, set `CODER_INPUT_DIR` and
`CODER_OUTPUT_DIR` in `.env`; container paths remain `/in` and `/out`.

## 1. Anthropic API — Docker

Install Docker with the Compose plugin. Copy the environment template once
(keep your existing `.env` if already configured):

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

Before building, download the complete `semantic/` folder from
[Google Drive](https://drive.google.com/drive/folders/14W-Rx1LamEf1SWjIal2o-ODjqUqciBfQ?usp=sharing)
and place it at `assets/semantic/`. Both `english/` and `multilingual/` must
contain `model/`, `metadata.json` and `title_embeddings.npy`. The matrices alone
are not enough. Alternatively, generate these assets using section 3 first.

Build and run:

```bash
docker compose build coder
docker compose run --rm coder /in/notes.jsonl /out/predictions.jsonl
```

The build copies the prepared encoders and catalogue index. It does not download
models or regenerate embeddings. Internet access is still needed to install
Python dependencies on the first build. Pip has a 180-second read timeout and
10 connection retries; these do not guarantee recovery from every interrupted
download. No GPU is required for this path.

The tokenizer loader supplies `extra_special_tokens={}` for these SapBERT
encoders. This handles copied tokenizer metadata that serializes the field as
an empty list, which Docker's pinned Transformers version cannot load directly.

To check asset loading before inference:

```bash
docker compose run --rm --entrypoint python coder -c '
from coder.__main__ import BM25, load_hybrid, DEFAULT_MMS, DEFAULT_SEMANTIC_ASSETS
load_hybrid(BM25.from_file(DEFAULT_MMS), DEFAULT_SEMANTIC_ASSETS)
print("Both semantic encoders loaded successfully.")
'
```

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

Build the app image using `docker compose build coder` if you skipped section 1.
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

Download the prepared semantic assets from
[Google Drive](https://drive.google.com/drive/folders/14W-Rx1LamEf1SWjIal2o-ODjqUqciBfQ?usp=sharing).
Place the downloaded `semantic/` folder under `assets/`, so both
`assets/semantic/english/` and `assets/semantic/multilingual/` contain `model/`,
`metadata.json` and `title_embeddings.npy`. With the complete assets in place,
run `python scripts/finalize_tokenizers.py`, then skip the encoder preparation
commands below and proceed to running the app.
The catalogue index stays the same for 60 or 600 notes; it is not rebuilt per
input dataset. The same downloaded assets work for both native Python and Docker.

Alternatively, prepare the query encoders and catalogue vectors once with the
commands below. They download the pinned encoders and generate the title
matrices, or reuse matching bundled matrices when supported by your checkout:

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

To score inside Docker instead:

```bash
docker compose run --rm --entrypoint python coder score.py \
  --gold /in/dev_notes.jsonl \
  --pred /out/predictions.jsonl \
  --mms data/mms-2026-01.json
```
