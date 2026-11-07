# Setup and private inputs

Only the public ICD-11 catalogue is tracked. Keep consultation notes, labels,
translations, candidates, predictions and traces out of Git. Supply your original
assessment files locally in `private-data/` or set `CODER_INPUT_DIR` to another
private directory. This directory and every JSONL file are Git-ignored and
excluded from image builds. Gold labels are used only by `score.py` after inference.

```sh
docker compose build coder
docker compose run --rm coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold private-data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```

The image builds pinned encoder title vectors from the public catalogue, never
from clinical notes. The grading box needs no GPU or runtime downloads. Compose
forwards ANTHROPIC_API_KEY and HTTPS_PROXY. The API model defaults to
claude-haiku-5-5. No key means configured Qwen, then offline retrieval. With a key,
Claude is first; provider failures use Qwen then retrieval. Budget reservation
is $5.50 by default, never configured above $6; inference is bounded at 3,300s.
OpenAI-only credentials are not used. A genuine Claude account/model still needs
live access validation; simulated protocol checks cannot establish that.

For host Ollama set QWEN_BACKEND=ollama in .env and use:

```sh
docker compose -f compose.yaml -f compose.qwen.yaml run --rm coder /in/notes.jsonl /out/predictions.jsonl
```

The optional override enables host networking only. Docker Desktop requires host
networking enabled; Linux Docker supports it directly. --backend ollama forces
Qwen; omit it for automatic API-first routing. The measured local model is
hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M. For optional llama.cpp use
--cache-ram 0 --no-cache-idle-slots with one slot; this bounds prompt caching,
not total process RAM. Python client cleanup cannot free a server's model weights.

Native entry point: python run.py INPUT OUTPUT. Install declared requirements and
CPU/GPU Torch, and supply the built assets locally. Unit checks use
PYTHONPATH=src python -m unittest discover -s tests. Real-encoder smoke checks are
optional: set CODER_TEST_ASSETS and CODER_TEST_NOTES to private local paths. They
read at most three supplied notes. No clinical fixture is required in Git.
