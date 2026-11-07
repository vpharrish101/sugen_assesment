# Qwen3 4B reranking over hybrid retrieval

This branch preserves the corresponding prototype. See `final-prototype` or
`main` for the final API-first application. Public catalogue only: supply your
own notes/labels in ignored `private-data/` (or CODER_INPUT_DIR), never in Git.

```sh
mkdir -p private-data out
docker compose build coder
docker compose run --rm coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold private-data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```


Ollama must run separately with the selected model. Set QWEN_MODEL/QWEN_ENDPOINT
as needed. For host Ollama use -f compose.yaml -f compose.qwen.yaml (host
networking; Docker Desktop must enable it). Each note gets fresh candidates;
no frozen candidate file or labels are used at inference. This branch's Docker
entrypoint runs its named prototype, not retrieval-only code.

Tests use small protocol fixtures; clinical input for a smoke test is mounted
privately and limited externally to three notes. Runtime output stays in ignored
out/. Models/indexes are built from public catalogue titles, not consultation
notes. No training, API credentials or private notes are embedded in the image.
Original aggregate benchmark results are recorded in the final branch README.
