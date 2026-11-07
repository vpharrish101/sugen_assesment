# Diagnosis-focused BM25 with catalogue-family candidates

This branch preserves the corresponding prototype. See `final-prototype` or
`main` for the final API-first application. Public catalogue only: supply your
own notes/labels in ignored `private-data/` (or CODER_INPUT_DIR), never in Git.

```sh
mkdir -p private-data out
docker compose build coder
docker compose run --rm coder /in/notes.jsonl /out/predictions.jsonl
python score.py --gold private-data/dev_notes.jsonl --pred out/predictions.jsonl --mms data/mms-2026-01.json
```


Tests use small protocol fixtures; clinical input for a smoke test is mounted
privately and limited externally to three notes. Runtime output stays in ignored
out/. Models/indexes are built from public catalogue titles, not consultation
notes. No training, API credentials or private notes are embedded in the image.
Original aggregate benchmark results are recorded in the final branch README.
