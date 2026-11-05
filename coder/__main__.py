"""Usage: python -m coder INPUT.jsonl OUTPUT.jsonl [--mms PATH]."""

import argparse
import json
from pathlib import Path
import sys
import time

from .bm25 import BM25


DEFAULT_MMS = Path(__file__).resolve().parent.parent / "data" / "mms-2026-01.json"


def read_notes(handle):
    seen = set()
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            note = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"input line {line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(note, dict) or not isinstance(note.get("id"), str):
            raise ValueError(f"input line {line_number}: id must be a string")
        if not isinstance(note.get("text"), str):
            raise ValueError(f"input line {line_number}: text must be a string")
        if note["id"] in seen:
            raise ValueError(f"input line {line_number}: duplicate id {note['id']!r}")
        seen.add(note["id"])
        # Ignore all other fields, including gold/style if present in a dev file.
        yield {"id": note["id"], "text": note["text"]}


def run(input_path: Path, output_path: Path, mms_path: Path) -> dict:
    if output_path.resolve() in {input_path.resolve(), mms_path.resolve()}:
        raise ValueError("output must not overwrite the input or catalogue")
    started = time.perf_counter()
    index = BM25.from_file(mms_path)
    valid = set(index.codes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = zero_overlap = 0
    with input_path.open(encoding="utf-8") as source:
        with output_path.open("w", encoding="utf-8") as destination:
            for note in read_notes(source):
                ranked = index.search(note["text"], limit=3)
                codes = [code for code, _ in ranked]
                if not codes or len(codes) != len(set(codes)) or not set(codes) <= valid:
                    raise ValueError(f"invalid internal prediction for {note['id']!r}")
                destination.write(json.dumps({"id": note["id"], "codes": codes}, ensure_ascii=False) + "\n")
                destination.flush()
                count += 1
                zero_overlap += ranked[0][1] == 0.0
    return {
        "notes": count,
        "catalogue_codes": len(valid),
        "zero_overlap_notes": zero_overlap,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Title-only BM25 ICD-11 baseline")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mms", type=Path, default=DEFAULT_MMS)
    args = parser.parse_args()
    try:
        summary = run(args.input, args.output, args.mms)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"coder: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
