"""Usage: python -m coder INPUT.jsonl OUTPUT.jsonl [--mms PATH]."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

from .bm25 import BM25
from .retriever import CandidateRetriever, HybridRetriever
from .semantic import DenseIndex


DEFAULT_MMS = Path(__file__).resolve().parent.parent / "data" / "mms-2026-01.json"
DEFAULT_SEMANTIC_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "semantic"
BATCH_SIZE = 32


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


def load_hybrid(index, asset_dir: Path, threads: int = 4):
    lexical = CandidateRetriever(index)
    english = DenseIndex(index, asset_dir / "english", threads)
    multilingual = DenseIndex(index, asset_dir / "multilingual", threads)
    return HybridRetriever(lexical, english, multilingual)


def run(
    input_path: Path,
    output_path: Path,
    mms_path: Path,
    strategy: str = "hybrid",
    semantic_assets: Path = DEFAULT_SEMANTIC_ASSETS,
    threads: int = 4,
) -> dict:
    if output_path.resolve() in {input_path.resolve(), mms_path.resolve()}:
        raise ValueError("output must not overwrite the input or catalogue")
    started = time.perf_counter()
    index = BM25.from_file(mms_path)
    retriever = CandidateRetriever(index)
    hybrid = None
    fallback_reason = None
    if strategy == "hybrid":
        try:
            hybrid = load_hybrid(index, semantic_assets, threads)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            fallback_reason = str(error)
            print(f"coder: semantic fallback: {error}", file=sys.stderr)
    valid = set(index.codes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = zero_overlap = fallback_batches = 0
    with input_path.open(encoding="utf-8") as source:
        notes = list(read_notes(source))
        with output_path.open("w", encoding="utf-8") as destination:
            for start in range(0, len(notes), BATCH_SIZE):
                batch = notes[start:start + BATCH_SIZE]
                if strategy == "hybrid" and hybrid is not None:
                    try:
                        batch_codes = hybrid.rank_many(
                            [note["text"] for note in batch], limit=3
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        print(f"coder: semantic batch fallback: {error}", file=sys.stderr)
                        fallback_batches += 1
                        batch_codes = [
                            [code for code, _ in retriever.focused_ranking(note["text"], 3)]
                            for note in batch
                        ]
                elif strategy in {"hybrid", "focused"}:
                    batch_codes = []
                    for note in batch:
                        ranked = retriever.focused_ranking(note["text"], limit=3)
                        batch_codes.append([code for code, _ in ranked])
                        zero_overlap += ranked[0][1] == 0.0
                elif strategy == "baseline":
                    batch_codes = []
                    for note in batch:
                        ranked = index.search(note["text"], limit=3)
                        batch_codes.append([code for code, _ in ranked])
                        zero_overlap += ranked[0][1] == 0.0
                else:
                    raise ValueError(f"unknown strategy {strategy!r}")
                for note, codes in zip(batch, batch_codes):
                    if not codes or len(codes) != len(set(codes)) or not set(codes) <= valid:
                        raise ValueError(f"invalid internal prediction for {note['id']!r}")
                    destination.write(
                        json.dumps({"id": note["id"], "codes": codes}, ensure_ascii=False) + "\n"
                    )
                    destination.flush()
                    count += 1
    return {
        "strategy": strategy,
        "semantic_enabled": hybrid is not None,
        "semantic_models": (
            [hybrid.english.model_id, hybrid.multilingual.model_id]
            if hybrid is not None else []
        ),
        "fallback_reason": fallback_reason,
        "fallback_batches": fallback_batches,
        "notes": count,
        "catalogue_codes": len(valid),
        "zero_overlap_notes": zero_overlap,
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid ICD-11 candidate retrieval")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mms", type=Path, default=DEFAULT_MMS)
    parser.add_argument(
        "--strategy", choices=("hybrid", "focused", "baseline"), default="hybrid"
    )
    parser.add_argument("--semantic-assets", type=Path, default=DEFAULT_SEMANTIC_ASSETS)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        summary = run(
            args.input, args.output, args.mms, args.strategy,
            args.semantic_assets, args.threads,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"coder: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
