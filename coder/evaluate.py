"""Report retrieval recall; official accuracy is calculated by unmodified score.py."""

import argparse
import json
from pathlib import Path
import time

from .__main__ import DEFAULT_MMS, read_notes
from .bm25 import BM25
from .retriever import CandidateRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--mms", type=Path, default=DEFAULT_MMS)
    parser.add_argument("--strategy", choices=("focused", "baseline"), default="focused")
    args = parser.parse_args()
    started = time.perf_counter()
    index = BM25.from_file(args.mms)
    retriever = CandidateRetriever(index)
    with args.gold.open(encoding="utf-8") as source:
        gold_rows = [json.loads(line) for line in source if line.strip()]
    if not gold_rows:
        parser.error("gold file contains no notes")
    hits = {k: 0 for k in (1, 3, 25, 50)}
    per_note = []
    # Retrieval gets only id/text. Gold is consulted after ranking to measure recall.
    for row in read_notes(json.dumps(row, ensure_ascii=False) for row in gold_rows):
        if args.strategy == "baseline":
            ranked = [code for code, _ in index.search(row["text"], limit=50)]
        else:
            ranked = retriever.candidate_pool(row["text"], limit=50)
        per_note.append((row["id"], ranked))
    gold_by_id = {row["id"]: set(row["gold"]) for row in gold_rows}
    for note_id, ranked in per_note:
        for k in hits:
            hits[k] += bool(set(ranked[:k]) & gold_by_id[note_id])
    print(json.dumps({
        "strategy": args.strategy,
        "notes": len(gold_rows),
        "hits": hits,
        "recall": {k: count / len(gold_rows) for k, count in hits.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
