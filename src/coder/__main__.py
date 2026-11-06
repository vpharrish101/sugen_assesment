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


DEFAULT_MMS = Path(__file__).resolve().parents[2] / "data" / "mms-2026-01.json"
DEFAULT_SEMANTIC_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "semantic"
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
    *, backend="auto", model="hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M",
    endpoint="http://127.0.0.1:11434", timeout_seconds=45, max_seconds=3300,
    batch_size=32, candidate_limit=50, trace_path=None,
) -> dict:
    from contextlib import ExitStack
    from .reranker import Reranker

    if output_path.resolve() in {input_path.resolve(), mms_path.resolve()}:
        raise ValueError("output must not overwrite the input or catalogue")
    if trace_path and trace_path.resolve() in {input_path.resolve(), mms_path.resolve(), output_path.resolve()}:
        raise ValueError("trace must not overwrite the input, output or catalogue")
    if backend not in {"auto", "none", "anthropic", "ollama", "llama"}:
        raise ValueError("unknown reranking backend")
    if not (1 <= threads <= 8 and 1 <= batch_size <= 32 and 3 <= candidate_limit <= 100):
        raise ValueError("threads must be 1..8, batch size 1..32, candidates 3..100")
    if not (0 < timeout_seconds <= 600 and 0 < max_seconds <= 3500):
        raise ValueError("timeout must be 0..600s and inference budget 0..3500s")
    if strategy != "hybrid":
        if backend not in {"auto", "none"}:
            raise ValueError("reranking requires the hybrid strategy")
        backend = "none"
    started = time.perf_counter()
    with input_path.open(encoding="utf-8") as source:
        notes = list(read_notes(source))
    index = BM25.from_file(mms_path)
    retriever = CandidateRetriever(index)
    hybrid, fallback_reason = None, None
    if strategy == "hybrid":
        try:
            hybrid = load_hybrid(index, semantic_assets, threads)
        except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
            fallback_reason = str(error)
            print(f"coder: semantic fallback: {error}", file=sys.stderr)
    reranker = Reranker(backend, model, endpoint, timeout_seconds) if backend != "none" else None
    valid = set(index.codes)
    titles = {row['code']: row['title'] for row in index.records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = fallback_batches = reranked = llm_failures = 0
    disabled_reason = None
    with ExitStack() as stack:
        if reranker:
            stack.callback(reranker.close)
        destination = stack.enter_context(output_path.open("w", encoding="utf-8"))
        trace = None
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace = stack.enter_context(trace_path.open("w", encoding="utf-8"))
        for start in range(0, len(notes), batch_size):
            batch = notes[start:start + batch_size]
            batch_fallbacks = pools = None
            if hybrid is not None:
                try:
                    batch_fallbacks, pools = hybrid.retrieve_many([note['text'] for note in batch], candidate_limit)
                    if len(batch_fallbacks) != len(batch) or len(pools) != len(batch):
                        raise ValueError('semantic retriever returned the wrong number of notes')
                    for codes in batch_fallbacks + pools:
                        if not codes or len(set(codes)) != len(codes) or not set(codes) <= valid:
                            raise ValueError('semantic retriever returned invalid candidates')
                    batch_fallbacks = [codes[:3] for codes in batch_fallbacks]
                except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
                    print(f"coder: semantic batch fallback: {error}", file=sys.stderr)
                    fallback_batches += 1
                    batch_fallbacks = pools = None
            if batch_fallbacks is None:
                rank = index.search if strategy == 'baseline' else retriever.focused_ranking
                batch_fallbacks = [[code for code, _ in rank(note['text'], 3)] for note in batch]
                pools = [retriever.candidate_pool(note['text'], candidate_limit) for note in batch]
            for note, codes, pool in zip(batch, batch_fallbacks, pools):
                fallback = codes
                error = None
                remaining = max_seconds - (time.perf_counter() - started)
                if reranker is not None and disabled_reason is None:
                    if remaining <= 1:
                        disabled_reason = 'inference time budget reached'
                    else:
                        ranked = reranker.rerank(note['text'], pool, titles, timeout=remaining)
                        if ranked is not None:
                            codes = ranked
                            reranked += 1
                        else:
                            error = dict(reranker.errors) or None
                            llm_failures += bool(error)
                if not codes or len(codes) > 3 or len(codes) != len(set(codes)) or not set(codes) <= valid:
                    raise ValueError(f"invalid internal prediction for {note['id']!r}")
                destination.write(json.dumps({"id": note['id'], "codes": codes}, ensure_ascii=False) + "\n")
                destination.flush()
                if trace:
                    trace.write(json.dumps({'id': note['id'], 'candidates': pool, 'fallback_codes': fallback,
                                            'codes': codes, 'reranker_error': error,
                                            'disabled_reason': disabled_reason}, ensure_ascii=False) + "\n")
                    trace.flush()
                count += 1
    return {
        "strategy": strategy, "backend": backend,
        "api_model": reranker.api.model if reranker and reranker.api else None,
        "qwen_model": next((client.model for name, client in reranker.clients if name == "qwen"), None) if reranker else None,
        "semantic_enabled": hybrid is not None, "fallback_reason": fallback_reason,
        "fallback_batches": fallback_batches, "reranked": reranked, "reranker_failures": llm_failures,
        "reranker_disabled_reason": disabled_reason,
        "routes": dict(reranker.routes) if reranker else {"retrieval": count},
        "provider_errors": reranker.errors if reranker else {},
        "api_calls": reranker.api.calls if reranker and reranker.api else 0,
        "api_reserved_usd": round(reranker.api.reserved_usd, 6) if reranker and reranker.api else 0,
        "language_counts": dict(hybrid.language_counts) if hybrid is not None else {},
        "notes": count, "catalogue_codes": len(valid),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="ICD-11 retrieval: Anthropic API, optional local Qwen, retrieval fallback")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / 'config/default.json')
    parser.add_argument("--mms", type=Path, default=DEFAULT_MMS)
    parser.add_argument("--strategy", choices=("hybrid", "focused", "baseline"), default="hybrid")
    parser.add_argument("--semantic-assets", type=Path, default=DEFAULT_SEMANTIC_ASSETS)
    parser.add_argument("--backend", choices=("auto", "none", "anthropic", "ollama", "llama"))
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    try:
        options = json.loads(args.config.read_text(encoding='utf-8'))
        allowed = {'backend', 'model', 'endpoint', 'threads', 'timeout_seconds', 'max_seconds', 'batch_size', 'candidate_limit'}
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError('configuration contains unknown settings')
        for key in allowed:
            if getattr(args, key, None) is not None:
                options[key] = getattr(args, key)
        summary = run(args.input, args.output, args.mms, args.strategy, args.semantic_assets,
                      trace_path=args.trace, **options)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"coder: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
