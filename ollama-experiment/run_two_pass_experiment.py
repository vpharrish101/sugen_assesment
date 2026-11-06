"""Run original Qwen ranking and one hierarchy verifier, or verify saved Qwen predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from hierarchy_reranker import Catalogue, parse_result
from run_hierarchy_experiment import read_jsonl, metrics
from run_ollama_experiment import MODEL
from two_pass_reranker import (first_pass_payload, parse_first_pass, shortlist,
                              verifier_payload, validate_cached_predictions)


def infer(client, payload):
    request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
        json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with client.open(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output-prefix', default='two-pass')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--first-pass-predictions', type=Path,
                        help='Reuse saved ORIGINAL Qwen baseline predictions; only call the verifier')
    args = parser.parse_args()
    if not args.output_prefix.replace('-', '').replace('_', '').isalnum():
        parser.error('invalid output prefix')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    root = Path(__file__).resolve().parent
    rows = read_jsonl(root / 'candidates.jsonl')
    catalogue = Catalogue(json.loads((root / 'mms-2026-01.json').read_text(encoding='utf-8')))
    cached = validate_cached_predictions(read_jsonl(args.first_pass_predictions), rows) if args.first_pass_predictions else None
    config = {'experiment': 'hierarchy-two-pass-v1', 'model': args.model,
        'input_sha256': hashlib.sha256((root / 'candidates.jsonl').read_bytes()).hexdigest(),
        'catalogue_sha256': hashlib.sha256((root / 'mms-2026-01.json').read_bytes()).hexdigest(),
        'first_pass_sha256': hashlib.sha256(args.first_pass_predictions.read_bytes()).hexdigest() if cached is not None else None,
        'mode': 'saved-first-pass' if cached is not None else 'fresh-two-pass', 'verifier_candidate_limit': 24}
    output = root / (args.output_prefix + '_predictions.jsonl')
    trace_path = root / (args.output_prefix + '_trace.jsonl')
    config_path = root / (args.output_prefix + '_run.json')
    if args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            parser.error('resume requires the same model, inputs and first-pass source')
    elif any(p.exists() for p in (output, trace_path, config_path)):
        parser.error('output files exist; use a fresh prefix or --resume')
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open('http://127.0.0.1:11434/api/tags', timeout=10) as response:
        models = json.load(response)['models']
    if args.model not in {m['name'] for m in models}:
        raise SystemExit('Install the model first:\nollama pull ' + args.model)
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    done = {r['id'] for r in read_jsonl(output)} if args.resume and output.exists() else set()
    mode = 'a' if args.resume else 'w'
    with output.open(mode, encoding='utf-8') as out, trace_path.open(mode, encoding='utf-8') as traces:
        for row in rows[:args.limit] if args.limit else rows:
            if row['id'] in done:
                continue
            first_result = verify_result = evidence = None
            first_error = verify_error = None
            evidence_valid, calls = False, 0
            started = time.perf_counter()
            if cached is not None:
                first_codes = cached[row['id']]
            else:
                try:
                    calls += 1
                    first_result = infer(client, first_pass_payload(row, args.model))
                    first_codes = parse_first_pass(first_result, row)
                except Exception as exc:
                    first_error, first_codes = str(exc), row['fallback_codes']
            first_seconds = time.perf_counter() - started
            candidates = shortlist(row, first_codes, catalogue)
            payload = verifier_payload(row, candidates, args.model)
            verify_started = time.perf_counter()
            try:
                calls += 1
                verify_result = infer(client, payload)
                codes, evidence, evidence_valid = parse_result(verify_result, candidates, row['text'])
                if not evidence_valid:
                    raise ValueError('Verifier excerpt is not a short literal note excerpt')
            except Exception as exc:
                verify_error, codes = str(exc), first_codes
            trace = {'id': row['id'], 'first_pass_codes': first_codes,
                'first_pass_error': first_error, 'verifier_error': verify_error,
                'first_pass_response': first_result, 'verifier_response': verify_result,
                'candidate_codes': [c['code'] for c in candidates],
                'evidence': evidence, 'evidence_valid': evidence_valid,
                'first_pass_seconds': round(first_seconds, 3),
                'verifier_seconds': round(time.perf_counter() - verify_started, 3), 'local_inference_calls': calls}
            traces.write(json.dumps(trace) + '\n')
            traces.flush()
            out.write(json.dumps({'id': row['id'], 'codes': codes}) + '\n')
            out.flush()
            print(f'{row["id"]}: {time.perf_counter()-started:.1f}s {codes}' +
                  (f' KEEP FIRST PASS: {verify_error}' if verify_error else ''), flush=True)
    preds = {r['id']: r['codes'] for r in read_jsonl(output)}
    if set(preds) != {r['id'] for r in rows}:
        print(f'Partial run: {len(preds)}/{len(rows)} notes. Continue with the same options and --resume.')
        return
    # Evaluation labels are never used to construct either model request.
    gold = {r['id']: set(r['gold']) for r in read_jsonl(root / 'dev_notes.jsonl')}
    traces = {t['id']: t for t in read_jsonl(trace_path) if t['id'] in preds}
    first = {i: t['first_pass_codes'] for i, t in traces.items()}
    report = dict(config, notes=len(preds),
        retrieval_baseline=metrics({r['id']: r['fallback_codes'] for r in rows}, gold),
        first_pass=metrics(first, gold), two_pass=metrics(preds, gold),
        first_pass_fallbacks=sum(bool(t['first_pass_error']) for t in traces.values()),
        verifier_fallbacks=sum(bool(t['verifier_error']) for t in traces.values()),
        valid_evidence=sum(t['evidence_valid'] for t in traces.values()),
        total_seconds=sum(t['first_pass_seconds'] + t['verifier_seconds'] for t in traces.values()),
        local_inference_calls=sum(t['local_inference_calls'] for t in traces.values()), external_inference_calls=0)
    for k in (1, 3):
        before = {i for i, g in gold.items() if set(first[i][:k]) & g}
        after = {i for i, g in gold.items() if set(preds[i][:k]) & g}
        report[f'top{k}_gains'], report[f'top{k}_losses'] = sorted(after - before), sorted(before - after)
    (root / (args.output_prefix + '_results.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
