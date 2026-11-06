"""One Qwen call per note, using the frozen retrieval pool plus hierarchy context."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from hierarchy_reranker import Catalogue, make_payload, parse_result
from run_ollama_experiment import MODEL


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def metrics(predictions, gold):
    top1 = sum(predictions[id_][0] in codes for id_, codes in gold.items())
    top3 = sum(bool(set(predictions[id_][:3]) & codes) for id_, codes in gold.items())
    return {'top1': top1, 'top3': top3, 'score_out_of_75': (50 * top1 + 25 * top3) / len(gold)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output-prefix', default='hierarchy')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--compare-predictions', type=Path, help='Optional earlier Qwen predictions for scoring after inference')
    args = parser.parse_args()
    if not args.output_prefix.replace('-', '').replace('_', '').isalnum():
        parser.error('output prefix must contain only letters, numbers, hyphens or underscores')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    root = Path(__file__).resolve().parent
    rows = read_jsonl(root / 'candidates.jsonl')
    catalogue = Catalogue(json.loads((root / 'mms-2026-01.json').read_text(encoding='utf-8')))
    config = {'experiment': 'hierarchy-single-pass-v1', 'model': args.model,
              'input_sha256': hashlib.sha256((root / 'candidates.jsonl').read_bytes()).hexdigest(),
              'catalogue_sha256': hashlib.sha256((root / 'mms-2026-01.json').read_bytes()).hexdigest(),
              'candidate_limit': 80, 'expansion_seeds': 10}
    output = root / (args.output_prefix + '_predictions.jsonl')
    trace_path = root / (args.output_prefix + '_trace.jsonl')
    config_path = root / (args.output_prefix + '_run.json')
    if args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            parser.error('resume requires matching model, inputs and experiment; use a fresh prefix')
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
            candidates = catalogue.expand(row, config['candidate_limit'], config['expansion_seeds'])
            payload = make_payload(row, candidates, args.model)
            started = time.perf_counter()
            result = evidence = None
            error, evidence_valid = None, False
            try:
                request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
                    json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                with client.open(request, timeout=600) as response:
                    result = json.load(response)
                codes, evidence, evidence_valid = parse_result(result, candidates, row['text'])
            except Exception as exc:
                error, codes = str(exc), row['fallback_codes']
            seconds = round(time.perf_counter() - started, 3)
            traces.write(json.dumps({'id': row['id'], 'seconds': seconds, 'fallback_error': error,
                'evidence': evidence, 'evidence_valid': evidence_valid, 'response': result,
                'candidate_codes': [c['code'] for c in candidates],
                'added_codes': [c['code'] for c in candidates if not c['retrieved']],
                'prompt_sha256': hashlib.sha256(json.dumps(payload['messages']).encode()).hexdigest()}) + '\n')
            traces.flush()
            out.write(json.dumps({'id': row['id'], 'codes': codes}) + '\n')
            out.flush()
            status = f' FALLBACK: {error}' if error else ('' if evidence_valid else ' EVIDENCE FLAGGED')
            print(f'{row["id"]}: {seconds:.1f}s {codes}{status}', flush=True)
    preds = {r['id']: r['codes'] for r in read_jsonl(output)}
    if set(preds) != {r['id'] for r in rows}:
        print(f'Partial run: {len(preds)}/{len(rows)} predictions; continue with the same prefix and --resume.')
        return
    # Gold and optional previous predictions enter only the scoring phase.
    gold = {r['id']: set(r['gold']) for r in read_jsonl(root / 'dev_notes.jsonl')}
    traces = {t['id']: t for t in read_jsonl(trace_path) if t['id'] in preds}
    report = dict(config, notes=len(preds),
        baseline=metrics({r['id']: r['fallback_codes'] for r in rows}, gold),
        hierarchy_with_fallback=metrics(preds, gold),
        fallbacks=sum(bool(t['fallback_error']) for t in traces.values()),
        valid_evidence=sum(t['evidence_valid'] for t in traces.values()),
        total_seconds=sum(t['seconds'] for t in traces.values()), external_inference_calls=0)
    if args.compare_predictions:
        previous = {r['id']: r['codes'] for r in read_jsonl(args.compare_predictions)}
        if set(previous) != set(gold):
            raise SystemExit('Comparison predictions must cover exactly the gold IDs; inference outputs are saved.')
        report['previous_qwen'] = metrics(previous, gold)
        report['top1_gains'] = [i for i, g in gold.items() if preds[i][0] in g and previous[i][0] not in g]
        report['top1_losses'] = [i for i, g in gold.items() if preds[i][0] not in g and previous[i][0] in g]
        report['top3_gains'] = [i for i, g in gold.items() if set(preds[i]) & g and not set(previous[i]) & g]
        report['top3_losses'] = [i for i, g in gold.items() if not set(preds[i]) & g and set(previous[i]) & g]
    (root / (args.output_prefix + '_results.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
