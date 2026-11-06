"""Run the frozen ICD reranking experiment through local Ollama. Standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

MODEL = 'medgemma1.5:4b'
RUNNER_VERSION = 5


def make_payload(row, model, available, rank, cpu=False):
    """Ask for one catalogue code; the JSON schema limits the model's vocabulary."""
    catalogue = '\n'.join(f'{c["code"]}: {c["title"]}' for c in available)
    messages = [
        {'role': 'system', 'content': (
            'Select one ICD-11 code for the current consultation from the supplied '
            'catalogue. Prefer the clinician-supported current diagnosis over history, '
            'negated findings, and speculation. Choose the most specific supported '
            'code. Return only the code in a JSON object with key "code".')},
        {'role': 'user', 'content': (
            f'CLINICAL NOTE\n{row["text"]}\n\n'
            f'DIAGNOSIS FOCUS (retrieval hint)\n{row["focus"]}\n\n'
            f'AVAILABLE ICD-11 CODES\n{catalogue}\n\n'
            f'Choose rank {rank}: the best remaining code for this visit.')},
    ]
    schema = {'type': 'object', 'properties': {
        'code': {'type': 'string', 'enum': [c['code'] for c in available]}},
        'required': ['code'], 'additionalProperties': False}
    options = {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 64}
    if cpu:
        options['num_gpu'] = 0
    return {'model': model, 'messages': messages, 'stream': False,
            'format': schema, 'keep_alive': '30m', 'options': options}


def read_code(result, available):
    content = (result.get('message') or {}).get('content', '')
    response = json.loads(content)
    if not isinstance(response, dict) or set(response) != {'code'}:
        raise ValueError('Expected a JSON object with only a code field')
    code = response['code']
    if type(code) is not str or code not in {c['code'] for c in available}:
        raise ValueError('Code is not in the available catalogue: ' + repr(code))
    return code


def valid_resume_rows(output, trace_path):
    """Keep completed results, dropping old fallback rows and half-written pairs."""
    predictions = [json.loads(x) for x in output.read_text(encoding='utf-8').splitlines()] if output.exists() else []
    traces = [json.loads(x) for x in trace_path.read_text(encoding='utf-8').splitlines()] if trace_path.exists() else []
    trace_by_id = {r['id']: r for r in traces}
    valid = {r['id'] for r in predictions if r['id'] in trace_by_id
             and not trace_by_id[r['id']].get('fallback_error')}
    for path, records in ((output, predictions), (trace_path, traces)):
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(''.join(json.dumps(r) + '\n' for r in records if r['id'] in valid), encoding='utf-8')
        os.replace(tmp, path)
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='Use CPU and system RAM if GPU inference freezes the desktop')
    parser.add_argument('--limit', type=int, help='Run only the first N notes as a technical check')
    parser.add_argument('--output-prefix', default='medgemma-schema', help='Prefix for output files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rows = [json.loads(x) for x in (root / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    if not args.output_prefix.replace('-', '').replace('_', '').isalnum():
        parser.error('output prefix must contain only letters, numbers, hyphens or underscores')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    output = root / (args.output_prefix + '_predictions.jsonl')
    trace_path = root / (args.output_prefix + '_trace.jsonl')
    if not args.resume and (output.exists() or trace_path.exists()):
        parser.error('output files already exist; choose a new --output-prefix or use --resume')
    if args.resume and trace_path.exists():
        existing = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
        if any(t.get('runner_version') != RUNNER_VERSION or t.get('model') != args.model
               for t in existing):
            parser.error('this output prefix belongs to an older runner or another model; use a new prefix')
    done = valid_resume_rows(output, trace_path) if args.resume else set()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # Fail before creating predictions if Ollama or the requested model is unavailable.
    with client.open('http://127.0.0.1:11434/api/tags', timeout=10) as response:
        models = json.load(response)['models']
    if args.model not in {m['name'] for m in models}:
        raise SystemExit('Install the model first:\nollama pull ' + args.model)
    mode = 'a' if args.resume else 'w'
    with output.open(mode, encoding='utf-8') as out, trace_path.open(mode, encoding='utf-8') as traces:
        for row in rows[:args.limit] if args.limit else rows:
            if row['id'] in done:
                continue
            started = time.perf_counter()
            attempts = []
            try:
                codes = []
                error = None
                for rank in range(1, 4):
                    available = [c for c in row['candidates'] if c['code'] not in codes]
                    payload = make_payload(row, args.model, available, rank, cpu=args.cpu)
                    request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
                        json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                    with client.open(request, timeout=600) as response:
                        result = json.load(response)
                    attempt = {'rank': rank, 'done_reason': result.get('done_reason'),
                               'content': (result.get('message') or {}).get('content', ''),
                               'eval_count': result.get('eval_count'),
                               'prompt_eval_count': result.get('prompt_eval_count')}
                    attempts.append(attempt)
                    try:
                        codes.append(read_code(result, available))
                    except (ValueError, TypeError) as exc:
                        error = f'Rank {rank}: {exc}'
                        attempt['parse_error'] = error
                        codes = row['fallback_codes']
                        break
            except urllib.error.HTTPError as exc:
                error = f'Ollama HTTP {exc.code}: {exc.read(1000).decode("utf-8", errors="replace")}'
                traces.write(json.dumps({'id': row['id'], 'fallback_error': error,
                    'attempts': attempts, 'runner_version': RUNNER_VERSION,
                    'model': args.model}) + '\n')
                traces.flush()
                print(f'{row["id"]}: STOPPED: {error}', flush=True)
                raise SystemExit('Completed notes are saved. Check Ollama, then run --resume.') from exc
            except Exception as exc:
                traces.write(json.dumps({'id': row['id'], 'fallback_error': str(exc),
                    'attempts': attempts, 'runner_version': RUNNER_VERSION,
                    'model': args.model}) + '\n')
                traces.flush()
                print(f'{row["id"]}: STOPPED: {exc}', flush=True)
                raise SystemExit('Completed notes are saved. Check Ollama, then run --resume.') from exc
            seconds = round(time.perf_counter() - started, 3)
            traces.write(json.dumps({'id': row['id'], 'seconds': seconds,
                'fallback_error': error, 'attempts': attempts,
                'runner_version': RUNNER_VERSION, 'cpu': args.cpu,
                'model': args.model}) + '\n')
            traces.flush()
            out.write(json.dumps({'id': row['id'], 'codes': codes}) + '\n')
            out.flush()
            print(f'{row["id"]}: {seconds:.1f}s {codes}' +
                  (f' FALLBACK: {error}' if error else ''), flush=True)
    if len({json.loads(x)['id'] for x in output.read_text(encoding='utf-8').splitlines()}) != len(rows):
        print(f'Partial run complete. Predictions: {output.name}; continue with --resume.')
        return
    gold = {r['id']: set(r['gold']) for r in map(json.loads,
        (root / 'dev_notes.jsonl').read_text(encoding='utf-8').splitlines())}
    preds = {r['id']: r['codes'] for r in map(json.loads, output.read_text().splitlines())}
    baseline = {r['id']: r['fallback_codes'] for r in rows}
    def metrics(ranks):
        a = sum(ranks[i][0] in g for i, g in gold.items())
        b = sum(bool(set(ranks[i][:3]) & g) for i, g in gold.items())
        return {'top1': a, 'top3': b, 'score_out_of_75': (50*a + 25*b)/len(gold)}
    traces = [json.loads(x) for x in trace_path.read_text(encoding='utf-8').splitlines()]
    report = {'model': args.model, 'notes': len(preds), 'baseline': metrics(baseline),
        'ollama': metrics(preds),
        'fallbacks': sum(bool(t.get('fallback_error')) for t in traces),
        'runner_versions': sorted({t.get('runner_version', 1) for t in traces}),
        'total_seconds': sum(t['seconds'] for t in traces),
        'external_inference_calls': 0,
        'input_sha256': hashlib.sha256((root / 'candidates.jsonl').read_bytes()).hexdigest()}
    (root / (args.output_prefix + '_results.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
