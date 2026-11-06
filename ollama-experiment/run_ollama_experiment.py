"""Run the frozen ICD reranking experiment through local Ollama. Standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from test_local_reranker import make_messages

MODEL = 'medgemma1.5:4b'


def parse_choices(content, candidates):
    """Accept candidate positions or exact catalogue codes in the choices array."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Ollama returned empty content')
    stripped = content.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    value = None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    if isinstance(value, dict):
        value = value.get('choices')
    if value is None:
        match = re.fullmatch(r'\s*\{\s*"choices"\s*:\s*\[([^\]]+)\]\s*\}\s*', stripped, re.S)
        if match:
            value = [part.strip().strip('"') for part in match.group(1).split(',')]
    if value is None:
        match = re.fullmatch(r'\s*(\d{1,2})\s*[,;\s]+\s*(\d{1,2})\s*[,;\s]+\s*(\d{1,2})\s*', stripped)
        if match:
            value = [int(x) for x in match.groups()]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError('No valid three-choice ranking in response: ' + repr(content[:160]))
    by_code = {candidate['code']: i for i, candidate in enumerate(candidates, 1)}
    indices = []
    for item in value:
        if type(item) is int and 1 <= item <= len(candidates):
            indices.append(item)
        elif isinstance(item, str) and item in by_code:
            indices.append(by_code[item])
        else:
            raise ValueError('Unknown candidate in response: ' + repr(item))
    if len(set(indices)) != 3:
        raise ValueError('Repeated candidate in response: ' + repr(content[:160]))
    return indices


def make_payload(row, model, cpu=False, retry=False):
    messages = make_messages(row)
    messages[0]['content'] = (
        'Choose the three best ICD-11 candidates for the current diagnosis. '
        'Use the clinical note and candidate titles; ignore negated or historical conditions. '
        'Reply with one JSON object whose "choices" field contains three distinct '
        'ICD codes copied exactly from the supplied candidate list, best first. '
        'Do not invent codes or repeat a code. No explanation.'
    )
    if retry:
        messages[0]['content'] = (
            'The previous output was invalid. Return a JSON "choices" array with '
            'three distinct ICD codes copied exactly from the candidate list, '
            'best first. Do not invent codes. No other text.'
        )
        messages[1]['content'] += '\n\nReturn only three distinct listed ICD codes in a JSON choices array.'
    options = {'temperature': 0, 'seed': 42, 'num_ctx': 8192,
               'num_predict': 384 if retry else 256}
    if cpu:
        options['num_gpu'] = 0
    return {'model': model, 'messages': messages, 'stream': False,
            'format': 'json', 'keep_alive': '30m', 'options': options}


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
    parser.add_argument('--output-prefix', default='ollama', help='Prefix for output files')
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
                for retry in (False, True):
                    payload = make_payload(row, args.model, cpu=args.cpu, retry=retry)
                    request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
                        json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                    with client.open(request, timeout=600) as response:
                        result = json.load(response)
                    content = (result.get('message') or {}).get('content', '')
                    attempt = {'done_reason': result.get('done_reason'), 'content': content,
                               'eval_count': result.get('eval_count'),
                               'prompt_eval_count': result.get('prompt_eval_count')}
                    attempts.append(attempt)
                    try:
                        choices = parse_choices(content, row['candidates'])
                        codes = [row['candidates'][i - 1]['code'] for i in choices]
                        break
                    except ValueError as exc:
                        attempt['parse_error'] = str(exc)
                else:
                    raise ValueError('Ollama did not return three valid choices after two attempts')
            except urllib.error.HTTPError as exc:
                error = f'Ollama HTTP {exc.code}: {exc.read(1000).decode("utf-8", errors="replace")}'
                traces.write(json.dumps({'id': row['id'], 'fallback_error': error,
                    'attempts': attempts}) + '\n')
                traces.flush()
                print(f'{row["id"]}: STOPPED: {error}', flush=True)
                raise SystemExit('Completed notes are saved. Check Ollama, then run --resume.') from exc
            except Exception as exc:
                traces.write(json.dumps({'id': row['id'], 'fallback_error': str(exc),
                    'attempts': attempts}) + '\n')
                traces.flush()
                print(f'{row["id"]}: STOPPED: {exc}', flush=True)
                raise SystemExit('Completed notes are saved. Check Ollama, then run --resume.') from exc
            seconds = round(time.perf_counter() - started, 3)
            traces.write(json.dumps({'id': row['id'], 'seconds': seconds,
                'fallback_error': None, 'attempts': attempts,
                'runner_version': 4, 'cpu': args.cpu}) + '\n')
            traces.flush()
            out.write(json.dumps({'id': row['id'], 'codes': codes}) + '\n')
            out.flush()
            print(f'{row["id"]}: {seconds:.1f}s {codes}', flush=True)
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
        'runner_versions': sorted({t.get('runner_version', 1) for t in traces}),
        'total_seconds': sum(t['seconds'] for t in traces),
        'external_inference_calls': 0,
        'input_sha256': hashlib.sha256((root / 'candidates.jsonl').read_bytes()).hexdigest()}
    (root / (args.output_prefix + '_results.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
