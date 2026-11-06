"""Run the frozen ICD reranking experiment through local Ollama. Standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from test_local_reranker import make_messages

MODEL = 'hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, help='Run only the first N notes as a technical check')
    parser.add_argument('--output-prefix', default='ollama', help='Prefix for output files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rows = [json.loads(x) for x in (root / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    if not args.output_prefix.replace('-', '').replace('_', '').isalnum():
        parser.error('output prefix must contain only letters, numbers, hyphens or underscores')
    output = root / (args.output_prefix + '_predictions.jsonl')
    trace_path = root / (args.output_prefix + '_trace.jsonl')
    if not args.resume and (output.exists() or trace_path.exists()):
        parser.error('output files already exist; choose a new --output-prefix or use --resume')
    done = set()
    if args.resume and output.exists():
        done = {json.loads(x)['id'] for x in output.read_text(encoding='utf-8').splitlines()}
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
            messages = make_messages(row)
            messages[0]['content'] = messages[0]['content'].split('First write one short sentence')[0] + (
                'Return only a JSON object in exactly this shape: '
                '{"choices":[1,2,3]}. Replace the numbers with three distinct candidate '
                'numbers, best first. No explanation or markdown.'
            )
            payload = {'model': args.model, 'messages': messages,
                'stream': False, 'format': 'json', 'keep_alive': '30m',
                'options': {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 256}}
            started = time.perf_counter()
            result, error = None, None
            try:
                request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
                    json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                with client.open(request, timeout=600) as response:
                    result = json.load(response)
                choices = json.loads(result['message']['content'])['choices']
                if len(choices) != 3 or len(set(choices)) != 3 or any(
                    type(i) is not int or not 1 <= i <= len(row['candidates']) for i in choices):
                    raise ValueError('Invalid or repeated candidate numbers')
                codes = [row['candidates'][i - 1]['code'] for i in choices]
            except Exception as exc:
                error = str(exc)
                codes = row['fallback_codes']
            seconds = round(time.perf_counter() - started, 3)
            out.write(json.dumps({'id': row['id'], 'codes': codes}) + '\n')
            out.flush()
            traces.write(json.dumps({'id': row['id'], 'seconds': seconds,
                'fallback_error': error, 'response': result}) + '\n')
            traces.flush()
            print(f'{row["id"]}: {seconds:.1f}s' + (f' FALLBACK: {error}' if error else ''), flush=True)
            if error:
                raise SystemExit('Stopped after an invalid response; inspect the trace before continuing.')
    if len({json.loads(x)['id'] for x in output.read_text(encoding='utf-8').splitlines()}) != len(rows):
        print(f'Partial run complete. Predictions: {output.name}; no full-set score yet.')
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
        'ollama_with_fallback': metrics(preds),
        'fallbacks': sum(bool(t['fallback_error']) for t in traces),
        'total_seconds': sum(t['seconds'] for t in traces),
        'external_inference_calls': 0,
        'input_sha256': hashlib.sha256((root / 'candidates.jsonl').read_bytes()).hexdigest()}
    (root / (args.output_prefix + '_results.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
