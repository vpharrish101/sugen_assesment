"""Offline CPU acceptance checks; run in the built image with the repo mounted.

Uses real encoders and the actual SDK. API HTTP is simulated by unit tests.
Repeated clinical texts below are a throughput fixture, not new clinical data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('/app/assets/semantic'))
    parser.add_argument('--output-dir', type=Path, default=Path('/out'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_source = Path('/app/src') if Path('/app/src').exists() else root/'src'
    if runtime_source != root/'src':
        for source_file in (root/'src').rglob('*.py'):
            assert source_file.read_bytes() == (runtime_source/source_file.relative_to(root/'src')).read_bytes(), source_file
    env = dict(os.environ, PYTHONPATH=str(runtime_source), CODER_TEST_ASSETS=str(args.assets))
    for key in ('ANTHROPIC_API_KEY','OPENAI_API_KEY','QWEN_BACKEND'):
        env.pop(key, None)

    def command(arguments):
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=root, env=env,
                                capture_output=True, text=True, check=True)
        return result

    def predict(source, name, trace=False):
        destination = args.output_dir/(name+'.jsonl')
        cli = ['-m','coder',source,destination,'--backend','none','--semantic-assets',args.assets]
        if trace:
            cli.extend(['--trace',args.output_dir/(name+'.trace.jsonl')])
        result = command(cli)
        summary = json.loads(result.stderr.splitlines()[-1])
        assert summary['semantic_enabled'] and not summary['fallback_batches'], summary
        return rows(destination), summary

    valid = {row['code'] for row in rows_catalogue(root/'data/mms-2026-01.json')}
    tests = command(['-m','unittest','discover','-s',root/'tests','-q'])
    gold = rows(root/'data/dev_notes.jsonl')
    predictions, summary = predict(root/'data/notes.jsonl','dev',trace=True)
    frozen = {r['id']:r for r in rows(root/'experiments/ollama/candidates.jsonl')}
    traces = rows(args.output_dir/'dev.trace.jsonl')
    for row in traces:
        assert row['candidates'] == [c['code'] for c in frozen[row['id']]['candidates']], row['id']
        assert row['fallback_codes'] == frozen[row['id']]['fallback_codes'], row['id']
    official = command([root/'score.py','--gold',root/'data/dev_notes.jsonl','--pred',args.output_dir/'dev.jsonl',
                        '--mms',root/'data/mms-2026-01.json']).stdout
    print(official, end='')
    translated = rows(root/'data/translation_eval/notes.jsonl')
    translated_predictions, translated_summary = predict(root/'data/translation_eval/notes.jsonl','translated')
    paired = []
    for language in ('en','es','zh'):
        language_ids = {row['id'] for row in translated if row['language'] == language}
        paired.append({'language':language, **metrics(
            [row for row in translated if row['id'] in language_ids],
            [row for row in translated_predictions if row['id'] in language_ids])})
    load_source = args.output_dir/'load-input.jsonl'
    load_notes = [{'id':f'load-{i:04d}','text':gold[i % len(gold)]['text']} for i in range(600)]
    load_source.write_text(''.join(json.dumps(row)+'\n' for row in load_notes), encoding='utf-8')
    load_predictions, load_summary = predict(load_source,'load')
    assert [row['id'] for row in load_predictions] == [row['id'] for row in load_notes]
    assert all(0 < len(row['codes']) <= 3 and len(set(row['codes'])) == len(row['codes'])
               and set(row['codes']) <= valid for row in load_predictions)
    report = {'unit_tests':tests.stderr.strip(), 'official_scorer':official.strip(),
              'dev':{**metrics(gold,predictions), 'runtime':summary},
              'candidate_order_regression':'all 60 pools and fallback lists exactly match frozen Qwen inputs',
              'translated':{'paired':paired,'runtime':translated_summary,'independent_validation':False},
              'load_600':{'runtime':load_summary,'missing':0,'invalid':0,'clinical_generalisation_test':False},
              'api_validation':'actual SDK + MockTransport; no live API key/inference',
              'catalogue_sha256':hashlib.sha256((root/'data/mms-2026-01.json').read_bytes()).hexdigest()}
    (args.output_dir/'final_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


def rows_catalogue(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(gold, predictions):
    by_id = {row['id']:row['codes'] for row in predictions}
    top1 = sum(bool(by_id.get(row['id'])) and by_id[row['id']][0] in row['gold'] for row in gold)
    top3 = sum(bool(set(by_id.get(row['id'],[])[:3]) & set(row['gold'])) for row in gold)
    return {'notes':len(gold),'top1':top1,'top3':top3,'score_out_of_75':(50*top1+25*top3)/len(gold)}


if __name__ == '__main__':
    main()
