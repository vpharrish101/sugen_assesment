"""Original Qwen ranking followed by a focused hierarchy/evidence check."""

import json

from hierarchy_reranker import make_payload as hierarchy_payload
from run_ollama_experiment import make_messages


def first_pass_payload(row, model):
    # Keep this identical to the measured run_ollama_experiment.py request.
    messages = make_messages(row)
    messages[0]['content'] = messages[0]['content'].split('First write one short sentence')[0] + (
        'Return only a JSON object in exactly this shape: '
        '{"choices":[1,2,3]}. Replace the numbers with three distinct candidate '
        'numbers, best first. No explanation or markdown.'
    )
    return {'model': model, 'messages': messages, 'stream': False, 'format': 'json',
            'keep_alive': '30m', 'options': {'temperature': 0, 'seed': 42,
                                            'num_ctx': 8192, 'num_predict': 256}}


def parse_first_pass(result, row):
    choices = json.loads(result['message']['content'])['choices']
    if (not isinstance(choices, list) or len(choices) != 3 or
            any(type(i) is not int or not 1 <= i <= len(row['candidates']) for i in choices) or
            len(set(choices)) != 3):
        raise ValueError('Invalid first-pass candidate numbers')
    return [row['candidates'][i - 1]['code'] for i in choices]


def shortlist(row, first_codes, catalogue):
    codes = list(dict.fromkeys(first_codes + [c['code'] for c in row['candidates'][:10]]))
    seed_row = dict(row, candidates=[catalogue.by_code[c] for c in codes])
    return catalogue.expand(seed_row, limit=24, seeds=3)


def verifier_payload(row, candidates, model):
    payload = hierarchy_payload(row, candidates, model)
    payload['messages'][0]['content'] = (
        'Review a provisional diagnosis-code ranking against the clinical note. '
        'The provisional ranking may be wrong: verify it using the documented facts. '
        'A parent category should not replace an applicable child, including an '
        'unspecified child when the required subtype detail is absent. '
    ) + payload['messages'][0]['content']
    payload['messages'][1]['content'] += (
        '\n\nPROVISIONAL RANKING: C001, C002, C003. '
        'Compare these with the nearby alternatives and return the three best supported entries.'
    )
    return payload


def validate_cached_predictions(predictions, rows):
    if len(predictions) != len(rows) or {p['id'] for p in predictions} != {r['id'] for r in rows}:
        raise ValueError('Saved first-pass predictions must cover every input ID exactly once')
    by_id = {p['id']: p['codes'] for p in predictions}
    for row in rows:
        codes = by_id[row['id']]
        allowed = {c['code'] for c in row['candidates']}
        if (not isinstance(codes, list) or len(codes) != 3 or
                any(type(c) is not str or c not in allowed for c in codes) or len(set(codes)) != 3):
            raise ValueError(f'Invalid saved first-pass prediction for {row["id"]}')
    return by_id
