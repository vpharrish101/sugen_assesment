"""Opt-in key/model check: one real API request, no inference fallback.

Run only when ready: PYTHONPATH=src python tests/smoke_live_api.py
This command may incur API charges. Without a key it exits without any request.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coder.reranker import Reranker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes',type=int,choices=(1,2,3),default=1)
    args=parser.parse_args()
    key=os.getenv('ANTHROPIC_API_KEY')
    if not key:
        print('NOT TESTED: ANTHROPIC_API_KEY is not configured; zero requests made.',file=sys.stderr)
        return 2
    router=Reranker('anthropic','unused','http://127.0.0.1:11434',20)
    try:
        if router.api is None:
            raise RuntimeError(str(router.errors) or 'API client unavailable')
        path=Path(__file__).resolve().parents[1]/'experiments/ollama/candidates.jsonl'
        notes=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()][:args.notes]
        for row in notes:
            candidates=[c['code'] for c in row['candidates']]
            titles={c['code']:c['title'] for c in row['candidates']}
            codes=router.api.rerank(row['text'],candidates,titles)
            print(json.dumps({'id':row['id'],'provider':'anthropic','model':router.api.model,'codes':codes}))
        print('PASS: real API authentication, model access and ranking response confirmed. No fallback used.')
        return 0
    except Exception as exc:
        print(('FAIL: '+type(exc).__name__+': '+str(exc)).replace(key,'[REDACTED]'),file=sys.stderr)
        return 1
    finally:
        router.close()


if __name__=='__main__':sys.exit(main())
