"""Opt-in real API smoke on external notes. Potentially billable; no-key exits 2."""
import argparse,json,os,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coder.__main__ import run,read_notes,DEFAULT_MMS,DEFAULT_SEMANTIC_ASSETS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=DEFAULT_SEMANTIC_ASSETS)
    args=p.parse_args()
    if not os.getenv('ANTHROPIC_API_KEY'):
        print('NOT TESTED: key missing; zero requests made.',file=sys.stderr);return 2
    with args.input.open(encoding='utf-8') as f:notes=list(read_notes(f))[:1]
    if not notes:raise ValueError('input has no notes')
    with tempfile.TemporaryDirectory() as d:
        source,output=Path(d)/'notes.jsonl',Path(d)/'predictions.jsonl'
        source.write_text(json.dumps(notes[0]),encoding='utf-8')
        summary=run(source,output,DEFAULT_MMS,semantic_assets=args.assets,backend='anthropic')
        if summary['routes'].get('anthropic')!=1:
            print('FAIL: API did not succeed; fallback is not a live PASS.',file=sys.stderr);return 1
        print('PASS: real API returned a valid ranking.');return 0
if __name__=='__main__':sys.exit(main())
