"""Fresh-note prototype runner; no frozen clinical data or evaluation labels."""
import argparse,json,os,sys,time
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
from coder.__main__ import load_hybrid,read_notes,DEFAULT_MMS,DEFAULT_SEMANTIC_ASSETS
from coder.bm25 import BM25
from coder.clinical_query import diagnosis_focus
from coder.retriever import CandidateRetriever
from .prompt import MODEL,make_messages

ARCHITECTURE='qwen'

def infer(client,endpoint,payload):
    request=urllib.request.Request(endpoint+'/api/chat',json.dumps(payload).encode(),{'Content-Type':'application/json'})
    with client.open(request,timeout=45) as response:return json.load(response)

def basic_payload(row,model):
    messages=make_messages(row)
    messages[0]['content']=messages[0]['content'].split('First write one short sentence')[0]+('Return only a JSON object in exactly this shape: {"choices":[1,2,3]}. Replace the numbers with three distinct candidate numbers, best first. No explanation or markdown.')
    return {'model':model,'messages':messages,'stream':False,'format':'json','keep_alive':'30m',
            'options':{'temperature':0,'seed':42,'num_ctx':8192,'num_predict':256}}

def parse_basic(result,row):
    choices=json.loads(result['message']['content'])['choices']
    if not isinstance(choices,list) or len(choices)!=3 or any(type(i) is not int or not 1<=i<=len(row['candidates']) for i in choices) or len(set(choices))!=3:
        raise ValueError('expected three distinct candidate numbers')
    return [row['candidates'][i-1]['code'] for i in choices]

def run(source,output,mms=DEFAULT_MMS,assets=DEFAULT_SEMANTIC_ASSETS,model=MODEL,endpoint='http://127.0.0.1:11434',limit=None):
    if output.resolve() in {source.resolve(),mms.resolve()}:raise ValueError('output must not overwrite input/catalogue')
    parsed=urlparse(endpoint)
    if parsed.scheme!='http' or parsed.hostname not in {'localhost','127.0.0.1','::1'} or parsed.path not in {'','/'} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Ollama endpoint must be HTTP loopback')
    with source.open(encoding='utf-8') as f:notes=list(read_notes(f))
    if limit:notes=notes[:limit]
    index=BM25.from_file(mms);lexical=CandidateRetriever(index);valid=set(index.codes)
    titles={r['code']:r['title'] for r in index.records}
    hybrid=None
    try:hybrid=load_hybrid(index,assets)
    except (ImportError,OSError,RuntimeError,ValueError,KeyError,TypeError) as exc:print('semantic fallback: '+str(exc),file=sys.stderr)
    catalogue=None
    if ARCHITECTURE!='qwen':
        from .hierarchy_reranker import Catalogue,make_payload,parse_result
        catalogue=Catalogue(index.records)
    client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    output.parent.mkdir(parents=True,exist_ok=True)
    calls=fallbacks=failures=0;started=time.monotonic()
    with output.open('w',encoding='utf-8') as destination:
        for start in range(0,len(notes),32):
            batch=notes[start:start+32];texts=[r['text'] for r in batch]
            try:
                if hybrid is None:raise RuntimeError('semantic unavailable')
                pools=hybrid.candidate_pool_many(texts,50);baselines=hybrid.rank_many(texts,3)
                if len(pools)!=len(batch) or len(baselines)!=len(batch):raise ValueError('incomplete retrieval batch')
            except (OSError,RuntimeError,ValueError,KeyError,TypeError):
                pools=[lexical.candidate_pool(t,50) for t in texts]
                baselines=[[c for c,_ in lexical.focused_ranking(t,3)] for t in texts]
            for note,pool,baseline in zip(batch,pools,baselines):
                row=dict(note,focus=diagnosis_focus(note['text']),candidates=[{'code':c,'title':titles[c]} for c in pool],fallback_codes=baseline)
                codes=baseline;error=None
                if failures<3 and time.monotonic()-started<3300:
                    try:
                        if ARCHITECTURE=='hierarchy':
                            candidates=catalogue.expand(row,80,10);calls+=1
                            codes,_,_=parse_result(infer(client,endpoint,make_payload(row,candidates,model)),candidates,note['text'])
                        else:
                            calls+=1;codes=parse_basic(infer(client,endpoint,basic_payload(row,model)),row)
                        failures=0
                    except Exception as exc:error=str(exc);failures+=1;fallbacks+=1
                    if ARCHITECTURE=='two-pass' and failures==0:
                        from .two_pass_reranker import shortlist,verifier_payload
                        first=codes
                        try:
                            candidates=shortlist(row,first,catalogue);calls+=1
                            codes,_,grounded=parse_result(infer(client,endpoint,verifier_payload(row,candidates,model)),candidates,note['text'])
                            if not grounded:raise ValueError('verifier evidence not in note')
                        except Exception as exc:error=str(exc);codes=first;fallbacks+=1
                if not codes or len(codes)>3 or len(set(codes))!=len(codes) or not set(codes)<=valid:codes=baseline
                destination.write(json.dumps({'id':note['id'],'codes':codes},ensure_ascii=False)+'\n');destination.flush()
                if error:print('prototype fallback: '+error,file=sys.stderr)
    return {'architecture':ARCHITECTURE,'notes':len(notes),'local_inference_calls':calls,'fallbacks':fallbacks,'semantic_enabled':hybrid is not None}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mms',type=Path,default=DEFAULT_MMS);p.add_argument('--semantic-assets',type=Path,default=DEFAULT_SEMANTIC_ASSETS)
    p.add_argument('--model',default=os.getenv('QWEN_MODEL',MODEL));p.add_argument('--endpoint',default=os.getenv('QWEN_ENDPOINT','http://127.0.0.1:11434'))
    p.add_argument('--limit',type=int,choices=(1,2,3))
    args=p.parse_args()
    try:result=run(args.input,args.output,args.mms,args.semantic_assets,args.model,args.endpoint,args.limit)
    except (OSError,ValueError,KeyError,TypeError) as exc:print('prototype: '+str(exc),file=sys.stderr);return 2
    print(json.dumps(result),file=sys.stderr);return 0
if __name__=='__main__':sys.exit(main())
