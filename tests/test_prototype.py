"""Small protocol tests, independent of any supplied clinical dataset."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import prototype.__main__ as runner
from coder.bm25 import BM25

class PrototypeTests(unittest.TestCase):
    def test_fresh_input_and_invalid_response_never_drop_ids(self):
        index=BM25.from_file(runner.DEFAULT_MMS);codes=index.codes[:3]
        class Retrieval:
            def candidate_pool_many(self,texts,limit):return [codes[:] for _ in texts]
            def rank_many(self,texts,limit):return [codes[:] for _ in texts]
        calls=[]
        def infer(client,endpoint,payload):
            calls.append(payload)
            if len(calls)==2:return {'message':{'content':'not JSON'}}
            if payload['format']=='json':value={'choices':[2,1,3]}
            else:value={'choices':['C002','C001','C003'],'evidence':'protocol-test'}
            return {'message':{'content':json.dumps(value)}}
        with tempfile.TemporaryDirectory() as directory:
            source,output=Path(directory)/'in.jsonl',Path(directory)/'out.jsonl'
            source.write_text(''.join(json.dumps({'id':str(i),'text':'protocol-test'})+'\n' for i in range(3)))
            with patch.object(runner,'load_hybrid',return_value=Retrieval()),patch.object(runner,'infer',side_effect=infer):
                summary=runner.run(source,output)
            results=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r['id'] for r in results],['0','1','2'])
            self.assertTrue(all(len(set(r['codes']))==3 and set(r['codes'])<=set(index.codes) for r in results))
            self.assertEqual(summary['architecture'],runner.ARCHITECTURE)
            self.assertGreaterEqual(summary['fallbacks'],1)
            self.assertEqual(len(calls),6 if runner.ARCHITECTURE=='two-pass' else 3)
