"""Optional real-encoder + actual SDK tests, with inference HTTP simulated."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coder.__main__ import run


@unittest.skipUnless((os.getenv('CODER_TEST_ASSETS') and os.getenv('CODER_TEST_NOTES')) and importlib.util.find_spec('anthropic'),
                     'set CODER_TEST_ASSETS and install API requirements for integration')
class RuntimeIntegrationTests(unittest.TestCase):
    def test_mixed_notes_use_api_qwen_then_retrieval_without_losing_ids(self):
        import httpx2
        from anthropic import Anthropic
        root = Path(__file__).resolve().parents[1]
        assets = Path(os.environ['CODER_TEST_ASSETS'])
        notes = [json.loads(x) for x in Path(os.environ['CODER_TEST_NOTES']).read_text().splitlines()][:3]
        requests, local_calls = [], []

        def api_response(request):
            requests.append(request)
            if len(requests) > 1:
                return httpx2.Response(429, json={'type':'error','error':{'type':'rate_limit_error','message':'simulated outage'}})
            return httpx2.Response(200, json={
                'id':'msg_test', 'type':'message', 'role':'assistant', 'model':'claude-haiku-5-5',
                'content':[{'type':'tool_use','id':'tool_test','name':'rank_icd_codes','input':{'choices':[2,1,3]}}],
                'stop_reason':'tool_use','stop_sequence':None,'usage':{'input_tokens':1000,'output_tokens':30},
            })

        def local_response(text, candidates, titles, timeout=None):
            local_calls.append(text)
            if len(local_calls) > 1:
                raise ConnectionError('simulated local outage')
            return candidates[:3]

        client = Anthropic(api_key='test-only', max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(api_response)))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'notes.jsonl', Path(directory)/'predictions.jsonl'
            source.write_text('\n'.join(json.dumps({'id':f'unseen-{i}','text':row['text']}, ensure_ascii=False)
                                        for i,row in enumerate(notes)), encoding='utf-8')
            with patch.dict(os.environ, {'ANTHROPIC_API_KEY':'test-only','QWEN_BACKEND':'ollama'}, clear=True), \
                 patch('anthropic.Anthropic',return_value=client), \
                 patch('coder.providers.local.LocalReranker.rerank',side_effect=local_response):
                summary = run(source, output, root/'data/mms-2026-01.json', semantic_assets=assets)
            rows = [json.loads(x) for x in output.read_text().splitlines()]
            self.assertEqual([row['id'] for row in rows], ['unseen-0','unseen-1','unseen-2'])
            self.assertTrue(all(len(row['codes']) == 3 and len(set(row['codes'])) == 3 for row in rows))
            self.assertEqual(summary['routes'], {'anthropic':1,'qwen':1,'retrieval':1})
            self.assertEqual(sum(summary['language_counts'].values()), 3)
            self.assertTrue(summary['semantic_enabled'])
            self.assertEqual(summary['api_calls'], 3)
            self.assertEqual(len(requests), 3)
            self.assertEqual(len(local_calls), 2)
            client.close()
