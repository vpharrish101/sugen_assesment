"""Provider contract tests use the actual SDK with an in-memory HTTP transport."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coder.__main__ import run
from coder.prompt import make_messages, parse_choices
from coder.reranker import Reranker


class OutputContractTests(unittest.TestCase):
    def test_ollama_request_keeps_the_measured_model_settings(self):
        import io
        from unittest.mock import Mock
        from coder.providers.local import LocalReranker
        local = LocalReranker('ollama', 'http://127.0.0.1:11434', 'configured-qwen', 30)
        local.client = Mock()
        local.client.open.return_value = io.BytesIO(json.dumps({
            'done_reason':'stop', 'message':{'content':'{"choices":[3,1,2]}'},
        }).encode())
        self.assertEqual(local.rerank('Note', ['A','B','C'], dict(A='A',B='B',C='C')), ['C','A','B'])
        request = local.client.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/chat')
        self.assertEqual(payload['model'], 'configured-qwen')
        self.assertEqual(payload['options'], {'temperature':0,'seed':42,'num_ctx':8192,'num_predict':256})
        self.assertEqual(payload['format'], 'json')
        self.assertFalse(payload['stream'])

    def test_numbers_are_strict_distinct_and_bounded(self):
        self.assertEqual(parse_choices('{"choices":[2,1,3]}', ['A', 'B', 'C']), ['B', 'A', 'C'])
        for choices in ([1, 1, 3], [1, 2, 4], [True, 2, 3], [1, 2], ['1', '2', '3']):
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                parse_choices(json.dumps({'choices': choices}), ['A', 'B', 'C'])

    def test_measured_system_prompt_is_preserved(self):
        import hashlib
        from coder.prompt import SYSTEM
        self.assertEqual(hashlib.sha256(SYSTEM.encode()).hexdigest(), 'ba9ade57a1bc39b064aebc8bcea3b1944776ae2fc7a731b20736246401a41e27')

    def test_api_first_then_qwen_then_retrieval(self):
        with patch.dict(os.environ, {}, clear=True):
            router = Reranker('none', 'Qwen', 'http://127.0.0.1:11434', 30)
        from unittest.mock import Mock
        api, qwen = Mock(), Mock()
        api.rerank.return_value = ['A', 'B', 'C']
        qwen.rerank.return_value = ['C', 'B', 'A']
        router.clients = [('anthropic', api), ('qwen', qwen)]
        self.assertEqual(router.rerank('note', ['A','B','C'], {}, 30), ['A','B','C'])
        qwen.rerank.assert_not_called()
        api.rerank.side_effect = TimeoutError('API down')
        self.assertEqual(router.rerank('note', ['A','B','C'], {}, 30), ['C','B','A'])
        qwen.rerank.side_effect = ConnectionError('local down')
        for _ in range(6):
            self.assertIsNone(router.rerank('note', ['A','B','C'], {}, 30))
        self.assertEqual(api.rerank.call_count, 4)  # first success, then three failures
        self.assertEqual(qwen.rerank.call_count, 4) # first success, then three failures

    def test_outages_still_write_every_unseen_input_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalogue, source, output = root/'mms.json', root/'notes.jsonl', root/'out.jsonl'
            catalogue.write_text(json.dumps([{'code':'A','title':'Asthma'}, {'code':'B','title':'Pneumonia'}, {'code':'C','title':'Diabetes'}]))
            source.write_text('\n'.join(json.dumps({'id':f'new-{i}', 'text':'Assessment: Asthma', 'gold':['B']}) for i in range(7)))
            with patch.dict(os.environ, {'QWEN_BACKEND':'ollama'}, clear=True), \
                 patch('coder.providers.local.LocalReranker.rerank', side_effect=ConnectionError('offline')) as call:
                summary = run(source, output, catalogue, semantic_assets=root/'missing')
            rows = [json.loads(x) for x in output.read_text().splitlines()]
            self.assertEqual([row['id'] for row in rows], [f'new-{i}' for i in range(7)])
            self.assertTrue(all(row['codes'][0] == 'A' for row in rows))
            self.assertEqual(call.call_count, 3)
            self.assertEqual(summary['routes'], {'retrieval':7})

    def test_time_cap_skips_inference_but_completes_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalogue, source, output = root/'mms.json', root/'notes.jsonl', root/'out.jsonl'
            catalogue.write_text(json.dumps([{'code':'A','title':'Asthma'}, {'code':'B','title':'Pneumonia'}, {'code':'C','title':'Diabetes'}]))
            source.write_text('\n'.join(json.dumps({'id':str(i), 'text':'Asthma'}) for i in range(4)))
            with (patch.dict(os.environ, {'QWEN_BACKEND':'ollama'}, clear=True),
                  patch('coder.providers.local.LocalReranker.rerank') as call):
                summary = run(source, output, catalogue, semantic_assets=root/'missing', max_seconds=0.000001)
            call.assert_not_called()
            self.assertEqual(len(output.read_text().splitlines()), 4)
            self.assertEqual(summary['reranker_disabled_reason'], 'inference time budget reached')


@unittest.skipUnless(importlib.util.find_spec('anthropic'), 'install requirements-api.txt')
class AnthropicSDKTests(unittest.TestCase):
    def client(self, handler, **options):
        import httpx2
        from anthropic import Anthropic
        from coder.providers.anthropic import AnthropicReranker
        # SDK receives fake credentials; MockTransport prevents all network IO.
        client = Anthropic(api_key='test-only', max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        with patch('anthropic.Anthropic', return_value=client):
            result = AnthropicReranker('test-only', 'claude-haiku-5-5', 20, **options)
        self.addCleanup(result.close)
        return result

    def response(self, choices):
        import httpx2
        return httpx2.Response(200, json={
            'id':'msg_test', 'type':'message', 'role':'assistant', 'model':'claude-haiku-5-5',
            'content':[{'type':'tool_use', 'id':'tool_test', 'name':'rank_icd_codes', 'input':{'choices':choices}}],
            'stop_reason':'tool_use', 'stop_sequence':None,
            'usage':{'input_tokens':1000, 'output_tokens':30},
        })

    def test_real_sdk_serializes_tool_request_and_parses_response(self):
        requests = []
        def handle(request):
            requests.append(request)
            return self.response([2, 1, 3])
        api = self.client(handle)
        self.assertEqual(api.rerank('Assessment: asthma', ['A','B','C'], dict(A='Asthma',B='Pneumonia',C='Diabetes')), ['B','A','C'])
        request = requests[0]
        payload = json.loads(request.content)
        self.assertEqual(str(request.url), 'https://api.anthropic.com/v1/messages')
        self.assertEqual(payload['model'], 'claude-haiku-5-5')
        self.assertEqual(payload['tool_choice'], {'type':'tool','name':'rank_icd_codes'})
        self.assertEqual(payload['max_tokens'], 256)
        self.assertEqual(api.calls, 1)
        self.assertGreater(api.reserved_usd, 0)

    def test_budget_exhaustion_never_makes_a_request(self):
        def forbidden(request):
            self.fail('over-budget request reached transport')
        api = self.client(forbidden, budget_usd=0.000001)
        with self.assertRaisesRegex(ValueError, 'spend reservation'):
            api.rerank('Note', ['A','B','C'], dict(A='A',B='B',C='C'))
        self.assertEqual(api.calls, 0)

    def test_http_error_is_not_retried_and_keeps_reservation(self):
        import httpx2
        requests = []
        def handle(request):
            requests.append(request)
            return httpx2.Response(429, json={'type':'error','error':{'type':'rate_limit_error','message':'rate limited'}})
        api = self.client(handle)
        with self.assertRaises(Exception):
            api.rerank('Note', ['A','B','C'], dict(A='A',B='B',C='C'))
        self.assertEqual(len(requests), 1)
        self.assertGreater(api.reserved_usd, 0)

    def test_sdk_retains_https_proxy_from_environment(self):
        from coder.providers.anthropic import AnthropicReranker
        with patch.dict(os.environ, {'HTTPS_PROXY':'http://test-proxy.invalid:8080', 'HTTP_PROXY':'', 'ALL_PROXY':'', 'NO_PROXY':'', 'https_proxy':'http://test-proxy.invalid:8080', 'http_proxy':'', 'all_proxy':'', 'no_proxy':''}):
            api = AnthropicReranker('test-only','claude-haiku-5-5',20)
            self.addCleanup(api.close)
            mounts = api.client._client._mounts.values()
            proxies = [str(transport._pool._proxy_url) for transport in mounts
                       if transport is not None and hasattr(transport._pool,'_proxy_url')]
            self.assertTrue(any('test-proxy.invalid' in value for value in proxies))


if __name__ == '__main__':
    unittest.main()
