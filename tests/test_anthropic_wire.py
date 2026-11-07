"""Real SDK/socket tests against a loopback-only fake API and HTTPS proxy.

No provider traffic: the proxy terminates CONNECT locally and never forwards.
Inspired by fake-anthropic's endpoint approach; adds our forced-tool contract.
"""
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import anthropic
import httpx2
from coder.providers.anthropic import AnthropicReranker
from coder.reranker import Reranker


def reply(choices=None, stop='tool_use'):
    return {'id':'msg_test','type':'message','role':'assistant','model':'claude-haiku-5-5',
            'content':[{'type':'tool_use','id':'tool_test','name':'rank_icd_codes',
                        'input':{'choices':choices if choices is not None else [2,1,3]}}],
            'stop_reason':stop,'stop_sequence':None,'usage':{'input_tokens':1000,'output_tokens':20}}


@contextmanager
def server(responses, tls_context=None):
    requests, connects = [], []
    queue = deque(responses)

    class API(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path,dict(self.headers),body))
            status, response, delay = queue.popleft() if queue else (500,{},0)
            if delay == -1:
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            time.sleep(delay)
            raw = response if isinstance(response,bytes) else json.dumps(response).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError): pass

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_CONNECT(self):
            connects.append(self.path)
            if self.path != 'api.anthropic.com:443':
                self.send_error(403)
                return
            self.send_response(200)
            self.end_headers()
            try:
                with tls_context.wrap_socket(self.connection,server_side=True) as connection:
                    API(connection,self.client_address,self.server)
            except (ssl.SSLError,ConnectionResetError): pass

    httpd = ThreadingHTTPServer(('127.0.0.1',0),Proxy if tls_context else API)
    thread = threading.Thread(target=httpd.serve_forever,daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', requests, connects
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


class AnthropicWireTests(unittest.TestCase):
    @contextmanager
    def client(self, responses, timeout=2):
        with server(responses) as (url,requests,_):
            # Replace only the destination in tests. Actual SDK serializer,
            # socket transport, response parser and production adapter execute.
            real_constructor = anthropic.Anthropic
            def factory(**options):
                options['base_url'] = url
                return real_constructor(**options,http_client=httpx2.Client(trust_env=False))
            with patch('anthropic.Anthropic',side_effect=factory):
                api = AnthropicReranker('test-key-never-real','claude-haiku-5-5',timeout)
            try: yield api, requests
            finally: api.close()

    def rank(self,api):
        return api.rerank('Assessment: asthma', ['A','B','C'], dict(A='Asthma',B='Pneumonia',C='Diabetes'))

    def test_success_and_actual_auth_version_tool_headers(self):
        with self.client([(200,reply(),0)]) as (api,requests):
            self.assertEqual(self.rank(api),['B','A','C'])
            path,headers,body=requests[0]
            self.assertEqual(path,'/v1/messages')
            headers={k.lower():v for k,v in headers.items()}
            self.assertEqual(headers['authorization'],'Bearer test-key-never-real')
            self.assertTrue(headers['anthropic-version'])
            self.assertEqual(body['model'],'claude-haiku-5-5')
            self.assertEqual(body['tool_choice'],{'type':'tool','name':'rank_icd_codes',
                                                'disable_parallel_tool_use':True})
            self.assertEqual(body['thinking'],{'type':'disabled'})
            self.assertEqual(body['tools'][0]['input_schema']['properties']['choices']['items']['enum'],[1,2,3])

    def test_http_failure_matrix_no_hidden_retries(self):
        for status in (400,401,403,404,413,429,500,529):
            with self.subTest(status=status), self.client([(status,{'type':'error','error':{
                'type':'api_error','message':'deliberate test failure'}},0)]) as (api,requests):
                with self.assertRaises(anthropic.APIStatusError): self.rank(api)
                self.assertEqual(len(requests),1)
                self.assertEqual(api.calls,1)
                self.assertGreater(api.reserved_usd,0)

    def test_invalid_and_truncated_output_matrix(self):
        missing_tool=reply();missing_tool['content']=[{'type':'text','text':'Hello world!'}]
        duplicate_tools=reply();duplicate_tools['content']*=2
        wrong_tool=reply();wrong_tool['content'][0]['name']='different_tool'
        empty=reply();empty['content']=[]
        cases=[b'not json',missing_tool,duplicate_tools,wrong_tool,empty,reply(stop='max_tokens'),
               reply([1,1,2]),reply([0,2,3]),reply([1,2,4]),reply([True,2,3]),
               reply(['1','2','3']),reply([1,2]),reply([1,2,3,4])]
        for case in cases:
            with self.subTest(case=case), self.client([(200,case,0)]) as (api,requests):
                with self.assertRaises(Exception): self.rank(api)
                self.assertEqual(len(requests),1)

    def test_timeout_and_dropped_connection(self):
        for delay in (.2,-1):
            with self.subTest(delay=delay),self.client([(200,reply(),delay)],timeout=.05) as (api,requests):
                with self.assertRaises(anthropic.APIConnectionError): self.rank(api)
                self.assertEqual(len(requests),1)

    def test_circuit_breaker_after_three_real_http_failures(self):
        with self.client([(429,{'type':'error','error':{'type':'rate_limit_error','message':'limited'}},0)]*3) as (api,requests):
            with patch.dict(os.environ,{},clear=True): router=Reranker('none','unused','http://127.0.0.1:11434',1)
            router.clients=[('anthropic',api)]
            for _ in range(12): self.assertIsNone(router.rerank('Note',['A','B','C'],dict(A='A',B='B',C='C'),1))
            self.assertEqual(len(requests),3)
            self.assertEqual(router.routes['retrieval'],12)

    def test_real_https_connect_proxy_and_certificate_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            cert,key=Path(directory)/'cert.pem',Path(directory)/'key.pem'
            supplied=os.getenv('CODER_TEST_TLS_DIR')
            if supplied:
                cert,key=Path(supplied)/'cert.pem',Path(supplied)/'key.pem'
            else:
                subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1',
                                '-keyout',str(key),'-out',str(cert),'-subj','/CN=api.anthropic.com',
                                '-addext','subjectAltName=DNS:api.anthropic.com'],check=True,capture_output=True)
            tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(cert,key)
            with server([(200,reply(),0)],tls) as (proxy,requests,connects):
                environment={'HTTPS_PROXY':proxy,'https_proxy':proxy,'NO_PROXY':'','no_proxy':'',
                             'ALL_PROXY':'','all_proxy':''}
                real_constructor=anthropic.Anthropic
                def factory(**options):
                    # Trust just the ephemeral test CA; hostname verification
                    # remains enabled. Production keeps its normal trust store.
                    return real_constructor(**options,http_client=httpx2.Client(verify=ssl.create_default_context(cafile=str(cert))))
                with patch.dict(os.environ,environment),patch('anthropic.Anthropic',side_effect=factory):
                    api=AnthropicReranker('test-key-never-real','claude-haiku-5-5',2)
                    try: self.assertEqual(self.rank(api),['B','A','C'])
                    finally:api.close()
                self.assertEqual(connects,['api.anthropic.com:443'])
                self.assertEqual(len(requests),1)

    def test_untrusted_tls_certificate_fails_without_sending_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            cert, key = Path(directory)/'cert.pem', Path(directory)/'key.pem'
            subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1',
                            '-keyout',str(key),'-out',str(cert),'-subj','/CN=api.anthropic.com',
                            '-addext','subjectAltName=DNS:api.anthropic.com'],
                           check=True,capture_output=True)
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(cert, key)
            with server([(200,reply(),0)],tls) as (proxy,requests,connects):
                real_constructor = anthropic.Anthropic

                def factory(**options):
                    # The test certificate is deliberately absent from the CA store.
                    return real_constructor(**options, http_client=httpx2.Client(
                        verify=ssl.create_default_context()))

                environment = {'HTTPS_PROXY':proxy,'https_proxy':proxy,'NO_PROXY':'',
                               'no_proxy':'','ALL_PROXY':'','all_proxy':''}
                with patch.dict(os.environ,environment),patch('anthropic.Anthropic',side_effect=factory):
                    api = AnthropicReranker('test-key-never-real','claude-haiku-5-5',2)
                    try:
                        with self.assertRaises(anthropic.APIConnectionError):
                            self.rank(api)
                    finally:
                        api.close()
                self.assertEqual(connects,['api.anthropic.com:443'])
                self.assertEqual(requests, [])


@unittest.skipUnless((os.getenv('CODER_TEST_ASSETS') and os.getenv('CODER_TEST_NOTES')),'set CODER_TEST_ASSETS for the three-note integration check')
class ThreeNoteWireTests(unittest.TestCase):
    def test_three_notes_real_retrieval_sdk_http_and_failure_fallback(self):
        from coder.__main__ import run
        assets=Path(os.environ['CODER_TEST_ASSETS'])
        root=Path(__file__).resolve().parents[1]
        notes=[json.loads(line) for line in Path(os.environ['CODER_TEST_NOTES']).read_text().splitlines()][:3]
        responses=[(200,reply(),0),(429,{'type':'error','error':{'type':'rate_limit_error','message':'test-only'}},0),(200,reply(),0)]
        with tempfile.TemporaryDirectory() as directory:
            source,output=Path(directory)/'input.jsonl',Path(directory)/'predictions.jsonl'
            source.write_text(''.join(json.dumps({'id':f'wire-{i}','text':row['text']})+'\n' for i,row in enumerate(notes)))
            with server(responses) as (url,requests,_):
                real_constructor=anthropic.Anthropic
                def factory(**options):
                    options['base_url']=url
                    return real_constructor(**options,http_client=httpx2.Client(trust_env=False))
                with patch.dict(os.environ,{'ANTHROPIC_API_KEY':'test-key-never-real'},clear=True),patch('anthropic.Anthropic',side_effect=factory):
                    summary=run(source,output,root/'data/mms-2026-01.json',semantic_assets=assets)
            result=[json.loads(line) for line in output.read_text().splitlines()]
            valid={r['code'] for r in json.loads((root/'data/mms-2026-01.json').read_text())}
            self.assertEqual([r['id'] for r in result],['wire-0','wire-1','wire-2'])
            self.assertTrue(all(len(set(r['codes']))==3 and set(r['codes'])<=valid for r in result))
            self.assertEqual(len(requests),3)
            self.assertEqual(summary['routes'],{'anthropic':2,'retrieval':1})
            self.assertEqual(sum(summary['language_counts'].values()),3)
            self.assertTrue(summary['semantic_enabled'])


if __name__=='__main__':unittest.main()
