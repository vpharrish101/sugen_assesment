"""Three-case CLI smoke: real encoders, SDK, proxy, TLS, and output files.

No SDK monkeypatching in these subprocesses. The local proxy never forwards.
"""
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import unittest

from test_anthropic_wire import reply, server


@unittest.skipUnless((os.getenv('CODER_TEST_ASSETS') and os.getenv('CODER_TEST_NOTES')),'set CODER_TEST_ASSETS for app smoke')
class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.assets=Path(os.environ['CODER_TEST_ASSETS'])
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.work=Path(self.temp.name)
        self.source=self.work/'notes.jsonl'
        self.output=self.work/'predictions.jsonl'
        self.notes=[json.loads(line) for line in Path(os.environ['CODER_TEST_NOTES']).read_text().splitlines()][:3]
        self.source.write_text(''.join(json.dumps({'id':r['id'],'text':r['text']},ensure_ascii=False)+'\n' for r in self.notes),encoding='utf-8')
        self.environment=dict(os.environ)
        for name in ('ANTHROPIC_API_KEY','ANTHROPIC_AUTH_TOKEN','ANTHROPIC_MODEL','QWEN_BACKEND','QWEN_ENDPOINT',
                     'ANTHROPIC_INPUT_USD_PER_MILLION','ANTHROPIC_OUTPUT_USD_PER_MILLION','CODER_API_BUDGET_USD'):
            self.environment.pop(name,None)
        self.valid={r['code'] for r in json.loads((self.root/'data/mms-2026-01.json').read_text())}

    def invoke(self,extra=(),environment=None):
        command=[sys.executable,'-m','coder',str(self.source),str(self.output),'--mms',str(self.root/'data/mms-2026-01.json'),
                 '--semantic-assets',str(self.assets),*extra]
        result=subprocess.run(command,env=environment or self.environment,cwd=self.root,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'')
        rows=[json.loads(x) for x in self.output.read_text().splitlines()]
        self.assertEqual([r['id'] for r in rows],[r['id'] for r in self.notes])
        self.assertTrue(all(0<len(r['codes'])<=3 and len(r['codes'])==len(set(r['codes'])) and set(r['codes'])<=self.valid for r in rows))
        self.assertNotIn('test-key-never-real',result.stderr)
        return json.loads(result.stderr.splitlines()[-1])

    def certificate(self):
        supplied=os.getenv('CODER_TEST_TLS_DIR')
        cert,key=(Path(supplied)/'cert.pem',Path(supplied)/'key.pem') if supplied else (self.work/'cert.pem',self.work/'key.pem')
        if not supplied:
            subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-keyout',str(key),
                            '-out',str(cert),'-subj','/CN=api.anthropic.com','-addext','subjectAltName=DNS:api.anthropic.com'],check=True,capture_output=True)
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(cert,key)
        return cert,context

    def test_cli_key_proxy_tls_success_and_failures_without_sdk_patches(self):
        cert,tls=self.certificate()
        scenarios={
            'success':[(200,reply(),0)]*3,
            'bad_key_model_and_rate_limit':[(status,{'type':'error','error':{'type':'api_error','message':'test-only'}},0) for status in (401,404,429)],
        }
        for name,responses in scenarios.items():
            with self.subTest(name=name),server(responses,tls) as (proxy,requests,connects):
                env=dict(self.environment,ANTHROPIC_API_KEY='test-key-never-real',HTTPS_PROXY=proxy,https_proxy=proxy,
                         NO_PROXY='',no_proxy='',ALL_PROXY='',all_proxy='',SSL_CERT_FILE=str(cert))
                if name!='success':
                    env.update(QWEN_BACKEND='ollama',QWEN_ENDPOINT='http://127.0.0.1:9')
                summary=self.invoke(environment=env)
                self.assertTrue(summary['semantic_enabled'])
                self.assertEqual(len(requests),3)
                self.assertEqual(summary['routes'],{'anthropic':3} if name=='success' else {'retrieval':3})
                self.assertEqual(summary['api_calls'],3)
                self.assertEqual(sum(summary['language_counts'].values()),3)
                self.assertTrue(all(c=='api.anthropic.com:443' for c in connects))
                for _,headers,payload in requests:
                    self.assertEqual(headers.get('Authorization'),'Bearer test-key-never-real')
                    self.assertEqual(payload['model'],'claude-haiku-5-5')
                    self.assertEqual(len(payload['tools'][0]['input_schema']['properties']['choices']['items']['enum']),50)

    def test_no_key_cli_with_proxy_unreachable_stays_offline(self):
        env=dict(self.environment,HTTPS_PROXY='http://127.0.0.1:9',https_proxy='http://127.0.0.1:9')
        summary=self.invoke(environment=env)
        self.assertEqual(summary['api_calls'],0)
        self.assertEqual(summary['routes'],{'retrieval':3})
        self.assertTrue(summary['semantic_enabled'])

    def test_cli_api_priority_then_qwen_then_retrieval_with_real_http(self):
        cert, tls = self.certificate()
        api_replies = [
            (200, reply(), 0),
            (429, {'type':'error','error':{'type':'rate_limit_error','message':'simulated'}}, 0),
            (200, reply([1, 1, 2]), 0),
        ]
        qwen_replies = [
            (200, {'done_reason':'stop','message':{'content':'{"choices":[3,2,1]}'}}, 0),
            (200, {'done_reason':'stop','message':{'content':'not JSON'}}, 0),
        ]
        with server(api_replies, tls) as (proxy, api_requests, _), \
             server(qwen_replies) as (local, qwen_requests, _):
            env = dict(self.environment, ANTHROPIC_API_KEY='test-key-never-real',
                       HTTPS_PROXY=proxy, https_proxy=proxy, NO_PROXY='', no_proxy='',
                       ALL_PROXY='', all_proxy='', SSL_CERT_FILE=str(cert),
                       QWEN_BACKEND='ollama', QWEN_ENDPOINT=local)
            summary = self.invoke(environment=env)
        self.assertEqual(summary['routes'], {'anthropic':1,'qwen':1,'retrieval':1})
        self.assertEqual(len(api_requests), 3)
        self.assertEqual(len(qwen_requests), 2)
        self.assertTrue(all(path == '/api/chat' for path, _, _ in qwen_requests))
        self.assertTrue(all('Authorization' not in headers for _, headers, _ in qwen_requests))

    def test_cli_budget_exhaustion_and_custom_model_configuration(self):
        cert, tls = self.certificate()
        for name, settings in (
            ('budget', {'CODER_API_BUDGET_USD':'0.000001'}),
            ('custom_model', {'ANTHROPIC_MODEL':'custom-test-model',
                              'ANTHROPIC_INPUT_USD_PER_MILLION':'1',
                              'ANTHROPIC_OUTPUT_USD_PER_MILLION':'5'}),
        ):
            with self.subTest(name=name), server([(200, reply(), 0)] * 3, tls) as (proxy, requests, _):
                env = dict(self.environment, ANTHROPIC_API_KEY='test-key-never-real',
                           HTTPS_PROXY=proxy, https_proxy=proxy, NO_PROXY='', no_proxy='',
                           ALL_PROXY='', all_proxy='', SSL_CERT_FILE=str(cert), **settings)
                summary = self.invoke(environment=env)
                if name == 'budget':
                    self.assertEqual(requests, [])
                    self.assertEqual(summary['routes'], {'retrieval':3})
                    self.assertEqual(summary['api_calls'], 0)
                else:
                    self.assertEqual(len(requests), 3)
                    self.assertEqual(summary['routes'], {'anthropic':3})
                    self.assertTrue(all(body['model'] == 'custom-test-model' for _, _, body in requests))

    def test_missing_assets_and_bad_optional_settings_still_complete(self):
        env=dict(self.environment,ANTHROPIC_API_KEY='test-key-never-real',SSL_CERT_FILE='/no/such/ca.pem',QWEN_BACKEND='typo')
        summary=self.invoke(['--semantic-assets',str(self.work/'missing'),'--timeout-seconds','0.1'],env)
        self.assertFalse(summary['semantic_enabled'])
        self.assertEqual(summary['routes'],{'retrieval':3})
        self.assertIn('anthropic_setup',summary['provider_errors'])
        self.assertIn('qwen_setup',summary['provider_errors'])

    def test_invalid_config_and_input_return_clear_error_without_truncation(self):
        before=self.source.read_bytes()
        config=self.work/'bad.json';config.write_text('{invalid')
        result=subprocess.run([sys.executable,'-m','coder',str(self.source),str(self.output),'--config',str(config)],
                              env=self.environment,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2)
        self.assertNotIn('Traceback',result.stderr)
        self.assertEqual(self.source.read_bytes(),before)
        self.assertFalse(self.output.exists())

        self.source.write_text('{bad input')
        malformed=self.source.read_bytes()
        result=subprocess.run([sys.executable,'-m','coder',str(self.source),str(self.output)],
                              env=self.environment,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2)
        self.assertIn('input line 1',result.stderr)
        self.assertNotIn('Traceback',result.stderr)
        self.assertEqual(self.source.read_bytes(),malformed)
        self.assertFalse(self.output.exists())
