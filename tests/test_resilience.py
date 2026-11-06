"""Optional components must fail visibly without taking predictions down."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from coder.__main__ import run
from coder.reranker import Reranker


class ResilienceTests(unittest.TestCase):
    def test_api_initialization_os_error_falls_back_and_redacts_key(self):
        with patch.dict(os.environ,{'ANTHROPIC_API_KEY':'secret-test-key'},clear=True), \
             patch('coder.providers.anthropic.AnthropicReranker',side_effect=OSError('bad CA secret-test-key')):
            router=Reranker('auto','unused','http://127.0.0.1:11434',1)
        self.assertEqual(router.clients,[])
        self.assertNotIn('secret-test-key',str(router.errors))
        self.assertIn('anthropic_setup',router.errors)

    def test_bad_optional_qwen_settings_do_not_block_api_or_retrieval(self):
        for settings in ({'QWEN_BACKEND':'typo'},{'QWEN_BACKEND':'ollama','QWEN_ENDPOINT':'https://bad.invalid'}):
            with self.subTest(settings=settings),patch.dict(os.environ,settings,clear=True):
                router=Reranker('auto','unused','http://127.0.0.1:11434',1)
                self.assertIsNone(router.rerank('Note',['A','B','C'],dict(A='A',B='B',C='C'),1))
                self.assertIn('qwen_setup',router.errors)

    def test_client_close_failure_does_not_turn_completed_run_into_failure(self):
        with patch.dict(os.environ,{},clear=True):router=Reranker('none','unused','http://127.0.0.1:11434',1)
        router.api=Mock();router.api.close.side_effect=OSError('socket cleanup failed')
        router.close()
        self.assertIn('anthropic_close',router.errors)

    def test_optional_failures_keep_every_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.jsonl';output=root/'out.jsonl';catalogue=root/'mms.json'
            catalogue.write_text(json.dumps([{'code':'A','title':'Asthma'},{'code':'B','title':'Pneumonia'},{'code':'C','title':'Diabetes'}]))
            source.write_text(''.join(json.dumps({'id':str(i),'text':'Asthma'})+'\n' for i in range(3)))
            cases=[KeyError('damaged metadata'),TypeError('bad metadata')]
            for error in cases:
                with self.subTest(error=error),patch('coder.__main__.load_hybrid',side_effect=error),patch.dict(os.environ,{},clear=True):
                    summary=run(source,output,catalogue)
                self.assertEqual(len(output.read_text().splitlines()),3)
                self.assertFalse(summary['semantic_enabled'])
            for result in (([['A','B','C']], [['A','B','C']]),
                           ([['A','B','C']]*3, [['BAD']]*3)):
                hybrid=Mock();hybrid.retrieve_many.return_value=result;hybrid.language_counts={}
                with self.subTest(result=result),patch('coder.__main__.load_hybrid',return_value=hybrid),patch.dict(os.environ,{},clear=True):
                    summary=run(source,output,catalogue)
                self.assertEqual([json.loads(x)['id'] for x in output.read_text().splitlines()],['0','1','2'])
                self.assertEqual(summary['fallback_batches'],1)

    def test_error_diagnostics_include_no_key(self):
        key='secret-test-key'
        with patch.dict(os.environ,{'ANTHROPIC_API_KEY':key},clear=True):
            with patch('coder.providers.anthropic.AnthropicReranker'):
                router=Reranker('auto','unused','http://127.0.0.1:11434',1)
            client=Mock();client.rerank.side_effect=ValueError('bad header '+key)
            router.clients=[('anthropic',client)]
            self.assertIsNone(router.rerank('Note',['A','B','C'],{},1))
        self.assertNotIn(key,str(router.errors))
