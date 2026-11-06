"""Protocol checks for the measured Qwen Ollama and bounded llama-server paths."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import run_ollama_experiment as runner
import test_local_reranker as llama_runner


SOURCE = Path(__file__).parent


class Response:
    def __init__(self, data):
        self.data = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, *_):
        return self.data


class OllamaMock:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout=600):
        if isinstance(request, str):
            return Response({'models': [{'name': runner.MODEL}]})
        payload = json.loads(request.data)
        self.requests.append(payload)
        return Response({'message': {'content': '{"choices":[2,3,1]}'},
                         'done_reason': 'stop', 'eval_count': 16})


class QwenRunnerTests(unittest.TestCase):
    def test_four_note_ollama_smoke_preserves_measured_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('candidates.jsonl', 'dev_notes.jsonl'):
                shutil.copyfile(SOURCE / name, root / name)
            fake = OllamaMock()
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', ['runner.py', '--limit', '4', '--output-prefix', 'check']), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(len(fake.requests), 4)
            self.assertTrue(all(p['model'] == runner.MODEL and p['format'] == 'json'
                                and p['options']['num_ctx'] == 8192 for p in fake.requests))
            self.assertIn('three distinct candidate numbers', fake.requests[0]['messages'][0]['content'])
            preds = [json.loads(s) for s in (root / 'check_predictions.jsonl').read_text().splitlines()]
            rows = [json.loads(s) for s in (root / 'candidates.jsonl').read_text().splitlines()]
            self.assertEqual(len(preds), 4)
            for pred, row in zip(preds, rows):
                self.assertEqual(pred['codes'], [row['candidates'][i - 1]['code'] for i in (2, 3, 1)])

    def test_llama_server_disables_prompt_cache(self):
        row = json.loads((SOURCE / 'candidates.jsonl').read_text().splitlines()[0])

        class LlamaMock:
            def open(self, request, timeout=300):
                payload = json.loads(request.data)
                self_payloads.append(payload)
                return Response({'choices': [{'message': {'content':
                    '{"assessment":"malaria","choices":[2,3,1]}'}}]})

        self_payloads = []
        with mock.patch.object(llama_runner.urllib.request, 'build_opener', return_value=LlamaMock()):
            codes, *_ = llama_runner.rerank(row, 'http://127.0.0.1:8097')
        self.assertFalse(self_payloads[0]['cache_prompt'])
        self.assertEqual(codes[0], row['candidates'][1]['code'])


if __name__ == '__main__':
    unittest.main()
