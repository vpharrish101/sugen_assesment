"""Two-pass contract and failure-recovery checks; no GPU required."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import run_ollama_experiment as baseline
import run_two_pass_experiment as runner
from hierarchy_reranker import Catalogue
from run_hierarchy_experiment import read_jsonl
from two_pass_reranker import (first_pass_payload, shortlist, verifier_payload,
                              validate_cached_predictions)

ROOT = Path(__file__).parent


class Response:
    def __init__(self, value):
        self.value = json.dumps(value).encode()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def read(self, *_):
        return self.value


class Ollama:
    def __init__(self, failures=False):
        self.payloads = []
        self.failures = failures
    def open(self, request, timeout=600):
        if isinstance(request, str):
            return Response({'models': [{'name': baseline.MODEL}]})
        payload = json.loads(request.data)
        self.payloads.append(payload)
        if payload['format'] == 'json':
            content = '{"choices":[2,3,1]}'
        else:
            note = payload['messages'][1]['content'].split('CLINICAL NOTE\n')[1].split('\n\nDIAGNOSIS-FOCUS HINT')[0]
            choices = ['C002', 'C001', 'C003']
            if self.failures and len(self.payloads) == 4:
                choices = ['C001', 'C001', 'C002']
            evidence = 'invented quote' if self.failures and len(self.payloads) == 6 else note[:30]
            content = json.dumps({'choices': choices, 'evidence': evidence})
        return Response({'message': {'content': content}})


class TwoPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(ROOT / 'candidates.jsonl')
        cls.catalogue = Catalogue(json.loads((ROOT / 'mms-2026-01.json').read_text()))

    def copy_inputs(self, folder):
        root = Path(folder)
        for name in ('candidates.jsonl', 'mms-2026-01.json', 'dev_notes.jsonl'):
            shutil.copyfile(ROOT / name, root / name)
        return root

    def test_first_pass_matches_original_runner_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            fake = Ollama()
            with mock.patch.object(baseline, '__file__', str(root / 'baseline.py')), \
                 mock.patch.object(baseline.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', ['baseline.py', '--limit', '1']), \
                 contextlib.redirect_stdout(io.StringIO()):
                baseline.main()
            self.assertEqual(fake.payloads[0], first_pass_payload(self.rows[0], baseline.MODEL))

    def test_shortlist_is_focused_on_first_pass_families(self):
        first = ['BC81.3', 'BC81.31', 'BC81.30']
        candidates = shortlist(self.rows[49], first, self.catalogue)
        codes = [c['code'] for c in candidates]
        self.assertEqual(codes[:3], first)
        self.assertIn('BC81.3Z', codes)
        self.assertLessEqual(len(codes), 24)
        self.assertTrue({c['code'] for c in self.rows[49]['candidates'][:10]} <= set(codes))
        payload = verifier_payload(self.rows[49], candidates, baseline.MODEL)
        self.assertIn('PROVISIONAL RANKING: C001, C002, C003', payload['messages'][1]['content'])

    def test_fresh_two_pass_and_resume_preserve_first_pass_on_bad_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            fake = Ollama(failures=True)
            original_read = runner.read_jsonl
            def check_read(path):
                if path.name == 'dev_notes.jsonl':
                    self.assertEqual(len(fake.payloads), 120, 'Gold read before inference completed')
                return original_read(path)
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner, 'read_jsonl', side_effect=check_read), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(sys, 'argv', ['runner.py', '--limit', '4']):
                    runner.main()
                self.assertEqual(len(fake.payloads), 8)
                with mock.patch.object(sys, 'argv', ['runner.py', '--resume']):
                    runner.main()
            self.assertEqual(len(fake.payloads), 120)
            traces = original_read(root / 'two-pass_trace.jsonl')
            preds = original_read(root / 'two-pass_predictions.jsonl')
            for i in (1, 2):
                self.assertTrue(traces[i]['verifier_error'])
                self.assertEqual(preds[i]['codes'], traces[i]['first_pass_codes'])
            report = json.loads((root / 'two-pass_results.json').read_text())
            self.assertEqual(report['verifier_fallbacks'], 2)
            self.assertEqual(report['first_pass_fallbacks'], 0)
            self.assertEqual(report['local_inference_calls'], 120)

    def test_cached_mode_only_calls_verifier_and_rejects_bad_cache(self):
        saved = [{'id': row['id'], 'codes': row['fallback_codes']} for row in self.rows]
        with self.assertRaises(ValueError):
            validate_cached_predictions(saved[:-1], self.rows)
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root / 'cached.jsonl'
            path.write_text(''.join(json.dumps(p) + '\n' for p in saved))
            fake = Ollama()
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', ['runner.py', '--first-pass-predictions', str(path)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(len(fake.payloads), 60)
            self.assertTrue(all(isinstance(p['format'], dict) for p in fake.payloads))
            report = json.loads((root / 'two-pass_results.json').read_text())
            self.assertEqual(report['mode'], 'saved-first-pass')
            self.assertEqual(report['first_pass']['top1'], 27)
            self.assertEqual(report['first_pass']['top3'], 44)


if __name__ == '__main__':
    unittest.main()
