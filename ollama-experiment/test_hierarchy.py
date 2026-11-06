"""Catalogue integrity and one-call experiment checks; no GPU required."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from hierarchy_reranker import Catalogue, make_payload, parse_result
import run_hierarchy_experiment as runner

ROOT = Path(__file__).parent


class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = Catalogue(json.loads((ROOT / 'mms-2026-01.json').read_text()))
        cls.rows = runner.read_jsonl(ROOT / 'candidates.jsonl')

    def test_nested_residuals_have_the_correct_coded_parent(self):
        self.assertEqual(self.catalogue.parent['BC81.3Z'], 'BC81.3')
        self.assertEqual(self.catalogue.parent['5A00.2Z'], '5A00.2')
        self.assertEqual(self.catalogue.kind('CA40.Z'), 'unspecified')
        self.assertEqual(len(self.catalogue.parent), 12917)

    def test_every_pool_keeps_original_candidates_and_bounds_expansion(self):
        for row in self.rows:
            before = json.dumps(row)
            candidates = self.catalogue.expand(row)
            codes = [c['code'] for c in candidates]
            self.assertEqual(codes[:50], [c['code'] for c in row['candidates']])
            self.assertLessEqual(len(codes), 80)
            self.assertEqual(len(set(codes)), len(codes))
            self.assertTrue(set(codes) <= self.catalogue.by_code.keys())
            self.assertEqual(json.dumps(row), before)
        row = dict(self.rows[0], candidates=[self.catalogue.by_code[c]
                                            for c in ('CA40', '1A00', '1A01')])
        self.assertIn('CA40.Z', [c['code'] for c in self.catalogue.expand(row)])

    def test_code_validity_and_quote_grounding_are_separate(self):
        row = self.rows[0]
        candidates = self.catalogue.expand(row)
        response = lambda choices, evidence: {'message': {'content': json.dumps(
            {'choices': choices, 'evidence': evidence})}}
        codes, _, grounded = parse_result(response(['C002', 'C003', 'C001'], row['text'][:30]), candidates, row['text'])
        self.assertEqual(codes, [candidates[i - 1]['code'] for i in (2, 3, 1)])
        self.assertTrue(grounded)
        self.assertFalse(parse_result(response(['C002', 'C003', 'C001'], 'invented quotation'), candidates, row['text'])[2])
        for choices in (['C002', 'C002', 'C001'], ['C000', 'C003', 'C001'], [2, 3, 1], ['1F40', 'C003', 'C001']):
            with self.assertRaises(ValueError):
                parse_result(response(choices, ''), candidates, row['text'])
        payload = make_payload(row, candidates, runner.MODEL)
        self.assertIsInstance(payload['format'], dict)
        self.assertIn('parent=', payload['messages'][1]['content'])
        for candidate in candidates:
            self.assertNotIn(candidate['code'], payload['messages'][1]['content'])

    def test_four_note_resume_then_full_run_has_exactly_one_call_per_note(self):
        class Response:
            def __init__(self, value):
                self.data = json.dumps(value).encode()
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self, *_):
                return self.data

        class Ollama:
            calls = 0
            def open(self, request, timeout=600):
                if isinstance(request, str):
                    return Response({'models': [{'name': runner.MODEL}]})
                self.calls += 1
                payload = json.loads(request.data)
                note = payload['messages'][1]['content'].split('CLINICAL NOTE\n')[1].split('\n\nDIAGNOSIS-FOCUS HINT')[0]
                choices = ['C001', 'C001', 'C002'] if self.calls == 3 else ['C002', 'C003', 'C001']
                return Response({'message': {'content': json.dumps({'choices': choices, 'evidence': note[:30]})}})

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('candidates.jsonl', 'mms-2026-01.json', 'dev_notes.jsonl'):
                shutil.copyfile(ROOT / name, root / name)
            fake = Ollama()
            original_read = runner.read_jsonl
            def checked_read(path):
                if path.name == 'dev_notes.jsonl':
                    self.assertEqual(fake.calls, 60, 'Gold was read during inference')
                return original_read(path)
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner, 'read_jsonl', side_effect=checked_read), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(sys, 'argv', ['runner.py', '--limit', '4']):
                    runner.main()
                self.assertEqual(fake.calls, 4)
                with mock.patch.object(sys, 'argv', ['runner.py', '--resume']):
                    runner.main()
            self.assertEqual(fake.calls, 60)
            predictions = original_read(root / 'hierarchy_predictions.jsonl')
            self.assertEqual(len(predictions), 60)
            self.assertEqual(predictions[2]['codes'], self.rows[2]['fallback_codes'])
            report = json.loads((root / 'hierarchy_results.json').read_text())
            self.assertEqual(report['fallbacks'], 1)
            self.assertEqual(report['valid_evidence'], 59)
            self.assertEqual(report['baseline']['top1'], 27)
            self.assertEqual(report['baseline']['top3'], 44)


if __name__ == '__main__':
    unittest.main()
