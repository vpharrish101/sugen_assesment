"""Contract tests for MedGemma's constrained catalogue selection."""

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


SOURCE = Path(__file__).parent


class Response:
    def __init__(self, value):
        self.value = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, *_):
        return self.value


class FakeOllama:
    def __init__(self, invalid_on=None, crash_on=None):
        self.calls = []
        self.invalid_on = invalid_on
        self.crash_on = crash_on

    def open(self, request, timeout=600):
        if isinstance(request, str):
            return Response({'models': [{'name': runner.MODEL}]})
        payload = json.loads(request.data)
        self.calls.append(payload)
        if len(self.calls) == self.crash_on:
            self.crash_on = None
            raise OSError('simulated Ollama outage')
        codes = payload['format']['properties']['code']['enum']
        code = 'NOT_IN_CATALOGUE' if len(self.calls) == self.invalid_on else codes[0]
        return Response({'message': {'content': json.dumps({'code': code})},
                         'done_reason': 'stop', 'eval_count': 12})


class MedGemmaRunnerTests(unittest.TestCase):
    def setUp(self):
        self.row = json.loads((SOURCE / 'candidates.jsonl').read_text().splitlines()[0])

    def test_one_choice_schema_contains_only_remaining_codes(self):
        first = runner.make_payload(self.row, runner.MODEL, self.row['candidates'], 1)
        self.assertEqual(first['format']['properties']['code']['enum'],
                         [c['code'] for c in self.row['candidates']])
        self.assertEqual(first['format']['properties']['code']['type'], 'string')
        self.assertNotIn('1. ', first['messages'][1]['content'])
        self.assertEqual(first['keep_alive'], '30m')
        chosen = self.row['candidates'][0]['code']
        remaining = [c for c in self.row['candidates'] if c['code'] != chosen]
        second = runner.make_payload(self.row, runner.MODEL, remaining, 2)
        self.assertNotIn(chosen, second['format']['properties']['code']['enum'])

    def test_only_exact_catalogue_code_is_accepted(self):
        candidates = self.row['candidates']
        selected = candidates[1]['code']
        self.assertEqual(runner.read_code({'message': {'content': json.dumps({'code': selected})}},
                                          candidates), selected)
        for content in ('{"code":"1E90.9"}', '{"code":33}',
                        '{"choices":[1,2,3]}', '{"code":"' + selected + '","extra":1}'):
            with self.assertRaises((ValueError, TypeError)):
                runner.read_code({'message': {'content': content}}, candidates)

    def test_four_note_smoke_and_invalid_response_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('candidates.jsonl', 'dev_notes.jsonl'):
                shutil.copyfile(SOURCE / name, root / name)
            fake = FakeOllama(invalid_on=10)
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', ['runner.py', '--limit', '4', '--output-prefix', 'smoke']), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            predictions = [json.loads(s) for s in (root / 'smoke_predictions.jsonl').read_text().splitlines()]
            traces = [json.loads(s) for s in (root / 'smoke_trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(predictions), 4)
            self.assertEqual(len(fake.calls), 10)
            self.assertEqual([t['fallback_error'] is None for t in traces], [True, True, True, False])
            rows = [json.loads(s) for s in (root / 'candidates.jsonl').read_text().splitlines()]
            for predicted, row in zip(predictions[:3], rows[:3]):
                self.assertEqual(predicted['codes'], [c['code'] for c in row['candidates'][:3]])
            self.assertEqual(predictions[3]['codes'], rows[3]['fallback_codes'])
            old_trace = root / 'old_trace.jsonl'
            old_output = root / 'old_predictions.jsonl'
            old_trace.write_text(json.dumps({'id': 'dev-001', 'runner_version': 4,
                                             'model': runner.MODEL}) + '\n')
            old_output.write_text(json.dumps(predictions[0]) + '\n')
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(sys, 'argv', ['runner.py', '--resume', '--output-prefix', 'old']), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                runner.main()
            self.assertEqual(len(old_output.read_text().splitlines()), 1)

    def test_server_outage_can_resume_same_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('candidates.jsonl', 'dev_notes.jsonl'):
                shutil.copyfile(SOURCE / name, root / name)
            fake = FakeOllama(crash_on=2)
            args = ['runner.py', '--limit', '1', '--output-prefix', 'recovery']
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', args), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaises(SystemExit):
                runner.main()
            self.assertEqual((root / 'recovery_predictions.jsonl').read_text(), '')
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', args + ['--resume']), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(len((root / 'recovery_predictions.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
