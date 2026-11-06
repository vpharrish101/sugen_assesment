"""Small protocol and recovery tests; no model or GPU required."""

import json
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import run_ollama_experiment as runner
from run_ollama_experiment import make_payload, parse_choices, valid_resume_rows


class OllamaRunnerTests(unittest.TestCase):
    def setUp(self):
        self.row = json.loads((Path(__file__).parent / 'candidates.jsonl').read_text().splitlines()[0])

    def test_memory_options_and_prompt(self):
        payload = make_payload(self.row, 'local-model')
        self.assertEqual(payload['keep_alive'], '30m')
        self.assertEqual(payload['options']['num_ctx'], 8192)
        self.assertEqual(payload['format'], 'json')
        self.assertEqual(make_payload(self.row, 'local-model', cpu=True)['options']['num_gpu'], 0)
        self.assertIn('not ICD codes', make_payload(self.row, 'local-model', retry=True)['messages'][0]['content'])

    def test_response_parsing(self):
        for content in ('{"choices":[2,3,9]}',
                        '```json\n{"choices":[2,3,9]}\n```',
                        '2,3,9'):
            self.assertEqual(parse_choices(content, 50), [2, 3, 9])
        for content in ('The answer is 2. 1F40', '{"choices":[2]}',
                        '{"choices":[2,2,9]}', '{"choices":[51,3,9]}',
                        'Example {"choices":[2,3,9]} is wrong',
                        '{"choices":[1E90, 1E90, 1E90]}'):
            with self.assertRaises(ValueError):
                parse_choices(content, 50)

    def test_resume_retries_failed_and_half_written_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'predictions.jsonl'
            trace = Path(folder) / 'trace.jsonl'
            output.write_text(''.join(json.dumps(r) + '\n' for r in (
                {'id': 'dev-001', 'codes': ['1F40']},
                {'id': 'dev-002', 'codes': ['CA23']},
                {'id': 'dev-003', 'codes': ['A']},
            )))
            trace.write_text(''.join(json.dumps(r) + '\n' for r in (
                {'id': 'dev-001', 'fallback_error': None},
                {'id': 'dev-002', 'fallback_error': 'bad JSON'},
            )))
            self.assertEqual(valid_resume_rows(output, trace), {'dev-001'})
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(len(trace.read_text().splitlines()), 1)

    def test_thirty_note_failure_then_complete_resume(self):
        class Response:
            def __init__(self, value):
                self.value = json.dumps(value).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *_):
                return self.value

        class FakeOllama:
            def __init__(self):
                self.calls = 0
                self.fail_once = True

            def open(self, request, timeout=600):
                if isinstance(request, str):
                    return Response({'models': [{'name': runner.MODEL}]})
                payload = json.loads(request.data)
                assert payload['keep_alive'] == '30m'
                assert payload['options']['num_ctx'] == 8192
                self.calls += 1
                if self.calls == 31 and self.fail_once:
                    self.fail_once = False
                    raise OSError('simulated Ollama crash')
                samples = ('{"choices":[2,12,1]}', '{"choices":[1,2,3]}',
                           '{"choices":[2, 6, 30]}', '{"choices":[1, 18, 30]}')
                content = samples[self.calls - 1] if self.calls <= 4 else '{"choices":[2,3,9]}'
                return Response({'message': {'content': content},
                                 'done_reason': 'stop', 'eval_count': 11})

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('candidates.jsonl', 'dev_notes.jsonl'):
                shutil.copyfile(Path(__file__).parent / name, root / name)
            fake = FakeOllama()
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', ['runner.py', '--output-prefix', 'test']), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main()
            output = root / 'test_predictions.jsonl'
            self.assertEqual(len(output.read_text().splitlines()), 30)
            first_four = [json.loads(line)['codes'] for line in output.read_text().splitlines()[:4]]
            candidates = [json.loads(line)['candidates'] for line in (root / 'candidates.jsonl').read_text().splitlines()[:4]]
            for codes, row_candidates, indices in zip(first_four, candidates,
                    ([2, 12, 1], [1, 2, 3], [2, 6, 30], [1, 18, 30])):
                self.assertEqual(codes, [row_candidates[i - 1]['code'] for i in indices])
            with mock.patch.object(runner, '__file__', str(root / 'runner.py')), \
                 mock.patch.object(runner.urllib.request, 'build_opener', return_value=fake), \
                 mock.patch.object(sys, 'argv', ['runner.py', '--output-prefix', 'test', '--resume']), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(len(output.read_text().splitlines()), 60)
            self.assertEqual(fake.calls, 61)
            self.assertEqual(json.loads((root / 'test_results.json').read_text())['notes'], 60)


if __name__ == '__main__':
    unittest.main()
