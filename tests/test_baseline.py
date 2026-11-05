import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from coder.__main__ import read_notes
from coder.bm25 import BM25


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = BM25([
            {"code": "A", "title": "Asthma"},
            {"code": "B", "title": "Asthma with exacerbation"},
            {"code": "C", "title": "Pneumonia"},
        ])

    def test_specific_term_changes_first_choice(self):
        self.assertEqual(self.index.search("asthma", 1)[0][0], "A")
        self.assertEqual(self.index.search("asthma exacerbation", 1)[0][0], "B")

    def test_no_overlap_and_empty_notes_still_have_distinct_valid_codes(self):
        for text in ("", "无匹配词", "🩺"):
            ranked = self.index.search(text)
            self.assertEqual([code for code, _ in ranked], ["A", "B", "C"])
            self.assertTrue(all(score == 0 for _, score in ranked))

    def test_repetition_does_not_change_ranking(self):
        self.assertEqual(self.index.search("asthma"), self.index.search("ASTHMA asthma asthma"))

    def test_gold_is_removed_and_duplicate_ids_are_rejected(self):
        note = {"id": "x", "text": "asthma", "gold": ["C"]}
        self.assertEqual(list(read_notes([json.dumps(note)])), [{"id": "x", "text": "asthma"}])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            list(read_notes([json.dumps(note)] * 2))


class RunnerTests(unittest.TestCase):
    def test_cli_contract_with_unseen_ids_no_key_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, catalogue = root / "notes.jsonl", root / "nested" / "pred.jsonl", root / "mms.json"
            catalogue.write_text(json.dumps([
                {"code": "A", "title": "Asthma"}, {"code": "B", "title": "Pneumonia"},
                {"code": "C", "title": "Cholera"},
            ]))
            source.write_text('\n{"id":"新-1","text":"pneumonia","gold":["A"]}\n{"id":"empty","text":""}\n', encoding="utf-8")
            result = subprocess.run([
                sys.executable, "-m", "coder", str(source), str(output), "--mms", str(catalogue),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["新-1", "empty"])
            self.assertEqual(rows[0]["codes"][0], "B")
            self.assertTrue(all(len(row["codes"]) == 3 and set(row["codes"]) == {"A", "B", "C"} for row in rows))
            # An accidental same-file invocation must not truncate the input.
            before = source.read_bytes()
            failed = subprocess.run([
                sys.executable, "-m", "coder", str(source), str(source), "--mms", str(catalogue),
            ], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
