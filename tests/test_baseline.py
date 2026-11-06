import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from coder.__main__ import read_notes
from coder.bm25 import BM25
from coder.clinical_query import diagnosis_focus
from coder.retriever import CandidateRetriever, HybridRetriever, reciprocal_rank_fusion
from coder.semantic import catalogue_digest


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

    def test_diagnosis_focus_uses_generic_cues_and_falls_back(self):
        self.assertEqual(
            diagnosis_focus("S: coughing. A: Asthma with exacerbation. P: inhaler."),
            "Asthma with exacerbation.",
        )
        self.assertEqual(diagnosis_focus("Dx viral upper respiratory infection."), "viral upper respiratory infection.")
        original = "Cough and fever without an explicit assessment"
        self.assertEqual(diagnosis_focus(original), original)

    def test_diagnosis_focus_ignores_patient_suspicion_and_ruled_out_cues(self):
        note = (
            "Patient: I think it's my thyroid.\n"
            "Doctor: Tests are normal, so it's not the thyroid.\n"
            "Doctor: BMI is in the obesity range."
        )
        self.assertEqual(diagnosis_focus(note), note)

    def test_family_pool_uses_only_real_catalogue_codes(self):
        index = BM25([
            {"code": "CA23", "title": "Asthma"},
            {"code": "CA23.30", "title": "Asthma with exacerbation"},
            {"code": "CA23.32", "title": "Asthma uncomplicated"},
            {"code": "CA24", "title": "Bronchiectasis"},
            {"code": "ZZ00", "title": "Unrelated condition"},
        ])
        pool = CandidateRetriever(index).candidate_pool(
            "Assessment: asthma attack", limit=5, focused_seeds=1,
            full_note_seeds=1, expanded_families=1,
        )
        self.assertEqual(len(pool), len(set(pool)))
        self.assertLessEqual(set(pool), set(index.codes))
        self.assertNotIn("CA23.3212", pool)

    def test_rank_fusion_is_deterministic_and_validates_weights(self):
        self.assertEqual(
            reciprocal_rank_fusion(
                [["A", "B", "C"], ["B", "C", "D"]], limit=4, constant=0
            ),
            ["B", "A", "C", "D"],
        )
        with self.assertRaisesRegex(ValueError, "one fusion weight"):
            reciprocal_rank_fusion([["A"]], [1.0, 2.0])

    def test_catalogue_digest_tracks_code_title_order(self):
        rows = [{"code": "A", "title": "One"}, {"code": "B", "title": "Two"}]
        self.assertEqual(catalogue_digest(rows), catalogue_digest(list(rows)))
        self.assertNotEqual(catalogue_digest(rows), catalogue_digest(list(reversed(rows))))

    def test_hybrid_reverses_model_priority_for_non_english_notes(self):
        class FakeDense:
            def __init__(self, codes):
                self.codes = codes

            def search_many(self, texts, limit):
                return [[(code, 1.0) for code in self.codes[:limit]] for _ in texts]

        hybrid = HybridRetriever(
            CandidateRetriever(self.index),
            FakeDense(["A", "B", "C"]),
            FakeDense(["B", "C", "A"]),
        )
        with patch.object(HybridRetriever, "language", side_effect=["en", "sw"]):
            ranked = hybrid.rank_many(["English note", "Dokezo la Kiswahili"], 3)
        self.assertEqual(ranked[0], ["A", "B", "C"])
        self.assertEqual(ranked[1], ["B", "C", "A"])


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
