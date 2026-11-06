"""Check mixed-language routing without changing the English mixture."""
import unittest
from unittest.mock import patch
from coder.bm25 import BM25
from coder.retriever import CandidateRetriever, HybridRetriever


class DenseSpy:
    def __init__(self, codes):
        self.codes, self.calls = codes, []

    def search_many(self, texts, limit):
        self.calls.append(list(texts))
        return [[(code, 1.0) for code in self.codes[:limit]] for _ in texts]


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.index = BM25([{'code':'A','title':'Asthma'}, {'code':'B','title':'Pneumonia'}, {'code':'C','title':'Diabetes'}])
        self.en, self.multi = DenseSpy(['A','B','C']), DenseSpy(['C','B','A'])
        self.hybrid = HybridRetriever(CandidateRetriever(self.index), self.en, self.multi)

    def test_mixed_batch_keeps_english_mixture_and_routes_other_languages(self):
        texts = ['English one','中文病历','Consulta española','English two']
        with patch.object(HybridRetriever, 'language', side_effect=['en','zh','es','en']):
            fallback, pools = self.hybrid.retrieve_many(texts, 3)
        self.assertEqual(self.en.calls, [[texts[0], texts[3]]])
        self.assertEqual(self.multi.calls, [texts])
        self.assertEqual([row[0] for row in fallback], ['A','C','C','A'])
        self.assertEqual(self.hybrid.language_counts, {'en':2,'zh':1,'es':1})
        self.assertTrue(all(len(set(pool)) == 3 for pool in pools))

    def test_non_english_never_uses_english_encoder_or_zero_bm25_votes(self):
        with patch.object(HybridRetriever, 'language', return_value='zh'):
            fallback, pools = self.hybrid.retrieve_many(['中文病历'], 3)
        self.assertEqual(self.en.calls, [])
        self.assertEqual(fallback, [['C','B','A']])
        self.assertEqual(pools, [['C','B','A']])

    def test_empty_batch_and_zero_limit_do_not_invoke_encoders(self):
        self.assertEqual(self.hybrid.retrieve_many([], 50), ([], []))
        self.assertEqual(self.hybrid.retrieve_many(['x'], 0), ([[]], [[]]))
        self.assertEqual(self.en.calls + self.multi.calls, [])


if __name__ == '__main__':
    unittest.main()
