"""Dependency-free BM25 with an inverted index over the supplied titles."""

from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
import re


TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


class BM25:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        if not records:
            raise ValueError("catalogue must contain at least one code")
        self.records = sorted(records, key=lambda row: row["code"])
        self.codes = [row["code"] for row in self.records]
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("catalogue contains duplicate codes")
        documents = [Counter(tokenize(row["title"])) for row in self.records]
        lengths = [sum(document.values()) for document in documents]
        average_length = sum(lengths) / len(lengths) or 1.0
        postings = defaultdict(list)
        for index, document in enumerate(documents):
            normalization = k1 * (1 - b + b * lengths[index] / average_length)
            for term, frequency in document.items():
                weight = frequency * (k1 + 1) / (frequency + normalization)
                postings[term].append((index, weight))
        self.postings = {}
        for term, entries in postings.items():
            idf = math.log1p((len(documents) - len(entries) + 0.5) / (len(entries) + 0.5))
            self.postings[term] = [(index, weight * idf) for index, weight in entries]

    @classmethod
    def from_file(cls, path: Path):
        with path.open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def search(self, text: str, limit: int = 3) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        scores = defaultdict(float)
        # Query terms count once so repeated transcript mentions do not multiply scores.
        for term in sorted(set(tokenize(text))):
            for index, contribution in self.postings.get(term, ()):
                scores[index] += contribution
        best = heapq.nsmallest(
            limit, scores, key=lambda index: (-scores[index], self.codes[index])
        )
        results = [(self.codes[index], scores[index]) for index in best]
        # Unmatched codes tie at zero; use catalogue code order deterministically.
        # This guarantees coverage, not a clinically meaningful zero-overlap answer.
        if len(results) < min(limit, len(self.codes)):
            selected = {code for code, _ in results}
            for code in self.codes:
                if code not in selected:
                    results.append((code, 0.0))
                    if len(results) == min(limit, len(self.codes)):
                        break
        return results
