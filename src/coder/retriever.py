"""Diagnosis-focused retrieval with bounded, catalogue-backed family expansion."""

from collections import Counter, defaultdict

from .bm25 import BM25
from .clinical_query import diagnosis_focus


def code_family(code: str) -> str:
    """Return the supplied catalogue's dot-prefix family key."""
    return code.split(".", 1)[0]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    limit: int = 50,
    constant: int = 60,
) -> list[str]:
    """Fuse rank positions without assuming comparable score scales."""
    if limit <= 0:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(rankings) != len(weights):
        raise ValueError("each ranking needs one fusion weight")
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for position, code in enumerate(ranking):
            scores[code] += weight / (constant + position + 1)
    return sorted(scores, key=lambda code: (-scores[code], code))[:limit]


class HybridRetriever:
    """Keep the measured English mixture; route other languages to XLM-R."""

    def __init__(self, lexical: "CandidateRetriever", english, multilingual):
        self.lexical, self.english, self.multilingual = lexical, english, multilingual
        self.language_counts = Counter()

    @staticmethod
    def language(text: str) -> str:
        import langid
        return langid.classify(text)[0]

    def ranking(self, text: str, limit: int = 50) -> list[str]:
        return self.rank_many([text], limit)[0]

    def retrieve_many(self, texts: list[str], limit: int = 50):
        """Compute fallback ranks and reranker pools with one embedding per route.

        English fallback remains the measured 3:1 English/multilingual mixture.
        English candidate fusion intentionally preserves its original ordering.
        Other languages use XLM-R; BM25 contributes only positive lexical matches.
        """
        if not texts:
            return [], []
        if limit <= 0:
            return [[] for _ in texts], [[] for _ in texts]
        search_limit = min(len(self.lexical.index.codes), max(limit * 10, 500))
        languages = [self.language(text) for text in texts]
        self.language_counts.update(languages)
        focused = [diagnosis_focus(text) for text in texts]
        multilingual = self.multilingual.search_many(focused, search_limit)
        english_positions = [i for i, lang in enumerate(languages) if lang == "en"]
        english = {}
        if english_positions:
            rows = self.english.search_many([focused[i] for i in english_positions], search_limit)
            english = dict(zip(english_positions, rows))
        fallbacks, pools = [], []
        for i, text in enumerate(texts):
            multi_codes = [code for code, _ in multilingual[i]]
            primary = [code for code, _ in english[i]] if i in english else multi_codes
            if i in english:
                fallback = reciprocal_rank_fusion([primary, multi_codes], [3.0, 1.0], limit, constant=10)
            else:
                fallback = multi_codes[:limit]
            lexical = [code for code, score in self.lexical.focused_ranking(text, search_limit)
                       if i in english or score > 0]
            pools.append(reciprocal_rank_fusion([primary, lexical], [1.0, 1.0], limit, constant=60))
            fallbacks.append(fallback)
        return fallbacks, pools

    def rank_many(self, texts: list[str], limit: int = 3) -> list[list[str]]:
        if limit <= 0:
            return [[] for _ in texts]
        fallback, _ = self.retrieve_many(texts, max(limit, 50))
        return [row[:limit] for row in fallback]

    def candidate_pool_many(self, texts: list[str], limit: int = 50) -> list[list[str]]:
        return self.retrieve_many(texts, limit)[1]


class CandidateRetriever:
    """Build a diverse candidate pool without inventing hierarchy or codes."""

    def __init__(self, index: BM25):
        self.index = index
        self.families = defaultdict(list)
        for code in index.codes:
            self.families[code_family(code)].append(code)
        for members in self.families.values():
            members.sort()

    def focused_ranking(self, text: str, limit: int = 50) -> list[tuple[str, float]]:
        focused = self.index.search(diagnosis_focus(text), limit)
        # A colloquial assessment such as "jiggers" may not overlap any title.
        # Fall back to the full note instead of returning arbitrary zero-score codes.
        if focused and focused[0][1] == 0.0:
            return self.index.search(text, limit)
        return focused

    def candidate_pool(
        self,
        text: str,
        limit: int = 50,
        focused_seeds: int = 25,
        full_note_seeds: int = 10,
        expanded_families: int = 5,
    ) -> list[str]:
        """Combine focused, full-note, and catalogue-family candidates.

        The first three remain the focused lexical fallback. Expansion starts
        after the focused seeds, so it supports a future reranker without
        pretending string-family membership is clinical evidence.
        """
        if limit <= 0:
            return []
        search_limit = min(len(self.index.codes), max(limit * 10, 500))
        focused = [code for code, _ in self.focused_ranking(text, search_limit)]
        full_note = [code for code, _ in self.index.search(text, search_limit)]
        selected = []
        seen = set()

        def add(code: str):
            if code not in seen and len(selected) < limit:
                selected.append(code)
                seen.add(code)

        for code in focused[:focused_seeds]:
            add(code)

        added_full = 0
        for code in full_note:
            if code not in seen:
                add(code)
                added_full += 1
                if added_full >= full_note_seeds or len(selected) >= limit:
                    break

        roots = []
        for code in focused[:focused_seeds]:
            root = code_family(code)
            if root not in roots:
                roots.append(root)
            if len(roots) >= expanded_families:
                break
        # Round-robin expansion prevents a large family from consuming the pool.
        queues = [
            [code for code in self.families[root] if code not in seen]
            for root in roots
        ]
        while len(selected) < limit and any(queues):
            for queue in queues:
                if queue:
                    add(queue.pop(0))
                if len(selected) >= limit:
                    break

        for ranking in (focused, full_note):
            for code in ranking:
                add(code)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
        return selected
