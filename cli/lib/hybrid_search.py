import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]) -> None:
    """
    Normalize a list of scores to [0, 1] using min-max normalization.
    Handles edge cases: empty list, all scores equal.
    """
    if not scores:
        return  # no output for empty input

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        # All scores equal → normalize to 1.0
        for _ in scores:
            print(f"* 1.0000")
        return

    # Normal min-max normalization
    denom = max_score - min_score
    for score in scores:
        normalized = (score - min_score) / denom
        print(f"* {normalized:.4f}")
