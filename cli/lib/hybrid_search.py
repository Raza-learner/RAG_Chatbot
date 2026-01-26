import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.document_map = {doc["id"]: doc for doc in documents}
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
        """
        Hybrid search: combine BM25 and chunked semantic scores with alpha weighting.
        Returns top `limit` movies with hybrid scores.
        """
        # 1. Get lots of candidates (500× limit for good coverage)
        bm25_results = self._bm25_search(query, limit=500 * limit)
        sem_results = self.semantic_search.search_chunks(query, limit=500 * limit)

        # 2. Normalize BM25 scores (0–1)
        bm25_scores = [res["score"] for res in bm25_results]
        bm25_norm = normalize_scores(bm25_scores) if bm25_scores else []
        bm25_norm_dict = {bm25_results[i]["id"]: bm25_norm[i] 
                         for i in range(len(bm25_norm))}

        # 3. Normalize semantic scores (0–1)
        sem_scores = [res["score"] for res in sem_results]
        sem_norm = normalize_scores(sem_scores) if sem_scores else []
        sem_norm_dict = {sem_results[i]["id"]: sem_norm[i] 
                        for i in range(len(sem_norm))}

        # 4. Collect all unique movie IDs and compute hybrid score
        all_movie_ids = set(bm25_norm_dict.keys()) | set(sem_norm_dict.keys())
        hybrid_scores = {}

        for movie_id in all_movie_ids:
            bm25_normed = bm25_norm_dict.get(movie_id, 0.0)
            sem_normed = sem_norm_dict.get(movie_id, 0.0)
            hybrid = alpha * bm25_normed + (1 - alpha) * sem_normed
            hybrid_scores[movie_id] = hybrid

        # 5. Sort by hybrid score descending
        sorted_movies = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 6. Build final formatted results
        top_results = []
        for movie_id, hybrid_score in sorted_movies[:limit]:
            movie = self.document_map.get(movie_id)   # ← FIXED HERE
            if movie is None:
                continue

            top_results.append({
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "score": round(hybrid_score, 4),
                "metadata": {
                    "bm25": round(bm25_norm_dict.get(movie_id, 0.0), 4),
                    "semantic": round(sem_norm_dict.get(movie_id, 0.0), 4)
                }
            })

        
        return top_results

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]) -> list[float]:
    """
    Return list of scores normalized to [0, 1] using min-max.
    Returns empty list if input empty.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    denom = max_score - min_score
    return [(s - min_score) / denom for s in scores]
