# lib/inverted_index.py
import os
import pickle
import math
from typing import Dict, Set, List, Tuple

from .keyword_search import tokenize_text


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}

    def add_document(self, doc_id: int, movie: dict) -> None:
        full_text = f"{movie['title']} {movie['description']}"
        tokens = tokenize_text(full_text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
        self.docmap[doc_id] = movie

    def build(self, movies: List[dict]) -> None:
        self.index.clear()
        self.docmap.clear()
        for doc_id, movie in enumerate(movies, start=0):
            self.add_document(doc_id, movie)

    def save(self, cache_dir: str = "cache") -> None:
        os.makedirs(cache_dir, existed_ok=True)
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        print(f"Index saved to {cache_dir}/")

    def load(self) -> None:
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {docmap_path}")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        print(f"Loaded index ({len(self.index):,} terms) and docmap ({len(self.docmap):,} docs)")

    def search(self, query: str, max_results: int = 5) -> List[Tuple[int, str]]:
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        seen_docs = set()
        results = []

        for token in query_tokens:
            if len(results) >= max_results:
                break
            for doc_id in self.index.get(token, set()):
                if doc_id not in seen_docs and len(results) < max_results:
                    seen_docs.add(doc_id)
                    movie = self.docmap[doc_id]
                    results.append((doc_id, movie["title"]))

        return results

    def idf(self, term: str) -> float:
        """
        Calculate IDF using the exact formula from the assignment:
        log((total_docs + 1) / (doc_freq + 1))
        """
        term = term.lower()
        total_docs = len(self.docmap)
        doc_freq = len(self.index.get(term, set()))  # number of docs containing term

        # +1 smoothing to prevent division by zero
        idf_value = math.log((total_docs + 1) / (doc_freq + 1))
        return idf_value
