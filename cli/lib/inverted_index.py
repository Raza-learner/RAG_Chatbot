# lib/inverted_index.py
import os
import pickle
from typing import Dict, Set, List

from .keyword_search import tokenize_text  # reuse our solid tokenizer + stemming


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}   # token → set of doc_ids
        self.docmap: Dict[int, dict] = {}      # doc_id → full movie dict

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenize text (with stopword removal + stemming) and add all tokens
        to the index pointing to doc_id.
        """
        tokens = tokenize_text(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

        # Store the full document
        self.docmap[doc_id] = text  # we don't actually need the full movie here, but keeping dict for clarity later

    def add_document(self, doc_id: int, movie: dict) -> None:
        """
        Public method: concatenate title + description and index it.
        Also stores the full movie object in docmap.
        """
        full_text = f"{movie['title']} {movie['description']}"
        self.__add_document(doc_id, full_text)
        self.docmap[doc_id] = movie  # store actual movie object

    def get_documents(self, term: str) -> List[int]:
        """
        Return sorted list of document IDs containing the term (case-insensitive).
        """
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self, movies: List[dict]) -> None:
        """
        Build the full index from the list of movie dictionaries.
        """
        self.index.clear()
        self.docmap.clear()
        for doc_id, movie in enumerate(movies, start=0):
            self.add_document(doc_id, movie)

    def save(self, cache_dir: str = "cache") -> None:
        """
        Save index and docmap to disk using pickle.
        Creates the cache directory if it doesn't exist.
        """
        os.makedirs(cache_dir, exist_ok=True)

        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")

        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        print(f"Index saved to {cache_dir}/")
