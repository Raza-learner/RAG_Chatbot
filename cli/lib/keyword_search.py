# lib/keyword_search.py
import os
import string
from typing import List
from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

stemmer = PorterStemmer()

def _load_stopwords() -> set[str]:
    """
    Reads data/stopwords.txt and returns a set of stop words (for O(1) lookup).
    """
    # Find the project root the same way search_utils does
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    stopwords_path = os.path.join(project_root, "data", "stopwords.txt")

    with open(stopwords_path, "r", encoding="utf-8") as f:
        # .read().splitlines() gives us clean lines without \n
        stopwords = {line.strip().lower() for line in f.read().splitlines() if line.strip()}
        
    return stopwords


STOPWORDS = _load_stopwords()

stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """Lowercase + remove punctuation"""
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def tokenize_text(text: str) -> List[str]:
    """
    1. Lowercase + remove punctuation
    2. Split into tokens
    3. Remove empty tokens and stop words
    """
    text = preprocess_text(text)
    raw_tokens = text.split()
    stemmed_tokens = []
    for token in raw_tokens:
        if not token:
            continue
        if token in STOPWORDS:
            continue
        stemmed = stemmer.stem(token)
        stemmed_tokens.append(stemmed)

    return stemmed_tokens
    tokens = [token for token in raw_tokens if token and token not in STOPWORDS]
    return tokens

def tokens_match(query_tokens: List[str], title_tokens: List[str]) -> bool:
    """
    Returns True if **any** token from the query appears in the title tokens.
    """
    if not query_tokens:               
        return False
    return any(qtoken in title_tokens for qtoken in query_tokens)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[dict]:
   
    movies = load_movies()

    query_tokens = tokenize_text(query)
    results = []

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])

        if tokens_match(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results