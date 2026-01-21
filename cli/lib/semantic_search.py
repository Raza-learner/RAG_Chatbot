from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
import json
from .search_utils import load_movies, CACHE_DIR


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or text.strip() == "":
            raise ValueError("Text is Empty")
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.embeddings = embeddings
        embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to: {embeddings_path}")
        return embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(embeddings_path):
            print(f"File exists: {embeddings_path}")
            embeddings = np.load(embeddings_path)
            if len(embeddings) == len(documents):
                self.embeddings = embeddings
                print("Cached embeddings loaded successfully")
                return embeddings
            else:
                print("Cached embeddings length mismatch → rebuilding...")
        
        print("No valid cache found → generating new embeddings...")
        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        result = []
        for i, movie_embedding in enumerate(self.embeddings):
            score = cosine_similarity(movie_embedding, query_embedding)
            movie = self.documents[i]
            result.append((score, movie))
        result.sort(key=lambda x: x[0], reverse=True)

        top_results = []
        for score, movie in result[:limit]:
            top_results.append({
                "score": round(score, 4),
                "title": movie["title"],
                "description": movie["description"]
            })
        return top_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        all_chunks = []
        chunk_metadata = []

        for doc_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description.strip():
                continue

            chunks = semantic_chunking(description, max_chunk_size=4, overlap=1)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                metadata = {
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                }
                chunk_metadata.append(metadata)

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.chunk_metadata = chunk_metadata

        embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(embeddings_path, self.chunk_embeddings)

        metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.chunk_metadata,
                "total_chunks": len(self.chunk_metadata)
            }, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            self.chunk_embeddings = np.load(embeddings_path)
            with open(metadata_path, "r", encoding='utf-8') as f:
                metadata_data = json.load(f)
                self.chunk_metadata = metadata_data["chunks"]
            print("Loaded cached chunk embeddings and metadata")
            return self.chunk_embeddings

        print("No valid cache found → generating new chunk embeddings...")
        return self.build_chunk_embeddings(documents)


# Helper functions

def verify_model() -> None:
    searcher = SemanticSearch()
    MODEL = searcher.model
    MAX_LENGTH = searcher.model.max_seq_length
    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")


def embed_text(text) -> None:
    searcher = SemanticSearch()    
    embedding = searcher.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3].tolist()}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    searcher = SemanticSearch()
    documents = load_movies()
    embeddings = searcher.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
    searcher = SemanticSearch()
    embedding = searcher.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5].tolist()}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    if not text.strip():
        print("Chunking 0 characters")
        return

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    words = text.split()
    total_chars = len(text)
    print(f"Chunking {total_chars} characters")

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        start += chunk_size - overlap
        if start >= len(words):
            break

    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semantic_chunking(text: str, max_chunk_size: int = 4, overlap: int = 0):
    if not text.strip():
        print("Semantically chunking 0 characters")
        return []

    if overlap < 0:
        overlap = 0
    if overlap >= max_chunk_size:
        overlap = max_chunk_size - 1

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    total_chars = len(text)
    print(f"Semantically chunking {total_chars} characters")

    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + max_chunk_size, len(sentences))
        chunk_sentences = sentences[start:end]
        chunk_str = " ".join(chunk_sentences)
        chunks.append(chunk_str)
        start += max_chunk_size - overlap
        if start >= len(sentences):
            break

    return chunks
