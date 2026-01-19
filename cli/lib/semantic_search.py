from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import load_movies,CACHE_DIR
import os
class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or text.strip() == "":
            raise ValueError("Text is Empty")
        embedding = self.model.encode([text])
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
            print(f"File exists{embeddings_path}")
            embeddings = np.load(embeddings_path)
            if len(embeddings) == len(documents):
                self.embeddings = embeddings
                print("Cached embeddings loaded successfully")
                return embeddings
            else:
                print("Cached embeddings length mismatch → rebuilding...")
        
        
        print("No valid cache found → generating new embeddings...")
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embeddings = self.generate_embedding(query)
        result = []
        for i,movie_embeddings in enumerate(self.embeddings):
            score = cosine_similarity(movie_embeddings,query_embeddings)
            movie = self.documents[i]
            result.append((score,movie))
        result.sort(key=lambda x:x[0],reverse=True)

        top_results = []
        for score,movie in result[:limit]:
            top_results.append({
                "score": round(score, 4),
                "title": movie["title"],
                "description": movie["description"]
            })

        return top_results

def verify_model()-> None:
    searcher =SemanticSearch()
    MODEL = searcher.model
    MAX_LENGTH = searcher.model.max_seq_length
    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")

def embed_text(text):
    searcher = SemanticSearch()    
    embedding = searcher.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    searcher =  SemanticSearch()
    documents = load_movies()
    embeddings = searcher.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    searcher = SemanticSearch()
    embedding = searcher.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(text: str, chunk_size: int = 200) -> None:
    words = text.split()
    
    if not words:
        print("Chunking 0 characters")
        return
    
    total_chars = len(text)
    print(f"Chunking {total_chars} characters")
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunk_number = (i // chunk_size) + 1
        print(f"{chunk_number}. {chunk_str}")
