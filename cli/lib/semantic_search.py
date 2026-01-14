from sentence_transformers import SentenceTransformer
import numpy as np

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
        pass
        #continue from here...
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


