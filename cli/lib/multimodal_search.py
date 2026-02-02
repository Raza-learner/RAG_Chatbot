from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import load_movies, CACHE_DIR

class MultimodalSearch:
    def __init__(self, documents: list[dict] = None, model_name: str = "clip-ViT-B-32"):
        """
        Initialize multimodal search with movie documents.
        Precomputes text embeddings for all movies if documents are provided.
        """
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.text_embeddings = None

        if self.documents:
            # Create text representations: title + description
            self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]

            # Generate embeddings for all movie texts
            print("Encoding all movie texts for multimodal search...")
            self.text_embeddings = self.model.encode(
                self.texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"Generated {len(self.text_embeddings)} text embeddings")

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate CLIP embedding for a single image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        embeddings = self.model.encode([image], convert_to_numpy=True)
        return embeddings[0]

    def search_with_image(self, image_path: str, top_k: int = 5) -> list[dict]:
        """
        Find top_k movies most similar to the given image using CLIP.
        """
        if self.text_embeddings is None or not self.documents:
            raise ValueError("No text embeddings loaded. Provide documents during initialization.")

        image_embedding = self.embed_image(image_path)

        
        similarities = np.dot(self.text_embeddings, image_embedding) / (
            np.linalg.norm(self.text_embeddings, axis=1) * np.linalg.norm(image_embedding)
        )

        
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": round(float(similarities[idx]), 3)
            })

        return results


def image_search_command(image_path: str, top_k: int = 5) -> list[dict]:
    """
    Load movies, initialize MultimodalSearch, search by image.
    """
    documents = load_movies()
    searcher = MultimodalSearch(documents)
    return searcher.search_with_image(image_path, top_k=top_k)


def verify_image_embedding(image_path: str) -> None:
    """
    Generate image embedding and print its dimension count.
    """
    searcher = MultimodalSearch()  # no documents needed for single image
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
