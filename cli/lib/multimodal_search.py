from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image at the given path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        
        # encode expects list of images → wrap single image
        embeddings = self.model.encode([image], convert_to_numpy=True)
        
        return embeddings[0]  # single embedding vector


def verify_image_embedding(image_path: str) -> None:
    """
    Load image, generate embedding, print its dimension count.
    """
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
