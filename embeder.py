
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image


class MultimodalEmbedder:
    """A multimodal embedder that supports text and image embeddings."""
    def __init__(
            self,
            text_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
            image_model: str = 'facebook/dinov2-small'
    ):
        self.text_model = SentenceTransformer(text_model)
        self.image_model = pipeline(
            'image-feature-extraction',
            model=image_model,
            device=0 if torch.cuda.is_available() else -1,
            pool=True
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts"""
        return self.text_model.encode(
            texts,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            show_progress_bar=True
        ).tolist()

    def embed_images(self, images: list[str | Image.Image]) -> list[list[float]]:
        """Embed a list of images, which can be file paths or PIL Image objects."""
        images = [Image.open(img) if isinstance(img, str) else img for img in images]
        images = [img.convert('RGB') for img in images]

        embeddings = self.image_model(images)

        return [emb[0] for emb in embeddings]

