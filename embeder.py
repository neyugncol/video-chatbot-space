from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from transformers.utils import ModelOutput


class MultimodalEmbedder:
    """A multimodal embedder that supports text and image embeddings."""
    def __init__(
            self,
            text_model: str = 'nomic-ai/nomic-embed-text-v1.5',
            image_model: str = 'nomic-ai/nomic-embed-vision-v1.5'
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model, trust_remote_code=True)
        self.text_model.eval()
        self.text_embedding_size = self.text_model.config.hidden_size

        self.processor = AutoImageProcessor.from_pretrained(image_model)
        self.image_model = AutoModel.from_pretrained(image_model, trust_remote_code=True)
        self.image_embedding_size = self.image_model.config.hidden_size

    def embed_texts(
            self,
            texts: list[str],
            kind: Literal['query', 'document'] = 'document',
            device: str = 'cpu'
    ) -> list[list[float]]:
        """Embed a list of texts"""
        texts = [f'search_query: {text}' if kind == 'query' else f'search_document: {text}' for text in texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            outputs = self.text_model.to(device)(**inputs.to(device), batch_size=64)

        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def embed_images(self, images: list[str | Image.Image], device: str = 'cpu') -> list[list[float]]:
        """Embed a list of images, which can be file paths or PIL Image objects."""
        images = [Image.open(img) if isinstance(img, str) else img for img in images]
        images = [img.convert('RGB') for img in images]

        inputs = self.processor(images, return_tensors='pt')

        embeddings = self.image_model.to(device)(**inputs.to(device), batch_size=64).last_hidden_state

        embeddings = F.normalize(embeddings[:, 0], p=2, dim=1)

        return embeddings.cpu().tolist()

    def similarity(
            self,
            embeddings1: list[list[float]],
            embeddings2: list[list[float]],
            pair_type: Literal['text-text', 'image-image', 'text-image']
    ) -> list[list[float]]:
        """Calculate cosine similarity between two sets of embeddings."""
        pair_min_max = {
            'text-text': (0.4, 1.0),
            'image-image': (0.75, 1.0),
            'text-image': (0.01, 0.09)
        }
        min_val, max_val = pair_min_max[pair_type]

        similarities = np.dot(embeddings1, np.transpose(embeddings2))
        similarities = np.clip((similarities - min_val) / (max_val - min_val), 0, 1)

        return similarities.tolist()


def mean_pooling(model_output: ModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for the model output."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
