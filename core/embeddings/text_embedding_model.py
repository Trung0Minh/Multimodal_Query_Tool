import torch

from typing import List
from sentence_transformers import SentenceTransformer
from utils.logger import logger
from config.model_configs import TEXT_EMBEDDING_MODEL

class TextEmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Text Embedding Model '{TEXT_EMBEDDING_MODEL}' to device: {self.device}")
        
        self.model = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=self.device)
        logger.info("Text Embedding Model loaded successfully.")
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        logger.debug(f"Generated {len(embeddings)} embeddings for {len(texts)} texts.")
        return embeddings