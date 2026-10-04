import hashlib
from typing import List
import numpy as np
from loguru import logger


class EmbeddingGenerator:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates deterministic dense embeddings for testing and reproducible simulation."""
        logger.info(f"Computing dense vector embeddings for {len(texts)} chunks.")
        embeddings = []

        for text in texts:
            # Vector determinista normalizado basado en hash para reproducibilidad local
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            vector = np.random.uniform(-1.0, 1.0, self.dimension)
            norm = np.linalg.norm(vector)
            normalized_vector = (vector / norm).tolist()
            embeddings.append(normalized_vector)

        return embeddings
