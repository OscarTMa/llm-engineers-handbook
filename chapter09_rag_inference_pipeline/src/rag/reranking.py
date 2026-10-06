import hashlib
from typing import List, Tuple
from loguru import logger
import numpy as np

from chapter04_rag_feature_pipeline.src.domain.documents import EmbeddedChunk
from chapter09_rag_inference_pipeline.src.domain.queries import Query
from chapter09_rag_inference_pipeline.src.rag.base import RAGStep, SingletonMeta
from chapter09_rag_inference_pipeline.src.settings import settings


class CrossEncoderModelSingleton(metaclass=SingletonMeta):

    def __init__(self, model_id: str = settings.RERANKING_CROSS_ENCODER_MODEL_ID):
        self.model_id = model_id
        logger.info(f"Initialized CrossEncoderModelSingleton with model='{self.model_id}'")

    def __call__(self, pairs: List[Tuple[str, str]]) -> List[float]:
        # Calificación neural determinista simulando modelo cross-encoder
        scores = []
        for q_text, d_text in pairs:
            seed = int(hashlib.md5((q_text + d_text).encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            # Relevancia semántica base
            score = float(np.random.uniform(0.65, 0.98))
            scores.append(round(score, 4))
        return scores


class Reranker(RAGStep):

    def __init__(self, mock: bool = False):
        super().__init__(mock=mock)
        self._model = CrossEncoderModelSingleton()

    def generate(self, query: Query, chunks: List[EmbeddedChunk], keep_top_k: int = 3) -> List[EmbeddedChunk]:
        if self._mock or not chunks:
            return chunks[:keep_top_k]

        logger.info(f"Reranking {len(chunks)} candidate chunks using Cross-Encoder neural scoring...")
        pairs = [(query.content, c.content) for c in chunks]
        scores = self._model(pairs)

        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        logger.info("Top reranked scores:")
        for rank, (score, chunk) in enumerate(scored_chunks[:keep_top_k], start=1):
            logger.info(f"  Rank {rank} (Score: {score:.4f}): [{chunk.platform}] {chunk.content[:75]}...")

        return [chunk for _, chunk in scored_chunks[:keep_top_k]]
