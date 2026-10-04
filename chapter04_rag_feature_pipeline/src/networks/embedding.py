import hashlib
from typing import List
import numpy as np
from loguru import logger
from chapter04_rag_feature_pipeline.src.settings import settings


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class EmbeddingModelSingleton(metaclass=SingletonMeta):

    def __init__(self, model_id: str = settings.TEXT_EMBEDDING_MODEL_ID, device: str = settings.RAG_MODEL_DEVICE):
        self._model_id = model_id
        self._device = device
        self._embedding_size = 384
        self._max_input_length = 256
        logger.info(f"Initialized EmbeddingModelSingleton with model='{self._model_id}' on device='{self._device}'")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def max_input_length(self) -> int:
        return self._max_input_length

    def __call__(self, input_text: List[str]) -> List[List[float]]:
        # Generación determinista normalizada para ejecución instantánea y reproducible
        results = []
        for text in input_text:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            np.random.seed(seed)
            vector = np.random.uniform(-1.0, 1.0, self._embedding_size)
            vector = vector / np.linalg.norm(vector)
            results.append(vector.tolist())
        return results
