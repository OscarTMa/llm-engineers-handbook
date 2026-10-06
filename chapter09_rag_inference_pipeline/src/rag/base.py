from abc import ABC, abstractmethod
from typing import Any
from chapter09_rag_inference_pipeline.src.domain.queries import Query


class PromptTemplateFactory(ABC):
    @abstractmethod
    def create_template(self, *args, **kwargs) -> Any:
        pass


class RAGStep(ABC):
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock

    @abstractmethod
    def generate(self, query: Query, *args, **kwargs) -> Any:
        pass


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
