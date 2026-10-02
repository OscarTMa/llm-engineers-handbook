"""
Core contracts and abstract interfaces for the LLM Twin architecture.
Follows the Feature/Training/Inference (FTI) design pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    content: str
    category: str = Field(description="'posts', 'articles', or 'code'")
    source_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseDataCollectionPipeline(ABC):
    """ETL Pipeline: Extracts from personal sources, normalizes into NoSQL DW."""

    @abstractmethod
    def extract(self, source_name: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def transform(self, raw_records: List[Dict[str, Any]]) -> List[Document]:
        pass

    @abstractmethod
    def load(self, documents: List[Document]) -> bool:
        pass


class BaseFeaturePipeline(ABC):
    """Feature Pipeline: Cleans, chunks, embeds, and outputs artifacts + vector DB."""

    @abstractmethod
    def clean(self, documents: List[Document]) -> List[Document]:
        pass

    @abstractmethod
    def chunk_and_embed(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def generate_instruction_dataset(self, documents: List[Document]) -> str:
        """Returns the artifact path/identifier for training."""
        pass


class BaseTrainingPipeline(ABC):
    """Training Pipeline: Fine-tunes model using instruction dataset artifacts."""

    @abstractmethod
    def train(self, dataset_artifact_id: str, hyperparams: Dict[str, Any]) -> str:
        """Returns the registered model identifier."""
        pass

    @abstractmethod
    def evaluate(self, model_id: str) -> Dict[str, float]:
        pass


class BaseInferencePipeline(ABC):
    """Inference Pipeline: Handles live RAG requests using Vector DB and fine-tuned LLM."""

    @abstractmethod
    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        pass

    @abstractmethod
    def generate(self, prompt: str, retrieved_context: List[str]) -> str:
        pass