from abc import ABC
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
import uuid
from pydantic import BaseModel, Field
from loguru import logger


class DataCategory(str, Enum):
    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"


T = TypeVar("T", bound="VectorBaseDocument")


class MockQdrantStorage:
    """Almacén local en memoria para desacoplar el OVM sin requerir un demonio Docker activo."""
    collections: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @classmethod
    def upsert(cls, collection_name: str, points: List[Dict[str, Any]]):
        if collection_name not in cls.collections:
            cls.collections[collection_name] = {}
        for pt in points:
            cls.collections[collection_name][str(pt["id"])] = pt

    @classmethod
    def scroll(cls, collection_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        points = list(cls.collections.get(collection_name, {}).values())
        return points[:limit]


class VectorBaseDocument(BaseModel, Generic[T], ABC):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def get_collection_name(cls) -> str:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"):
            raise ValueError(f"Class {cls.__name__} must define a Config.name property.")
        return cls.Config.name

    def to_point(self) -> Dict[str, Any]:
        data = self.model_dump()
        _id = str(data.pop("id"))
        vector = data.pop("embedding", None)
        return {
            "id": _id,
            "vector": vector,
            "payload": data
        }

    @classmethod
    def from_record(cls: Type[T], record: Dict[str, Any]) -> T:
        payload = record.get("payload", {})
        payload["id"] = record["id"]
        if "vector" in record and record["vector"]:
            payload["embedding"] = record["vector"]
        return cls(**payload)

    @classmethod
    def bulk_insert(cls: Type[T], documents: List["VectorBaseDocument"]) -> bool:
        collection = cls.get_collection_name()
        points = [doc.to_point() for doc in documents]
        MockQdrantStorage.upsert(collection, points)
        logger.info(f"OVM: Inserted {len(points)} points into Qdrant collection '{collection}'")
        return True

    @classmethod
    def group_by_class(cls, documents: List["VectorBaseDocument"]) -> Dict[Type["VectorBaseDocument"], List["VectorBaseDocument"]]:
        grouped: Dict[Type["VectorBaseDocument"], List["VectorBaseDocument"]] = {}
        for doc in documents:
            cls_type = type(doc)
            grouped.setdefault(cls_type, []).append(doc)
        return grouped
