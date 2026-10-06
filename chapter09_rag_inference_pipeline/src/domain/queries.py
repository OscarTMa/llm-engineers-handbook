from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from chapter04_rag_feature_pipeline.src.domain.base import DataCategory, VectorBaseDocument


class Query(VectorBaseDocument):
    content: str
    author_id: Optional[str] = None
    author_full_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        name = "queries"
        category = DataCategory.POSTS

    @classmethod
    def from_str(cls, query_str: str) -> "Query":
        return cls(content=query_str.strip("\n "))

    def replace_content(self, new_content: str) -> "Query":
        return Query(
            id=self.id,
            content=new_content,
            author_id=self.author_id,
            author_full_name=self.author_full_name,
            metadata=self.metadata,
        )


class EmbeddedQuery(Query):
    embedding: List[float] = Field(default_factory=list)
