from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User prompt or task instruction")
    author: Optional[str] = Field(default=None, description="Optional author metadata constraint")


class QueryResponse(BaseModel):
    query: str
    answer: str
    grounding_references: int = Field(default=0, description="Number of retrieved chunks injected")
    metadata: Dict[str, Any] = Field(default_factory=dict)
