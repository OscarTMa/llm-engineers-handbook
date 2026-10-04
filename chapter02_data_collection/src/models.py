"""
Data models for the Data Collection Pipeline.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    id: str
    category: Literal["posts", "articles", "code"]
    content: str
    author: str
    source_url: str
    platform: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)