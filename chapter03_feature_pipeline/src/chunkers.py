import hashlib
from typing import Any, Dict, List
from pydantic import BaseModel
from chapter02_data_collection.src.models import RawDocument


class TextChunk(BaseModel):
    chunk_id: str
    parent_doc_id: str
    category: str
    text: str
    metadata: Dict[str, Any]


class DocumentChunker:

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def chunk_document(self, document: RawDocument) -> List[TextChunk]:
        raw_chunks = self.split_text(document.content)
        chunk_objects = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_hash = hashlib.sha256(
                f"{document.id}-{idx}-{chunk_text[:20]}".encode()
            ).hexdigest()[:16]

            chunk_objects.append(
                TextChunk(
                    chunk_id=f"chk_{chunk_hash}",
                    parent_doc_id=document.id,
                    category=document.category,
                    text=chunk_text,
                    metadata={
                        "source_url": document.source_url,
                        "platform": document.platform,
                        "chunk_index": idx,
                    },
                )
            )
        return chunk_objects
