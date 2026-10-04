from abc import ABC, abstractmethod
import hashlib
from typing import List
from chapter04_rag_feature_pipeline.src.domain.documents import (
    ArticleChunk,
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
    PostChunk,
    RepositoryChunk,
)


class ChunkingDataHandler(ABC):
    @abstractmethod
    def chunk(self, data_model: CleanedDocument) -> List[Any]:
        pass


class ArticleChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: CleanedArticleDocument) -> List[ArticleChunk]:
        words = data_model.content.split(" ")
        step = 40
        chunks = []
        for i in range(0, len(words), step):
            segment = " ".join(words[i : i + step])
            chunk_id = hashlib.md5(segment.encode()).hexdigest()[:16]
            chunks.append(
                ArticleChunk(
                    id=f"chk_art_{chunk_id}",
                    content=segment,
                    platform=data_model.platform,
                    document_id=data_model.id,
                    author_id=data_model.author_id,
                    author_full_name=data_model.author_full_name,
                    link=data_model.link,
                    metadata={"min_length": 50, "max_length": 300},
                )
            )
        return chunks


class PostChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: CleanedPostDocument) -> List[PostChunk]:
        chunk_id = hashlib.md5(data_model.content.encode()).hexdigest()[:16]
        return [
            PostChunk(
                id=f"chk_pst_{chunk_id}",
                content=data_model.content,
                platform=data_model.platform,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
            )
        ]


class RepositoryChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: CleanedRepositoryDocument) -> List[RepositoryChunk]:
        chunk_id = hashlib.md5(data_model.content.encode()).hexdigest()[:16]
        return [
            RepositoryChunk(
                id=f"chk_rep_{chunk_id}",
                name=data_model.name,
                content=data_model.content,
                platform=data_model.platform,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                link=data_model.link,
            )
        ]
