from abc import ABC, abstractmethod
from typing import List
from chapter04_rag_feature_pipeline.src.domain.documents import (
    ArticleChunk,
    Chunk,
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
    PostChunk,
    RepositoryChunk,
)
from chapter04_rag_feature_pipeline.src.networks.embedding import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC):

    def embed_batch(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        texts = [chunk.content for chunk in chunks]
        vectors = embedding_model(texts)
        return [self.map_model(chk, vec) for chk, vec in zip(chunks, vectors)]

    @abstractmethod
    def map_model(self, chunk: Chunk, vector: List[float]) -> EmbeddedChunk:
        pass


class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, chunk: ArticleChunk, vector: List[float]) -> EmbeddedArticleChunk:
        return EmbeddedArticleChunk(
            id=chunk.id,
            content=chunk.content,
            embedding=vector,
            platform=chunk.platform,
            document_id=chunk.document_id,
            author_id=chunk.author_id,
            author_full_name=chunk.author_full_name,
            link=chunk.link,
            metadata={"model": embedding_model.model_id},
        )


class PostEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, chunk: PostChunk, vector: List[float]) -> EmbeddedPostChunk:
        return EmbeddedPostChunk(
            id=chunk.id,
            content=chunk.content,
            embedding=vector,
            platform=chunk.platform,
            document_id=chunk.document_id,
            author_id=chunk.author_id,
            author_full_name=chunk.author_full_name,
            metadata={"model": embedding_model.model_id},
        )


class RepositoryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, chunk: RepositoryChunk, vector: List[float]) -> EmbeddedRepositoryChunk:
        return EmbeddedRepositoryChunk(
            id=chunk.id,
            name=chunk.name,
            content=chunk.content,
            embedding=vector,
            platform=chunk.platform,
            document_id=chunk.document_id,
            author_id=chunk.author_id,
            author_full_name=chunk.author_full_name,
            link=chunk.link,
            metadata={"model": embedding_model.model_id},
        )
