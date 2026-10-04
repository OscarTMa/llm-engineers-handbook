from typing import Any, List
from chapter04_rag_feature_pipeline.src.domain.base import DataCategory
from chapter04_rag_feature_pipeline.src.domain.documents import (
    ArticleChunk,
    CleanedArticleDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
    PostChunk,
    RepositoryChunk,
)
from chapter04_rag_feature_pipeline.src.preprocessing.handlers.cleaning import (
    ArticleCleaningHandler,
    CleaningDataHandler,
    PostCleaningHandler,
    RepositoryCleaningHandler,
)
from chapter04_rag_feature_pipeline.src.preprocessing.handlers.chunking import (
    ArticleChunkingHandler,
    ChunkingDataHandler,
    PostChunkingHandler,
    RepositoryChunkingHandler,
)
from chapter04_rag_feature_pipeline.src.preprocessing.handlers.embedding import (
    ArticleEmbeddingHandler,
    EmbeddingDataHandler,
    PostEmbeddingHandler,
    RepositoryEmbeddingHandler,
)


class CleaningHandlerFactory:
    @staticmethod
    def create_handler(category: DataCategory) -> CleaningDataHandler:
        if category == DataCategory.POSTS:
            return PostCleaningHandler()
        elif category == DataCategory.ARTICLES:
            return ArticleCleaningHandler()
        elif category == DataCategory.REPOSITORIES:
            return RepositoryCleaningHandler()
        raise ValueError(f"Unsupported category {category}")


class CleaningDispatcher:
    @classmethod
    def dispatch(cls, raw_doc: Any) -> Any:
        cat_map = {"posts": DataCategory.POSTS, "articles": DataCategory.ARTICLES, "code": DataCategory.REPOSITORIES}
        category = cat_map.get(raw_doc.category, DataCategory.ARTICLES)
        handler = CleaningHandlerFactory.create_handler(category)
        return handler.clean(raw_doc)


class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(doc: Any) -> ChunkingDataHandler:
        if isinstance(doc, CleanedPostDocument):
            return PostChunkingHandler()
        elif isinstance(doc, CleanedArticleDocument):
            return ArticleChunkingHandler()
        elif isinstance(doc, CleanedRepositoryDocument):
            return RepositoryChunkingHandler()
        raise ValueError(f"Unsupported document type: {type(doc)}")


class ChunkingDispatcher:
    @classmethod
    def dispatch(cls, cleaned_doc: Any) -> List[Any]:
        handler = ChunkingHandlerFactory.create_handler(cleaned_doc)
        return handler.chunk(cleaned_doc)


class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(chunk: Any) -> EmbeddingDataHandler:
        if isinstance(chunk, PostChunk):
            return PostEmbeddingHandler()
        elif isinstance(chunk, ArticleChunk):
            return ArticleEmbeddingHandler()
        elif isinstance(chunk, RepositoryChunk):
            return RepositoryEmbeddingHandler()
        raise ValueError(f"Unsupported chunk type: {type(chunk)}")


class EmbeddingDispatcher:
    @classmethod
    def dispatch(cls, chunks: List[Any]) -> List[Any]:
        if not chunks:
            return []
        handler = EmbeddingHandlerFactory.create_handler(chunks[0])
        return handler.embed_batch(chunks)
