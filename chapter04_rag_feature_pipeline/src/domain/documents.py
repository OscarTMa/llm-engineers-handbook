from abc import ABC
from typing import Any, Dict, List, Optional
from chapter04_rag_feature_pipeline.src.domain.base import DataCategory, VectorBaseDocument


# --- 1. Snapshot de Documentos Limpios (Cleaned Documents) ---
class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: str
    author_full_name: str


class CleanedPostDocument(CleanedDocument):
    image: Optional[str] = None

    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


class CleanedArticleDocument(CleanedDocument):
    link: str

    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class CleanedRepositoryDocument(CleanedDocument):
    name: str
    link: str

    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False


# --- 2. Snapshot de Chunks Fragmentados ---
class Chunk(VectorBaseDocument, ABC):
    content: str
    platform: str
    document_id: str
    author_id: str
    author_full_name: str
    metadata: Dict[str, Any] = {}


class PostChunk(Chunk):
    class Config:
        name = "post_chunks"


class ArticleChunk(Chunk):
    link: str

    class Config:
        name = "article_chunks"


class RepositoryChunk(Chunk):
    name: str
    link: str

    class Config:
        name = "repository_chunks"


# --- 3. Snapshot de Chunks con Embeddings (RAG Online) ---
class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: List[float]
    platform: str
    document_id: str
    author_id: str
    author_full_name: str
    metadata: Dict[str, Any] = {}


class EmbeddedPostChunk(EmbeddedChunk):
    class Config:
        name = "embedded_posts"


class EmbeddedArticleChunk(EmbeddedChunk):
    link: str

    class Config:
        name = "embedded_articles"


class EmbeddedRepositoryChunk(EmbeddedChunk):
    name: str
    link: str

    class Config:
        name = "embedded_repositories"
