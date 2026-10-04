from abc import ABC, abstractmethod
import re
from typing import Any
from chapter04_rag_feature_pipeline.src.domain.documents import (
    CleanedArticleDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "[URL]", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CleaningDataHandler(ABC):
    @abstractmethod
    def clean(self, raw_doc: Any) -> Any:
        pass


class PostCleaningHandler(CleaningDataHandler):
    def clean(self, raw_doc: Any) -> CleanedPostDocument:
        return CleanedPostDocument(
            id=raw_doc.id,
            content=clean_text(raw_doc.content),
            platform=raw_doc.platform,
            author_id=raw_doc.author,
            author_full_name=raw_doc.author,
        )


class ArticleCleaningHandler(CleaningDataHandler):
    def clean(self, raw_doc: Any) -> CleanedArticleDocument:
        return CleanedArticleDocument(
            id=raw_doc.id,
            content=clean_text(raw_doc.content),
            platform=raw_doc.platform,
            author_id=raw_doc.author,
            author_full_name=raw_doc.author,
            link=raw_doc.source_url,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, raw_doc: Any) -> CleanedRepositoryDocument:
        repo_name = raw_doc.metadata.get("repository", "repo")
        return CleanedRepositoryDocument(
            id=raw_doc.id,
            name=repo_name,
            content=clean_text(raw_doc.content),
            platform=raw_doc.platform,
            author_id=raw_doc.author,
            author_full_name=raw_doc.author,
            link=raw_doc.source_url,
        )
