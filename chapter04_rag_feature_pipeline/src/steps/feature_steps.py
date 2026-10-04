import json
from pathlib import Path
from typing import Any, List
from loguru import logger

from chapter02_data_collection.src.models import RawDocument
from chapter04_rag_feature_pipeline.src.domain.base import VectorBaseDocument
from chapter04_rag_feature_pipeline.src.preprocessing.dispatchers import (
    ChunkingDispatcher,
    CleaningDispatcher,
    EmbeddingDispatcher,
)


def query_data_warehouse(author_names: List[str]) -> List[RawDocument]:
    logger.info(f"Querying Data Warehouse for authors: {author_names}")
    raw_path = Path("data/raw")
    docs = []
    if raw_path.exists():
        for f in raw_path.glob("*.json"):
            with open(f, "r", encoding="utf-8") as file:
                docs.append(RawDocument(**json.load(file)))

    if not docs:
        docs = [
            RawDocument(
                id="doc_01",
                category="articles",
                content="Decoupled RAG architectures prevent training-serving skew and isolate compute bottlenecks.",
                author="Oscar Tibaduiza",
                source_url="https://medium.com/@oscar/rag-feature-pipelines",
                platform="medium",
            ),
            RawDocument(
                id="doc_02",
                category="code",
                content="class VectorStoreConnector:\n    def connect(self): return True",
                author="Oscar Tibaduiza",
                source_url="https://github.com/OscarTMa/llm-engineers-handbook/blob/main/store.py",
                platform="github",
                metadata={"repository": "llm-engineers-handbook"},
            ),
        ]
    logger.success(f"Extracted {len(docs)} documents from Data Warehouse.")
    return docs


def clean_documents(raw_docs: List[RawDocument]) -> List[Any]:
    cleaned = [CleaningDispatcher.dispatch(doc) for doc in raw_docs]
    logger.success(f"Cleaned {len(cleaned)} documents via CleaningDispatcher.")
    return cleaned


def chunk_and_embed(cleaned_docs: List[Any]) -> List[Any]:
    all_embedded = []
    for doc in cleaned_docs:
        chunks = ChunkingDispatcher.dispatch(doc)
        embedded = EmbeddingDispatcher.dispatch(chunks)
        all_embedded.extend(embedded)
    logger.success(f"Generated {len(all_embedded)} embedded chunks via Dispatchers.")
    return all_embedded


def load_to_vector_db(documents: List[VectorBaseDocument]) -> bool:
    grouped = VectorBaseDocument.group_by_class(documents)
    for doc_class, docs in grouped.items():
        doc_class.bulk_insert(docs)
    return True
