import json
from pathlib import Path
from typing import List
from loguru import logger
import yaml

from chapter02_data_collection.src.models import RawDocument
from chapter03_feature_pipeline.src.cleaners import TextCleaner
from chapter03_feature_pipeline.src.chunkers import DocumentChunker
from chapter03_feature_pipeline.src.embeddings import EmbeddingGenerator
from chapter03_feature_pipeline.src.feature_store import LogicalFeatureStore


def load_raw_documents(raw_dir: str) -> List[RawDocument]:
    path = Path(raw_dir)
    documents = []

    if path.exists():
        for file in path.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents.append(RawDocument(**data))

    # Si no se ejecutó el crawler del capítulo 2, usar muestra de fallback
    if not documents:
        logger.warning(f"No raw documents found in '{raw_dir}'. Generating fallback sample documents.")
        documents = [
            RawDocument(
                id="doc_sample_1",
                category="articles",
                content="Decoupled machine learning systems use Feature/Training/Inference pipelines to isolate responsibilities.",
                author="Oscar Tibaduiza",
                source_url="https://medium.com/@sample/fti-pipelines",
                platform="medium",
            ),
            RawDocument(
                id="doc_sample_2",
                category="posts",
                content="Building an LLM Twin requires clean data and continuous fine-tuning pipelines.",
                author="Oscar Tibaduiza",
                source_url="https://linkedin.com/posts/sample",
                platform="linkedin",
            ),
        ]
    return documents


def main():
    logger.info("=== Starting Chapter 03: Feature Pipeline ===")

    # 1. Cargar configuraciones
    with open("chapter03_feature_pipeline/configs/features.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Cargar documentos brutos
    raw_docs = load_raw_documents(config["storage"]["raw_data_dir"])
    logger.info(f"Loaded {len(raw_docs)} raw documents for processing.")

    # 3. Limpiar documentos
    cleaned_docs = TextCleaner.clean_batch(raw_docs)
    logger.info(f"Cleaned {len(cleaned_docs)} documents.")

    # 4. Generar dataset de instrucciones para Fine-Tuning (Offline Store)
    feature_store = LogicalFeatureStore(
        artifacts_dir=config["storage"]["artifacts_dir"],
        vector_db_path=config["storage"]["vector_db_cache"],
    )
    artifact_path = feature_store.save_instruction_dataset(cleaned_docs)

    # 5. Segmentación (Chunking) para RAG
    chunker = DocumentChunker(
        chunk_size=config["chunking"]["articles"]["chunk_size"],
        chunk_overlap=config["chunking"]["articles"]["chunk_overlap"],
    )
    all_chunks = []
    for doc in cleaned_docs:
        all_chunks.extend(chunker.chunk_document(doc))
    logger.info(f"Generated {len(all_chunks)} semantic chunks.")

    # 6. Generar Embeddings Vectoriales
    embedder = EmbeddingGenerator(dimension=config["embedding"]["dimension"])
    texts = [c.text for c in all_chunks]
    vectors = embedder.generate_embeddings(texts)

    # 7. Indexar en Vector DB (Online Store)
    indexed_count = feature_store.index_vectors(all_chunks, vectors)

    logger.success(
        f"=== Feature Pipeline Completed: {indexed_count} chunks indexed, artifact ready at '{artifact_path}' ==="
    )


if __name__ == "__main__":
    main()
