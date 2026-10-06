import json
from pathlib import Path
from loguru import logger
import yaml

from chapter04_rag_feature_pipeline.src.domain.base import MockQdrantStorage
from chapter04_rag_feature_pipeline.src.domain.documents import (
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from chapter09_rag_inference_pipeline.src.rag.llm_client import format_context_string, rag
from chapter09_rag_inference_pipeline.src.rag.retriever import ContextRetriever


def seed_qdrant_store():
    """Siembra colecciones de prueba con payload completo alineado a Pydantic."""
    MockQdrantStorage.upsert("embedded_articles", [
        {
            "id": "art_101",
            "vector": [0.0] * 384,
            "payload": {
                "id": "art_101",
                "document_id": "doc_art_101",
                "content": "Decoupling feature pipelines from model inference prevents training-serving skew and makes RAG modular.",
                "platform": "medium",
                "link": "https://medium.com/@oscar/fti-pattern",
                "author_id": "900fec95-d621-4315-84c6-52e5229e0b96",
                "author_full_name": "Oscar Tibaduiza",
            }
        }
    ])
    MockQdrantStorage.upsert("embedded_posts", [
        {
            "id": "pst_201",
            "vector": [0.0] * 384,
            "payload": {
                "id": "pst_201",
                "document_id": "doc_pst_201",
                "content": "Production LLM Twins require continuous batching and cross-encoder reranking to optimize precision.",
                "platform": "linkedin",
                "author_id": "900fec95-d621-4315-84c6-52e5229e0b96",
                "author_full_name": "Oscar Tibaduiza",
            }
        }
    ])
    MockQdrantStorage.upsert("embedded_repositories", [
        {
            "id": "rep_301",
            "vector": [0.0] * 384,
            "payload": {
                "id": "rep_301",
                "document_id": "doc_rep_301",
                "name": "llm-engineers-handbook",
                "content": "def rag(query: str): retriever = ContextRetriever(); docs = retriever.search(query); return call_llm(docs)",
                "platform": "github",
                "link": "https://github.com/OscarTMa/llm-engineers-handbook",
                "author_id": "900fec95-d621-4315-84c6-52e5229e0b96",
                "author_full_name": "Oscar Tibaduiza",
            }
        }
    ])


def main():
    logger.info("=== Starting Chapter 09: Advanced RAG Inference Pipeline ===")

    # 1. Cargar configuraciones
    cfg_path = "chapter09_rag_inference_pipeline/configs/rag_inference.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Sembrar datos para simular el almacén vectorial de Qdrant
    seed_qdrant_store()

    # 3. Consulta de prueba
    user_query = "I am Oscar Tibaduiza. Could you draft a technical explanation on how advanced RAG works?"
    logger.info(f"User Query: '{user_query}'")

    # 4. Ejecución del módulo ContextRetriever
    retriever = ContextRetriever(mock=config["inference"]["mock_mode"])
    documents = retriever.search(
        query=user_query,
        k=config["retrieval"]["top_k"],
        expand_to_n_queries=config["retrieval"]["expand_to_n_queries"],
    )

    print("\n" + "=" * 65)
    print("TOP RERANKED GROUNDING DOCUMENTS RETRIEVED:")
    print("=" * 65)
    for rank, doc in enumerate(documents, start=1):
        print(f"[{rank}] Platform: {doc.platform} | Author: {doc.author_full_name}")
        print(f"    Excerpt: {doc.content[:110]}...")
    print("=" * 65 + "\n")

    # 5. Ejecutar canalización RAG completa
    response = rag(
        query=user_query,
        k=config["retrieval"]["top_k"],
        expand_to_n=config["retrieval"]["expand_to_n_queries"],
    )

    print("\n" + "=" * 65)
    print("FINAL LLM TWIN COMPLETION (SageMaker / TwinLlama):")
    print("=" * 65)
    print(response)
    print("=" * 65 + "\n")

    # 6. Registrar telemetría de inferencia
    telemetry_dir = Path(config["storage"]["telemetry_dir"])
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    report_file = telemetry_dir / config["storage"]["report_file"]

    report_payload = {
        "query": user_query,
        "k_retrieved": len(documents),
        "retrieved_documents": [d.model_dump() for d in documents],
        "generated_answer": response,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.success(f"Inference telemetry persisted to: {report_file}")
    logger.success("=== Chapter 09 Completed Successfully ===")


if __name__ == "__main__":
    main()
