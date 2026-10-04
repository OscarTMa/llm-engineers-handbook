from loguru import logger
import yaml

from chapter04_rag_feature_pipeline.src.steps.feature_steps import (
    clean_documents,
    chunk_and_embed,
    load_to_vector_db,
    query_data_warehouse,
)


def main():
    logger.info("=== Starting Chapter 04: RAG Feature Pipeline ===")

    # 1. Cargar configuración YAML
    config_path = "chapter04_rag_feature_pipeline/configs/feature_engineering.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    authors = config["parameters"]["author_full_names"]

    # 2. Paso 1: Extraer datos brutos del Data Warehouse
    raw_docs = query_data_warehouse(author_names=authors)

    # 3. Paso 2: Limpiar documentos
    cleaned_docs = clean_documents(raw_docs)

    # 4. Paso 3: Snapshot 1 -> Almacenar documentos limpios en Vector DB (metadata NoSQL)
    logger.info("--- Storing Snapshot 1: Cleaned Documents (for Offline Dataset / Fine-Tuning) ---")
    load_to_vector_db(cleaned_docs)

    # 5. Paso 4: Segmentar y calcular embeddings vectoriales
    embedded_chunks = chunk_and_embed(cleaned_docs)

    # 6. Paso 5: Snapshot 2 -> Almacenar Chunks con Vectores Densos (para Online RAG)
    logger.info("--- Storing Snapshot 2: Embedded Chunks (for Real-Time RAG Search) ---")
    load_to_vector_db(embedded_chunks)

    logger.success("=== Chapter 04: RAG Feature Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
