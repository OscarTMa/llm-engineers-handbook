import json
from pathlib import Path
from loguru import logger
import yaml

from chapter04_rag_feature_pipeline.src.domain.base import MockQdrantStorage
from chapter04_rag_feature_pipeline.src.domain.documents import (
    CleanedArticleDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)
from chapter05_instruction_dataset.src.curator import DatasetCurator
from chapter05_instruction_dataset.src.generator import InstructionGenerator


def fetch_cleaned_documents_from_store() -> list:
    """Extrae los documentos limpios guardados en Qdrant (Snapshot 1) durante el capítulo 4."""
    logger.info("Fetching Cleaned Documents (Snapshot 1) from Logical Feature Store.")
    documents = []

    # Consultar colecciones de Qdrant
    for doc_cls in [CleanedArticleDocument, CleanedPostDocument, CleanedRepositoryDocument]:
        collection = doc_cls.get_collection_name()
        records = MockQdrantStorage.scroll(collection, limit=100)
        for rec in records:
            documents.append(doc_cls.from_record(rec))

    # Muestra de fallback en caso de ejecución independiente
    if not documents:
        logger.warning("No documents retrieved from Feature Store. Generating fallback cleaned documents.")
        documents = [
            CleanedArticleDocument(
                id="doc_art_01",
                content="Decoupling the ML system into FTI pipelines prevents training-serving skew and makes RAG applications modular.",
                platform="medium",
                author_id="Oscar Tibaduiza",
                author_full_name="Oscar Tibaduiza",
                link="https://medium.com/@oscar/fti-pattern",
            ),
            CleanedPostDocument(
                id="doc_pst_01",
                content="Production LLM Twins require robust data pipelines and reproducible instruction datasets.",
                platform="linkedin",
                author_id="Oscar Tibaduiza",
                author_full_name="Oscar Tibaduiza",
            ),
        ]
    return documents


def main():
    logger.info("=== Starting Chapter 05: Instruction Dataset Pipeline ===")

    # 1. Cargar configuraciones
    config_path = "chapter05_instruction_dataset/configs/dataset.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Extraer documentos limpios desde el Feature Store
    cleaned_docs = fetch_cleaned_documents_from_store()
    logger.info(f"Loaded {len(cleaned_docs)} cleaned documents for dataset synthesis.")

    # 3. Generar pares instrucción-respuesta
    raw_instructions = InstructionGenerator.generate_from_cleaned_documents(cleaned_docs)

    # 4. Curación y filtrado de calidad
    curator = DatasetCurator(
        min_len=config["curation"]["min_output_characters"],
        max_len=config["curation"]["max_output_characters"],
        deduplicate=config["curation"]["deduplicate_by_content"],
    )
    curated = curator.curate(raw_instructions)

    # 5. Exportar artefacto versionado
    target_format = config["dataset_generation"].get("target_format", "chatml")
    output_dir = Path(config["storage"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / config["storage"]["output_filename"]

    with open(output_file, "w", encoding="utf-8") as f:
        for item in curated:
            if target_format == "alpaca":
                payload = item.to_alpaca().model_dump()
            else:
                payload = item.to_chatml().model_dump()
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    logger.success(
        f"=== Instruction Dataset Pipeline Completed: {len(curated)} records exported in {target_format.upper()} format to '{output_file}' ==="
    )


if __name__ == "__main__":
    main()
