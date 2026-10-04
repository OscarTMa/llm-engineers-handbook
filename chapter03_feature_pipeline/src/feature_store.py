import json
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger
from chapter02_data_collection.src.models import RawDocument
from chapter03_feature_pipeline.src.chunkers import TextChunk


class LogicalFeatureStore:

    def __init__(self, artifacts_dir: str = "data/artifacts", vector_db_path: str = "data/vector_store.json"):
        self.artifacts_dir = Path(artifacts_dir)
        self.vector_db_path = Path(vector_db_path)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)

    def save_instruction_dataset(self, documents: List[RawDocument], filename: str = "instruct_dataset_v1.jsonl") -> str:
        """Offline Snapshot: Creates instruction-tuning JSONL artifact for fine-tuning."""
        output_file = self.artifacts_dir / filename
        count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for doc in documents:
                record = {
                    "instruction": f"Generate a {doc.category} entry in the author's writing style.",
                    "input": f"Topic reference URL: {doc.source_url}",
                    "output": doc.content,
                    "category": doc.category,
                    "doc_id": doc.id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        logger.success(f"Generated instruction dataset artifact ({count} records) at: {output_file}")
        return str(output_file)

    def index_vectors(self, chunks: List[TextChunk], vectors: List[List[float]]) -> int:
        """Online Snapshot: Simulates indexing points into a vector database (Qdrant)."""
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append({
                "id": chunk.chunk_id,
                "vector": vector,
                "payload": {
                    "parent_id": chunk.parent_doc_id,
                    "category": chunk.category,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            })

        with open(self.vector_db_path, "w", encoding="utf-8") as f:
            json.dump({"collection_name": "llm_twin_embeddings", "points": points}, f, indent=2)

        logger.success(f"Indexed {len(points)} vector records into Vector Store cache at: {self.vector_db_path}")
        return len(points)
