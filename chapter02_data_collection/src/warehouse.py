"""
Data Warehouse adapter (supports MongoDB and local JSON fallback).
"""

import json
from pathlib import Path
from typing import List
from loguru import logger
from chapter02_data_collection.src.models import RawDocument


class DataWarehouse:

    def __init__(self, local_cache_dir: str = "data/raw"):
        self.local_cache_path = Path(local_cache_dir)
        self.local_cache_path.mkdir(parents=True, exist_ok=True)

    def load_documents(self, documents: List[RawDocument]) -> int:
        """Stores documents in local file storage (simulating MongoDB collection)."""
        inserted_count = 0
        for doc in documents:
            target_file = self.local_cache_path / f"{doc.id}.json"
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(doc.model_dump(), f, indent=2)
            inserted_count += 1

        logger.info(
            f"Persisted {inserted_count} documents into Data Warehouse at: {self.local_cache_path}"
        )
        return inserted_count