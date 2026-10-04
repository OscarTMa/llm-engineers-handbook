from typing import List
from loguru import logger
from chapter05_instruction_dataset.src.models import InstructionRecord


class DatasetCurator:

    def __init__(self, min_len: int = 30, max_len: int = 4000, deduplicate: bool = True):
        self.min_len = min_len
        self.max_len = max_len
        self.deduplicate = deduplicate

    def curate(self, records: List[InstructionRecord]) -> List[InstructionRecord]:
        logger.info(f"Curating dataset: evaluating {len(records)} records for quality and length bounds.")
        seen_hashes = set()
        curated_records = []

        for rec in records:
            out_len = len(rec.target_output)

            # 1. Filtro de longitud
            if out_len < self.min_len:
                continue
            if out_len > self.max_len:
                continue

            # 2. Deduplicación por contenido
            if self.deduplicate:
                content_hash = hash(rec.target_output.strip().lower())
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

            curated_records.append(rec)

        logger.success(f"Dataset curation complete: {len(curated_records)}/{len(records)} samples retained.")
        return curated_records
