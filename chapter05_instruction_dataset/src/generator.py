import hashlib
from typing import Any, List
from loguru import logger

from chapter05_instruction_dataset.src.models import InstructionRecord
from chapter05_instruction_dataset.src.prompts import get_instruction_for_category


class InstructionGenerator:

    @staticmethod
    def generate_from_cleaned_documents(documents: List[Any]) -> List[InstructionRecord]:
        logger.info(f"Synthesizing instruction pairs from {len(documents)} cleaned documents.")
        records = []

        for doc in documents:
            category = getattr(doc, "category", "articles")
            if hasattr(doc, "Config") and hasattr(doc.Config, "category"):
                category = doc.Config.category.value

            instruction = get_instruction_for_category(category)
            source_link = getattr(doc, "link", "")
            content = getattr(doc, "content", "").strip()

            record_id = hashlib.sha256(f"{doc.id}-{content[:30]}".encode()).hexdigest()[:16]

            records.append(
                InstructionRecord(
                    id=f"inst_{record_id}",
                    category=category,
                    source_url=source_link,
                    instruction=instruction,
                    context=f"Reference Source: {source_link}" if source_link else "",
                    target_output=content,
                )
            )

        logger.success(f"Synthesized {len(records)} raw instruction records.")
        return records
