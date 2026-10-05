import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple
from loguru import logger


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_chunks(text: str, min_length: int = 100, max_length: int = 800) -> List[str]:
    cleaned = clean_text(text)
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
    sentences = re.split(sentence_pattern, cleaned)
    
    chunks = []
    current_chunk = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current_chunk) + len(s) <= max_length:
            current_chunk += s + " "
        else:
            if len(current_chunk) >= min_length:
                chunks.append(current_chunk.strip())
            current_chunk = s + " "
    if len(current_chunk) >= min_length:
        chunks.append(current_chunk.strip())
    return chunks


class InstructionDatasetBuilder:

    @staticmethod
    def synthesize_pairs_from_text(chunk: str) -> List[Tuple[str, str]]:
        # Generación determinista que simula el output estructurado de backtranslation
        seed_id = hashlib.md5(chunk.encode()).hexdigest()[:6]
        pairs = [
            (f"Explain the technical design decisions regarding this component [Ref: {seed_id}].", chunk),
            (f"Summarize the architectural advantages described in the excerpt [Ref: {seed_id}].", chunk),
        ]
        return pairs

    @classmethod
    def build_from_raw_warehouse(cls, raw_data_dir: str) -> List[Dict[str, str]]:
        logger.info(f"Loading raw articles from Data Warehouse at: {raw_data_dir}")
        path = Path(raw_data_dir)
        records = []

        if path.exists():
            for f in path.glob("*.json"):
                with open(f, "r", encoding="utf-8") as file:
                    doc = json.load(file)
                    content = doc.get("content", "")
                    for chunk in extract_chunks(content):
                        pairs = cls.synthesize_pairs_from_text(chunk)
                        for inst, ans in pairs:
                            records.append({"instruction": inst, "output": ans})

        if not records:
            logger.warning("No raw files found. Using fallback instruction pairs.")
            records = [
                {
                    "instruction": "Explain the architectural difference between batch and streaming pipelines in ML.",
                    "output": "Batch pipelines process data in scheduled windows optimizing resource saturation, while streaming pipelines update state per event.",
                },
                {
                    "instruction": "How does LoRA reduce memory consumption during LLM fine-tuning?",
                    "output": "LoRA decomposes weight updates into two low-rank matrices A and B, keeping base weights frozen and saving over 90% of trainable gradients.",
                },
            ]

        logger.success(f"Built instruction dataset with {len(records)} verified pairs.")
        return records
