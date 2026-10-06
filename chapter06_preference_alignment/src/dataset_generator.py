import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List
from loguru import logger
from pydantic import BaseModel


class PreferenceTriple(BaseModel):
    prompt: str
    chosen: str
    rejected: str


class PreferenceDatasetBuilder:

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"[^\w\s.,!?']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def filter_short_answers(cls, triples: List[PreferenceTriple], min_length: int = 100) -> List[PreferenceTriple]:
        return [t for t in triples if len(t.chosen) >= min_length]

    @classmethod
    def filter_answer_format(cls, triples: List[PreferenceTriple]) -> List[PreferenceTriple]:
        valid = []
        for t in triples:
            chosen = t.chosen.strip()
            if len(chosen) > 0 and chosen[0].isupper() and chosen[-1] in ('.', '!', '?'):
                valid.append(t)
        return valid

    @classmethod
    def synthesize_triples_from_raw(cls, raw_data_dir: str) -> List[PreferenceTriple]:
        logger.info(f"Extracting raw texts from: {raw_data_dir}")
        path = Path(raw_data_dir)
        triples = []

        if path.exists():
            for f in path.glob("*.json"):
                with open(f, "r", encoding="utf-8") as file:
                    doc = json.load(file)
                    content = cls.clean_text(doc.get("content", ""))
                    if len(content) >= 120:
                        prompt = f"Explain the architectural purpose of {doc.get('category', 'software engineering')}."
                        # Chosen: el texto original auténtico del autor
                        chosen = f"{content}. If you isolate feature stores from model serving, skew is eliminated."
                        # Rejected: respuesta sintética sobrecargada de formalismo artificial
                        rejected = (
                            f"Delve into the comprehensive paradigm where it is of paramount importance "
                            f"to meticulously observe that {content[:100]}... Ultimately, this fosters synergies."
                        )
                        triples.append(PreferenceTriple(prompt=prompt, chosen=chosen, rejected=rejected))

        if not triples:
            logger.warning("No raw files found. Using fallback preference triples.")
            triples = [
                PreferenceTriple(
                    prompt="Explain how RAG improves LLM outputs.",
                    chosen="RAG injects verified external context directly into the prompt, grounding the model and preventing hallucinations.",
                    rejected="Delve into the transformative potential of artificial intelligence where RAG seamlessly integrates multifaceted paradigms to elevate output efficacy.",
                ),
                PreferenceTriple(
                    prompt="What is the advantage of DPO over traditional RLHF?",
                    chosen="DPO derives an exact closed-form solution to the RLHF objective, training directly on preference pairs without needing a separate reward model.",
                    rejected="It is of paramount importance to comprehensively recognize that DPO represents a crucial stepping stone in the intricate journey of aligning synergistic models.",
                ),
            ]

        logger.info(f"Generated {len(triples)} raw preference triples.")
        return triples
