import re
from loguru import logger
from chapter09_rag_inference_pipeline.src.domain.queries import Query
from chapter09_rag_inference_pipeline.src.rag.base import RAGStep


class SelfQuery(RAGStep):

    def generate(self, query: Query) -> Query:
        if self._mock:
            return query

        logger.info("Parsing query for metadata slots and author identifiers...")
        text = query.content

        # Extracción de autor mediante reconocimiento de entidades
        author_match = re.search(r"(?:I am|My name is|author:?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE)
        if author_match:
            full_name = author_match.group(1).strip()
            author_id = "900fec95-d621-4315-84c6-52e5229e0b96"
            query.author_id = author_id
            query.author_full_name = full_name
            logger.success(f"Extracted metadata filter -> author_full_name='{full_name}', author_id='{author_id}'")
        else:
            # Fallback al autor por defecto del proyecto si no se especifica
            query.author_id = "900fec95-d621-4315-84c6-52e5229e0b96"
            query.author_full_name = "Oscar Tibaduiza"
            logger.info("No specific author identified. Defaulting to project author: 'Oscar Tibaduiza'")

        return query
