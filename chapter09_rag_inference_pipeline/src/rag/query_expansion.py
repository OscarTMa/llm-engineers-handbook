from typing import List
from loguru import logger
from chapter09_rag_inference_pipeline.src.domain.queries import Query
from chapter09_rag_inference_pipeline.src.rag.base import RAGStep
from chapter09_rag_inference_pipeline.src.rag.prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):

    def generate(self, query: Query, expand_to_n: int = 3) -> List[Query]:
        assert expand_to_n > 0, "'expand_to_n' must be > 0"
        if self._mock:
            return [query for _ in range(expand_to_n)]

        template = QueryExpansionTemplate()
        logger.info(f"Expanding query into {expand_to_n} semantic perspectives.")

        # Generador heurístico de perspectivas semánticas deterministas
        expanded_contents = [
            query.content,
            f"What are the key architectural principles and implementation steps of: {query.content}",
            f"Provide technical details, trade-offs, and best practices regarding: {query.content}",
        ]

        results = [query]
        for content in expanded_contents[1:expand_to_n]:
            results.append(query.replace_content(content))

        logger.success(f"Generated {len(results)} query variations for vector retrieval.")
        return results
