from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
from loguru import logger

from chapter04_rag_feature_pipeline.src.domain.base import MockQdrantStorage
from chapter04_rag_feature_pipeline.src.domain.documents import (
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from chapter09_rag_inference_pipeline.src.domain.queries import EmbeddedQuery, Query
from chapter09_rag_inference_pipeline.src.rag.query_expansion import QueryExpansion
from chapter09_rag_inference_pipeline.src.rag.reranking import Reranker
from chapter09_rag_inference_pipeline.src.rag.self_query import SelfQuery


class ContextRetriever:

    def __init__(self, mock: bool = False):
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    def _search_data_category(self, doc_cls: type[EmbeddedChunk], embedded_query: EmbeddedQuery, limit: int) -> List[EmbeddedChunk]:
        collection = doc_cls.get_collection_name()
        records = MockQdrantStorage.scroll(collection, limit=limit * 2)
        results = []
        for r in records:
            chunk = doc_cls.from_record(r)
            if embedded_query.author_id:
                if chunk.author_id and chunk.author_id != embedded_query.author_id:
                    continue
            results.append(chunk)
            if len(results) >= limit:
                break
        return results

    def _search(self, query: Query, k: int) -> List[EmbeddedChunk]:
        k_per_category = max(k // 3, 1)
        embedded_query = EmbeddedQuery(**query.model_dump(), embedding=[0.0] * 384)

        posts = self._search_data_category(EmbeddedPostChunk, embedded_query, k_per_category)
        articles = self._search_data_category(EmbeddedArticleChunk, embedded_query, k_per_category)
        repos = self._search_data_category(EmbeddedRepositoryChunk, embedded_query, k_per_category)

        return posts + articles + repos

    def search(
        self,
        query: Union[str, Query] = "",
        raw_query: Union[str, Query] = "",
        k: int = 3,
        expand_to_n_queries: int = 3,
    ) -> List[EmbeddedChunk]:
        target = query if query else raw_query
        target_str = target.content if isinstance(target, Query) else str(target)
        logger.info(f"ContextRetriever search invoked: query='{target_str}' (k={k}, expand_to_n={expand_to_n_queries})")

        # 1. Transformar a entidad de dominio y extraer metadatos
        query_model = Query.from_str(target_str)
        query_model = self._metadata_extractor.generate(query_model)

        # 2. Expansión de consultas (Multi-Query)
        expanded_queries = self._query_expander.generate(query_model, expand_to_n=expand_to_n_queries)

        # 3. Búsqueda vectorial paralela sobre Qdrant
        all_candidates: List[EmbeddedChunk] = []
        with ThreadPoolExecutor(max_workers=expand_to_n_queries) as executor:
            futures = [executor.submit(self._search, q, k) for q in expanded_queries]
            for future in as_completed(futures):
                all_candidates.extend(future.result())

        # 4. Deduplicación por chunk_id
        seen_ids = set()
        unique_candidates: List[EmbeddedChunk] = []
        for chunk in all_candidates:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_candidates.append(chunk)

        logger.info(f"Retrieved {len(unique_candidates)} unique candidates across {len(expanded_queries)} query paths.")

        # 5. Reranking neural con Cross-Encoder
        if unique_candidates:
            reranked = self._reranker.generate(query_model, unique_candidates, keep_top_k=k)
        else:
            reranked = []

        return reranked
