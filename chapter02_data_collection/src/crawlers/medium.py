"""
Medium article crawler and cleaner.
"""

from typing import List
import hashlib
from loguru import logger
from chapter02_data_collection.src.crawlers.base import BaseCrawler
from chapter02_data_collection.src.models import RawDocument


class MediumCrawler(BaseCrawler):

    def __init__(self, author: str, article_urls: List[str]):
        self.author = author
        self.article_urls = article_urls

    def extract(self) -> List[RawDocument]:
        logger.info(f"Extracting articles for author: {self.author}")

        # Simulación de artículos de blog
        documents = []
        for url in self.article_urls:
            doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]
            article_body = (
                "Decoupling the ML lifecycle with Feature/Training/Inference pipelines "
                "prevents training-serving skew and makes RAG applications modular."
            )

            documents.append(
                RawDocument(
                    id=f"med_{doc_id}",
                    category="articles",
                    content=article_body,
                    author=self.author,
                    source_url=url,
                    platform="medium",
                    metadata={"reading_time_min": 4},
                )
            )

        logger.success(f"Extracted {len(documents)} articles from Medium.")
        return documents

