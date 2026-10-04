"""
GitHub repository and file crawler.
"""

from typing import List
import hashlib
from loguru import logger
from chapter02_data_collection.src.crawlers.base import BaseCrawler
from chapter02_data_collection.src.models import RawDocument


class GitHubCrawler(BaseCrawler):

    def __init__(self, username: str):
        self.username = username

    def extract(self) -> List[RawDocument]:
        logger.info(f"Extracting public repositories for GitHub user: {self.username}")

        # Simulación de extracción de código de repositorios
        mock_code_files = [
            {
                "file": "pipeline.py",
                "repo": "llm-twin-service",
                "code": "def process_documents(docs: list) -> list:\n    return [d.strip() for d in docs]",
            },
            {
                "file": "model.py",
                "repo": "rag-inference",
                "code": "class LLMInferenceEngine:\n    def __init__(self, model_name: str):\n        self.model = model_name",
            },
        ]

        documents = []
        for item in mock_code_files:
            doc_id = hashlib.sha256(
                f"{self.username}-{item['repo']}-{item['file']}".encode()
            ).hexdigest()[:16]

            documents.append(
                RawDocument(
                    id=f"gh_{doc_id}",
                    category="code",
                    content=item["code"],
                    author=self.username,
                    source_url=f"https://github.com/{self.username}/{item['repo']}/blob/main/{item['file']}",
                    platform="github",
                    metadata={"repository": item["repo"], "file_path": item["file"]},
                )
            )

        logger.success(f"Extracted {len(documents)} code documents from GitHub.")
        return documents