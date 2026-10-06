"""
Main entrypoint for Chapter 02 Data Collection Pipeline.
Run with: python -m chapter02_data_collection.src.run
"""

from loguru import logger
import yaml

from chapter02_data_collection.src.crawlers.github import GitHubCrawler
from chapter02_data_collection.src.crawlers.medium import MediumCrawler
from chapter02_data_collection.src.warehouse import DataWarehouse


def main():
    logger.info("=== Starting Data Collection Pipeline (ETL) ===")

    # 1. Cargar configuración
    with open("chapter02_data_collection/configs/crawlers.yaml", "r") as f:
        config = yaml.safe_load(f)

    author = config.get("author", "User")
    warehouse = DataWarehouse(local_cache_dir="data/raw")
    all_documents = []

    # 2. Extraer Código (GitHub)
    if config["targets"]["github"]["enabled"]:
        for username in config["targets"]["github"]["usernames"]:
            gh_crawler = GitHubCrawler(username=username)
            all_documents.extend(gh_crawler.extract())

    # 3. Extraer Artículos (Medium)
    if config["targets"]["medium"]["enabled"]:
        article_urls = [s["url"] for s in config["targets"]["medium"]["sources"]]
        medium_crawler = MediumCrawler(author=author, article_urls=article_urls)
        all_documents.extend(medium_crawler.extract())

    # 4. Cargar en el Data Warehouse
    total_stored = warehouse.load_documents(all_documents)

    logger.success(
        f"=== Data Collection Complete: {total_stored} documents persisted ==="
    )


if __name__ == "__main__":
    main()