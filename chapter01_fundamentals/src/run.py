"""
Main execution script to simulate the end-to-end LLM Twin lifecycle.
Run from root with: python -m chapter01_fundamentals.src.run
"""

from typing import Any, Dict, List
from loguru import logger
import yaml

from chapter01_fundamentals.src.pipeline_interfaces import (
    BaseDataCollectionPipeline,
    BaseFeaturePipeline,
    BaseInferencePipeline,
    BaseTrainingPipeline,
    Document,
)


class MockDataCollectionPipeline(BaseDataCollectionPipeline):
    def extract(self, source_name: str) -> List[Dict[str, Any]]:
        logger.info(f"Extracting data from: {source_name}")
        return [
            {
                "id": "doc-001",
                "raw_text": "Building LLM applications requires decoupling pipelines.",
                "url": f"https://{source_name}.com/article/1",
            }
        ]

    def transform(self, raw_records: List[Dict[str, Any]]) -> List[Document]:
        return [
            Document(
                id=rec["id"],
                content=rec["raw_text"],
                category="posts",
                source_url=rec["url"],
            )
            for rec in raw_records
        ]

    def load(self, documents: List[Document]) -> bool:
        logger.info(f"Saved {len(documents)} documents to Data Warehouse.")
        return True


class MockFeaturePipeline(BaseFeaturePipeline):
    def clean(self, documents: List[Document]) -> List[Document]:
        return documents

    def chunk_and_embed(self, documents: List[Document]) -> None:
        logger.info("Indexed chunks into Vector DB.")

    def generate_instruction_dataset(self, documents: List[Document]) -> str:
        artifact_path = "artifacts/dataset_v1.jsonl"
        logger.info(f"Artifact created: {artifact_path}")
        return artifact_path


class MockTrainingPipeline(BaseTrainingPipeline):
    def train(self, dataset_artifact_id: str, hyperparams: Dict[str, Any]) -> str:
        logger.info(f"Training on {dataset_artifact_id} with params: {hyperparams}")
        return "models/llm_twin_v1"

    def evaluate(self, model_id: str) -> Dict[str, float]:
        return {"rouge_score": 0.85, "perplexity": 4.2}


class MockInferencePipeline(BaseInferencePipeline):
    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        logger.info(f"Retrieving {top_k} context matches for: '{query}'")
        return ["Historical context sample regarding MLOps."]

    def generate(self, prompt: str, retrieved_context: List[str]) -> str:
        return f"Twin response to '{prompt}' using context: {retrieved_context[0]}"


def main():
    logger.info("=== Simulating LLM Twin End-to-End Pipeline ===")

    # 1. Extraction
    collector = MockDataCollectionPipeline()
    raw_data = collector.extract("linkedin")
    docs = collector.transform(raw_data)
    collector.load(docs)

    # 2. Features
    features = MockFeaturePipeline()
    cleaned = features.clean(docs)
    features.chunk_and_embed(cleaned)
    dataset_artifact = features.generate_instruction_dataset(cleaned)

    # 3. Training
    trainer = MockTrainingPipeline()
    model_id = trainer.train(dataset_artifact, {"epochs": 3, "lr": 2e-4})
    metrics = trainer.evaluate(model_id)
    logger.info(f"Evaluation metrics: {metrics}")

    # 4. Inference
    inference = MockInferencePipeline()
    query = "Draft an introduction about LLM Twin architecture."
    context = inference.retrieve_context(query)
    output = inference.generate(query, context)

    print("\n" + "=" * 50)
    print("OUTPUT RESULT:")
    print(output)
    print("=" * 50 + "\n")

    logger.success("=== Simulation Completed Successfully ===")


if __name__ == "__main__":
    main()