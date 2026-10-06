import json
from pathlib import Path
from loguru import logger
import yaml

from chapter07_evaluation.src.benchmark import BenchmarkRunner
from chapter07_evaluation.src.rag_metrics import RagasMetricsEvaluator


def main():
    logger.info("=== Starting Chapter 07: LLM & RAG Evaluation Pipeline ===")

    # 1. Cargar configuraciones
    cfg_path = "chapter07_evaluation/configs/evaluation.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Evaluación RAG (Framework Ragas)
    logger.info("--- 1. Evaluating RAG Pipeline Triad (Ragas Metrics) ---")
    rag_sample = RagasMetricsEvaluator.evaluate_sample(
        query="Why do we decouple Feature and Inference pipelines in RAG?",
        context="Decoupled pipelines avoid training-serving skew and allow caching vector embeddings independently.",
        answer="Decoupling feature extraction from serving avoids training-serving skew and makes embeddings reusable.",
        ground_truth="Decoupling prevents training-serving skew and optimizes serving latency.",
    )

    # 3. Benchmark Comparativo Multi-Modelo (LLM-as-a-Judge)
    logger.info("--- 2. Running Comparative Benchmark (LLM-as-a-Judge) ---")
    test_prompts = [
        "Discuss the concept of algorithm bias and its implications.",
        "Explain the advantage of DPO over standard RLHF algorithms.",
        "Describe the FTI architecture design pattern for LLMs.",
    ]

    summaries = BenchmarkRunner.run_comparative_benchmark(test_prompts)

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS (Scale 1.0 to 3.0)")
    print("=" * 65)
    print(f"{'Model Name':<25} | {'Accuracy Score':<15} | {'Style Score':<15}")
    print("-" * 65)
    for s in summaries:
        print(f"{s.model_name:<25} | {s.average_accuracy:<15} | {s.average_style:<15}")
    print("=" * 65 + "\n")

    # 4. Guardar resultados
    out_dir = Path(config["storage"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config["storage"]["output_report"]

    report_payload = {
        "rag_evaluation": rag_sample.model_dump(),
        "model_benchmarks": [s.model_dump() for s in summaries],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.success(f"Evaluation report exported to: {out_path}")
    logger.success("=== Chapter 07 Completed Successfully ===")


if __name__ == "__main__":
    main()
