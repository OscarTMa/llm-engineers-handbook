import json
from pathlib import Path
from loguru import logger
import yaml

from chapter08_inference_optimization.src.batching import (
    ContinuousBatchingScheduler,
    InferenceRequest,
)
from chapter08_inference_optimization.src.kv_cache import KVCacheManager
from chapter08_inference_optimization.src.quantization import QuantizationAlgorithms
from chapter08_inference_optimization.src.speculative import SpeculativeDecodingEngine


def main():
    logger.info("=== Starting Chapter 08: Inference Optimization Pipeline ===")

    # 1. Cargar configuraciones
    cfg_path = "chapter08_inference_optimization/configs/inference.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Análisis y cálculo del KV Cache
    logger.info("--- 1. Evaluating KV Cache Scaling & Footprint ---")
    kv_metrics = KVCacheManager.calculate_cache_size(
        num_tokens=config["kv_cache"]["max_sequence_length"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        head_dim=config["model"]["head_dim"],
        bytes_per_param=config["model"]["bytes_per_param"],
    )
    logger.info(
        f"Llama 3.1 8B KV Cache @ {kv_metrics.num_tokens} tokens: "
        f"{kv_metrics.cache_size_mb} MB ({kv_metrics.cache_size_gb} GB)"
    )
    KVCacheManager.evaluate_growth_profiles(max_seq_len=config["kv_cache"]["max_sequence_length"])

    # 3. Simulación de Continuous Batching
    logger.info("--- 2. Continuous / In-Flight Batching Execution ---")
    scheduler = ContinuousBatchingScheduler(max_batch_size=config["continuous_batching"]["max_batch_size"])
    sample_requests = [
        InferenceRequest(request_id="req_01", prompt_tokens=64, generated_tokens_target=12),
        InferenceRequest(request_id="req_02", prompt_tokens=128, generated_tokens_target=45),
        InferenceRequest(request_id="req_03", prompt_tokens=32, generated_tokens_target=8),
        InferenceRequest(request_id="req_04", prompt_tokens=256, generated_tokens_target=60),
        InferenceRequest(request_id="req_05", prompt_tokens=48, generated_tokens_target=14),
        InferenceRequest(request_id="req_06", prompt_tokens=80, generated_tokens_target=22),
    ]
    scheduler.add_requests(sample_requests)
    batching_telemetry = scheduler.run_simulation()

    # 4. Evaluación de Speculative Decoding
    logger.info("--- 3. Speculative Decoding Validation ---")
    avg_tokens, speedup = SpeculativeDecodingEngine.simulate_verification(
        lookahead_k=config["speculative_decoding"]["lookahead_k"],
        acceptance_rate=config["speculative_decoding"]["estimated_acceptance_rate"],
    )

    # 5. Algoritmos de Cuantización (Absmax vs Zero-Point)
    logger.info("--- 4. Post-Training Quantization Error Analysis ---")
    quant_telemetry = QuantizationAlgorithms.evaluate_quantization_error()

    # 6. Exportar reporte consolidado
    out_dir = Path(config["storage"]["telemetry_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / config["storage"]["report_file"]

    report_payload = {
        "model_spec": config["model"],
        "kv_cache_analysis": kv_metrics.model_dump(),
        "continuous_batching": batching_telemetry,
        "speculative_decoding": {
            "lookahead_k": config["speculative_decoding"]["lookahead_k"],
            "avg_accepted_tokens": avg_tokens,
            "net_speedup_factor": speedup,
        },
        "quantization_benchmarks": quant_telemetry,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.success(f"Inference optimization telemetry saved to: {out_file}")
    logger.success("=== Chapter 08 Completed Successfully ===")


if __name__ == "__main__":
    main()
