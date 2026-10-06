import json
from pathlib import Path
from loguru import logger
import yaml

from chapter09_rag_inference_pipeline.src.run import seed_qdrant_store
from chapter10_inference_deployment.src.api.schemas import QueryRequest
from chapter10_inference_deployment.src.api.server import app, rag_endpoint
from chapter10_inference_deployment.src.aws.autoscaling import SageMakerAutoScaler
from chapter10_inference_deployment.src.aws.sagemaker_deployer import SageMakerDeploymentService


def main():
    logger.info("=== Starting Chapter 10: Production Inference Deployment Pipeline ===")

    # 1. Cargar configuración YAML
    cfg_path = "chapter10_inference_deployment/configs/deployment.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Sembrar datos para el almacén vectorial de Qdrant
    seed_qdrant_store()

    # 3. Paso 1: Despliegue de AWS SageMaker Endpoint (Simulación / Boto3)
    logger.info("--- 1. Deploying AWS SageMaker Real-Time Endpoint ---")
    deployer = SageMakerDeploymentService(mock_mode=True)
    deploy_metadata = deployer.deploy_huggingface_endpoint()

    # 4. Paso 2: Configurar Reglas de Application Auto Scaling
    logger.info("--- 2. Setting Up Application Auto Scaling ---")
    scaling_metadata = SageMakerAutoScaler.configure_target_tracking(
        min_capacity=config["autoscaling"]["min_capacity"],
        max_capacity=config["autoscaling"]["max_capacity"],
        target_invocations=config["autoscaling"]["target_invocations_per_copy"],
        scale_out_cooldown=config["autoscaling"]["scale_out_cooldown"],
        scale_in_cooldown=config["autoscaling"]["scale_in_cooldown"],
    )

    # 5. Paso 3: Probar el endpoint de negocio FastAPI /rag
    logger.info("--- 3. Testing Business Microservice /rag Endpoint ---")
    test_request = QueryRequest(
        query="I am Oscar Tibaduiza. How do microservice architectures optimize LLM serving?",
        author="Oscar Tibaduiza",
    )

    import asyncio
    response = asyncio.run(rag_endpoint(test_request))

    print("\n" + "=" * 65)
    print("FASTAPI SERVING RESPONSE (/rag):")
    print("=" * 65)
    print(f"QUERY: {response.query}\n")
    print(f"ANSWER: {response.answer}\n")
    print(f"GROUNDING REFS: {response.grounding_references}")
    print("=" * 65 + "\n")

    # 6. Registrar telemetría de despliegue
    telemetry_dir = Path(config["storage"]["telemetry_dir"])
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    report_path = telemetry_dir / config["storage"]["report_file"]

    report_payload = {
        "deployment": deploy_metadata,
        "autoscaling": scaling_metadata,
        "test_call": response.model_dump(),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.success(f"Deployment telemetry recorded to: {report_path}")
    logger.success("=== Chapter 10 Completed Successfully ===")


if __name__ == "__main__":
    main()
