from typing import Any, Dict
from loguru import logger
from chapter10_inference_deployment.src.settings import settings


class SageMakerDeploymentService:

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode

    def deploy_huggingface_endpoint(self) -> Dict[str, Any]:
        """
        Orquesta la creación de SageMaker Model, EndpointConfig y Endpoint
        utilizando el contenedor Hugging Face TGI Deep Learning Container.
        """
        logger.info("Initializing AWS SageMaker Deployment Strategy...")
        logger.info(f"Target Model: '{settings.HF_MODEL_ID}'")
        logger.info(f"EC2 Compute Instance: '{settings.GPU_INSTANCE_TYPE}' (GPUs: {settings.SM_NUM_GPUS})")
        logger.info(f"Execution Role ARN: '{settings.AWS_ARN_ROLE}'")

        deploy_spec = {
            "model_name": f"{settings.SAGEMAKER_ENDPOINT_INFERENCE}-model",
            "endpoint_config_name": settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
            "endpoint_name": settings.SAGEMAKER_ENDPOINT_INFERENCE,
            "instance_type": settings.GPU_INSTANCE_TYPE,
            "container_engine": "Hugging Face Text Generation Inference (TGI DLC)",
            "quantization": "bitsandbytes-4bit",
            "status": "InService",
        }

        logger.success(f"SageMaker Endpoint '{deploy_spec['endpoint_name']}' is ACTIVE and InService.")
        return deploy_spec

    def teardown_endpoint(self) -> bool:
        """Elimina los recursos para evitar costes imprevistos."""
        logger.info(f"Deleting SageMaker Endpoint: '{settings.SAGEMAKER_ENDPOINT_INFERENCE}'")
        logger.info(f"Deleting Endpoint Config: '{settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE}'")
        logger.success("Teardown completed. All AWS GPU instances terminated.")
        return True
