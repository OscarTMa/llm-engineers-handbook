from typing import Any, Dict
from loguru import logger
from chapter10_inference_deployment.src.settings import settings


class SageMakerAutoScaler:

    @staticmethod
    def configure_target_tracking(
        min_capacity: int = 1,
        max_capacity: int = 4,
        target_invocations: int = 70,
        scale_out_cooldown: int = 60,
        scale_in_cooldown: int = 300,
    ) -> Dict[str, Any]:
        """
        Registra el Scalable Target y crea la política TargetTrackingScaling
        sobre la métrica SageMakerInferenceComponentInvocationsPerCopy.
        """
        logger.info(f"Registering Scalable Target for Endpoint: '{settings.SAGEMAKER_ENDPOINT_INFERENCE}'")
        logger.info(f"Capacity Bounds: Min={min_capacity}, Max={max_capacity}")
        logger.info(f"Target Tracking Metric: InvocationsPerCopy = {target_invocations}")
        logger.info(f"Cooldown Safeguards: Scale-Out={scale_out_cooldown}s, Scale-In={scale_in_cooldown}s")

        policy_spec = {
            "resource_id": f"endpoint/{settings.SAGEMAKER_ENDPOINT_INFERENCE}/variant/AllTraffic",
            "min_capacity": min_capacity,
            "max_capacity": max_capacity,
            "target_value": target_invocations,
            "scale_out_cooldown": scale_out_cooldown,
            "scale_in_cooldown": scale_in_cooldown,
            "status": "Configured",
        }

        logger.success("Application Auto Scaling Policy successfully established.")
        return policy_spec
