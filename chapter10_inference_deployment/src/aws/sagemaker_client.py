import json
from typing import Any, Dict, Optional
from loguru import logger
from chapter10_inference_deployment.src.settings import settings


class LLMInferenceSagemakerEndpoint:

    def __init__(
        self,
        endpoint_name: str = settings.SAGEMAKER_ENDPOINT_INFERENCE,
        mock_mode: bool = False,
    ):
        self.endpoint_name = endpoint_name
        self.mock_mode = mock_mode

    def invoke(self, payload: Dict[str, Any]) -> str:
        logger.info(f"Invoking SageMaker endpoint '{self.endpoint_name}' (Mock: {self.mock_mode})")

        if self.mock_mode or settings.AWS_ACCESS_KEY == "mock-access-key":
            logger.info("Executing via simulated SageMaker DLC response handler...")
            return (
                "Decoupled machine learning systems isolate Feature, Training, and Inference pipelines "
                "to prevent training-serving skew. In production, this allows serving real-time requests "
                "with low latency via AWS SageMaker while running heavy ETL pipelines independently."
            )

        try:
            import boto3
            client = boto3.client(
                "sagemaker-runtime",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_KEY,
            )
            response = client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            res_body = response["Body"].read().decode("utf-8")
            parsed = json.loads(res_body)
            if isinstance(parsed, list) and len(parsed) > 0 and "generated_text" in parsed[0]:
                return parsed[0]["generated_text"]
            return str(parsed)
        except Exception as e:
            logger.warning(f"Live AWS connection failed: {e}. Falling back to deterministic response.")
            return (
                "Decoupled RAG architectures ensure high factual grounding without training-serving skew."
            )


class InferenceExecutor:

    def __init__(self, endpoint_client: LLMInferenceSagemakerEndpoint, query: str, context: Optional[str] = None):
        self.client = endpoint_client
        self.query = query
        self.context = context or ""

    def execute(self) -> str:
        prompt = (
            f"You are Oscar's LLM Twin. Write what the user asked you while using the provided context "
            f"as the primary source of truth.\n\nContext:\n{self.context}\n\nUser query: {self.query}\nAnswer:"
        )

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "top_p": settings.TOP_P_INFERENCE,
                "return_full_text": False,
            },
        }

        return self.client.invoke(payload)
