from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS IAM & Regional Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY: str = "mock-access-key"
    AWS_SECRET_KEY: str = "mock-secret-key"
    AWS_ARN_ROLE: str = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

    # SageMaker Endpoint Identifiers
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin-llama-3-1-8b-endpoint"
    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin-llama-3-1-8b-config"
    GPU_INSTANCE_TYPE: str = "ml.g5.xlarge"

    # Hugging Face Model & Deployment Parameters
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    HUGGINGFACE_ACCESS_TOKEN: str = "mock-hf-token"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096

    # Inference Behavior Hyperparameters
    MAX_NEW_TOKENS_INFERENCE: int = 256
    TEMPERATURE_INFERENCE: float = 0.7
    TOP_P_INFERENCE: float = 0.9

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
