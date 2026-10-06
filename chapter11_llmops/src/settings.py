from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ZenML Cloud
    ZENML_SERVER_URL: str = "https://mock-zenml.cloudinfra.zenml.io"
    ZENML_TENANT: str = "twin"
    ZENML_STACK_NAME: str = "aws-stack"

    # Opik / Comet ML
    OPIK_API_KEY: str = "mock-opik-key"
    OPIK_PROJECT_NAME: str = "llm-twin-production"
    OPIK_WORKSPACE: str = "oscar-workspace"

    # AWS
    AWS_REGION: str = "eu-central-1"
    AWS_ECR_NAME: str = "zenml-llmtwin"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
