from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = "mock-openai-key"

    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin-llama-3-1-8b-endpoint"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
