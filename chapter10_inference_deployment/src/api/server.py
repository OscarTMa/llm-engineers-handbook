from fastapi import FastAPI, HTTPException
from loguru import logger

from chapter09_rag_inference_pipeline.src.rag.llm_client import format_context_string
from chapter09_rag_inference_pipeline.src.rag.retriever import ContextRetriever
from chapter10_inference_deployment.src.api.schemas import QueryRequest, QueryResponse
from chapter10_inference_deployment.src.aws.sagemaker_client import (
    InferenceExecutor,
    LLMInferenceSagemakerEndpoint,
)
from chapter10_inference_deployment.src.settings import settings

app = FastAPI(
    title="LLM Twin Inference Service",
    description="Business Microservice exposing the End-to-End RAG Inference Pipeline",
    version="1.0.0",
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "endpoint": settings.SAGEMAKER_ENDPOINT_INFERENCE,
        "region": settings.AWS_REGION,
    }


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        logger.info(f"POST /rag received: '{request.query}'")

        # 1. Recuperar contexto mediante el ContextRetriever del Capítulo 9
        retriever = ContextRetriever(mock=False)
        documents = retriever.search(query=request.query, k=3, expand_to_n_queries=3)
        context_str = format_context_string(documents)

        # 2. Invocar el LLM Microservice alojado en SageMaker
        sm_client = LLMInferenceSagemakerEndpoint(mock_mode=True)
        executor = InferenceExecutor(sm_client, query=request.query, context=context_str)
        answer = executor.execute()

        return QueryResponse(
            query=request.query,
            answer=answer,
            grounding_references=len(documents),
            metadata={"endpoint": settings.SAGEMAKER_ENDPOINT_INFERENCE},
        )
    except Exception as e:
        logger.exception("Inference processing error.")
        raise HTTPException(status_code=500, detail=str(e))
