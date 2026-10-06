from loguru import logger
from chapter04_rag_feature_pipeline.src.domain.documents import EmbeddedChunk
from chapter09_rag_inference_pipeline.src.rag.prompt_templates import RAGPromptTemplate
from chapter09_rag_inference_pipeline.src.rag.retriever import ContextRetriever
from chapter09_rag_inference_pipeline.src.settings import settings


class LLMInferenceSagemakerEndpoint:
    """Cliente para la llamada al endpoint de inferencia de SageMaker desplegado en el Capítulo 10."""
    def __init__(self, endpoint_name: str = settings.SAGEMAKER_ENDPOINT_INFERENCE):
        self.endpoint_name = endpoint_name

    def invoke(self, prompt: str) -> str:
        logger.info(f"Invoking SageMaker endpoint: '{self.endpoint_name}'")
        # Simulación de la respuesta generada por TwinLlama-3.1-8B-DPO
        completion = (
            "Decoupled RAG architectures prevent training-serving skew by isolating feature pipelines "
            "from real-time serving requests. In the LLM Twin system, data is continuously indexed in Qdrant, "
            "expanded across multi-query angles, and reranked using neural cross-encoders before prompt injection. "
            "This delivers high factual grounding without robotic verbosity."
        )
        return completion


def format_context_string(documents: list) -> str:
    sections = []
    for idx, doc in enumerate(documents, start=1):
        sections.append(f"[{idx}] Source: {getattr(doc, 'platform', 'internal')} | URL: {getattr(doc, 'link', 'N/A')}\n{doc.content}")
    return "\n\n".join(sections)


def call_llm_service(query: str, context: str) -> str:
    template = RAGPromptTemplate().create_template()
    prompt = template.format(query=query, context=context)
    endpoint = LLMInferenceSagemakerEndpoint()
    return endpoint.invoke(prompt)


def rag(query: str, k: int = 3, expand_to_n: int = 3) -> str:
    """Función de nivel superior que ejecuta la canalización RAG completa de extremo a extremo."""
    logger.info("=== Executing Full RAG Pipeline ===")
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query=query, k=k, expand_to_n_queries=expand_to_n)
    context = format_context_string(documents)
    answer = call_llm_service(query=query, context=context)
    return answer
