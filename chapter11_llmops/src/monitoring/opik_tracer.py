import time
from typing import Any, Callable, Dict, List
import uuid
from loguru import logger
from pydantic import BaseModel, Field


class OpikSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    duration_ms: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


class OpikTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str
    user_query: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_duration_ms: float
    time_to_first_token_ms: float
    tokens_per_second: float
    spans: List[OpikSpan] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OpikTracker:
    """Simulador e integrador del cliente Opik de Comet ML para seguimiento de trazas jerárquicas."""

    current_spans: List[OpikSpan] = []

    @classmethod
    def trace_span(cls, name: str):
        """Decorador para instrumentar funciones intermedias como spans de Opik."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                span = OpikSpan(
                    name=name,
                    duration_ms=round(elapsed_ms, 2),
                    inputs={"args_count": len(args)},
                    outputs={"result_type": str(type(result))},
                )
                cls.current_spans.append(span)
                return result
            return wrapper
        return decorator

    @classmethod
    def create_trace(
        cls,
        user_query: str,
        generated_text: str,
        total_time_ms: float,
        project_name: str = "llm-twin-production",
    ) -> OpikTrace:
        # Cálculo de tokens y métricas de inferencia
        p_tokens = len(user_query.split()) * 2
        c_tokens = len(generated_text.split()) * 2
        t_tokens = p_tokens + c_tokens

        ttft_ms = round(total_time_ms * 0.25, 2)
        gen_duration_sec = max((total_time_ms - ttft_ms) / 1000.0, 0.001)
        tps = round(c_tokens / gen_duration_sec, 2)

        trace = OpikTrace(
            project_name=project_name,
            user_query=user_query,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            total_duration_ms=round(total_time_ms, 2),
            time_to_first_token_ms=ttft_ms,
            tokens_per_second=tps,
            spans=list(cls.current_spans),
            tags=["rag", "production", "llm-twin"],
            metadata={"model": "TwinLlama-3.1-8B-DPO", "environment": "aws-production"},
        )

        cls.current_spans.clear()
        logger.info(
            f"Opik Trace Emitted -> ID: {trace.trace_id[:8]} | Tokens: {trace.total_tokens} | "
            f"TTFT: {trace.time_to_first_token_ms}ms | TPS: {trace.tokens_per_second} tokens/s"
        )
        return trace
