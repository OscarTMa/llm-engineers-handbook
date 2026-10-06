from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationCriteriaScore(BaseModel):
    analysis: str
    score: int = Field(ge=1, le=3)


class JudgeVerdict(BaseModel):
    model_id: str
    prompt: str
    generated_answer: str
    accuracy: EvaluationCriteriaScore
    style: EvaluationCriteriaScore


class RagasSampleResult(BaseModel):
    query: str
    context: str
    answer: str
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)


class BenchmarkSummary(BaseModel):
    model_name: str
    average_accuracy: float
    average_style: float
    total_evaluated: int
