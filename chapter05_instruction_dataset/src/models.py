from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class AlpacaRecord(BaseModel):
    instruction: str
    input: str = ""
    output: str
    system: str = "You are Oscar's LLM Twin. Answer authoritatively in his technical style."


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatMLRecord(BaseModel):
    messages: List[ChatMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InstructionRecord(BaseModel):
    id: str
    category: str
    source_url: Optional[str] = None
    instruction: str
    context: str = ""
    target_output: str

    def to_alpaca(self) -> AlpacaRecord:
        return AlpacaRecord(
            instruction=self.instruction,
            input=self.context,
            output=self.target_output,
        )

    def to_chatml(self) -> ChatMLRecord:
        return ChatMLRecord(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are Oscar's LLM Twin. Answer authoritatively and concisely in his technical style.",
                ),
                ChatMessage(
                    role="user",
                    content=f"{self.instruction}\n{self.context}".strip(),
                ),
                ChatMessage(
                    role="assistant",
                    content=self.target_output,
                ),
            ],
            metadata={"category": self.category, "record_id": self.id},
        )
