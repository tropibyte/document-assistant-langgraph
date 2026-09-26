from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal, TypedDict
from datetime import datetime


# The four intents the router understands. Kept as a module-level alias so the
# schema, the router in agent.py and the tests all agree on one definition.
IntentType = Literal["qa", "summarization", "calculation", "unknown"]


def _dedupe_doc_ids(ids: List[str]) -> List[str]:
    """Normalise document IDs (strip, upper-case) and drop duplicates, keeping order."""
    seen, out = set(), []
    for doc_id in ids or []:
        norm = str(doc_id).strip().upper()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


class DocumentChunk(BaseModel):
    """Represents a chunk of document content"""
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="The actual text content")
    # The starter used `lambda: dict`, which returns the dict *type*, not an empty dict.
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score for retrieval")


class AnswerResponse(BaseModel):
    """Structured response for Q&A tasks (Task 1.1)."""
    question: str = Field(description="The original user question, restated verbatim")
    answer: str = Field(description="The answer to the question, grounded in the retrieved documents")
    sources: List[str] = Field(
        default_factory=list,
        description="IDs of the documents the answer relies on, e.g. ['INV-001', 'CON-001']",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer from 0.0 (a guess) to 1.0 (stated explicitly in a source document)",
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")

    @field_validator("sources")
    @classmethod
    def _normalise_sources(cls, v: List[str]) -> List[str]:
        return _dedupe_doc_ids(v)


class SummarizationResponse(BaseModel):
    """Structured response for summarization tasks"""
    original_length: int = Field(description="Length of original text")
    summary: str = Field(description="The generated summary")
    key_points: List[str] = Field(description="List of key points extracted")
    document_ids: List[str] = Field(default_factory=list, description="Documents summarized")
    timestamp: datetime = Field(default_factory=datetime.now)


class CalculationResponse(BaseModel):
    """Structured response for calculation tasks"""
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Step-by-step explanation")
    units: Optional[str] = Field(default=None, description="Units if applicable")
    timestamp: datetime = Field(default_factory=datetime.now)


class UpdateMemoryResponse(BaseModel):
    """Response after updating memory"""
    summary: str = Field(description="Summary of the conversation up to this point")
    document_ids: List[str] = Field(default_factory=list, description="List of documents ids that are relevant to the users last message")

    @field_validator("document_ids")
    @classmethod
    def _normalise_ids(cls, v: List[str]) -> List[str]:
        return _dedupe_doc_ids(v)


class UserIntent(BaseModel):
    """User intent classification (Task 1.2)."""
    intent_type: IntentType = Field(
        description="One of 'qa', 'summarization', 'calculation' or 'unknown'"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the classification, from 0.0 to 1.0",
    )
    reasoning: str = Field(description="One or two sentences explaining why this intent was chosen")

    @field_validator("intent_type", mode="before")
    @classmethod
    def _normalise_intent(cls, v: Any) -> Any:
        # Tolerate 'QA' or ' Summarization ' from the model; anything outside
        # the four labels still fails validation.
        return v.strip().lower() if isinstance(v, str) else v


class SessionState(BaseModel):
    """Session state"""
    session_id: str
    user_id: str
    # One JSON-safe record per turn (see DocumentAssistant._turn_record).
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    document_context: List[str] = Field(default_factory=list, description="Active document IDs")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
