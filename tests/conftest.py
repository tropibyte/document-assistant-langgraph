"""
Shared fixtures. The offline tests drive the real LangGraph workflow with a scripted
chat model, so routing, reducers, checkpointing, tools and session files are all
exercised without an API key. tests/test_live.py covers the real model.
"""
import os
import sys
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from schemas import (  # noqa: E402
    UserIntent, AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse,
)


class ScriptedChatModel(BaseChatModel):
    """
    A chat model that replays scripted messages, for testing.

    - Plain calls (the ReAct loop) pop the next message from `responses`; an AIMessage
      with tool_calls makes the ReAct agent run real tools.
    - with_structured_output(Schema) returns `structured[Schema.__name__]`, so the intent,
      the specialist's structured response and the memory update are all scripted.
    - Every prompt the model sees is recorded in `seen` for assertions.
    """
    responses: List[Any] = []
    structured: Dict[str, Any] = {}
    seen: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(("chat", messages))
        msg = self.responses.pop(0) if self.responses else AIMessage(content="(no scripted reply)")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def _respond(prompt):
            self.seen.append((schema.__name__, prompt))
            value = self.structured[schema.__name__]
            if isinstance(value, Exception):
                raise value
            return value(prompt) if callable(value) else value
        return RunnableLambda(_respond)


def script_turn(llm: ScriptedChatModel, intent: str, *, tool_calls: Optional[list] = None,
                answer: str = "Done.", structured: Any = None, memory_docs=("INV-001",)):
    """Queue everything the model will be asked for during one turn of the graph."""
    llm.structured["UserIntent"] = UserIntent(intent_type=intent, confidence=0.9, reasoning="scripted")
    if tool_calls:
        llm.responses.append(AIMessage(content="", tool_calls=tool_calls))
    llm.responses.append(AIMessage(content=answer))
    if structured is not None:
        llm.structured[type(structured).__name__] = structured
    llm.structured["UpdateMemoryResponse"] = UpdateMemoryResponse(
        summary=f"Summary after: {answer}", document_ids=list(memory_docs))


@pytest.fixture
def llm():
    return ScriptedChatModel(responses=[], structured={
        "AnswerResponse": AnswerResponse(question="q", answer="a", sources=["INV-001"], confidence=0.9),
        "SummarizationResponse": SummarizationResponse(
            original_length=100, summary="s", key_points=["k"], document_ids=["CON-001"]),
        "CalculationResponse": CalculationResponse(expression="1 + 1", result=2, explanation="e"),
    }, seen=[])


@pytest.fixture
def assistant(llm, tmp_path):
    from assistant import DocumentAssistant
    return DocumentAssistant(
        openai_api_key="test-key",
        llm=llm,
        session_storage_path=str(tmp_path / "sessions"),
        logs_path=str(tmp_path / "logs"),
    )
