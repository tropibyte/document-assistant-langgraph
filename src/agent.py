from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
import warnings
from datetime import datetime
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import (
    get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT,
    get_structured_response_prompt,
)

# create_react_agent moved to langchain.agents in LangGraph 1.0 but still works;
# the deprecation warning is noise for this project.
warnings.filterwarnings("ignore", message=".*create_react_agent.*")

# Intent label -> graph node. Anything not listed (including "unknown") goes to the Q&A agent.
INTENT_TO_NODE = {
    "qa": "qa_agent",
    "summarization": "summarization_agent",
    "calculation": "calculation_agent",
}
DEFAULT_NODE = "qa_agent"

# How many recent human/assistant messages the intent classifier sees.
HISTORY_WINDOW = 8


class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    # operator.add reducer: each node returns ["<node name>"] and LangGraph appends it,
    # so the list records every node that ran (Task 2.6).
    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_react_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        # (prompt, schema): the prompt pins the structured response to the current turn.
        # The structured call sees the whole session, and without the current question
        # spelled out it often describes an earlier question instead.
        response_format=(get_structured_response_prompt(_last_human_text(messages)), response_schema),
    )

    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in current_turn_messages(result.get("messages", []))
                  if isinstance(t, ToolMessage)]

    return result, tools_used


def _last_human_text(messages: List[BaseMessage]) -> str:
    return next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def current_turn_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Messages after the last human message, i.e. this turn's tool calls, tool results and
    answer. A ReAct result repeats the whole chat history, including earlier turns' tool
    calls, so anything counted per turn has to start here.
    """
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[last_human + 1:]


def format_conversation_history(messages: List[BaseMessage], summary: Optional[str] = None,
                                window: int = HISTORY_WINDOW) -> str:
    """
    Render the recent conversation as plain text for the intent classifier.

    Only human turns and final assistant answers are kept; tool calls, tool results and
    system prompts would only add noise to a routing decision.
    """
    lines = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            lines.append(f"Assistant: {m.content}")
    lines = lines[-window:]

    parts = []
    if summary and summary != "No previous conversation.":
        parts.append(f"Summary of earlier conversation: {summary}")
    parts.extend(lines)
    return "\n".join(parts) if parts else "No previous conversation."


def _conversation_messages(result: Dict[str, Any]) -> List[BaseMessage]:
    """
    Messages from a ReAct run that belong in the conversation state.

    The run's message list starts with the system prompt the node built. add_messages
    gives it a fresh ID each turn, so returning it as-is would stack one more system prompt
    into the history on every turn. Everything else (the new human message, tool calls,
    tool results, the final answer) is kept; earlier history messages already carry IDs,
    so add_messages de-duplicates them.
    """
    return [m for m in result.get("messages", []) if not isinstance(m, SystemMessage)]


def _structured(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The ReAct agent's structured response as a JSON-safe dict, or None.

    The model fills in every schema field, including ones it cannot know. Two are
    replaced with measured values:
    - timestamp: the model invents one (e.g. "2023-10-06T14:30:00Z"); use the real time.
    - original_length (summaries): the model guesses (often a round 1000); use the
      character count of the documents actually read this turn.
    """
    structured = result.get("structured_response")
    if not isinstance(structured, BaseModel):
        return structured
    fields = type(structured).model_fields
    update: Dict[str, Any] = {}
    if "timestamp" in fields:
        update["timestamp"] = datetime.now()
    if "original_length" in fields:
        read = [m.content for m in current_turn_messages(result.get("messages", []))
                if isinstance(m, ToolMessage) and m.name == "document_reader"
                and not str(m.content).startswith(("Document with ID", "Error"))]
        if read:
            update["original_length"] = sum(len(str(c)) for c in read)
    return structured.model_copy(update=update).model_dump(mode="json")


def _run_specialist(intent: str, schema: type[BaseModel], node_name: str,
                    state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Shared body of the three specialist nodes: build the intent-specific prompt, run the
    ReAct agent with tools and structured output, and return the state update.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template(intent)

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    try:
        result, tools_used = invoke_react_agent(schema, messages, llm, tools)
    except Exception as e:
        # A failed model or tool call should end the turn politely, not crash the graph;
        # update_memory still runs so the session stays consistent.
        error = AIMessage(content=f"Sorry, I couldn't complete that request ({type(e).__name__}: {e}).")
        return {
            "messages": [HumanMessage(content=state["user_input"]), error],
            "actions_taken": [node_name],
            "current_response": {"error": str(e)},
            "tools_used": [],
            "next_step": "update_memory",
        }

    return {
        "messages": _conversation_messages(result),
        "actions_taken": [node_name],
        "current_response": _structured(result),
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    # Configure the llm chat model for structured output
    structured_llm = llm.with_structured_output(UserIntent)

    # Create a formatted prompt with conversation history and user input
    prompt = get_intent_classification_prompt().format(
        user_input=state["user_input"],
        conversation_history=format_conversation_history(history, state.get("conversation_summary")),
    )

    try:
        intent = structured_llm.invoke(prompt)
    except Exception as e:
        # If classification itself fails, fall back to the Q&A agent rather than failing the turn.
        intent = UserIntent(intent_type="unknown", confidence=0.0,
                            reasoning=f"Intent classification failed ({type(e).__name__}); defaulting to Q&A.")

    # "qa" -> qa_agent, "summarization" -> summarization_agent,
    # "calculation" -> calculation_agent, anything else -> qa_agent
    next_step = INTENT_TO_NODE.get(intent.intent_type, DEFAULT_NODE)

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent,
        "next_step": next_step,
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    return _run_specialist("qa", AnswerResponse, "qa_agent", state, config)


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    return _run_specialist("summarization", SummarizationResponse, "summarization_agent", state, config)


def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    return _run_specialist("calculation", CalculationResponse, "calculation_agent", state, config)


def _documents_from_response(response: Optional[Dict[str, Any]]) -> List[str]:
    """Document IDs cited by the specialist's structured response (AnswerResponse.sources etc.)."""
    if not isinstance(response, dict):
        return []
    ids = response.get("sources") or response.get("document_ids") or []
    return [str(i).strip().upper() for i in ids if str(i).strip()]


def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    # Retrieve the LLM from config
    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(UpdateMemoryResponse)

    try:
        response = structured_llm.invoke(prompt_with_history)
        summary = response.summary
        document_ids = response.document_ids
    except Exception:
        # Keep the previous memory rather than losing it if summarisation fails.
        summary = state.get("conversation_summary", "")
        document_ids = []

    # Prefer the memory model's view of the last message; fall back to what the
    # specialist cited, then to the documents already in play.
    active_documents = (document_ids
                        or _documents_from_response(state.get("current_response"))
                        or state.get("active_documents") or [])

    return {
        "conversation_summary": summary,
        "active_documents": active_documents,
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


def create_workflow(llm, tools, checkpointer=None):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.

    The llm and tools are not captured here: every node reads them from
    config["configurable"], so one compiled graph can serve any model.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END
        }
    )

    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    workflow.add_edge("update_memory", END)

    # Compiled with an InMemorySaver checkpointer (Task 2.6). The serializer only
    # registers UserIntent so reading it back from a checkpoint is allowed; see below.
    # (Tests may inject a different checkpointer.)
    return workflow.compile(checkpointer=checkpointer or InMemorySaver(serde=_checkpoint_serde()))


def _checkpoint_serde():
    """
    Serializer for the InMemorySaver with our Pydantic state types registered. The
    checkpointer stores `intent` as a UserIntent; unregistered types log a warning on every
    read and will be refused by a future LangGraph release. Returns None (InMemorySaver's
    default serializer) on an older langgraph-checkpoint without allow-listing.
    """
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        return JsonPlusSerializer(allowed_msgpack_modules=[
            (UserIntent.__module__, UserIntent.__name__),
        ])
    except TypeError:
        return None
