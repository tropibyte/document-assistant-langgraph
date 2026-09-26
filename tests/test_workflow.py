"""Tasks 2.2-2.6: the LangGraph workflow, routing, reducers and checkpointer."""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import agent
from agent import (
    AgentState, create_workflow, classify_intent, update_memory, format_conversation_history,
)
from conftest import script_turn
from retrieval import SimulatedRetriever
from schemas import CalculationResponse, UserIntent
from tools import ToolLogger, get_all_tools


@pytest.fixture
def tools(tmp_path):
    return get_all_tools(SimulatedRetriever(), ToolLogger(logs_dir=str(tmp_path)))


def run(workflow, llm, tools, text, thread="t1"):
    config = {"configurable": {"thread_id": thread, "llm": llm, "tools": tools}}
    return workflow.invoke({"messages": [], "user_input": text, "actions_taken": [], "tools_used": []}, config)


def test_graph_has_all_nodes_and_edges(llm, tools):
    graph = create_workflow(llm, tools).get_graph()
    assert set(graph.nodes) >= {"classify_intent", "qa_agent", "summarization_agent",
                                "calculation_agent", "update_memory"}
    edges = {(e.source, e.target) for e in graph.edges}
    for node in ("qa_agent", "summarization_agent", "calculation_agent"):
        assert ("classify_intent", node) in edges
        assert (node, "update_memory") in edges
    assert ("update_memory", "__end__") in edges
    assert ("__start__", "classify_intent") in edges


def test_compiled_with_in_memory_checkpointer(llm, tools):
    assert isinstance(create_workflow(llm, tools).checkpointer, InMemorySaver)


def test_actions_taken_uses_operator_add_reducer():
    import operator
    meta = AgentState.__annotations__["actions_taken"].__metadata__
    assert operator.add in meta


@pytest.mark.parametrize("intent, node", [
    ("qa", "qa_agent"),
    ("summarization", "summarization_agent"),
    ("calculation", "calculation_agent"),
    ("unknown", "qa_agent"),
])
def test_routes_each_intent_to_its_agent(llm, tools, intent, node):
    script_turn(llm, intent)
    state = run(create_workflow(llm, tools), llm, tools, "question")
    assert state["actions_taken"] == ["classify_intent", node, "update_memory"]
    assert state["intent"].intent_type == intent
    assert state["next_step"] == "end"


def test_specialists_request_their_own_schema_and_prompt(llm, tools):
    script_turn(llm, "summarization")
    run(create_workflow(llm, tools), llm, tools, "Summarize the contract")
    schemas_requested = [name for name, _ in llm.seen if name != "chat"]
    assert "SummarizationResponse" in schemas_requested
    chat_prompts = [p for name, p in llm.seen if name == "chat"]
    assert chat_prompts[0][0].content == agent.get_chat_prompt_template("summarization") \
        .invoke({"input": "x", "chat_history": []}).to_messages()[0].content


def test_calculation_agent_runs_the_calculator_tool(llm, tools):
    script_turn(
        llm, "calculation",
        tool_calls=[{"name": "calculator", "args": {"expression": "69300 + 214500"}, "id": "call_1"}],
        answer="The total is $283,800.",
        structured=CalculationResponse(expression="69300 + 214500", result=283800, explanation="sum"),
        memory_docs=["INV-002", "INV-003"],
    )
    state = run(create_workflow(llm, tools), llm, tools, "Add INV-002 and INV-003")
    assert state["tools_used"] == ["calculator"]
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[-1].content == "283800"
    assert state["current_response"]["result"] == 283800
    assert state["active_documents"] == ["INV-002", "INV-003"]
    assert state["messages"][-1].content == "The total is $283,800."


def test_state_persists_across_turns_via_checkpointer(llm, tools):
    workflow = create_workflow(llm, tools)
    script_turn(llm, "qa", answer="First answer.")
    run(workflow, llm, tools, "first question")
    script_turn(llm, "calculation",
                tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c2"}],
                answer="Second answer.")
    state = run(workflow, llm, tools, "second question")

    humans = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["first question", "second question"]
    # operator.add accumulated both turns' nodes on the same thread
    assert state["actions_taken"] == ["classify_intent", "qa_agent", "update_memory",
                                      "classify_intent", "calculation_agent", "update_memory"]
    # tools_used covers only the current turn
    assert state["tools_used"] == ["calculator"]
    # system prompts are not stacked into the saved history
    assert not any(isinstance(m, SystemMessage) for m in state["messages"])
    # the second turn's specialist saw the first turn in its chat history
    last_chat_prompt = [p for name, p in llm.seen if name == "chat"][-2]
    assert any(m.content == "First answer." for m in last_chat_prompt)


def test_threads_are_isolated(llm, tools):
    workflow = create_workflow(llm, tools)
    script_turn(llm, "qa")
    run(workflow, llm, tools, "thread one", thread="a")
    script_turn(llm, "qa")
    state = run(workflow, llm, tools, "thread two", thread="b")
    assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == ["thread two"]


def test_classify_intent_passes_history_to_prompt(llm):
    llm.structured["UserIntent"] = UserIntent(intent_type="calculation", confidence=0.8, reasoning="r")
    state = {"user_input": "and add them up",
             "messages": [HumanMessage(content="show invoices"), AIMessage(content="INV-001, INV-002")],
             "conversation_summary": "User looked at invoices"}
    update = classify_intent(state, {"configurable": {"llm": llm}})
    assert update == {"actions_taken": ["classify_intent"], "intent": llm.structured["UserIntent"],
                      "next_step": "calculation_agent"}
    prompt = llm.seen[-1][1]
    assert "and add them up" in prompt and "User: show invoices" in prompt
    assert "Summary of earlier conversation: User looked at invoices" in prompt


def test_classify_intent_falls_back_to_qa_on_error(llm):
    llm.structured["UserIntent"] = RuntimeError("model down")
    update = classify_intent({"user_input": "hi", "messages": []}, {"configurable": {"llm": llm}})
    assert update["next_step"] == "qa_agent"
    assert update["intent"].intent_type == "unknown"


def test_update_memory_sets_summary_documents_and_end(llm):
    from schemas import UpdateMemoryResponse
    llm.structured["UpdateMemoryResponse"] = UpdateMemoryResponse(summary="S", document_ids=["clm-001"])
    update = update_memory({"messages": [HumanMessage(content="q")]}, {"configurable": {"llm": llm}})
    assert update == {"conversation_summary": "S", "active_documents": ["CLM-001"],
                      "actions_taken": ["update_memory"], "next_step": "end"}


def test_update_memory_falls_back_to_cited_sources(llm):
    from schemas import UpdateMemoryResponse
    llm.structured["UpdateMemoryResponse"] = UpdateMemoryResponse(summary="S", document_ids=[])
    update = update_memory({"messages": [], "current_response": {"sources": ["INV-003"]}},
                           {"configurable": {"llm": llm}})
    assert update["active_documents"] == ["INV-003"]


def test_structured_timestamp_is_stamped_by_code(llm, tools):
    from datetime import datetime, timedelta
    from schemas import AnswerResponse
    llm.structured["AnswerResponse"] = AnswerResponse(
        question="q", answer="a", confidence=0.9, timestamp=datetime(2023, 10, 6))
    script_turn(llm, "qa")
    state = run(create_workflow(llm, tools), llm, tools, "q")
    stamped = datetime.fromisoformat(state["current_response"]["timestamp"])
    assert datetime.now() - stamped < timedelta(minutes=1)


def test_structured_response_is_pinned_to_current_turn(llm, tools):
    # Regression: without a prompt, the structured call sees the whole session and
    # described the FIRST question instead of the current one.
    workflow = create_workflow(llm, tools)
    script_turn(llm, "qa")
    run(workflow, llm, tools, "first {question}")
    script_turn(llm, "qa", answer="Second answer.")
    run(workflow, llm, tools, "second question")
    answer_prompts = [p for name, p in llm.seen if name == "AnswerResponse"]
    instruction = answer_prompts[-1][0].content
    assert "MOST RECENT user message" in instruction
    assert "second question" in instruction and "first {question}" not in instruction
    assert "first {question}" in answer_prompts[0][0].content   # braces survive
    assert answer_prompts[-1][-1].content == "Second answer."   # ends with this turn's answer


def test_summary_original_length_is_measured(llm, tools):
    from schemas import SummarizationResponse
    script_turn(
        llm, "summarization",
        tool_calls=[{"name": "document_reader", "args": {"doc_id": "CLM-001"}, "id": "r1"}],
        structured=SummarizationResponse(original_length=1000, summary="s", key_points=["k"]),
    )
    state = run(create_workflow(llm, tools), llm, tools, "Summarize the claim")
    read = [m for m in state["messages"] if isinstance(m, ToolMessage)][-1].content
    assert state["current_response"]["original_length"] == len(read) != 1000


def test_checkpointer_allows_user_intent_without_warning(llm, tools, caplog):
    import logging
    script_turn(llm, "qa")
    workflow = create_workflow(llm, tools)
    run(workflow, llm, tools, "q")
    with caplog.at_level(logging.WARNING):
        workflow.get_state({"configurable": {"thread_id": "t1"}})
    assert "unregistered type" not in caplog.text


def test_history_formatting_skips_tool_traffic():
    msgs = [HumanMessage(content="q1"),
            AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "x"}]),
            ToolMessage(content="2", tool_call_id="x"),
            AIMessage(content="a1")]
    assert format_conversation_history(msgs) == "User: q1\nAssistant: a1"
    assert format_conversation_history([]) == "No previous conversation."
