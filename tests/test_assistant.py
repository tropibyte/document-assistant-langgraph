"""Task 2.6 (config) and the session / log files the rubric asks for."""
import json
import os

from conftest import script_turn
from schemas import CalculationResponse


def test_process_message_end_to_end(assistant, llm):
    sid = assistant.start_session("tester")
    script_turn(llm, "calculation",
                tool_calls=[{"name": "calculator", "args": {"expression": "2 + 3"}, "id": "c1"}],
                answer="It is 5.",
                structured=CalculationResponse(expression="2 + 3", result=5, explanation="add"))
    result = assistant.process_message("what is 2 + 3?")

    assert result["success"] is True
    assert result["response"] == "It is 5."
    assert result["intent"]["intent_type"] == "calculation"
    assert result["tools_used"] == ["calculator"]
    assert result["actions_taken"] == ["classify_intent", "calculation_agent", "update_memory"]
    assert result["structured_response"]["expression"] == "2 + 3"
    assert result["sources"] == result["active_documents"] == ["INV-001"]
    assert result["summary"].startswith("Summary after")

    # config carried thread_id / llm / tools into the graph
    snapshot = assistant.workflow.get_state({"configurable": {"thread_id": sid}})
    assert snapshot.values["user_input"] == "what is 2 + 3?"


def test_session_file_is_valid_json_with_turn_records(assistant, llm):
    sid = assistant.start_session("tester")
    script_turn(llm, "qa", answer="A1")
    assistant.process_message("q1")
    script_turn(llm, "summarization", answer="A2")
    assistant.process_message("q2")

    path = os.path.join(assistant.session_storage_path, f"{sid}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["session_id"] == sid and data["user_id"] == "tester"
    turns = data["conversation_history"]
    assert [t["user_input"] for t in turns] == ["q1", "q2"]
    assert [t["intent"]["intent_type"] for t in turns] == ["qa", "summarization"]
    # per-turn action list, even though the graph state accumulates across turns
    assert turns[1]["actions_taken"] == ["classify_intent", "summarization_agent", "update_memory"]


def test_tool_calls_logged_per_session(assistant, llm, tmp_path):
    sid = assistant.start_session("tester")
    script_turn(llm, "calculation",
                tool_calls=[{"name": "calculator", "args": {"expression": "6 * 7"}, "id": "c1"}])
    assistant.process_message("6 times 7")
    log_path = tmp_path / "logs" / f"session_{sid}.json"
    entries = json.loads(log_path.read_text(encoding="utf-8"))
    assert entries[0]["tool_name"] == "calculator"
    assert entries[0]["input"] == {"expression": "6 * 7"}


def test_resumed_session_restores_conversation(llm, tmp_path):
    from assistant import DocumentAssistant
    kwargs = dict(openai_api_key="k", llm=llm, session_storage_path=str(tmp_path / "s"),
                  logs_path=str(tmp_path / "l"))
    first = DocumentAssistant(**kwargs)
    sid = first.start_session("tester")
    script_turn(llm, "qa", answer="The total is $22,000.")
    first.process_message("What is the INV-001 total?")

    # A new process: a fresh InMemorySaver, rehydrated from the session file
    second = DocumentAssistant(**kwargs)
    assert second.start_session("tester", sid) == sid
    script_turn(llm, "qa", answer="Follow-up answered.")
    result = second.process_message("And the client?")
    assert result["success"]
    state = second.workflow.get_state({"configurable": {"thread_id": sid}}).values
    contents = [m.content for m in state["messages"]]
    assert "What is the INV-001 total?" in contents and "The total is $22,000." in contents
    assert result["actions_taken"] == ["classify_intent", "qa_agent", "update_memory"]


def test_process_message_without_session_raises(assistant):
    import pytest
    with pytest.raises(ValueError):
        assistant.process_message("hello")
