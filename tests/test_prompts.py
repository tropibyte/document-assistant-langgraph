"""Tasks 3.1 / 3.2 and the intent-classification prompt."""
import pytest
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import prompts
from prompts import (
    get_chat_prompt_template, get_intent_classification_prompt,
    QA_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, CALCULATION_SYSTEM_PROMPT,
)


@pytest.mark.parametrize("intent, expected", [
    ("qa", QA_SYSTEM_PROMPT),
    ("summarization", SUMMARIZATION_SYSTEM_PROMPT),
    ("calculation", CALCULATION_SYSTEM_PROMPT),
    ("unknown", QA_SYSTEM_PROMPT),
    ("something-else", QA_SYSTEM_PROMPT),
])
def test_chat_prompt_selects_system_prompt(intent, expected):
    template = get_chat_prompt_template(intent)
    assert isinstance(template, ChatPromptTemplate)
    messages = template.invoke({"input": "hello", "chat_history": []}).to_messages()
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == expected
    assert isinstance(messages[-1], HumanMessage) and messages[-1].content == "hello"


def test_chat_prompt_structure_includes_history():
    template = get_chat_prompt_template("calculation")
    assert set(template.input_variables) == {"input", "chat_history"}
    history = [HumanMessage(content="earlier q"), AIMessage(content="earlier a")]
    messages = template.invoke({"input": "now", "chat_history": history}).to_messages()
    assert [m.content for m in messages[1:]] == ["earlier q", "earlier a", "now"]


def test_calculation_prompt_requirements():
    p = CALCULATION_SYSTEM_PROMPT
    assert p.strip(), "CALCULATION_SYSTEM_PROMPT must be implemented"
    assert "document_reader" in p          # retrieve the document with the reader tool
    assert "calculator" in p               # use the calculator tool
    assert "ALL arithmetic" in p and "no matter how simple" in p
    assert "expression" in p               # determine the expression
    assert "{" not in p and "}" not in p   # no stray template variables


def test_intent_prompt_has_categories_examples_and_scoring():
    template = get_intent_classification_prompt()
    assert isinstance(template, PromptTemplate)
    assert set(template.input_variables) == {"user_input", "conversation_history"}
    text = template.template
    for label in ("qa:", "summarization:", "calculation:", "unknown:"):
        assert label in text
    assert text.count("->") >= 8                   # worked examples for every category
    assert "Confidence scoring" in text
    assert "Reasoning" in text
    rendered = template.format(user_input="Sum the invoices", conversation_history="None")
    assert "Sum the invoices" in rendered


def test_memory_prompt_asks_for_document_ids():
    assert "document_ids" in prompts.MEMORY_SUMMARY_PROMPT
