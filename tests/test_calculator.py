"""Task 4.1: the calculator tool is a LangChain tool that validates, evaluates, logs and fails gracefully."""
import json

import pytest
from langchain_core.tools import BaseTool

from tools import ToolLogger, create_calculator_tool, get_all_tools
from retrieval import SimulatedRetriever


@pytest.fixture
def logger(tmp_path):
    lg = ToolLogger(logs_dir=str(tmp_path))
    lg.set_session("test-session")
    return lg


@pytest.fixture
def calculator(logger):
    return create_calculator_tool(logger)


def test_is_a_langchain_tool_with_docstring(calculator):
    assert isinstance(calculator, BaseTool)
    assert calculator.name == "calculator"
    assert "ALL arithmetic" in calculator.description
    assert list(calculator.args) == ["expression"]


@pytest.mark.parametrize("expression, expected", [
    ("2 + 3", "5"),
    ("69300 + 214500", "283800"),
    ("20000 * 0.1", "2000"),
    ("10 / 4", "2.5"),
    ("1 / 3", "0.3333333333"),
    ("(5000 + 12500 + 2500) * 1.1", "22000"),
    ("2 ** 10", "1024"),
    ("-3 + 1", "-2"),
    ("17 // 5", "3"),
    ("17 % 5", "2"),
    ("$20,000 + $2,000", "22000"),        # currency and thousands separators copied from a document
    ("180000 / 12", "15000"),
])
def test_evaluates_and_returns_a_string(calculator, expression, expected):
    result = calculator.invoke({"expression": expression})
    assert isinstance(result, str)
    assert result == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('dir')",
    "open('secrets.txt').read()",
    "exec('print(1)')",
    "abs(-1)",
    "x + 1",
    "[1, 2]",
    "1 if 1 else 2",
    "().__class__",
    "True + 1",
    "1, 2",
    "9 ** 9 ** 9",          # exponent bomb
    "2 ** (10 ** 3)",
    "",
    "1 +",
    "1" * 250,
])
def test_rejects_unsafe_or_invalid_expressions(calculator, expression):
    result = calculator.invoke({"expression": expression})
    assert isinstance(result, str)
    assert result.startswith("Error:")


def test_division_by_zero_is_graceful(calculator):
    assert calculator.invoke({"expression": "1 / 0"}) == "Error: division by zero"


def test_logs_every_call_to_the_session_file(calculator, logger):
    calculator.invoke({"expression": "1 + 1"})
    calculator.invoke({"expression": "1 / 0"})
    with open(logger.log_file, encoding="utf-8") as f:
        entries = json.load(f)
    assert logger.log_file.endswith("session_test-session.json")
    assert [e["tool_name"] for e in entries] == ["calculator", "calculator"]
    assert entries[0]["input"] == {"expression": "1 + 1"}
    assert "2" in entries[0]["output"]
    assert "division by zero" in entries[1]["output"]


def test_get_all_tools_includes_calculator(tmp_path):
    tools = get_all_tools(SimulatedRetriever(), ToolLogger(logs_dir=str(tmp_path)))
    assert [t.name for t in tools] == ["calculator", "document_search", "document_reader", "document_statistics"]


def test_document_search_keyword_honours_amount_filter(tmp_path):
    # Regression: search_type="keyword" + comparison="over" ignored the amount filter.
    search = get_all_tools(SimulatedRetriever(), ToolLogger(logs_dir=str(tmp_path)))[1]
    out = search.invoke({"query": "documents over $50,000", "search_type": "keyword",
                         "comparison": "over", "amount": 50000})
    assert out.startswith("Found 3 document(s)")
    for doc_id in ("INV-003", "CON-001", "INV-002"):
        assert doc_id in out


def test_document_search_all_returns_every_document(tmp_path):
    # Regression: the starter's `if ... if ... else` let search_type="all" fall through.
    search = get_all_tools(SimulatedRetriever(), ToolLogger(logs_dir=str(tmp_path)))[1]
    out = search.invoke({"query": "anything", "search_type": "all"})
    assert out.startswith("Found 5 document(s)")
