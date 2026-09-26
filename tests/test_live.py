"""
Live tests against the real model. Skipped unless RUN_LIVE=1 and OPENAI_API_KEY are set:

    RUN_LIVE=1 python -m pytest tests/test_live.py -v
"""
import os

import pytest
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE") != "1" or not os.getenv("OPENAI_API_KEY"),
                       reason="set RUN_LIVE=1 and OPENAI_API_KEY to run live tests"),
]

# A labelled set for the intent classifier, including the borderline cases the prompt's
# examples are there to settle.
INTENT_CASES = [
    ("What is the client name on invoice INV-002?", "qa"),
    ("When was the insurance claim incident?", "qa"),
    ("What are the payment terms for INV-003?", "qa"),
    ("Find documents with amounts over $50,000", "qa"),
    ("Summarize all contracts", "summarization"),
    ("Give me the key points of the insurance claim", "summarization"),
    ("Can you give me an overview of invoice INV-001?", "summarization"),
    ("Calculate the sum of all invoice totals", "calculation"),
    ("What is the average invoice amount?", "calculation"),
    ("How much more is INV-003 than INV-002?", "calculation"),
    ("What percentage of the claim is the hospital visit?", "calculation"),
    ("Hello!", "unknown"),
    ("What's the weather in Paris?", "unknown"),
]


@pytest.fixture(scope="module")
def live_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model=os.getenv("MODEL_NAME", "gpt-4o"),
                      temperature=0, base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"))


@pytest.mark.parametrize("text, expected", INTENT_CASES)
def test_intent_classification(live_llm, text, expected):
    from agent import classify_intent
    update = classify_intent({"user_input": text, "messages": []}, {"configurable": {"llm": live_llm}})
    assert update["intent"].intent_type == expected, update["intent"].reasoning
    assert 0.0 <= update["intent"].confidence <= 1.0


def test_end_to_end_all_intents(live_llm, tmp_path):
    from assistant import DocumentAssistant
    a = DocumentAssistant(openai_api_key="unused", llm=live_llm,
                          session_storage_path=str(tmp_path / "sessions"), logs_path=str(tmp_path / "logs"))
    a.start_session("live-test")

    qa = a.process_message("What is the total due on invoice INV-003?")
    assert qa["success"] and qa["intent"]["intent_type"] == "qa"
    assert "214,500" in qa["response"] or "214500" in qa["response"]
    assert "INV-003" in qa["sources"]

    calc = a.process_message("Add the totals of INV-002 and INV-003")
    assert calc["success"] and calc["intent"]["intent_type"] == "calculation"
    assert "calculator" in calc["tools_used"]
    assert calc["structured_response"]["result"] == 283800

    summ = a.process_message("Summarize the insurance claim")
    assert summ["success"] and summ["intent"]["intent_type"] == "summarization"
    assert summ["structured_response"]["key_points"]
    assert "CLM-001" in summ["sources"]

    # Memory: a follow-up that only makes sense with the previous turn
    follow = a.process_message("Who is the claimant on it?")
    assert follow["success"]
    assert "John Doe" in follow["response"]
    # The structured response describes THIS turn, not the first question of the session
    if follow["intent"]["intent_type"] == "qa":
        assert "claimant" in follow["structured_response"]["question"].lower()
        assert "CLM-001" in follow["structured_response"]["sources"]
