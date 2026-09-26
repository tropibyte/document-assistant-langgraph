"""Task 1.1 / 1.2: the Pydantic schemas define the right fields, types, defaults and constraints."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from schemas import AnswerResponse, UserIntent, SummarizationResponse, DocumentChunk, UpdateMemoryResponse


class TestAnswerResponse:
    def test_has_required_fields_with_types(self):
        fields = AnswerResponse.model_fields
        assert set(fields) == {"question", "answer", "sources", "confidence", "timestamp"}
        assert fields["question"].annotation is str
        assert fields["answer"].annotation is str
        assert fields["confidence"].annotation is float
        assert fields["timestamp"].annotation is datetime

    def test_defaults(self):
        r = AnswerResponse(question="What is the total?", answer="$22,000")
        assert r.sources == []
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.timestamp, datetime)
        # default_factory, not a shared mutable default
        assert AnswerResponse(question="a", answer="b").sources is not r.sources

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_in_range_accepted(self, confidence):
        assert AnswerResponse(question="q", answer="a", confidence=confidence).confidence == confidence

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 5, -1])
    def test_confidence_out_of_range_rejected(self, confidence):
        with pytest.raises(ValidationError):
            AnswerResponse(question="q", answer="a", confidence=confidence)

    def test_type_enforcement(self):
        with pytest.raises(ValidationError):
            AnswerResponse(question="q", answer="a", confidence="very sure")
        with pytest.raises(ValidationError):
            AnswerResponse(question="q", answer="a", sources="INV-001")  # must be a list
        with pytest.raises(ValidationError):
            AnswerResponse(answer="a")  # question is required

    def test_sources_are_normalised(self):
        r = AnswerResponse(question="q", answer="a", sources=[" inv-001", "INV-001", "con-001"])
        assert r.sources == ["INV-001", "CON-001"]

    def test_json_schema_carries_bounds_for_the_llm(self):
        schema = AnswerResponse.model_json_schema()["properties"]["confidence"]
        assert schema["minimum"] == 0.0 and schema["maximum"] == 1.0


class TestUserIntent:
    def test_has_required_fields(self):
        assert set(UserIntent.model_fields) == {"intent_type", "confidence", "reasoning"}

    @pytest.mark.parametrize("intent", ["qa", "summarization", "calculation", "unknown"])
    def test_valid_intents(self, intent):
        assert UserIntent(intent_type=intent, confidence=0.8, reasoning="r").intent_type == intent

    @pytest.mark.parametrize("intent", ["search", "chat", "", "calc"])
    def test_invalid_intents_rejected(self, intent):
        with pytest.raises(ValidationError):
            UserIntent(intent_type=intent, confidence=0.8, reasoning="r")

    def test_intent_is_case_insensitive(self):
        assert UserIntent(intent_type=" QA ", confidence=0.8, reasoning="r").intent_type == "qa"

    @pytest.mark.parametrize("confidence", [-0.1, 1.5])
    def test_confidence_bounds(self, confidence):
        with pytest.raises(ValidationError):
            UserIntent(intent_type="qa", confidence=confidence, reasoning="r")

    def test_json_schema_restricts_intent_to_enum(self):
        prop = UserIntent.model_json_schema()["properties"]["intent_type"]
        assert set(prop["enum"]) == {"qa", "summarization", "calculation", "unknown"}


def test_starter_default_factory_bug_is_fixed():
    # The starter's `default_factory=lambda: list` returned the list *type*.
    assert DocumentChunk(doc_id="X", content="c").metadata == {}
    assert SummarizationResponse(original_length=1, summary="s", key_points=[]).document_ids == []
    assert UpdateMemoryResponse(summary="s").document_ids == []
