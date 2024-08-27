from unittest.mock import patch

import pytest

from cog_safe_push.exceptions import AIError
from cog_safe_push.predict import make_predict_inputs


@pytest.fixture
def sample_schemas():
    return {
        "Input": {
            "properties": {
                "text": {"type": "string", "description": "A text input"},
                "number": {"type": "integer", "description": "A number input"},
                "choice": {
                    "allOf": [{"$ref": "#/components/schemas/choice"}],
                    "description": "A choice input",
                },
                "optional": {"type": "boolean", "description": "An optional input"},
                "seed": {"type": "int", "description": "Random seed"},
            },
            "required": ["text", "number", "choice"],
        },
        "choice": {
            "type": "string",
            "enum": ["A", "B", "C"],
            "description": "Available choices",
        },
    }


@patch("cog_safe_push.predict.ai.json_object")
def test_make_predict_inputs_basic(mock_json_object, sample_schemas):
    mock_json_object.return_value = {"text": "hello", "number": 42, "choice": "A"}

    inputs, is_deterministic = make_predict_inputs(
        sample_schemas,
        train=False,
        only_required=True,
        seed=None,
        fixed_inputs={},
        disabled_inputs=[],
    )

    assert inputs == {"text": "hello", "number": 42, "choice": "A"}
    assert not is_deterministic


def test_make_predict_inputs_with_seed(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {"text": "hello", "number": 42, "choice": "A"}

        inputs, is_deterministic = make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=123,
            fixed_inputs={},
            disabled_inputs=[],
        )

        assert inputs == {"text": "hello", "number": 42, "choice": "A", "seed": 123}
        assert is_deterministic


def test_make_predict_inputs_with_fixed_inputs(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {"text": "hello", "number": 42, "choice": "A"}

        inputs, _ = make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=None,
            fixed_inputs={"text": "fixed"},
            disabled_inputs=[],
        )

        assert inputs["text"] == "fixed"


def test_make_predict_inputs_with_disabled_inputs(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {
            "text": "hello",
            "number": 42,
            "choice": "A",
            "optional": True,
        }

        inputs, _ = make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=False,
            seed=None,
            fixed_inputs={},
            disabled_inputs=["optional"],
        )

        assert "optional" not in inputs


def test_make_predict_inputs_with_inputs_history(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {"text": "new", "number": 100, "choice": "C"}

        inputs_history = [
            {"text": "old", "number": 42, "choice": "A"},
            {"text": "older", "number": 21, "choice": "B"},
        ]

        inputs, _ = make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=None,
            fixed_inputs={},
            disabled_inputs=[],
            inputs_history=inputs_history,
        )

        assert inputs != inputs_history[0]
        assert inputs != inputs_history[1]


def test_make_predict_inputs_ai_error(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.side_effect = [
            {"text": "hello"},  # Missing required fields
            {"text": "hello", "number": 42, "choice": "A"},  # Correct input
        ]

        inputs, _ = make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=None,
            fixed_inputs={},
            disabled_inputs=[],
        )

        assert inputs == {"text": "hello", "number": 42, "choice": "A"}
        assert mock_json_object.call_count == 2


def test_make_predict_inputs_max_attempts_reached(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {
            "text": "hello"
        }  # Always missing required fields

        with pytest.raises(AIError):
            make_predict_inputs(
                sample_schemas,
                train=False,
                only_required=True,
                seed=None,
                fixed_inputs={},
                disabled_inputs=[],
            )
