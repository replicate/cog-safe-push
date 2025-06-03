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
async def test_make_predict_inputs_basic(mock_json_object, sample_schemas):
    mock_json_object.return_value = {"text": "hello", "number": 42, "choice": "A"}

    inputs, is_deterministic = await make_predict_inputs(
        sample_schemas,
        train=False,
        only_required=True,
        seed=None,
        fixed_inputs={},
        disabled_inputs=[],
        fuzz_prompt=None,
    )

    assert inputs == {"text": "hello", "number": 42, "choice": "A"}
    assert not is_deterministic


async def test_make_predict_inputs_with_seed(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {"text": "hello", "number": 42, "choice": "A"}

        inputs, is_deterministic = await make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=123,
            fixed_inputs={},
            disabled_inputs=[],
            fuzz_prompt=None,
        )

        assert inputs == {"text": "hello", "number": 42, "choice": "A", "seed": 123}
        assert is_deterministic


async def test_make_predict_inputs_with_fixed_inputs(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {"text": "hello", "number": 42, "choice": "A"}

        inputs, _ = await make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=None,
            fixed_inputs={"text": "fixed"},
            disabled_inputs=[],
            fuzz_prompt=None,
        )

        assert inputs["text"] == "fixed"


async def test_make_predict_inputs_with_disabled_inputs(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {
            "text": "hello",
            "number": 42,
            "choice": "A",
            "optional": True,
        }

        inputs, _ = await make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=False,
            seed=None,
            fixed_inputs={},
            disabled_inputs=["optional"],
            fuzz_prompt=None,
        )

        assert "optional" not in inputs


async def test_make_predict_inputs_with_inputs_history(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {"text": "new", "number": 100, "choice": "C"}

        inputs_history = [
            {"text": "old", "number": 42, "choice": "A"},
            {"text": "older", "number": 21, "choice": "B"},
        ]

        inputs, _ = await make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=None,
            fixed_inputs={},
            disabled_inputs=[],
            fuzz_prompt=None,
            inputs_history=inputs_history,
        )

        assert inputs != inputs_history[0]
        assert inputs != inputs_history[1]


async def test_make_predict_inputs_ai_error(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.side_effect = [
            {"text": "hello"},  # Missing required fields
            {"text": "hello", "number": 42, "choice": "A"},  # Correct input
        ]

        inputs, _ = await make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=True,
            seed=None,
            fixed_inputs={},
            disabled_inputs=[],
            fuzz_prompt=None,
        )

        assert inputs == {"text": "hello", "number": 42, "choice": "A"}
        assert mock_json_object.call_count == 2


async def test_make_predict_inputs_max_attempts_reached(sample_schemas):
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {
            "text": "hello"
        }  # Always missing required fields

        with pytest.raises(AIError):
            await make_predict_inputs(
                sample_schemas,
                train=False,
                only_required=True,
                seed=None,
                fixed_inputs={},
                disabled_inputs=[],
                fuzz_prompt=None,
            )


async def test_make_predict_inputs_filters_null_values(sample_schemas):
    """Test that null values are filtered out from AI-generated inputs."""
    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {
            "text": "hello",
            "number": 42,
            "choice": "A",
            "optional": None,  # This should be filtered out
            "input_image": None,  # This should be filtered out
        }

        inputs, _ = await make_predict_inputs(
            sample_schemas,
            train=False,
            only_required=False,
            seed=None,
            fixed_inputs={},
            disabled_inputs=[],
            fuzz_prompt=None,
        )

        # Null values should be filtered out
        assert "optional" not in inputs
        assert "input_image" not in inputs
        assert inputs == {"text": "hello", "number": 42, "choice": "A"}


async def test_make_predict_inputs_filters_various_null_representations():
    """Test that various representations of null are filtered out."""
    schemas = {
        "Input": {
            "properties": {
                "text": {"type": "string", "description": "A text input"},
                "image": {
                    "type": "string",
                    "format": "uri",
                    "description": "An image input",
                },
                "number": {"type": "integer", "description": "A number input"},
            },
            "required": ["text"],
        }
    }

    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {
            "text": "hello",
            "image": None,  # Null value that should be filtered
            "number": None,  # Another null value that should be filtered
            "optional_field": None,  # Optional field with null that should be filtered
        }

        inputs, _ = await make_predict_inputs(
            schemas,
            train=False,
            only_required=False,
            seed=None,
            fixed_inputs={},
            disabled_inputs=[],
            fuzz_prompt=None,
        )

        # Only non-null values should remain
        assert inputs == {"text": "hello"}
        assert "image" not in inputs
        assert "number" not in inputs
        assert "optional_field" not in inputs


async def test_make_predict_inputs_preserves_valid_values():
    """Test that valid values (including falsy ones) are preserved while null is filtered."""
    schemas = {
        "Input": {
            "properties": {
                "text": {"type": "string", "description": "A text input"},
                "flag": {"type": "boolean", "description": "A boolean input"},
                "count": {"type": "integer", "description": "A number input"},
                "empty_string": {
                    "type": "string",
                    "description": "An empty string input",
                },
            },
            "required": ["text"],
        }
    }

    with patch("cog_safe_push.predict.ai.json_object") as mock_json_object:
        mock_json_object.return_value = {
            "text": "hello",
            "flag": False,  # Should be preserved (falsy but not None)
            "count": 0,  # Should be preserved (falsy but not None)
            "empty_string": "",  # Should be preserved (falsy but not None)
            "null_field": None,  # Should be filtered out
        }

        inputs, _ = await make_predict_inputs(
            schemas,
            train=False,
            only_required=False,
            seed=None,
            fixed_inputs={},
            disabled_inputs=[],
            fuzz_prompt=None,
        )

        # Falsy values should be preserved, only None should be filtered
        expected = {
            "text": "hello",
            "flag": False,
            "count": 0,
            "empty_string": "",
        }
        assert inputs == expected
        assert "null_field" not in inputs
