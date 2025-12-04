from unittest.mock import Mock

import pytest

from cog_safe_push.schema import (
    IncompatibleSchemaError,
    SchemaLintError,
    check_backwards_compatible,
    lint,
)


def make_input_schema(properties: dict) -> dict:
    """Helper to create a properly structured Input schema."""
    return {"type": "object", "title": "Input", "properties": properties}


def test_identical_schemas():
    old = new = {
        "Input": make_input_schema(
            {"text": {"type": "string"}, "number": {"type": "integer"}}
        ),
        "Output": {"type": "string"},
    }
    check_backwards_compatible(new, old, train=False)  # Should not raise


def test_new_optional_input():
    old = {
        "Input": make_input_schema({"text": {"type": "string"}}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema(
            {
                "text": {"type": "string"},
                "optional": {"type": "string", "default": "value"},
            }
        ),
        "Output": {"type": "string"},
    }
    check_backwards_compatible(new, old, train=False)  # Should not raise


def test_removed_input():
    old = {
        "Input": make_input_schema(
            {"text": {"type": "string"}, "number": {"type": "integer"}}
        ),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema({"text": {"type": "string"}}),
        "Output": {"type": "string"},
    }
    with pytest.raises(IncompatibleSchemaError, match="Missing input number"):
        check_backwards_compatible(new, old, train=False)


def test_changed_input_type():
    old = {
        "Input": make_input_schema({"value": {"type": "integer"}}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema({"value": {"type": "string"}}),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError,
        match="Input value has changed type from integer to string",
    ):
        check_backwards_compatible(new, old, train=False)


def test_added_minimum_constraint():
    old = {
        "Input": make_input_schema({"value": {"type": "integer"}}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema({"value": {"type": "integer", "minimum": 0}}),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError, match="Input value has added a minimum constraint"
    ):
        check_backwards_compatible(new, old, train=False)


def test_increased_minimum():
    old = {
        "Input": make_input_schema({"value": {"type": "integer", "minimum": 0}}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema({"value": {"type": "integer", "minimum": 1}}),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError, match="Input value has a higher minimum"
    ):
        check_backwards_compatible(new, old, train=False)


def test_added_maximum_constraint():
    old = {
        "Input": make_input_schema({"value": {"type": "integer"}}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema({"value": {"type": "integer", "maximum": 100}}),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError, match="Input value has added a maximum constraint"
    ):
        check_backwards_compatible(new, old, train=False)


def test_decreased_maximum():
    old = {
        "Input": make_input_schema({"value": {"type": "integer", "maximum": 100}}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema({"value": {"type": "integer", "maximum": 99}}),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError, match="Input value has a lower maximum"
    ):
        check_backwards_compatible(new, old, train=False)


def test_changed_choice_type():
    """Test enum type change with inline allOf."""
    old = {
        "Input": make_input_schema(
            {"choice": {"allOf": [{"type": "string", "enum": ["A", "B", "C"]}]}}
        ),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema(
            {"choice": {"allOf": [{"type": "integer", "enum": [1, 2, 3]}]}}
        ),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError,
        match="Input choice choices has changed type from string to integer",
    ):
        check_backwards_compatible(new, old, train=False)


def test_added_choice():
    """Test adding enum choices is backwards compatible."""
    old = {
        "Input": make_input_schema(
            {"choice": {"allOf": [{"type": "string", "enum": ["A", "B", "C"]}]}}
        ),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema(
            {"choice": {"allOf": [{"type": "string", "enum": ["A", "B", "C", "D"]}]}}
        ),
        "Output": {"type": "string"},
    }
    check_backwards_compatible(new, old, train=False)  # Should not raise


def test_removed_choice():
    """Test removing enum choices breaks compatibility."""
    old = {
        "Input": make_input_schema(
            {"choice": {"allOf": [{"type": "string", "enum": ["A", "B", "C"]}]}}
        ),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema(
            {"choice": {"allOf": [{"type": "string", "enum": ["A", "B"]}]}}
        ),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError, match="Input choice is missing choices: 'C'"
    ):
        check_backwards_compatible(new, old, train=False)


def test_new_required_input():
    old = {
        "Input": make_input_schema({"text": {"type": "string"}}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema(
            {"text": {"type": "string"}, "new_required": {"type": "string"}}
        ),
        "Output": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError, match="Input new_required is new and is required"
    ):
        check_backwards_compatible(new, old, train=False)


def test_changed_output_type():
    old = {
        "Input": make_input_schema({}),
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema({}),
        "Output": {"type": "integer"},
    }
    with pytest.raises(IncompatibleSchemaError, match="Output has changed type"):
        check_backwards_compatible(new, old, train=False)


def test_multiple_incompatibilities():
    old = {
        "Input": make_input_schema(
            {
                "text": {"type": "string"},
                "number": {"type": "integer", "minimum": 0},
                "choice": {"allOf": [{"$ref": "#/components/schemas/choice"}]},
            }
        ),
        "choice": {"type": "string", "enum": ["A", "B", "C"]},
        "Output": {"type": "string"},
    }
    new = {
        "Input": make_input_schema(
            {
                "text": {"type": "integer"},
                "number": {"type": "integer", "minimum": 1},
                "choice": {"allOf": [{"$ref": "#/components/schemas/choice"}]},
                "new_required": {"type": "string"},
            }
        ),
        "choice": {"type": "string", "enum": ["A", "B"]},
        "Output": {"type": "integer"},
    }
    with pytest.raises(IncompatibleSchemaError) as exc_info:
        check_backwards_compatible(new, old, train=False)
    error_message = str(exc_info.value)
    assert "Input text has changed type from string to integer" in error_message
    assert "Input number has a higher minimum" in error_message
    assert "Input choice is missing choices: 'C'" in error_message
    assert "Input new_required is new and is required" in error_message
    assert "Output has changed type" in error_message


def test_training_input_schema():
    """Test that train=True uses TrainingInput instead of Input."""
    old = {
        "TrainingInput": make_input_schema({"data": {"type": "string"}}),
        "TrainingOutput": {"type": "string"},
    }
    new = {
        "TrainingInput": make_input_schema({"data": {"type": "integer"}}),
        "TrainingOutput": {"type": "string"},
    }
    with pytest.raises(
        IncompatibleSchemaError,
        match="TrainingInput data has changed type from string to integer",
    ):
        check_backwards_compatible(new, old, train=True)


def test_realistic_schema_structure():
    """Test with a realistic schema structure from get_schemas().

    This guards against the bug where check_backwards_compatible expected
    Input to be the properties dict directly, but get_schemas returns:
    {"Input": {"type": "object", "properties": {...}}}
    """
    old = {
        "Input": {
            "type": "object",
            "title": "Input",
            "properties": {
                "seed": {
                    "type": "integer",
                    "title": "Seed",
                    "x-order": 8,
                    "nullable": True,
                    "description": "Random seed. Set for reproducible generation",
                },
                "image": {
                    "type": "string",
                    "title": "Image",
                    "format": "uri",
                    "x-order": 0,
                    "nullable": True,
                    "description": "Image file",
                },
                "prompt": {
                    "type": "string",
                    "title": "Prompt",
                    "x-order": 6,
                    "nullable": True,
                    "description": "Text prompt for image generation",
                },
            },
        },
        "Output": {"type": "string"},
    }
    new = {
        "Input": {
            "type": "object",
            "title": "Input",
            "properties": {
                "seed": {
                    "type": "integer",
                    "title": "Seed",
                    "x-order": 8,
                    "nullable": True,
                    "description": "Random seed. Set for reproducible generation",
                },
                "image": {
                    "type": "string",
                    "title": "Image",
                    "format": "uri",
                    "x-order": 0,
                    "nullable": True,
                    "description": "Image file",
                },
                "prompt": {
                    "type": "string",
                    "title": "Prompt",
                    "x-order": 6,
                    "nullable": True,
                    "description": "Text prompt for image generation",
                },
            },
        },
        "Output": {"type": "string"},
    }
    check_backwards_compatible(new, old, train=False)  # Should not raise


def test_realistic_schema_structure_with_incompatibility():
    """Test realistic schema detects incompatibilities."""
    old = {
        "Input": {
            "type": "object",
            "title": "Input",
            "properties": {
                "steps": {
                    "type": "integer",
                    "title": "Steps",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 25,
                    "x-order": 0,
                    "description": "Number of steps",
                },
            },
        },
        "Output": {"type": "string"},
    }
    new = {
        "Input": {
            "type": "object",
            "title": "Input",
            "properties": {
                "steps": {
                    "type": "integer",
                    "title": "Steps",
                    "minimum": 10,  # Higher minimum - breaking change
                    "maximum": 100,
                    "default": 25,
                    "x-order": 0,
                    "description": "Number of steps",
                },
            },
        },
        "Output": {"type": "string"},
    }
    with pytest.raises(IncompatibleSchemaError, match="steps has a higher minimum"):
        check_backwards_compatible(new, old, train=False)


def test_lint_deprecated_input_without_description():
    mock_model = Mock()
    mock_model.versions.list.return_value = [
        Mock(
            openapi_schema={
                "components": {
                    "schemas": {
                        "Input": {
                            "properties": {
                                "steps": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 50,
                                    "default": 25,
                                    "deprecated": True,
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt to use",
                                },
                            }
                        }
                    }
                }
            }
        )
    ]
    lint(mock_model, train=False)


def test_lint_non_deprecated_input_without_description():
    mock_model = Mock()
    mock_model.versions.list.return_value = [
        Mock(
            openapi_schema={
                "components": {
                    "schemas": {
                        "Input": {
                            "properties": {
                                "steps": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 50,
                                    "default": 25,
                                },
                            }
                        }
                    }
                }
            }
        )
    ]
    with pytest.raises(SchemaLintError, match="steps: Missing description"):
        lint(mock_model, train=False)
