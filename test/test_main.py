import pytest
from cog_safe_push.main import (
    parse_model,
    parse_inputs,
    parse_input_value,
    parse_input_weight_percent,
    make_weighted_inputs,
)
from cog_safe_push.predict import WeightedInputValue, OMITTED_INPUT


def test_parse_model():
    assert parse_model("user/model-name") == ("user", "model-name")
    assert parse_model("user-123/model-name-456") == ("user-123", "model-name-456")

    with pytest.raises(ValueError):
        parse_model("invalid_format")

    with pytest.raises(ValueError):
        parse_model("user/model/extra")


def test_parse_input_value():
    assert parse_input_value("true")
    assert not parse_input_value("false")
    assert parse_input_value("123") == 123
    assert parse_input_value("3.14") == 3.14
    assert parse_input_value("hello") == "hello"


def test_parse_input_weight_percent():
    assert parse_input_weight_percent("value") == ("value", None)
    assert parse_input_weight_percent("value^50%") == ("value", 50.0)
    assert parse_input_weight_percent("value^75.5%") == ("value", 75.5)

    with pytest.raises(ValueError):
        parse_input_weight_percent("value^invalid%")

    with pytest.raises(ValueError):
        parse_input_weight_percent("value^0%")

    with pytest.raises(ValueError):
        parse_input_weight_percent("value^101%")


def test_parse_inputs():
    inputs = [
        "key1=value1",
        "key2=true",
        "key3=123",
        "key4=3.14",
        "key5=value5^50%",
        "key6=NULL^75%",
        "key7=value7a",
        "key7=value7b^25%",
    ]

    result = parse_inputs(inputs)

    assert set(result.keys()) == {
        "key1",
        "key2",
        "key3",
        "key4",
        "key5",
        "key6",
        "key7",
    }
    assert result["key1"] == [WeightedInputValue(value="value1", weight_percent=100.0)]
    assert result["key2"] == [WeightedInputValue(value=True, weight_percent=100.0)]
    assert result["key3"] == [WeightedInputValue(value=123, weight_percent=100.0)]
    assert result["key4"] == [WeightedInputValue(value=3.14, weight_percent=100.0)]
    assert result["key5"] == [WeightedInputValue(value="value5", weight_percent=50.0)]
    assert result["key6"] == [WeightedInputValue(value="NULL", weight_percent=75.0)]
    assert result["key7"] == [
        WeightedInputValue(value="value7a", weight_percent=75.0),
        WeightedInputValue(value="value7b", weight_percent=25.0),
    ]

    with pytest.raises(ValueError):
        parse_inputs(["invalid_format"])


def test_make_weighted_inputs():
    input_values = {
        "key1": ["value1"],
        "key2": ["value2a", "value2b"],
        "key3": ["value3a", "value3b", "value3c"],
    }
    input_weights = {
        "key1": [None],
        "key2": [50, None],
        "key3": [25, 25, None],
    }

    result = make_weighted_inputs(input_values, input_weights)

    assert set(result.keys()) == {"key1", "key2", "key3"}
    assert result["key1"] == [WeightedInputValue(value="value1", weight_percent=100.0)]
    assert result["key2"] == [
        WeightedInputValue(value="value2a", weight_percent=50.0),
        WeightedInputValue(value="value2b", weight_percent=50.0),
    ]
    assert result["key3"] == [
        WeightedInputValue(value="value3a", weight_percent=25.0),
        WeightedInputValue(value="value3b", weight_percent=25.0),
        WeightedInputValue(value="value3c", weight_percent=50.0),
    ]


def test_parse_input_value_omit():
    assert parse_input_value("(omit)") == OMITTED_INPUT


def test_parse_inputs_with_omitted():
    inputs = [
        "key1=value1",
        "key2=(omit)",
        "key3=(omit)^50%",
        "key4=value4",
        "key4=(omit)^25%",
    ]

    result = parse_inputs(inputs)

    assert set(result.keys()) == {"key1", "key2", "key3", "key4"}
    assert result["key1"] == [WeightedInputValue(value="value1", weight_percent=100.0)]
    assert result["key2"] == [
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=100.0)
    ]
    assert result["key3"] == [
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=50.0)
    ]
    assert result["key4"] == [
        WeightedInputValue(value="value4", weight_percent=75.0),
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=25.0),
    ]


def test_make_weighted_inputs_with_omitted():
    input_values = {
        "key1": ["value1"],
        "key2": [OMITTED_INPUT],
        "key3": ["value3a", OMITTED_INPUT],
    }
    input_weights = {
        "key1": [None],
        "key2": [None],
        "key3": [50, None],
    }

    result = make_weighted_inputs(input_values, input_weights)

    assert set(result.keys()) == {"key1", "key2", "key3"}
    assert result["key1"] == [WeightedInputValue(value="value1", weight_percent=100.0)]
    assert result["key2"] == [
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=100.0)
    ]
    assert result["key3"] == [
        WeightedInputValue(value="value3a", weight_percent=50.0),
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=50.0),
    ]


def test_parse_inputs_mixed_omitted_and_values():
    inputs = [
        "key1=value1",
        "key1=(omit)",
        "key2=(omit)^75%",
        "key2=value2^25%",
        "key3=(omit)",
        "key3=value3a",
        "key3=value3b^30%",
    ]

    result = parse_inputs(inputs)

    assert set(result.keys()) == {"key1", "key2", "key3"}
    assert result["key1"] == [
        WeightedInputValue(value="value1", weight_percent=50.0),
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=50.0),
    ]
    assert result["key2"] == [
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=75.0),
        WeightedInputValue(value="value2", weight_percent=25.0),
    ]
    assert result["key3"] == [
        WeightedInputValue(value=OMITTED_INPUT, weight_percent=35.0),
        WeightedInputValue(value="value3a", weight_percent=35.0),
        WeightedInputValue(value="value3b", weight_percent=30.0),
    ]
