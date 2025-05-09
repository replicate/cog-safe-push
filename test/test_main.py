import pytest

from cog_safe_push import log
from cog_safe_push.exceptions import ArgumentError
from cog_safe_push.main import (
    parse_args_and_config,
    parse_input_value,
    parse_inputs,
    parse_model,
)


def test_parse_args_minimal(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "user/model"])
    config, no_push = parse_args_and_config()
    assert config.model == "user/model"
    assert config.test_model == "user/model-test"
    assert not no_push


def test_parse_args_with_test_model(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "user/model", "--test-model", "user/test-model"]
    )
    config, no_push = parse_args_and_config()
    assert config.model == "user/model"
    assert config.test_model == "user/test-model"
    assert not no_push


def test_parse_args_no_push(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "user/model", "--no-push"])
    config, no_push = parse_args_and_config()
    assert config.model == "user/model"
    assert no_push


def test_parse_args_verbose(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "user/model", "-vv"])
    parse_args_and_config()
    assert log.level == log.VERBOSE2


def test_parse_args_too_verbose(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "user/model", "-vvvv"])
    with pytest.raises(ArgumentError, match="You can use a maximum of 3 -v"):
        parse_args_and_config()


def test_parse_args_predict_timeout(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "user/model", "--predict-timeout", "600"]
    )
    config, _ = parse_args_and_config()
    assert config.predict is not None
    assert config.predict.predict_timeout == 600


def test_parse_args_fuzz_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cog-safe-push",
            "user/model",
            "--fuzz-fixed-inputs",
            "key1=value1;key2=42",
            "--fuzz-disabled-inputs",
            "key3;key4",
            "--fuzz-iterations",
            "5",
        ],
    )
    config, _ = parse_args_and_config()
    assert config.predict is not None
    assert config.predict.fuzz is not None
    assert config.predict.fuzz.fixed_inputs == {"key1": "value1", "key2": 42}
    assert config.predict.fuzz.disabled_inputs == ["key3", "key4"]
    assert config.predict.fuzz.iterations == 5


def test_parse_args_test_case(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cog-safe-push",
            "user/model",
            "--test-case",
            "input1=value1;input2=42==expected output",
        ],
    )
    config, _ = parse_args_and_config()
    assert config.predict is not None
    assert len(config.predict.test_cases) == 1
    assert config.predict.test_cases[0].inputs == {"input1": "value1", "input2": 42}
    assert config.predict.test_cases[0].exact_string == "expected output"


def test_parse_args_multiple_test_cases(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cog-safe-push",
            "user/model",
            "--test-case",
            "input1=value1==output1",
            "--test-case",
            "input2=value2~=AI prompt",
        ],
    )
    config, _ = parse_args_and_config()
    assert config.predict is not None
    assert len(config.predict.test_cases) == 2
    assert config.predict.test_cases[0].inputs == {"input1": "value1"}
    assert config.predict.test_cases[0].exact_string == "output1"
    assert config.predict.test_cases[1].inputs == {"input2": "value2"}
    assert config.predict.test_cases[1].match_prompt == "AI prompt"


def test_parse_args_no_model(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cog-safe-push"])
    with pytest.raises(ArgumentError, match="Model was not specified"):
        parse_args_and_config()


def test_parse_config_file(tmp_path, monkeypatch):
    config_yaml = """
    model: user/model
    test_model: user/test-model
    test_hardware: gpu
    ignore_schema_compatibility: true
    predict:
      compare_outputs: false
      predict_timeout: 500
      test_cases:
        - inputs:
            input1: value1
          exact_string: expected output
      fuzz:
        fixed_inputs:
          key1: value1
        disabled_inputs:
          - key2
        iterations: 15
    """
    config_file = tmp_path / "cog-safe-push.yaml"
    config_file.write_text(config_yaml)
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "--config", str(config_file)])

    config, _ = parse_args_and_config()

    assert config.model == "user/model"
    assert config.test_model == "user/test-model"
    assert config.test_hardware == "gpu"
    assert config.ignore_schema_compatibility is True
    assert config.predict is not None
    assert config.predict.fuzz is not None
    assert not config.predict.compare_outputs
    assert config.predict.predict_timeout == 500
    assert len(config.predict.test_cases) == 1
    assert config.predict.test_cases[0].inputs == {"input1": "value1"}
    assert config.predict.test_cases[0].exact_string == "expected output"
    assert config.predict.fuzz.fixed_inputs == {"key1": "value1"}
    assert config.predict.fuzz.disabled_inputs == ["key2"]
    assert config.predict.fuzz.iterations == 15


def test_config_override_with_args(tmp_path, monkeypatch):
    config_yaml = """
    model: user/model
    test_model: user/test-model
    predict:
      predict_timeout: 500
    """
    config_file = tmp_path / "cog-safe-push.yaml"
    config_file.write_text(config_yaml)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cog-safe-push",
            "--config",
            str(config_file),
            "--test-model",
            "user/override-test-model",
            "--predict-timeout",
            "600",
        ],
    )

    config, _ = parse_args_and_config()

    assert config.model == "user/model"
    assert config.test_model == "user/override-test-model"
    assert config.predict is not None
    assert config.predict.predict_timeout == 600


def test_config_file_not_found(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "--config", "non_existent.yaml", "user/model"]
    )
    with pytest.raises(FileNotFoundError):
        parse_args_and_config()


def test_invalid_config_file(tmp_path, monkeypatch):
    invalid_yaml = "invalid: yaml: content"
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(invalid_yaml)
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "--config", str(config_file)])

    with pytest.raises(ArgumentError):
        parse_args_and_config()


def test_parse_args_help_config(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "--help-config"])
    with pytest.raises(SystemExit):
        parse_args_and_config()
    captured = capsys.readouterr()
    assert "model:" in captured.out
    assert "test_model:" in captured.out
    assert "predict:" in captured.out
    assert "train:" in captured.out


def test_parse_args_no_compare_outputs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "user/model", "--no-compare-outputs"]
    )
    config, _ = parse_args_and_config()
    assert config.predict is not None
    assert not config.predict.compare_outputs


def test_parse_args_fuzz_iterations(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "user/model", "--fuzz-iterations", "50"]
    )
    config, _ = parse_args_and_config()
    assert config.predict is not None
    assert config.predict.fuzz is not None
    assert config.predict.fuzz.iterations == 50


def test_parse_args_test_hardware(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "user/model", "--test-hardware", "gpu"]
    )
    config, _ = parse_args_and_config()
    assert config.test_hardware == "gpu"


def test_parse_config_with_train(tmp_path, monkeypatch):
    config_yaml = """
    model: user/model
    test_model: user/test-model
    train:
      destination: user/train-dest
      destination_hardware: gpu
      train_timeout: 3600
      test_cases:
        - inputs:
            input1: value1
          match_prompt: An AI generated output
      fuzz:
        iterations: 8
    """
    config_file = tmp_path / "cog-safe-push.yaml"
    config_file.write_text(config_yaml)
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "--config", str(config_file)])

    config, _ = parse_args_and_config()

    assert config.model == "user/model"
    assert config.test_model == "user/test-model"
    assert config.train is not None
    assert config.train.fuzz is not None
    assert config.train.destination == "user/train-dest"
    assert config.train.destination_hardware == "gpu"
    assert config.train.train_timeout == 3600
    assert len(config.train.test_cases) == 1
    assert config.train.test_cases[0].inputs == {"input1": "value1"}
    assert config.train.test_cases[0].match_prompt == "An AI generated output"
    assert config.train.fuzz.iterations == 8


def test_parse_args_with_default_config(tmp_path, monkeypatch):
    config_yaml = """
    model: user/default-model
    test_model: user/default-test-model
    """
    default_config_file = tmp_path / "cog-safe-push.yaml"
    default_config_file.write_text(config_yaml)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["cog-safe-push"])

    config, _ = parse_args_and_config()

    assert config.model == "user/default-model"
    assert config.test_model == "user/default-test-model"


def test_parse_args_override_default_config(tmp_path, monkeypatch):
    config_yaml = """
    model: user/default-model
    test_model: user/default-test-model
    """
    default_config_file = tmp_path / "cog-safe-push.yaml"
    default_config_file.write_text(config_yaml)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "user/override-model"])

    config, _ = parse_args_and_config()

    assert config.model == "user/override-model"
    assert config.test_model == "user/default-test-model"


def test_parse_args_invalid_test_case(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "user/model", "--test-case", "invalid_format"]
    )
    with pytest.raises(ArgumentError):
        parse_args_and_config()


def test_parse_args_invalid_fuzz_fixed_inputs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["cog-safe-push", "user/model", "--fuzz-fixed-inputs", "invalid_format"],
    )
    with pytest.raises(SystemExit):
        parse_args_and_config()


def test_parse_config_invalid_test_case(tmp_path, monkeypatch):
    config_yaml = """
    model: user/model
    predict:
      test_cases:
        - inputs:
            input1: value1
          exact_string: output1
          match_prompt: This should not be here
    """
    config_file = tmp_path / "cog-safe-push.yaml"
    config_file.write_text(config_yaml)
    monkeypatch.setattr("sys.argv", ["cog-safe-push", "--config", str(config_file)])

    with pytest.raises(ArgumentError):
        parse_args_and_config()


def test_parse_config_missing_predict_section(tmp_path, monkeypatch):
    config_yaml = """
    model: user/model
    """
    config_file = tmp_path / "cog-safe-push.yaml"
    config_file.write_text(config_yaml)
    monkeypatch.setattr(
        "sys.argv",
        ["cog-safe-push", "--config", str(config_file), "--predict-timeout", "600"],
    )

    with pytest.raises(ArgumentError, match="missing a predict section"):
        parse_args_and_config()


def test_parse_config_missing_fuzz_section(tmp_path, monkeypatch):
    config_yaml = """
    model: user/model
    predict:
      predict_timeout: 500
    """
    config_file = tmp_path / "cog-safe-push.yaml"
    config_file.write_text(config_yaml)
    monkeypatch.setattr(
        "sys.argv",
        ["cog-safe-push", "--config", str(config_file), "--fuzz-iterations", "20"],
    )

    with pytest.raises(ArgumentError, match="missing a predict.fuzz section"):
        parse_args_and_config()


def test_parse_args_ignore_schema_compatibility(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cog-safe-push", "user/model", "--ignore-schema-compatibility"]
    )
    config, _ = parse_args_and_config()
    assert config.ignore_schema_compatibility is True


def test_parse_model():
    assert parse_model("user/model-name") == ("user", "model-name")
    assert parse_model("user-123/model-name-456") == ("user-123", "model-name-456")

    with pytest.raises(ArgumentError):
        parse_model("invalid_format")

    with pytest.raises(ArgumentError):
        parse_model("user/model/extra")


def test_parse_input_value():
    assert parse_input_value("true")
    assert not parse_input_value("false")
    assert parse_input_value("123") == 123
    assert parse_input_value("3.14") == 3.14
    assert parse_input_value("hello") == "hello"


def test_parse_inputs():
    inputs = [
        "key1=value1",
        "key2=true",
        "key3=123",
        "key4=3.14",
    ]

    result = parse_inputs(inputs)

    assert set(result.keys()) == {
        "key1",
        "key2",
        "key3",
        "key4",
    }
    assert result["key1"] == "value1"
    assert result["key2"]
    assert result["key3"] == 123
    assert result["key4"] == 3.14

    with pytest.raises(ArgumentError):
        parse_inputs(["invalid_format"])
