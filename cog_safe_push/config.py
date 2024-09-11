import argparse

from pydantic import BaseModel, model_validator

from .exceptions import ArgumentError

DEFAULT_PREDICT_TIMEOUT = 300
DEFAULT_FUZZ_DURATION = 300

InputScalar = bool | int | float | str | list[int] | list[str] | list[float]


class TestCase(BaseModel):
    inputs: dict[str, InputScalar]
    exact_string: str | None = None
    match_url: str | None = None
    match_prompt: str | None = None

    @model_validator(mode="after")
    def check_mutually_exclusive(self):
        set_fields = sum(
            getattr(self, field) is not None
            for field in ["exact_string", "match_url", "match_prompt"]
        )
        if set_fields > 1:
            raise ArgumentError(
                "At most one of 'exact_string', 'match_url', or 'match_prompt' must be set"
            )
        return self


class FuzzConfig(BaseModel):
    fixed_inputs: dict[str, InputScalar] = {}
    disabled_inputs: list[str] = []
    duration: int = DEFAULT_FUZZ_DURATION
    iterations: int | None = None


class PredictConfig(BaseModel):
    compare_outputs: bool = True
    predict_timeout: int = DEFAULT_PREDICT_TIMEOUT
    test_cases: list[TestCase] = []
    fuzz: FuzzConfig | None = None


class TrainConfig(BaseModel):
    destination: str | None = None
    destination_hardware: str = "cpu"
    train_timeout: int = DEFAULT_PREDICT_TIMEOUT
    test_cases: list[TestCase] = []
    fuzz: FuzzConfig | None = None


class Config(BaseModel):
    model: str
    test_model: str | None = None
    test_hardware: str = "cpu"
    predict: PredictConfig | None = None
    train: TrainConfig | None = None

    def override(self, field: str, args: argparse.Namespace, arg: str):
        if hasattr(args, arg) and getattr(args, arg) is not None:
            setattr(self, field, getattr(args, arg))

    def predict_override(self, field: str, args: argparse.Namespace, arg: str):
        if not hasattr(args, arg):
            return
        if not self.predict:
            raise ArgumentError(
                f"--config is used but is missing a predict section and you are overriding predict {field} in the command line arguments."
            )
        setattr(self.predict, field, getattr(args, arg))

    def predict_fuzz_override(self, field: str, args: argparse.Namespace, arg: str):
        if not hasattr(args, arg):
            return
        if not self.predict:
            raise ArgumentError(
                f"--config is used but is missing a predict section and you are overriding fuzz {field} in the command line arguments."
            )
        if not self.predict.fuzz:
            raise ArgumentError(
                f"--config is used but is missing a predict.fuzz section and you are overriding fuzz {field} in the command line arguments."
            )
        setattr(self.predict.fuzz, field, getattr(args, arg))