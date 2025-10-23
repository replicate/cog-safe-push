import asyncio
from asyncio import Queue
from dataclasses import dataclass
from typing import Any, Protocol

from . import log, schema
from .exceptions import (
    FuzzError,
    OutputsDontMatchError,
    PredictionTimeoutError,
)
from .match_outputs import outputs_match
from .output_checkers import OutputChecker
from .predict import make_fuzz_inputs, predict
from .task_context import TaskContext


class Task(Protocol):
    async def run(self) -> None: ...


@dataclass
class CheckOutputsMatch(Task):
    context: TaskContext
    timeout_seconds: int
    first_test_case_inputs: dict[str, Any] | None
    fuzz_fixed_inputs: dict[str, Any]
    fuzz_disabled_inputs: list[str]
    fuzz_prompt: str | None
    prediction_index: int | None = None
    prediction_url: str | None = None

    async def run(self) -> None:
        if self.first_test_case_inputs is not None:
            inputs = self.first_test_case_inputs

            # TODO(andreas): This is weird, it means that if the first
            # input doesn't have a seed, the output comparison is
            # non-deterministic
            is_deterministic = "seed" in inputs
        else:
            schemas = schema.get_schemas(
                self.context.model, train=self.context.is_train()
            )
            inputs, is_deterministic = await make_fuzz_inputs(
                schemas,
                train=self.context.is_train(),
                only_required=True,
                seed=1,
                fixed_inputs=self.fuzz_fixed_inputs,
                disabled_inputs=self.fuzz_disabled_inputs,
                fuzz_prompt=self.fuzz_prompt,
            )

        prefix = (
            f"[{self.prediction_index}] " if self.prediction_index is not None else ""
        )
        log.v(
            f"{prefix}Checking outputs match between existing version and test version, with inputs: {inputs}"
        )
        test_output, test_error, test_url = await predict(
            model=self.context.test_model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=inputs,
            timeout_seconds=self.timeout_seconds,
            prediction_index=self.prediction_index,
        )
        self.prediction_url = test_url
        output, error, _ = await predict(
            model=self.context.model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=inputs,
            timeout_seconds=self.timeout_seconds,
            prediction_index=self.prediction_index,
        )

        if test_error is not None:
            raise OutputsDontMatchError(
                f"{prefix}Existing version raised an error: {test_error}"
            )
        if error is not None:
            raise OutputsDontMatchError(f"{prefix}New version raised an error: {error}")

        matches, match_error = await outputs_match(
            test_output, output, is_deterministic
        )
        if not matches:
            raise OutputsDontMatchError(
                f"{prefix}Outputs don't match:\n\ntest output:\n{test_output}\n\nmodel output:\n{output}\n\n{match_error}"
            )


@dataclass
class RunTestCase(Task):
    context: TaskContext
    inputs: dict[str, Any]
    checker: OutputChecker
    predict_timeout: int
    prediction_index: int | None = None

    async def run(self) -> None:
        prefix = (
            f"[{self.prediction_index}] " if self.prediction_index is not None else ""
        )
        log.v(f"{prefix}Running test case with inputs: {self.inputs}")
        output, error = await predict(
            model=self.context.test_model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=self.inputs,
            timeout_seconds=self.predict_timeout,
            prediction_index=self.prediction_index,
        )

        await self.checker(output, error)


@dataclass
class MakeFuzzInputs(Task):
    context: TaskContext
    num_inputs: int
    inputs_queue: Queue[dict[str, Any]]
    fixed_inputs: dict
    disabled_inputs: list[str]
    fuzz_prompt: str | None

    async def run(self) -> None:
        schemas = schema.get_schemas(
            self.context.test_model, train=self.context.is_train()
        )
        inputs_history = []
        for _ in range(self.num_inputs):
            inputs, _ = await make_fuzz_inputs(
                schemas,
                train=self.context.is_train(),
                only_required=False,
                seed=None,
                fixed_inputs=self.fixed_inputs,
                disabled_inputs=self.disabled_inputs,
                fuzz_prompt=self.fuzz_prompt,
                inputs_history=inputs_history,
            )
            await self.inputs_queue.put(inputs)
            inputs_history.append(inputs)


@dataclass
class FuzzModel(Task):
    context: TaskContext
    inputs_queue: Queue[dict[str, Any]]
    predict_timeout: int
    prediction_index: int | None = None

    async def run(self) -> None:
        inputs = await asyncio.wait_for(self.inputs_queue.get(), timeout=60)

        prefix = (
            f"[{self.prediction_index}] " if self.prediction_index is not None else ""
        )
        log.v(f"{prefix}Fuzzing with inputs: {inputs}")
        try:
            output, error = await predict(
                model=self.context.test_model,
                train=self.context.is_train(),
                train_destination=self.context.train_destination,
                inputs=inputs,
                timeout_seconds=self.predict_timeout,
                prediction_index=self.prediction_index,
            )
        except PredictionTimeoutError:
            raise FuzzError(f"{prefix}Prediction timed out")
        if error is not None:
            raise FuzzError(f"{prefix}Prediction raised an error: {error}")
        if not output:
            raise FuzzError(f"{prefix}No output")

        if error is not None:
            raise FuzzError(f"{prefix}Prediction failed: {error}")
