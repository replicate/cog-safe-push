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
from .predict import make_predict_inputs, predict
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
            inputs, is_deterministic = await make_predict_inputs(
                schemas,
                train=self.context.is_train(),
                only_required=True,
                seed=1,
                fixed_inputs=self.fuzz_fixed_inputs,
                disabled_inputs=self.fuzz_disabled_inputs,
            )

        log.v(
            f"Checking outputs match between existing version and test version, with inputs: {inputs}"
        )
        test_output, test_error = await predict(
            model=self.context.test_model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=inputs,
            timeout_seconds=self.timeout_seconds,
        )
        output, error = await predict(
            model=self.context.model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=inputs,
            timeout_seconds=self.timeout_seconds,
        )

        if test_error is not None:
            raise OutputsDontMatchError(
                f"Existing version raised an error: {test_error}"
            )
        if error is not None:
            raise OutputsDontMatchError(f"New version raised an error: {error}")

        matches, match_error = await outputs_match(
            test_output, output, is_deterministic
        )
        if not matches:
            raise OutputsDontMatchError(
                f"Outputs don't match:\n\ntest output:\n{test_output}\n\nmodel output:\n{output}\n\n{match_error}"
            )


@dataclass
class RunTestCase(Task):
    context: TaskContext
    inputs: dict[str, Any]
    checker: OutputChecker
    predict_timeout: int

    async def run(self) -> None:
        log.v(f"Running test case with inputs: {self.inputs}")
        output, error = await predict(
            model=self.context.test_model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=self.inputs,
            timeout_seconds=self.predict_timeout,
        )

        await self.checker(output, error)


@dataclass
class MakeFuzzInputs(Task):
    context: TaskContext
    num_inputs: int
    inputs_queue: Queue[dict[str, Any]]
    fixed_inputs: dict
    disabled_inputs: list[str]

    async def run(self) -> None:
        schemas = schema.get_schemas(
            self.context.test_model, train=self.context.is_train()
        )
        inputs_history = []
        for _ in range(self.num_inputs):
            inputs, _ = await make_predict_inputs(
                schemas,
                train=self.context.is_train(),
                only_required=False,
                seed=None,
                fixed_inputs=self.fixed_inputs,
                disabled_inputs=self.disabled_inputs,
                inputs_history=inputs_history,
            )
            await self.inputs_queue.put(inputs)
            inputs_history.append(inputs)


@dataclass
class FuzzModel(Task):
    context: TaskContext
    inputs_queue: Queue[dict[str, Any]]
    predict_timeout: int

    async def run(self) -> None:
        inputs = await asyncio.wait_for(self.inputs_queue.get(), timeout=60)

        log.v(f"Fuzzing with inputs: {inputs}")
        try:
            output, error = await predict(
                model=self.context.test_model,
                train=self.context.is_train(),
                train_destination=self.context.train_destination,
                inputs=inputs,
                timeout_seconds=self.predict_timeout,
            )
        except PredictionTimeoutError:
            raise FuzzError("Prediction timed out")
        if error is not None:
            raise FuzzError(f"Prediction raised an error: {error}")
        if not output:
            raise FuzzError("No output")

        if error is not None:
            raise FuzzError(f"Prediction failed: {error}")
