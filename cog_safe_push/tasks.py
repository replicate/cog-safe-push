import asyncio
from asyncio import Queue
from dataclasses import dataclass
from typing import Any, Protocol

from . import log, schema
from .exceptions import (
    AIError,
    FuzzError,
    OutputsDontMatchError,
    PredictionFailedError,
    PredictionTimeoutError,
    TestCaseFailedError,
)
from .match_outputs import is_url, output_matches_prompt, outputs_match, urls_match
from .predict import make_predict_inputs, predict, truncate
from .task_context import TaskContext


@dataclass
class ExactStringOutput:
    string: str


@dataclass
class ExactURLOutput:
    url: str


@dataclass
class AIOutput:
    prompt: str


ExpectedOutput = ExactStringOutput | ExactURLOutput | AIOutput | None


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
        test_output = await predict(
            model=self.context.test_model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=inputs,
            timeout_seconds=self.timeout_seconds,
        )
        output = await predict(
            model=self.context.model,
            train=self.context.is_train(),
            train_destination=self.context.train_destination,
            inputs=inputs,
            timeout_seconds=self.timeout_seconds,
        )
        matches, error = await outputs_match(test_output, output, is_deterministic)
        if not matches:
            raise OutputsDontMatchError(
                f"Outputs don't match:\n\ntest output:\n{test_output}\n\nmodel output:\n{output}\n\n{error}"
            )


@dataclass
class RunTestCase(Task):
    context: TaskContext
    inputs: dict[str, Any]
    output: ExactStringOutput | ExactURLOutput | AIOutput | None
    predict_timeout: int

    async def run(self) -> None:
        log.v(f"Running test case with inputs: {self.inputs}")
        try:
            output = await predict(
                model=self.context.test_model,
                train=self.context.is_train(),
                train_destination=self.context.train_destination,
                inputs=self.inputs,
                timeout_seconds=self.predict_timeout,
            )
        except PredictionFailedError as e:
            raise TestCaseFailedError(f"Test case failed: {str(e)}")

        if self.output is None:
            return

        if isinstance(self.output, ExactStringOutput):
            if output != self.output.string:
                raise TestCaseFailedError(
                    f"Test case failed: Expected '{self.output.string}', got '{truncate(output, 200)}'"
                )
        elif isinstance(self.output, ExactURLOutput):
            output_url = None
            if isinstance(output, str) and is_url(output):
                output_url = output
            if (
                isinstance(output, list)
                and len(output) == 1
                and isinstance(output[0], str)
                and is_url(output[0])
            ):
                output_url = output[0]
            if output_url is not None:
                matches, error = await urls_match(
                    self.output.url, output_url, is_deterministic=True
                )
                if not matches:
                    raise TestCaseFailedError(
                        f"Test case failed: file at URL {self.output.url} does not match file at URL {output_url}. {error}"
                    )
                log.info(
                    f"File at URL {self.output.url} matched file at URL {output_url}"
                )
            else:
                raise TestCaseFailedError(
                    f"Test case failed: Expected URL, got '{truncate(output, 200)}'"
                )
        elif isinstance(self.output, AIOutput):
            try:
                matches, error = await output_matches_prompt(output, self.output.prompt)
                if not matches:
                    raise TestCaseFailedError(f"Test case failed: {error}")
            except AIError as e:
                raise TestCaseFailedError(f"Test case failed: AI error: {str(e)}")


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
            output = await predict(
                model=self.context.test_model,
                train=self.context.is_train(),
                train_destination=self.context.train_destination,
                inputs=inputs,
                timeout_seconds=self.predict_timeout,
            )
        except PredictionTimeoutError:
            raise FuzzError("Prediction timed out")
        except PredictionFailedError as e:
            raise FuzzError(f"Prediction failed: {e}")
        if not output:
            raise FuzzError("No output")
