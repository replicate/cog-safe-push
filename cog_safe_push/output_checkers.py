import json
from dataclasses import dataclass
from typing import Any, Protocol

from . import log
from .exceptions import (
    AIError,
    TestCaseFailedError,
)
from .match_outputs import is_url, output_matches_prompt, urls_match
from .utils import truncate


class OutputChecker(Protocol):
    async def __call__(self, output: Any | None, error: str | None) -> None: ...


@dataclass
class NoChecker(OutputChecker):
    async def __call__(self, _: Any | None, error: str | None) -> None:
        check_no_error(error)


@dataclass
class ExactStringChecker(OutputChecker):
    string: str

    async def __call__(self, output: Any | None, error: str | None) -> None:
        check_no_error(error)

        if not isinstance(output, str):
            raise TestCaseFailedError(f"Expected string, got {truncate(output, 200)}")

        if output != self.string:
            raise TestCaseFailedError(
                f"Expected '{self.string}', got '{truncate(output, 200)}'"
            )


@dataclass
class MatchURLChecker(OutputChecker):
    url: str

    async def __call__(self, output: Any | None, error: str | None) -> None:
        check_no_error(error)

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
                self.url, output_url, is_deterministic=True
            )
            if not matches:
                raise TestCaseFailedError(
                    f"File at URL {self.url} does not match file at URL {output_url}. {error}"
                )
            log.info(f"File at URL {self.url} matched file at URL {output_url}")
        else:
            raise TestCaseFailedError(f"Expected URL, got '{truncate(output, 200)}'")


@dataclass
class AIChecker(OutputChecker):
    prompt: str

    async def __call__(self, output: Any | None, error: str | None) -> None:
        check_no_error(error)

        try:
            matches, error = await output_matches_prompt(output, self.prompt)
            if not matches:
                raise TestCaseFailedError(error)
        except AIError as e:
            raise TestCaseFailedError(f"AI error: {str(e)}")


@dataclass
class ErrorContainsChecker(OutputChecker):
    string: str

    async def __call__(self, _: Any | None, error: str | None) -> None:
        if error is None:
            raise TestCaseFailedError("Expected error, prediction succeeded")

        if self.string not in error:
            raise TestCaseFailedError(
                f"Expected error to contain {self.string}, got {error}"
            )


def check_no_error(error: str | None) -> None:
    if error is not None:
        raise TestCaseFailedError(f"Prediction raised unexpected error: {error}")
