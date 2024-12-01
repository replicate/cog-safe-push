import argparse
import asyncio
import re
import sys
from asyncio import Queue
from pathlib import Path
from typing import Any

import pydantic
import yaml
from replicate.exceptions import ReplicateError

from . import cog, lint, log, schema
from .config import (
    DEFAULT_PREDICT_TIMEOUT,
    Config,
    FuzzConfig,
    PredictConfig,
    TrainConfig,
)
from .config import TestCase as ConfigTestCase
from .exceptions import ArgumentError, CogSafePushError
from .task_context import TaskContext, make_task_context
from .tasks import (
    AIOutput,
    CheckOutputsMatch,
    ExactStringOutput,
    ExactURLOutput,
    ExpectedOutput,
    FuzzModel,
    MakeFuzzInputs,
    RunTestCase,
    Task,
)

DEFAULT_CONFIG_PATH = Path("cog-safe-push.yaml")


def main():
    try:
        config, no_push = parse_args_and_config()
        run_config(config, no_push)
    except CogSafePushError as e:
        print("💥 " + str(e), file=sys.stderr)
        sys.exit(1)


def parse_args_and_config() -> tuple[Config, bool]:
    parser = argparse.ArgumentParser(description="Safely push a Cog model, with tests")
    parser.add_argument(
        "--config",
        help="Path to the YAML config file. If --config is not passed, ./cog-safe-push.yaml will be used, if it exists. Any arguments you pass in will override fields on the predict configuration stanza.",
        type=Path,
    )
    parser.add_argument(
        "--help-config",
        help="Print a default cog-safe-push.yaml config to stdout.",
        action="store_true",
    )
    parser.add_argument(
        "--test-model",
        help="Replicate model to test on, in the format <username>/<model-name>. If omitted, <model>-test will be used. The test model is created automatically if it doesn't exist already",
        default=argparse.SUPPRESS,
        type=str,
    )
    parser.add_argument(
        "--no-push",
        help="Only test the model, don't push it to <model>",
        action="store_true",
    )
    parser.add_argument(
        "--test-hardware",
        help="Hardware to run the test model on. Only used when creating the test model, if it doesn't already exist.",
        default=argparse.SUPPRESS,
        type=str,
    )
    parser.add_argument(
        "--no-compare-outputs",
        help="Don't make predictions to compare that prediction outputs match the current version",
        dest="compare_outputs",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--predict-timeout",
        help=f"Timeout (in seconds) for predictions. Default: {DEFAULT_PREDICT_TIMEOUT}",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test-case",
        help="Inputs and expected output that will be used for testing, you can provide multiple --test-case options for multiple test cases. The first test case will be used when comparing outputs to the current version. Each --test-case is semicolon-separated key-value pairs in the format '<key1>=<value1>;<key2=value2>[<output-checker>]'. <output-checker> can either be '==<exact-string-or-url>' or '~=<ai-prompt>'. If you use '==<exact-string-or-url>' then the output of the model must match exactly the string or url you specify. If you use '~=<ai-prompt>' then the AI will verify your output based on <ai-prompt>. If you omit <output-checker>, it will just verify that the prediction doesn't throw an error.",
        action="append",
        dest="test_cases",
        type=parse_test_case,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fuzz-fixed-inputs",
        help="Inputs that should have fixed values during fuzzing. All other non-disabled input values will be generated by AI. If no test cases are specified, these will also be used when comparing outputs to the current version. Semicolon-separated key-value pairs in the format '<key1>=<value1>;<key2=value2>' (etc.)",
        type=parse_fuzz_fixed_inputs,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fuzz-disabled-inputs",
        help="Don't pass values for these inputs during fuzzing. Semicolon-separated keys in the format '<key1>;<key2>' (etc.). If no test cases are specified, these will also be disabled when comparing outputs to the current version. ",
        type=parse_fuzz_disabled_inputs,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fuzz-iterations",
        help="Maximum number of iterations to run fuzzing.",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (max 3)",
    )
    parser.add_argument(
        "model", help="Model in the format <owner>/<model-name>", nargs="?"
    )
    args = parser.parse_args()

    if args.verbose > 3:
        raise ArgumentError("You can use a maximum of 3 -v")
    log.set_verbosity(args.verbose)

    if args.help_config:
        print_help_config()
        sys.exit(0)

    config_path = None
    config = None
    if args.config:
        config_path = args.config
    elif DEFAULT_CONFIG_PATH.exists():
        config_path = DEFAULT_CONFIG_PATH

    if config_path is not None:
        with config_path.open() as f:
            try:
                config_dict = yaml.safe_load(f)
                config = Config.model_validate(config_dict)
            except (pydantic.ValidationError, yaml.YAMLError) as e:
                raise ArgumentError(str(e))

    else:
        if not args.model:
            raise ArgumentError("Model was not specified")
        config = Config(model=args.model, predict=PredictConfig(fuzz=FuzzConfig()))

    config.override("model", args, "model")
    config.override("test_model", args, "test_model")
    config.override("test_hardware", args, "test_hardware")
    config.predict_override("test_cases", args, "test_cases")
    config.predict_override("compare_outputs", args, "compare_outputs")
    config.predict_override("predict_timeout", args, "predict_timeout")
    config.predict_fuzz_override("fixed_inputs", args, "fuzz_fixed_inputs")
    config.predict_fuzz_override("disabled_inputs", args, "fuzz_disabled_inputs")
    config.predict_fuzz_override("iterations", args, "fuzz_iterations")

    if not config.test_model:
        config.test_model = config.model + "-test"

    return config, args.no_push


def run_config(config: Config, no_push: bool):
    assert config.test_model

    model_owner, model_name = parse_model(config.model)
    test_model_owner, test_model_name = parse_model(config.test_model)

    # small optimization
    task_context = None

    if config.train:
        # Don't push twice in case train and predict are both defined
        has_predict = config.predict is not None
        train_no_push = no_push or has_predict

        if not config.train.destination:
            config.train.destination = config.test_model + "-dest"
        destination_owner, destination_name = parse_model(config.train.destination)
        if config.train.fuzz:
            fuzz = config.train.fuzz
        else:
            fuzz = FuzzConfig(
                fixed_inputs={}, disabled_inputs=[], duration=0, iterations=0
            )
        task_context = make_task_context(
            model_owner=model_owner,
            model_name=model_name,
            test_model_owner=test_model_owner,
            test_model_name=test_model_name,
            test_hardware=config.test_hardware,
            train=True,
            train_destination_owner=destination_owner,
            train_destination_name=destination_name,
            dockerfile=config.dockerfile,
        )

        cog_safe_push(
            task_context=task_context,
            no_push=train_no_push,
            train=True,
            do_compare_outputs=False,
            predict_timeout=config.train.train_timeout,
            test_cases=parse_config_test_cases(config.train.test_cases),
            fuzz_fixed_inputs=fuzz.fixed_inputs,
            fuzz_disabled_inputs=fuzz.disabled_inputs,
            fuzz_iterations=fuzz.iterations,
            parallel=config.parallel,
        )

    if config.predict:
        if config.predict.fuzz:
            fuzz = config.predict.fuzz
        else:
            fuzz = FuzzConfig(
                fixed_inputs={}, disabled_inputs=[], duration=0, iterations=0
            )
        if task_context is None:  # has not been created in the training block above
            task_context = make_task_context(
                model_owner=model_owner,
                model_name=model_name,
                test_model_owner=test_model_owner,
                test_model_name=test_model_name,
                test_hardware=config.test_hardware,
                dockerfile=config.dockerfile,
            )

        cog_safe_push(
            task_context=task_context,
            no_push=no_push,
            train=False,
            do_compare_outputs=config.predict.compare_outputs,
            predict_timeout=config.predict.predict_timeout,
            test_cases=parse_config_test_cases(config.predict.test_cases),
            fuzz_fixed_inputs=fuzz.fixed_inputs,
            fuzz_disabled_inputs=fuzz.disabled_inputs,
            fuzz_iterations=fuzz.iterations,
            parallel=config.parallel,
        )


def cog_safe_push(
    task_context: TaskContext,
    no_push: bool = False,
    train: bool = False,
    do_compare_outputs: bool = True,
    predict_timeout: int = 300,
    test_cases: list[tuple[dict[str, Any], ExpectedOutput]] = [],
    fuzz_fixed_inputs: dict = {},
    fuzz_disabled_inputs: list = [],
    fuzz_iterations: int = 10,
    parallel=4,
):
    if no_push:
        log.info(
            f"Running in test-only mode, no model will be pushed to {task_context.model.owner}/{task_context.model.name}"
        )

    if train:
        lint.lint_train()
    else:
        lint.lint_predict()

    if set(fuzz_fixed_inputs.keys()) & set(fuzz_disabled_inputs):
        raise ArgumentError(
            "--fuzz-fixed-inputs keys must not be present in --fuzz-disabled-inputs"
        )

    log.info("Linting test model schema")
    schema.lint(task_context.test_model, train=train)

    model_has_versions = False
    try:
        model_has_versions = bool(task_context.model.versions.list())
    except ReplicateError as e:
        if e.status == 404:
            # Assume it's an official model
            model_has_versions = bool(task_context.model.latest_version)
        else:
            raise

    tasks = []

    if model_has_versions:
        log.info("Checking schema backwards compatibility")
        test_model_schemas = schema.get_schemas(task_context.test_model, train=train)
        model_schemas = schema.get_schemas(task_context.model, train=train)
        schema.check_backwards_compatible(
            test_model_schemas, model_schemas, train=train
        )
        if do_compare_outputs:
            tasks.append(
                CheckOutputsMatch(
                    context=task_context,
                    timeout_seconds=predict_timeout,
                    first_test_case_inputs=test_cases[0][0] if test_cases else None,
                    fuzz_fixed_inputs=fuzz_fixed_inputs,
                    fuzz_disabled_inputs=fuzz_disabled_inputs,
                )
            )

    if test_cases:
        for inputs, output in test_cases:
            tasks.append(
                RunTestCase(
                    context=task_context,
                    inputs=inputs,
                    output=output,
                    predict_timeout=predict_timeout,
                )
            )

    if fuzz_iterations > 0:
        fuzz_inputs_queue = Queue(maxsize=fuzz_iterations)
        tasks.append(
            MakeFuzzInputs(
                context=task_context,
                inputs_queue=fuzz_inputs_queue,
                num_inputs=fuzz_iterations,
                fixed_inputs=fuzz_fixed_inputs,
                disabled_inputs=fuzz_disabled_inputs,
            )
        )
        for _ in range(fuzz_iterations):
            tasks.append(
                FuzzModel(
                    context=task_context,
                    inputs_queue=fuzz_inputs_queue,
                    predict_timeout=predict_timeout,
                )
            )

    asyncio.run(run_tasks(tasks, parallel=parallel))

    log.info("Tests were successful ✨")

    if not no_push:
        log.info("Pushing model...")
        cog.push(task_context.model, task_context.dockerfile)


async def run_tasks(tasks: list[Task], parallel: int) -> None:
    log.info(f"Running tasks with parallelism {parallel}")

    semaphore = asyncio.Semaphore(parallel)
    errors: list[Exception] = []

    async def run_with_semaphore(task: Task) -> None:
        async with semaphore:
            try:
                print(f"starting task {type(task)}")
                await task.run()
                print(f"finished task {type(task)}")
            except Exception as e:
                errors.append(e)

    # Create task coroutines and run them concurrently
    task_coroutines = [run_with_semaphore(task) for task in tasks]

    # Use gather to run tasks concurrently
    await asyncio.gather(*task_coroutines, return_exceptions=True)

    if errors:
        # If there are multiple errors, we'll raise the first one
        # but log all of them
        for error in errors[1:]:
            log.error(f"Additional error occurred: {error}")
        raise errors[0]


def parse_inputs(inputs_list: list[str]) -> dict[str, Any]:
    inputs = {}
    for input_str in inputs_list:
        try:
            key, value_str = input_str.strip().split("=", 1)
            value = parse_input_value(value_str.strip())
            inputs[key] = value
        except ValueError:
            raise ArgumentError(f"Invalid input format: {input_str}")

    return inputs


def parse_input_value(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    # string
    return value


def parse_model(model_owner_name: str) -> tuple[str, str]:
    pattern = r"^([a-z0-9_-]+)/([a-z0-9-.]+)$"
    match = re.match(pattern, model_owner_name)
    if not match:
        raise ArgumentError(f"Invalid model URL format: {model_owner_name}")
    owner, name = match.groups()
    return owner, name


def parse_fuzz_fixed_inputs(
    fuzz_fixed_inputs_str: str,
) -> dict[str, Any]:
    if not fuzz_fixed_inputs_str:
        return {}
    return parse_inputs(
        [
            f"{k}={v}"
            for k, v in (pair.split("=") for pair in fuzz_fixed_inputs_str.split(";"))
        ]
    )


def parse_fuzz_disabled_inputs(fuzz_disabled_inputs_str: str) -> list[str]:
    return fuzz_disabled_inputs_str.split(";") if fuzz_disabled_inputs_str else []


def parse_test_case(test_case_str: str) -> ConfigTestCase:
    if "==" in test_case_str or "~=" in test_case_str:
        inputs_str, op, output_str = re.split("(==|~=)", test_case_str, 1)
    else:
        inputs_str = test_case_str
        op = output_str = None
    test_case = ConfigTestCase(
        inputs=parse_inputs([pair for pair in inputs_str.split(";") if pair])
    )

    if op is not None and output_str is not None:
        if op == "==":
            if output_str.startswith("http://") or output_str.startswith("https://"):
                test_case.match_url = output_str
            else:
                test_case.exact_string = output_str
        else:
            test_case.match_prompt = output_str

    return test_case


def parse_config_test_case(
    config_test_case: ConfigTestCase,
) -> tuple[dict[str, Any], ExpectedOutput]:
    output = None
    if config_test_case.exact_string:
        output = ExactStringOutput(string=config_test_case.exact_string)
    elif config_test_case.match_url:
        output = ExactURLOutput(url=config_test_case.match_url)
    elif config_test_case.match_prompt:
        output = AIOutput(prompt=config_test_case.match_prompt)

    return (config_test_case.inputs, output)


def parse_config_test_cases(
    config_test_cases: list[ConfigTestCase],
) -> list[tuple[dict[str, Any], ExpectedOutput]]:
    return [parse_config_test_case(tc) for tc in config_test_cases]


def print_help_config():
    print(
        yaml.dump(
            Config(
                model="<model>",
                test_model="<test model, or empty to append '-test' to model>",
                test_hardware="<hardware, e.g. cpu>",
                predict=PredictConfig(
                    fuzz=FuzzConfig(),
                    test_cases=[
                        ConfigTestCase(
                            inputs={"<input1>": "<value1>"},
                            exact_string="<exact string match>",
                        ),
                        ConfigTestCase(
                            inputs={"<input2>": "<value2>"},
                            match_url="<match output image against url>",
                        ),
                        ConfigTestCase(
                            inputs={"<input3>": "<value3>"},
                            match_prompt="<match output using AI prompt, e.g. 'an image of a cat'>",
                        ),
                    ],
                ),
                train=TrainConfig(
                    destination="<generated prediction model, e.g. andreasjansson/test-predict. leave blank to append '-dest' to the test model>",
                    destination_hardware="<hardware for the created prediction model, e.g. cpu>",
                    fuzz=FuzzConfig(),
                    test_cases=[
                        ConfigTestCase(
                            inputs={"<input1>": "<value1>"},
                            exact_string="<exact string match>",
                        ),
                        ConfigTestCase(
                            inputs={"<input2>": "<value2>"},
                            match_url="<match output image against url>",
                        ),
                        ConfigTestCase(
                            inputs={"<input3>": "<value3>"},
                            match_prompt="<match output using AI prompt, e.g. 'an image of a cat'>",
                        ),
                    ],
                ),
            ).model_dump(exclude_none=True),
            default_flow_style=False,
        )
    )
    print("# values between < and > should be edited")


if __name__ == "__main__":
    main()
