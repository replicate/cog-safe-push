from collections import defaultdict
import re
import argparse
import replicate
from replicate.exceptions import ReplicateException

from . import cog, lint, schema, predict, log


def main():
    parser = argparse.ArgumentParser(description="Safely push a Cog model, with tests")
    parser.add_argument(
        "--test-model",
        help="Replicate model to test on, in the format <username>/<model-name>. If omitted, <model>-test will be used. The test model is created automatically if it doesn't exist already",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--test-only",
        help="Only test the model, don't push it to <model>",
        action="store_true",
    )
    parser.add_argument(
        "--test-hardware",
        help="Hardware to run the test model on. Only used when creating the test model, if it doesn't already exist.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--train", action="store_true", help="Run test on trainer instead of predictor"
    )
    parser.add_argument(
        "--train-destination",
        type=str,
        help="Replicate model for the output of training, in the format <username>/<model-name>. Only used if --train is set. If omitted, <test-model>-test will be used. The model is created automatically if it doesn't exist.",
    )
    parser.add_argument(
        "--train-destination-hardware",
        help="Hardware to run the train destination model on. Only used if --train is set and if the train destination model doesn't already exist.",
        default="cpu",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input key-value pairs in the format <key>=<value>. These will be used when comparing outputs, as well as during fuzzing. The special value '(omit)' will unset the key. You can specify the same key multiple times, and during fuzzing a random value will be picked. If multiple values are provided, the first value will be used when comparing outputs. You can give weight to values by appending '^<weight>%' to the value. This is particularly useful in combination with '(omit)' during fuzzing, e.g. '-i extra_lora=(omit)^50%' will leave the extra_lora field blank half the time.",
        action="append",
        dest="inputs",
        default=[],
    )
    parser.add_argument(
        "-x",
        "--disable-input",
        help="Don't pass values to these inputs when comparing outputs or fuzzing",
        action="append",
        dest="disabled_inputs",
        default=[],
    )
    parser.add_argument(
        "--no-compare-outputs",
        help="Don't make predictions to compare that prediction outputs match",
        action="store_true",
    )
    parser.add_argument(
        "--predict-timeout",
        help="Timeout (in seconds) for predictions when comparing outputs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--fuzz-seconds",
        help="Number of seconds to run fuzzing. Set to 0 for no fuzzing",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--fuzz-iterations",
        help="Maximum number of iterations to run fuzzing. Leave blank to run for the full --fuzz-seconds",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (max 3)",
    )

    parser.add_argument("model", help="Model in the format <owner>/<model-name>")
    args = parser.parse_args()

    if args.verbose > 3:
        raise ValueError("You can use a maximum of 3 -v")
    log.set_verbosity(args.verbose)

    model_owner, model_name = parse_model(args.model)
    if args.test_model:
        test_model_owner, test_model_name = parse_model(args.test_model)
    else:
        test_model_owner = model_owner
        test_model_name = model_name + "-test"

    if args.train_destination and not args.train:
        raise ValueError("--train-destination should only be set if --train is set")
    if args.train and args.train_destination:
        train_destination_owner, train_destination_name = parse_model(
            args.train_destination
        )
    else:
        train_destination_owner = test_model_owner
        train_destination_name = test_model_name + "-dest"

    inputs = parse_inputs(args.inputs)

    cog_safe_push(
        model_owner=model_owner,
        model_name=model_name,
        test_model_owner=test_model_owner,
        test_model_name=test_model_name,
        test_only=args.test_only,
        test_hardware=args.test_hardware,
        train=args.train,
        train_destination_owner=train_destination_owner,
        train_destination_name=train_destination_name,
        inputs=inputs,
        disabled_inputs=args.disabled_inputs,
        do_compare_outputs=not args.no_compare_outputs,
        predict_timeout=args.predict_timeout,
        fuzz_seconds=args.fuzz_seconds,
        fuzz_iterations=args.fuzz_iterations,
    )


def cog_safe_push(
    model_owner: str,
    model_name: str,
    test_model_owner: str,
    test_model_name: str,
    test_hardware: str,
    test_only: bool = False,
    train: bool = False,
    train_destination_owner: str | None = None,
    train_destination_name: str | None = None,
    train_destination_hardware: str = "cpu",
    inputs: dict = {},
    disabled_inputs: list = [],
    do_compare_outputs: bool = True,
    predict_timeout: int = 300,
    fuzz_seconds: int = 30,
    fuzz_iterations: int | None = None,
):
    if model_owner == test_model_owner and model_name == test_model_name:
        raise ValueError("Can't use the same model as test model")

    if test_only:
        log.info(
            f"Running in test-only mode, no model will be pushed to {model_owner}/{model_name}"
        )

    if train:
        lint.lint_train()
    else:
        lint.lint_predict()

    if set(inputs.keys()) & set(disabled_inputs):
        raise ValueError("--input keys must not be present in --disabled-inputs")

    model = get_model(model_owner, model_name)
    if not model:
        raise ValueError(
            f"You need to create the model {model_owner}/{model_name} before running this script"
        )

    test_model = get_or_create_model(test_model_owner, test_model_name, test_hardware)

    if train:
        train_destination = get_or_create_model(
            train_destination_owner, train_destination_name, train_destination_hardware
        )
    else:
        train_destination = None

    log.info("Pushing test model")
    pushed_version_id = cog.push(test_model)
    test_model.reload()
    assert (
        test_model.versions.list()[0].id == pushed_version_id
    ), f"Pushed version ID {pushed_version_id} doesn't match latest version on {test_model_owner}/{test_model_name}: {test_model.versions.list()[0].id}"

    log.info("Linting test model schema")
    schema.lint(test_model, train=train)

    if model.latest_version:
        log.info("Checking schema backwards compatibility")
        test_model_schemas = schema.get_schemas(test_model, train=train)
        model_schemas = schema.get_schemas(model, train=train)
        schema.check_backwards_compatible(
            test_model_schemas, model_schemas, train=train
        )
        if do_compare_outputs:
            log.info(
                "Checking that outputs match between existing version and test version"
            )
            predict.check_outputs_match(
                test_model=test_model,
                model=model,
                train=train,
                train_destination=train_destination,
                timeout_seconds=predict_timeout,
                inputs=inputs,
                disabled_inputs=disabled_inputs,
            )

    if fuzz_seconds > 0:
        log.info("Fuzzing test model")
        predict.fuzz_model(
            model=test_model,
            train=train,
            train_destination=train_destination,
            timeout_seconds=fuzz_seconds,
            max_iterations=fuzz_iterations,
            inputs=inputs,
            disabled_inputs=disabled_inputs,
        )

    log.info("Tests were successful ✨")

    if not test_only:
        log.info("Pushing model...")
        cog.push(model)


def parse_inputs(inputs_list: list[str]) -> dict[str, list[predict.WeightedInputValue]]:
    input_values = defaultdict(list)
    input_weights = defaultdict(list)
    for input_str in inputs_list:
        try:
            key, weighted_value_str = input_str.strip().split("=", 1)
            value_str, weight_percent = parse_input_weight_percent(weighted_value_str)
            value = parse_input_value(value_str.strip())
            input_values[key.strip()].append(value)
            input_weights[key.strip()].append(weight_percent)
        except ValueError:
            raise ValueError(f"Invalid input format: {input_str}")

    inputs = make_weighted_inputs(input_values, input_weights)
    return inputs


def make_weighted_inputs(
    input_values: dict[str, list[str]], input_weights: dict[str, list[float]]
) -> dict[str, list[predict.WeightedInputValue]]:
    weighted_inputs = {}
    for key, values in input_values.items():
        weights = input_weights[key]
        weight_sum = sum(w for w in weights if w is not None)
        remaining_weight = 100 - weight_sum
        num_unweighted = len([w for w in weights if w is None])
        if num_unweighted > 0:
            unweighted_weight = remaining_weight / num_unweighted
            for i, weight in enumerate(weights):
                if weight is None:
                    weights[i] = unweighted_weight
        weighted_inputs[key] = [
            predict.WeightedInputValue(value=value, weight_percent=weight)
            for value, weight in zip(values, weights)
        ]
    return weighted_inputs


def parse_input_value(value: str) -> predict.InputValueType:
    if value == "(omit)":
        return predict.OMITTED_INPUT

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


def parse_input_weight_percent(value_str: str) -> tuple[str, float | None]:
    parts = value_str.rsplit("^")
    if len(parts) == 2 and value_str.endswith("%"):
        percent_str = parts[1][:-1]
        try:
            percent = float(percent_str)
        except ValueError:
            raise ValueError(f"Failed to parse input value weight {percent_str}")
        if percent <= 0:
            raise ValueError(
                f"Invalid value weight {percent_str}, must be greater than 0"
            )
        if percent > 100:
            raise ValueError(
                f"Invalid value weight {percent_str}, must be less or equal to 100"
            )
        return parts[0], percent
    return value_str, None


def get_or_create_model(model_owner, model_name, hardware) -> replicate.model.Model:
    model = get_model(model_owner, model_name)

    if not model:
        if not hardware:
            raise ValueError(
                f"Model {model_owner}/{model_name} doesn't exist, and you didn't specify hardware"
            )

        log.info(f"Creating model {model_owner}/{model_name} with hardware {hardware}")
        model = replicate.models.create(
            owner=model_owner,
            name=model_name,
            visibility="private",
            hardware=hardware,
        )
    return model


def get_model(owner, name) -> replicate.model.Model:
    try:
        model = replicate.models.get(f"{owner}/{name}")
    except ReplicateException as e:
        if e.status == 404:
            return None
        raise
    return model


def parse_model(model_owner_name: str) -> tuple[str, str]:
    pattern = r"^([a-z0-9_-]+)/([a-z0-9-]+)$"
    match = re.match(pattern, model_owner_name)
    if not match:
        raise ValueError(f"Invalid model URL format: {model_owner_name}")
    owner, name = match.groups()
    return owner, name


if __name__ == "__main__":
    main()
