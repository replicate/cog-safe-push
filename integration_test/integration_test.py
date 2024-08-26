import json
from pathlib import Path
import os
from contextlib import contextmanager
import pytest
import datetime
import replicate
from replicate.exceptions import ReplicateException

from cog_safe_push.main import cog_safe_push
from cog_safe_push.exceptions import *
from cog_safe_push import log, predict


log.set_verbosity(2)


def test_cog_safe_push():
    model_owner = "replicate-internal"
    model_name = "test-cog-safe-push-" + datetime.datetime.now().strftime(
        "%y%m%d-%H%M%S"
    )
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("base"):
            cog_safe_push(model_owner, model_name, model_owner, test_model_name, "cpu")

        with fixture_dir("same-schema"):
            cog_safe_push(model_owner, model_name, model_owner, test_model_name, "cpu")

        with fixture_dir("schema-lint-error"):
            with pytest.raises(SchemaLintError):
                cog_safe_push(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                )

        with fixture_dir("incompatible-schema"):
            with pytest.raises(IncompatibleSchema):
                cog_safe_push(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                )

        with fixture_dir("outputs-dont-match"):
            with pytest.raises(OutputsDontMatch):
                cog_safe_push(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                )

        with fixture_dir("additive-schema-fuzz-error"):
            with pytest.raises(FuzzError):
                cog_safe_push(
                    model_owner,
                    model_name,
                    model_owner,
                    test_model_name,
                    "cpu",
                    fuzz_seconds=120,
                )

        with fixture_dir("additive-schema-fuzz-error"):
            cog_safe_push(
                model_owner,
                model_name,
                model_owner,
                test_model_name,
                "cpu",
                fuzz_seconds=120,
                inputs={
                    "qux": [
                        predict.WeightedInputValue(2, 30),
                        predict.WeightedInputValue(3, 30),
                        predict.WeightedInputValue(predict.OMITTED_INPUT, 40),
                    ],
                },
            )

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_images():
    model_owner = "replicate-internal"
    model_name = "test-cog-safe-push-" + datetime.datetime.now().strftime(
        "%y%m%d-%H%M%S"
    )
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("image-base"):
            cog_safe_push(
                model_owner,
                model_name,
                model_owner,
                test_model_name,
                "cpu",
                fuzz_seconds=60,
            )

        with fixture_dir("image-base"):
            cog_safe_push(model_owner, model_name, model_owner, test_model_name, "cpu")

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_images_with_seed():
    model_owner = "replicate-internal"
    model_name = "test-cog-safe-push-" + datetime.datetime.now().strftime(
        "%y%m%d-%H%M%S"
    )
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("image-base-seed"):
            cog_safe_push(model_owner, model_name, model_owner, test_model_name, "cpu")

        with fixture_dir("image-base-seed"):
            cog_safe_push(model_owner, model_name, model_owner, test_model_name, "cpu")

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_train():
    model_owner = "replicate-internal"
    model_name = "test-cog-safe-push-" + datetime.datetime.now().strftime(
        "%y%m%d-%H%M%S"
    )
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("train"):
            cog_safe_push(
                model_owner,
                model_name,
                model_owner,
                test_model_name,
                "cpu",
                train=True,
                train_destination_owner=model_owner,
                train_destination_name=test_model_name + "-dest",
                fuzz_iterations=1,
            )

        with fixture_dir("train"):
            cog_safe_push(
                model_owner,
                model_name,
                model_owner,
                test_model_name,
                "cpu",
                train=True,
                train_destination_owner=model_owner,
                train_destination_name=test_model_name + "-dest",
                fuzz_iterations=1,
            )

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)
        delete_model(model_owner, test_model_name + "-dest")


def create_model(model_owner, model_name):
    replicate.models.create(
        owner=model_owner,
        name=model_name,
        visibility="private",
        hardware="cpu",
    )


def delete_model(model_owner, model_name):
    try:
        model = replicate.models.get(model_owner, model_name)
    except ReplicateException:
        # model likely doesn't exist
        return

    for version in model.versions.list():
        try:
            print(f"Deleting version {version.id}")
            model.versions.delete(version.id)
        except json.JSONDecodeError:
            pass  # bug in replicate-python causes delete to throw JSONDecodeError

    try:
        replicate.models.delete(model_owner, model_name)
    except json.JSONDecodeError:
        pass  # bug in replicate-python causes delete to throw JSONDecodeError


@contextmanager
def fixture_dir(fixture_name):
    current_file_path = Path(__file__).resolve()
    fixture_dir = current_file_path.parent / "fixtures" / fixture_name
    current_dir = Path.cwd()
    try:
        os.chdir(fixture_dir)
        yield
    finally:
        os.chdir(current_dir)
