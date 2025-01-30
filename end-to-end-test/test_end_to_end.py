import json
import os
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

import httpx
import pytest
import replicate
from replicate.exceptions import ReplicateException

from cog_safe_push import log
from cog_safe_push.exceptions import *
from cog_safe_push.main import cog_safe_push
from cog_safe_push.output_checkers import (
    AIChecker,
    ErrorContainsChecker,
    ExactStringChecker,
    MatchURLChecker,
    NoChecker,
)
from cog_safe_push.task_context import make_task_context

log.set_verbosity(2)


def test_cog_safe_push():
    model_owner = "replicate-internal"
    model_name = generate_model_name()
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("base"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                ),
                test_cases=[
                    (
                        {"text": "world"},
                        ExactStringChecker("hello world"),
                    ),
                    (
                        {"text": "world"},
                        AIChecker("the text hello world"),
                    ),
                ],
            )

        with fixture_dir("same-schema"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                )
            )

        with fixture_dir("schema-lint-error"):
            with pytest.raises(SchemaLintError):
                cog_safe_push(
                    make_task_context(
                        model_owner, model_name, model_owner, test_model_name, "cpu"
                    )
                )

        with fixture_dir("incompatible-schema"):
            with pytest.raises(IncompatibleSchemaError):
                cog_safe_push(
                    make_task_context(
                        model_owner, model_name, model_owner, test_model_name, "cpu"
                    )
                )

        with fixture_dir("outputs-dont-match"):
            with pytest.raises(OutputsDontMatchError):
                cog_safe_push(
                    make_task_context(
                        model_owner, model_name, model_owner, test_model_name, "cpu"
                    )
                )

        with fixture_dir("additive-schema-fuzz-error"):
            with pytest.raises(FuzzError):
                cog_safe_push(
                    make_task_context(
                        model_owner, model_name, model_owner, test_model_name, "cpu"
                    ),
                )

        with fixture_dir("additive-schema-fuzz-error"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                ),
                test_cases=[
                    (
                        {"text": "world", "qux": 2},
                        ExactStringChecker("hello world"),
                    ),
                    (
                        {"text": "world", "qux": 1},
                        ErrorContainsChecker("qux"),
                    ),
                ],
                fuzz_iterations=0,
            )

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_images():
    model_owner = "replicate-internal"
    model_name = generate_model_name()
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("image-base"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                ),
                test_cases=[
                    (
                        {
                            "image": "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg",
                            "width": 1024,
                            "height": 639,
                        },
                        MatchURLChecker(
                            "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg"
                        ),
                    ),
                    (
                        {
                            "image": "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg",
                            "width": 200,
                            "height": 100,
                        },
                        AIChecker("An image of a car"),
                    ),
                    (
                        {
                            "image": "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg",
                            "width": 200,
                            "height": 100,
                        },
                        AIChecker("A jpg image"),
                    ),
                    (
                        {
                            "image": "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg",
                            "width": 200,
                            "height": 100,
                        },
                        AIChecker("A image with width 200px and height 100px"),
                    ),
                    (
                        {
                            "image": "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg",
                            "width": 200,
                            "height": 100,
                        },
                        NoChecker(),
                    ),
                ],
            )

        with fixture_dir("image-base"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                )
            )

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_images_with_seed():
    model_owner = "replicate-internal"
    model_name = generate_model_name()
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("image-base-seed"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                )
            )

        with fixture_dir("image-base-seed"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                )
            )

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_train():
    model_owner = "replicate-internal"
    model_name = generate_model_name()
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        with fixture_dir("train"):
            cog_safe_push(
                make_task_context(
                    model_owner,
                    model_name,
                    model_owner,
                    test_model_name,
                    "cpu",
                    train=True,
                    train_destination_owner=model_owner,
                    train_destination_name=test_model_name + "-dest",
                ),
                fuzz_iterations=1,
            )

        with fixture_dir("train"):
            cog_safe_push(
                make_task_context(
                    model_owner,
                    model_name,
                    model_owner,
                    test_model_name,
                    "cpu",
                    train=True,
                    train_destination_owner=model_owner,
                    train_destination_name=test_model_name + "-dest",
                ),
                fuzz_iterations=1,
                do_compare_outputs=False,
            )

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)
        delete_model(model_owner, test_model_name + "-dest")


def generate_model_name():
    return "test-cog-safe-push-" + uuid.uuid4().hex


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

    with suppress(httpx.RemoteProtocolError):
        for version in model.versions.list():
            print(f"Deleting version {version.id}")
            with suppress(json.JSONDecodeError):
                # bug in replicate-python causes delete to throw JSONDecodeError
                model.versions.delete(version.id)

        print(f"Deleting model {model_owner}/{model_name}")
        with suppress(json.JSONDecodeError):
            # bug in replicate-python causes delete to throw JSONDecodeError
            replicate.models.delete(model_owner, model_name)


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
