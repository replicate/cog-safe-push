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
from cog_safe_push.config import Config
from cog_safe_push.exceptions import *
from cog_safe_push.main import cog_safe_push, run_config
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


def test_cog_safe_push_ignore_incompatible_schema():
    model_owner = "replicate-internal"
    model_name = generate_model_name()
    test_model_name = model_name + "-test"
    create_model(model_owner, model_name)

    try:
        # First push with base schema
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
                ],
            )

        # Then try to push with incompatible schema, but with ignore flag
        with fixture_dir("incompatible-schema"):
            cog_safe_push(
                make_task_context(
                    model_owner, model_name, model_owner, test_model_name, "cpu"
                ),
                ignore_schema_compatibility=True,
                do_compare_outputs=False,
            )

    finally:
        delete_model(model_owner, model_name)
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_deployment():
    """Test deployment functionality with a real model."""
    model_owner = "replicate-internal"
    model_name = "cog-safe-push-deployment-test"
    test_model_name = f"deployment-test-{generate_model_name()}"

    try:
        with fixture_dir("image-base"):
            cog_safe_push(
                make_task_context(
                    model_owner=model_owner,
                    model_name=model_name,
                    test_model_owner=model_owner,
                    test_model_name=test_model_name,
                    test_hardware="cpu",
                    deployment_name="cog-safe-push-deployment-test",
                    deployment_owner="replicate-internal",
                    deployment_hardware="cpu",
                ),
                ignore_schema_compatibility=True,
                test_cases=[
                    (
                        {
                            "image": "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg",
                            "width": 200,
                            "height": 100,
                        },
                        AIChecker("An image of a car"),
                    ),
                ],
            )

    finally:
        # Models associated with a deployment are not deleted by default
        # We only delete the test model
        delete_model(model_owner, test_model_name)


def test_cog_safe_push_create_official_model():
    model_owner = "replicate-internal"
    model_name = generate_model_name()
    test_model_name = model_name + "-test"
    official_model_name = model_name + "-official"

    try:
        with fixture_dir("image-base"):
            config = Config(
                model=f"{model_owner}/{model_name}",
                test_model=f"{model_owner}/{test_model_name}",
                official_model=f"{model_owner}/{official_model_name}",
                test_hardware="cpu",
            )
            run_config(config, no_push=False, push_official_model=True)

            # Verify the official model was created and has a version
            official_model = replicate.models.get(
                f"{model_owner}/{official_model_name}"
            )
            assert official_model.latest_version is not None

    finally:
        delete_model(model_owner, official_model_name)


def test_cog_safe_push_push_official_model():
    model_owner = "replicate-internal"
    model_name = generate_model_name()
    test_model_name = model_name + "-test"
    official_model_name = model_name + "-official"
    create_model(model_owner, official_model_name)

    try:
        with fixture_dir("image-base"):
            config = Config(
                model=f"{model_owner}/{model_name}",
                test_model=f"{model_owner}/{test_model_name}",
                official_model=f"{model_owner}/{official_model_name}",
                test_hardware="cpu",
            )

            official_model = replicate.models.get(
                f"{model_owner}/{official_model_name}"
            )
            initial_version_id = (
                official_model.latest_version.id
                if official_model.latest_version
                else None
            )

            run_config(config, no_push=False, push_official_model=True)

            official_model = replicate.models.get(
                f"{model_owner}/{official_model_name}"
            )
            assert official_model.latest_version is not None
            assert official_model.latest_version.id != initial_version_id

    finally:
        delete_model(model_owner, official_model_name)


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
