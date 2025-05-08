from dataclasses import dataclass

import replicate
from replicate.exceptions import ReplicateError
from replicate.model import Model

from . import cog, log
from .exceptions import ArgumentError


@dataclass(frozen=True)
class TaskContext:
    model: Model
    test_model: Model
    train_destination: Model | None
    dockerfile: str | None
    fast_push: bool

    def is_train(self):
        return self.train_destination is not None


def make_task_context(
    model_owner: str,
    model_name: str,
    test_model_owner: str,
    test_model_name: str,
    test_hardware: str,
    train: bool = False,
    dockerfile: str | None = None,
    train_destination_owner: str | None = None,
    train_destination_name: str | None = None,
    train_destination_hardware: str = "cpu",
    push_test_model=True,
    fast_push: bool = False,
) -> TaskContext:
    if model_owner == test_model_owner and model_name == test_model_name:
        raise ArgumentError("Can't use the same model as test model")

    model = get_model(model_owner, model_name)
    if not model:
        raise ArgumentError(
            f"You need to create the model {model_owner}/{model_name} before running this script"
        )

    test_model = get_or_create_model(test_model_owner, test_model_name, test_hardware)

    if train:
        train_destination = get_or_create_model(
            train_destination_owner, train_destination_name, train_destination_hardware
        )
    else:
        train_destination = None

    context = TaskContext(
        model=model,
        test_model=test_model,
        train_destination=train_destination,
        dockerfile=dockerfile,
        fast_push=fast_push,
    )

    if not push_test_model:
        log.info(
            "Not pushing test model; assume test model was already pushed for training"
        )
        return context

    log.info("Pushing test model")
    pushed_version_id = cog.push(test_model, dockerfile, fast_push)
    test_model.reload()
    try:
        assert test_model.versions.list()[0].id.strip() == pushed_version_id.strip(), (
            f"Pushed version ID {pushed_version_id} doesn't match latest version on {test_model_owner}/{test_model_name}: {test_model.versions.list()[0].id}"
        )
    except ReplicateError as e:
        if e.status == 404:
            # Assume it's an official model
            # If it's an official model, can't check that the version matches
            pass
        else:
            raise
    return context


def get_or_create_model(model_owner, model_name, hardware) -> Model:
    model = get_model(model_owner, model_name)

    if not model:
        if not hardware:
            raise ArgumentError(
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


def get_model(owner, name) -> Model | None:
    try:
        model = replicate.models.get(f"{owner}/{name}")
    except ReplicateError as e:
        if e.status == 404:
            return None
        raise
    return model
