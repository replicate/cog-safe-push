from . import cog
from .log import log
from .task_context import get_or_create_model


def push_official_model(official_model: str) -> None:
    owner, name = official_model.split("/")
    model = get_or_create_model(owner, name, "cpu")
    log.info(f"Pushing to official model {official_model}")
    cog.push(model)
