from . import cog
from .log import info
from .task_context import get_or_create_model


def push_official_model(
    official_model: str, dockerfile: str | None, fast_push: bool = False
) -> None:
    owner, name = official_model.split("/")
    model = get_or_create_model(owner, name, "cpu")
    info(f"Pushing to official model {official_model}")
    cog.push(model, dockerfile, fast_push)
