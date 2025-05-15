from replicate.exceptions import ReplicateError

from . import cog
from .log import info, warning
from .task_context import get_or_create_model
from .utils import parse_model


def push_official_model(
    official_model: str, dockerfile: str | None, fast_push: bool = False
) -> None:
    owner, name = parse_model(official_model)
    try:
        model = get_or_create_model(owner, name, "cpu")
        info(f"Pushing to official model {official_model}")
        cog.push(model.owner, model.name, dockerfile, fast_push)
    except ReplicateError as e:
        if e.status == 403:
            warning(
                f"Could not get or create model {official_model} due to permission issues. Continuing with push..."
            )
            cog.push(owner, name, dockerfile, fast_push)
        else:
            raise
