from replicate.exceptions import ReplicateError

from . import cog
from .log import info, warning
from .task_context import get_or_create_model
from .utils import parse_model


def push_official_model(
    official_model: str,
    dockerfile: str | None,
    fast_push: bool = False,
    use_cog_base_image: bool = True,
) -> None:
    owner, name = parse_model(official_model)
    try:
        model = get_or_create_model(owner, name, "cpu")
        info(f"Pushing to official model {official_model}")
        cog.push(
            model_owner=model.owner,
            model_name=model.name,
            dockerfile=dockerfile,
            fast_push=fast_push,
            use_cog_base_image=use_cog_base_image,
        )
    except ReplicateError as e:
        if e.status == 403:
            warning(
                f"Could not get or create model {official_model} due to permission issues. Continuing with push..."
            )
            cog.push(
                model_owner=owner,
                model_name=name,
                dockerfile=dockerfile,
                fast_push=fast_push,
                use_cog_base_image=use_cog_base_image,
            )
        else:
            raise
