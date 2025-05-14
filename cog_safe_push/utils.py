import re

from .exceptions import ArgumentError


def truncate(s, max_length=500) -> str:
    s = str(s)
    if len(s) <= max_length:
        return s
    return s[:max_length] + "..."


def parse_model(model_owner_name: str) -> tuple[str, str]:
    pattern = r"^([a-z0-9_-]+)/([a-z0-9-.]+)$"
    match = re.match(pattern, model_owner_name)
    if not match:
        raise ArgumentError(f"Invalid model URL format: {model_owner_name}")
    owner, name = match.groups()
    return owner, name
