import subprocess
from pathlib import Path
from typing import Any

import yaml

from .exceptions import CodeLintError


def lint_predict():
    cog_config = load_cog_config()
    predict_config = cog_config.get("predict", "")
    predict_filename = predict_config.split(":")[0]

    if not predict_filename:
        raise CodeLintError("cog.yaml doesn't have a valid predict stanza")

    lint_file(predict_filename)


def lint_train():
    cog_config = load_cog_config()
    train_config = cog_config.get("train", "")
    train_filename = train_config.split(":")[0]

    if not train_filename:
        raise CodeLintError("cog.yaml doesn't have a valid train stanza")

    lint_file(train_filename)


def load_cog_config() -> dict[str, Any]:
    with Path("cog.yaml").open() as f:
        return yaml.safe_load(f)


def lint_file(filename: str):
    if not Path(filename).exists():
        raise CodeLintError(f"{filename} doesn't exist")

    try:
        subprocess.run(
            ["ruff", "check", filename, "--ignore=E402"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CodeLintError(f"Linting {filename} failed: {e.stdout}\n{e.stderr}") from e
