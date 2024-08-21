import subprocess
import yaml

from .exceptions import CodeLintError


def lint_predict():
    with open("cog.yaml", "r") as f:
        cog_config = yaml.safe_load(f)

    predict_config = cog_config.get("predict", "")
    predict_filename = (
        predict_config.split(":")[0] if ":" in predict_config else predict_config
    )

    if not predict_filename:
        raise CodeLintError("cog.yaml doesn't have a valid predict stanza")

    try:
        subprocess.run(
            ["ruff", "check", predict_filename],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise CodeLintError(f"Linting failed: {e.stdout}\n{e.stderr}") from e
