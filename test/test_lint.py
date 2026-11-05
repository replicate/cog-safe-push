import pytest

from cog_safe_push import lint
from cog_safe_push.exceptions import CodeLintError


def test_lint_file_accepts_idiomatic_cog_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "predict.py").write_text(
        """from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def predict(self, image: Path = Input(description="Input image.")) -> str:
        return image.name
"""
    )

    lint.lint_file("predict.py")


def test_lint_file_rejects_undefined_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "predict.py").write_text("def predict():\n    return undefined\n")

    with pytest.raises(CodeLintError):
        lint.lint_file("predict.py")
