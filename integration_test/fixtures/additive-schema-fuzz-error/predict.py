# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.hello = "hello "

    def predict(
        self,
        text: str = Input(description="Text that will be prepended by 'hello '."),
        qux: int = Input(description="A number between 1 and 3", default=2, ge=1, le=3),
    ) -> str:
        if qux == 1:
            raise ValueError("qux!")
        return self.hello + text
