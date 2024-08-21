# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.hello = "goodbye "

    def predict(self, text: str = Input(description="Text that will be prepended by 'hello '.")) -> str:
        assert text
        return "1" * 100
