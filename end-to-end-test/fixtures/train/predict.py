import requests
from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.default_prefix = "hello "

    def predict(
        self,
        text: str = Input(description="Text that will be prepended by 'hello '."),
        replicate_weights: str = Input(
            description="Trained prefix string.",
            default=None,
        ),
    ) -> str:
        if replicate_weights:
            response = requests.get(replicate_weights)
            prefix = response.text
        else:
            prefix = self.default_prefix

        return prefix + text
