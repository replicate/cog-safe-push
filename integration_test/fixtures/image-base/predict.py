# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(description="Input image."),
        width: int = Input(description="New width.", ge=1, le=1000),
        height: int = Input(description="New height.", ge=1, le=1000),
    ) -> Path:
        img = Image.open(image)
        img = img.resize((width, height))
        img.save("out.png")
        return Path("out.png")
