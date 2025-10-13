# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: Path = Input(
            description="Input image. Valid file types are: jpg, png, webp, bmp, gif (not animated)"
        ),
        width: int = Input(description="New width.", ge=1, le=2000),
        height: int = Input(description="New height.", ge=1, le=1000),
    ) -> Path:
        img = Image.open(image)
        img = img.resize((width, height))
        out_path = Path("out" + image.suffix)
        img.save(str(out_path))
        return out_path
