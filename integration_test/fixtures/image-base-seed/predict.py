import random
from PIL import Image, ImageDraw
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        width: int = Input(description="Width.", ge=1, le=1000, default=256),
        height: int = Input(description="Height.", ge=1, le=1000, default=256),
        seed: int = Input(description="Random seed.", default=None),
    ) -> Path:
        if seed is not None:
            random.seed(seed)

        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)

        # Generate shapes
        for _ in range(random.randint(5, 50)):
            shape = random.choice(["rectangle", "ellipse", "line"])
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            if shape == "rectangle":
                draw.rectangle((x1, y1, x2, y2), fill=color)
            elif shape == "ellipse":
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.line([x1, y1, x2, y2], fill=color, width=3)

        img.save("out.png")
        return Path("out.png")
