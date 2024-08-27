import math
import os
import random

from cog import BasePredictor, Input, Path
from PIL import Image, ImageDraw


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        width: int = Input(description="Width.", ge=128, le=1440, default=256),
        height: int = Input(description="Height.", ge=128, le=1440, default=256),
        seed: int = Input(description="Random seed.", default=None),
    ) -> Path:
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        random.seed(seed)
        print(f"Using seed: {seed}")

        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)

        def random_color():
            return (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        def gradient_color(color1, color2, ratio):
            return tuple(
                int(color1[i] + (color2[i] - color1[i]) * ratio) for i in range(3)
            )

        for _ in range(random.randint(5, 50)):
            shape = random.choice(["rectangle", "ellipse", "line"])
            use_gradient = random.choice([True, False])

            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            if use_gradient:
                color1, color2 = random_color(), random_color()
                angle = random.uniform(0, 2 * math.pi)
                dx, dy = math.cos(angle), math.sin(angle)
            else:
                color = random_color()

            if shape == "rectangle":
                if use_gradient:
                    for x in range(x1, x2):
                        for y in range(y1, y2):
                            dist = (x - x1) * dx + (y - y1) * dy
                            max_dist = (x2 - x1) * dx + (y2 - y1) * dy
                            ratio = max(0, min(1, dist / max_dist))
                            draw.point(
                                (x, y), fill=gradient_color(color1, color2, ratio)
                            )
                else:
                    draw.rectangle((x1, y1, x2, y2), fill=color)
            elif shape == "ellipse":
                if use_gradient:
                    for x in range(x1, x2):
                        for y in range(y1, y2):
                            if (x - x1) * (x - x2) + (y - y1) * (y - y2) <= 0:
                                dist = (x - x1) * dx + (y - y1) * dy
                                max_dist = (x2 - x1) * dx + (y2 - y1) * dy
                                ratio = max(0, min(1, dist / max_dist))
                                draw.point(
                                    (x, y), fill=gradient_color(color1, color2, ratio)
                                )
                else:
                    draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                if use_gradient:
                    for t in range(100):
                        ratio = t / 99
                        x = x1 + (x2 - x1) * ratio
                        y = y1 + (y2 - y1) * ratio
                        draw.ellipse(
                            [x - 1, y - 1, x + 1, y + 1],
                            fill=gradient_color(color1, color2, ratio),
                        )
                else:
                    draw.line([x1, y1, x2, y2], fill=color, width=3)

        img.save("out.png")
        return Path("out.png")
