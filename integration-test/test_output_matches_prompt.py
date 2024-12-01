from pathlib import Path

import pytest

from cog_safe_push.match_outputs import output_matches_prompt

# log.set_verbosity(3)

positive_images = {
    "https://replicate.delivery/xezq/DyBtXhblvL7MApRBqeqiYnkw1xS9WpEf3nA7GRIlYFkQL31TA/out-0.webp": [
        "A bird",
        "A red bird",
        "A webp image of a bird",
        "A webp image of a red bird",
    ],
    "https://replicate.delivery/czjl/QFrZ9RF8VroFM5Ml9MKt3rm0vP8ZHTWaqfO1oT6bouj0m76JA/tmpn888w5a8.jpg": [
        "A jpg image of a formula one car",
        "a jpg image of a car",
        "A jpg image",
        "Formula 1 car",
        "car",
    ],
    "https://replicate.delivery/czjl/8C4OJCR6w7rQEFeernSerHH5e3xe2f9cYYsGTW8k5Eob57d9E/tmpjwitpu7f.png": [
        "480x320px png image",
        "480x320px image of a formula one car",
    ],
    "https://replicate.delivery/czjl/41MrDvJli4ZCAxeYMhEcKvAHNNcPaWJTicjqp7GYNFza476JA/tmpzs4y7hto.png": [
        "an anime illustration of a lake",
        "an anime illustration",
        "a lake",
    ],
    "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg": [
        "An image of a car",
        "A jpg image",
        "A image with width 1024px and height 639px",
    ],
}

negative_images = {
    "https://replicate.delivery/xezq/DyBtXhblvL7MApRBqeqiYnkw1xS9WpEf3nA7GRIlYFkQL31TA/out-0.webp": [
        "A cat",
        "A blue bird",
        "A png image of a bird",
        "A webp image of a blue bird",
    ],
    "https://replicate.delivery/czjl/QFrZ9RF8VroFM5Ml9MKt3rm0vP8ZHTWaqfO1oT6bouj0m76JA/tmpn888w5a8.jpg": [
        "A jpg image of a tractor",
        "a webp image of a road",
        "A webp image",
        "motorcycle",
    ],
    "https://replicate.delivery/czjl/8C4OJCR6w7rQEFeernSerHH5e3xe2f9cYYsGTW8k5Eob57d9E/tmpjwitpu7f.png": [
        "100x100px png image",
        "100x100px image of a formula one car",
    ],
    "https://replicate.delivery/czjl/41MrDvJli4ZCAxeYMhEcKvAHNNcPaWJTicjqp7GYNFza476JA/tmpzs4y7hto.png": [
        "an anime illustration of a cat",
        "a 3d render",
        "a potato patch",
    ],
}


def get_captioned_images(
    image_dict: dict[str, list[str]], iterations_per_image=3
) -> list[tuple[str, str]]:
    ret = []
    for url, captions in image_dict.items():
        for _ in range(iterations_per_image):
            for caption in captions:
                ret.append((url, caption))
    return ret


@pytest.mark.parametrize(
    ("file_url", "prompt"),
    get_captioned_images(positive_images),
    ids=lambda x: Path(x[0]).name if isinstance(x, tuple) else x,
)
async def test_image_output_matches_prompt_positive(file_url: str, prompt: str):
    """Test that images in the positive directory match their prompts."""
    matches, message = await output_matches_prompt(file_url, prompt)
    assert matches, f"Image should match prompt '{prompt}'. Error: {message}"


@pytest.mark.parametrize(
    ("file_url", "prompt"),
    get_captioned_images(negative_images),
    ids=lambda x: Path(x[0]).name if isinstance(x, tuple) else x,
)
async def test_image_output_matches_prompt_negative(file_url: str, prompt: str):
    """Test that images in the negative directory don't match their prompts."""
    matches, _ = await output_matches_prompt(file_url, prompt)
    assert not matches, f"Image should not match prompt '{prompt}'"
