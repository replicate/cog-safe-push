import pytest

from cog_safe_push.match_outputs import outputs_match


@pytest.mark.asyncio
async def test_output_match_similar_images():
    url1 = "https://replicate.delivery/xezq/AwVT92BrC2LjMph3Qr84eoTOBfUY14ms10oN0pr6GhXqeB8qA/out.jpg"
    url2 = "https://replicate.delivery/xezq/Nm38Rbi6wiqgJxTaqPw6Lwh58LWbJe8SruZnCVpD40HYfAeqA/out.jpg"
    matches, error_message = await outputs_match(url1, url2, is_deterministic=False)
    assert matches, error_message


@pytest.mark.asyncio
async def test_output_match_same_image():
    url = "https://replicate.delivery/xezq/AwVT92BrC2LjMph3Qr84eoTOBfUY14ms10oN0pr6GhXqeB8qA/out.jpg"
    matches, error_message = await outputs_match(url, url, is_deterministic=False)
    assert matches, error_message


@pytest.mark.asyncio
async def test_output_match_not_similar_images():
    url1 = "https://replicate.delivery/xezq/FC8AoQT9RlL1LNxCdM1scYfBKsk4A1rmOb67lYxfLYNcYBeqA/out-0.webp"
    url2 = "https://replicate.delivery/xezq/Zj0SX6yRmHbSM1SWXL583l4jg0N5UtiBPINOylKwq4zKWgXF/out-0.webp"
    matches, error_message = await outputs_match(url1, url2, is_deterministic=False)
    assert not matches
    assert error_message == "Images are not similar", (
        f"Expected 'Images are not similar' but got: {error_message}"
    )
