import pytest

from cog_safe_push.match_outputs import outputs_match


@pytest.mark.asyncio
async def test_output_match_similar_images():
    url1 = "https://replicate.delivery/xezq/OrGhA2j4ACZ8FdbZgTxyaav6EKSxZ4jBnNzZwXIZZleq8TvKA/out-0.webp"
    url2 = "https://replicate.delivery/xezq/Z4UKfUkAqp0RRaGQRIerW3ZGansA1Rqg6eodiOfYTfedZeTvKA/out-0.webp"
    matches, error_message = await outputs_match(url1, url2, is_deterministic=False)
    assert matches, error_message


@pytest.mark.asyncio
async def test_output_match_same_image():
    url = "https://replicate.delivery/xezq/OrGhA2j4ACZ8FdbZgTxyaav6EKSxZ4jBnNzZwXIZZleq8TvKA/out-0.webp"
    matches, error_message = await outputs_match(url, url, is_deterministic=False)
    assert matches, error_message


@pytest.mark.asyncio
async def test_output_match_not_similar_images():
    url1 = "https://replicate.delivery/xezq/OrGhA2j4ACZ8FdbZgTxyaav6EKSxZ4jBnNzZwXIZZleq8TvKA/out-0.webp"
    url2 = "https://replicate.delivery/xezq/NtEEOzxwpTaFFF5fhalpLevI1HwrmGc3bNX799EzWmf51P9qA/out-0.webp"
    matches, error_message = await outputs_match(url1, url2, is_deterministic=False)
    assert not matches
    assert error_message == "Images are not similar", (
        f"Expected 'Images are not similar' but got: {error_message}"
    )
