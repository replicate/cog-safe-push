import pytest

from cog_safe_push.match_outputs import outputs_match


@pytest.mark.asyncio
async def test_non_matching_replicate_delivery_images():
    """
    Test that reproduces the issue where two different replicate.delivery URLs
    for different images are correctly identified as not matching.
    
    This test uses the exact URLs from the reported error:
    - test output: https://replicate.delivery/xezq/AwVT92BrC2LjMph3Qr84eoTOBfUY14ms10oN0pr6GhXqeB8qA/out.jpg
    - model output: https://replicate.delivery/xezq/Nm38Rbi6wiqgJxTaqPw6Lwh58LWbJe8SruZnCVpD40HYfAeqA/out.jpg
    
    Expected behavior: Images should not match because they are different images.
    """
    test_output = "https://replicate.delivery/xezq/AwVT92BrC2LjMph3Qr84eoTOBfUY14ms10oN0pr6GhXqeB8qA/out.jpg"
    model_output = "https://replicate.delivery/xezq/Nm38Rbi6wiqgJxTaqPw6Lwh58LWbJe8SruZnCVpD40HYfAeqA/out.jpg"
    
    # Test with is_deterministic=False (uses AI comparison)
    matches, error_message = await outputs_match(test_output, model_output, is_deterministic=False)
    
    assert not matches, f"Images should not match. URLs are different: {test_output} vs {model_output}"
    assert error_message == "Images are not similar", f"Expected 'Images are not similar' but got: {error_message}"


@pytest.mark.asyncio
async def test_matching_replicate_delivery_images():
    """
    Test that the same URL matches itself (sanity check).
    """
    url = "https://replicate.delivery/xezq/AwVT92BrC2LjMph3Qr84eoTOBfUY14ms10oN0pr6GhXqeB8qA/out.jpg"
    
    # Test with is_deterministic=False (uses AI comparison)
    matches, error_message = await outputs_match(url, url, is_deterministic=False)
    
    assert matches, f"Same URL should match itself. Error: {error_message}"
