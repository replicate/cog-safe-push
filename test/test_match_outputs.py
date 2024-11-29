from unittest.mock import MagicMock, patch

import pytest

from cog_safe_push.match_outputs import (
    extensions_match,
    is_audio,
    is_image,
    is_url,
    is_video,
    outputs_match,
)


def test_identical_strings():
    assert outputs_match("hello", "hello", True) == (True, "")


def test_different_strings_deterministic():
    assert outputs_match("hello", "world", True) == (False, "Strings aren't the same")


@patch("cog_safe_push.predict.ai.boolean")
def test_different_strings_non_deterministic(mock_ai_boolean):
    mock_ai_boolean.return_value = True
    assert outputs_match("The quick brown fox", "A fast auburn canine", False) == (
        True,
        "",
    )
    mock_ai_boolean.assert_called_once()

    mock_ai_boolean.reset_mock()
    mock_ai_boolean.return_value = False
    assert outputs_match(
        "The quick brown fox", "Something completely different", False
    ) == (False, "Strings aren't similar")
    mock_ai_boolean.assert_called_once()


def test_identical_booleans():
    assert outputs_match(True, True, True) == (True, "")


def test_different_booleans():
    assert outputs_match(True, False, True) == (False, "Booleans aren't identical")


def test_identical_integers():
    assert outputs_match(42, 42, True) == (True, "")


def test_different_integers():
    assert outputs_match(42, 43, True) == (False, "Integers aren't identical")


def test_close_floats():
    assert outputs_match(3.14, 3.14001, True) == (True, "")


def test_different_floats():
    assert outputs_match(3.14, 3.25, True) == (False, "Floats aren't identical")


def test_identical_dicts():
    dict1 = {"a": 1, "b": "hello"}
    dict2 = {"a": 1, "b": "hello"}
    assert outputs_match(dict1, dict2, True) == (True, "")


def test_different_dict_values():
    dict1 = {"a": 1, "b": "hello"}
    dict2 = {"a": 1, "b": "world"}
    assert outputs_match(dict1, dict2, True) == (False, "In b: Strings aren't the same")


def test_different_dict_keys():
    dict1 = {"a": 1, "b": "hello"}
    dict2 = {"a": 1, "c": "hello"}
    assert outputs_match(dict1, dict2, True) == (False, "Dict keys don't match")


def test_identical_lists():
    list1 = [1, "hello", True]
    list2 = [1, "hello", True]
    assert outputs_match(list1, list2, True) == (True, "")


def test_different_list_values():
    list1 = [1, "hello", True]
    list2 = [1, "world", True]
    assert outputs_match(list1, list2, True) == (
        False,
        "At index 1: Strings aren't the same",
    )


def test_different_list_lengths():
    list1 = [1, 2, 3]
    list2 = [1, 2]
    assert outputs_match(list1, list2, True) == (False, "List lengths don't match")


def test_nested_structures():
    struct1 = {"a": [1, {"b": "hello"}], "c": True}
    struct2 = {"a": [1, {"b": "hello"}], "c": True}
    assert outputs_match(struct1, struct2, True) == (True, "")


def test_different_nested_structures():
    struct1 = {"a": [1, {"b": "hello"}], "c": True}
    struct2 = {"a": [1, {"b": "world"}], "c": True}
    assert outputs_match(struct1, struct2, True) == (
        False,
        "In a: At index 1: In b: Strings aren't the same",
    )


def test_different_types():
    assert outputs_match("42", 42, True) == (
        False,
        "The types of the outputs don't match",
    )


def test_is_url():
    assert is_url("http://example.com")
    assert is_url("https://example.com")
    assert not is_url("not_a_url")


def test_is_image():
    assert is_image("image.jpg")
    assert is_image("image.png")
    assert not is_image("not_an_image.txt")


def test_is_audio():
    assert is_audio("audio.mp3")
    assert is_audio("audio.wav")
    assert not is_audio("not_an_audio.txt")


def test_is_video():
    assert is_video("video.mp4")
    assert is_video("video.avi")
    assert not is_video("not_a_video.txt")


def test_extensions_match():
    assert extensions_match("file1.jpg", "file2.jpg")
    assert not extensions_match("file1.jpg", "file2.png")


async def test_urls_with_different_extensions():
    result, message = await outputs_match(
        "http://example.com/file1.jpg", "http://example.com/file2.png", False
    )
    assert not result
    assert message == "URL extensions don't match"


async def test_one_url_one_non_url():
    result, message = await outputs_match(
        "http://example.com/image.jpg", "not_a_url", False
    )
    assert not result
    assert message == "Only one output is a URL"


@patch("cog_safe_push.match_outputs.download")
@patch("PIL.Image.open")
async def test_images_with_different_sizes(mock_image_open, mock_download):
    assert mock_download
    mock_image1 = MagicMock()
    mock_image2 = MagicMock()
    mock_image1.size = (100, 100)
    mock_image2.size = (200, 200)
    mock_image_open.side_effect = [mock_image1, mock_image2]

    result, message = await outputs_match(
        "http://example.com/image1.jpg", "http://example.com/image2.jpg", False
    )
    assert not result
    assert message == "Image sizes don't match"


@patch("cog_safe_push.log.warning")
async def test_unknown_url_format(mock_warning):
    result, _ = await outputs_match(
        "http://example.com/unknown.xyz", "http://example.com/unknown.xyz", False
    )
    assert result
    mock_warning.assert_called_with(
        "Unknown URL format: http://example.com/unknown.xyz"
    )


@patch("cog_safe_push.log.warning")
async def test_unknown_output_type(mock_warning):
    class UnknownType:
        pass

    result, _ = await outputs_match(UnknownType(), UnknownType(), False)
    assert result
    mock_warning.assert_called_with(f"Unknown type: {type(UnknownType())}")


async def test_large_structure_performance():
    import time

    large_struct1 = {"key" + str(i): i for i in range(10000)}
    large_struct2 = {"key" + str(i): i for i in range(10000)}

    start_time = time.time()
    result, _ = await outputs_match(large_struct1, large_struct2, False)
    end_time = time.time()

    assert result
    assert end_time - start_time < 1  # Ensure it completes in less than 1 second


if __name__ == "__main__":
    pytest.main()
