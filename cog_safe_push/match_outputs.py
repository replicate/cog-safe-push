import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List
from urllib.parse import urlparse

import requests
from PIL import Image

from . import ai, log


def output_matches_prompt(output: Any, prompt: str) -> tuple[bool, str]:
    urls = []
    if isinstance(output, str) and is_url(output):
        urls = [output]
    elif isinstance(output, (list, dict)) and all(
        isinstance(item, str) and is_url(item)
        for item in (output if isinstance(output, list) else output.values())
    ):
        urls = output if isinstance(output, list) else list(output.values())

    with download_many(urls) as tmp_files:
        claude_prompt = """You are part of an automatic evaluation that compares media (text, audio, image, video, etc.) to captions. I want to know if the caption matches the text or file..

"""
        if urls:
            claude_prompt += f"""Does this file(s) and the attached content of the file(s) match the description? Pay close attention to the metadata about the attached files which is included below, especially if the description mentions file type, image dimensions, or any other aspect that is described in the metadata. Do not infer file type or image dimensions from the image content, but from the attached metadata.

Description to evaluate: {prompt}

Filename(s): {output}"""
        else:
            claude_prompt += f"""Do these outputs match the following description?

Output: {output}

Description to evaluate: {prompt}"""

        matches = ai.boolean(
            claude_prompt,
            files=tmp_files,
            include_file_metadata=True,
        )

        if matches:
            return True, ""

        # If it's not a match, do best of three to avoid flaky tests
        multiple_matches = [matches]
        for _ in range(2):
            matches = ai.boolean(
                claude_prompt,
                files=tmp_files,
                include_file_metadata=True,
            )
            multiple_matches.append(matches)

            if sum(multiple_matches) >= 2:
                return True, ""

    return False, "AI determined that the output does not match the description"


def outputs_match(test_output, output, is_deterministic: bool) -> tuple[bool, str]:
    if type(test_output) is not type(output):
        return False, "The types of the outputs don't match"

    if isinstance(output, str):
        if is_url(test_output) and is_url(output):
            return urls_match(test_output, output, is_deterministic)

        if is_url(test_output) or is_url(output):
            return False, "Only one output is a URL"

        return strings_match(test_output, output, is_deterministic)

    if isinstance(output, bool):
        if test_output == output:
            return True, ""
        return False, "Booleans aren't identical"

    if isinstance(output, int):
        if test_output == output:
            return True, ""
        return False, "Integers aren't identical"

    if isinstance(output, float):
        if abs(test_output - output) < 0.1:
            return True, ""
        return False, "Floats aren't identical"

    if isinstance(output, dict):
        if test_output.keys() != output.keys():
            return False, "Dict keys don't match"
        for key in output:
            matches, message = outputs_match(
                test_output[key], output[key], is_deterministic
            )
            if not matches:
                return False, f"In {key}: {message}"
        return True, ""

    if isinstance(output, list):
        if len(test_output) != len(output):
            return False, "List lengths don't match"
        for i in range(len(output)):
            matches, message = outputs_match(
                test_output[i], output[i], is_deterministic
            )
            if not matches:
                return False, f"At index {i}: {message}"
        return True, ""

    log.warning(f"Unknown type: {type(output)}")

    return True, ""


def strings_match(s1: str, s2: str, is_deterministic: bool) -> tuple[bool, str]:
    if is_deterministic:
        if s1 == s2:
            return True, ""
        return False, "Strings aren't the same"
    fuzzy_match = ai.boolean(
        f"""
Have these two strings been generated by the same generative AI model inputs/prompt?

String 1: '{s1}'
String 2: '{s2}'
    """
    )
    if fuzzy_match:
        return True, ""
    return False, "Strings aren't similar"


def urls_match(url1: str, url2: str, is_deterministic: bool) -> tuple[bool, str]:
    # New model must return same extension as previous model
    if not extensions_match(url1, url2):
        return False, "URL extensions don't match"

    if is_image(url1):
        return images_match(url1, url2, is_deterministic)

    if is_audio(url1):
        return audios_match(url1, url2, is_deterministic)

    if is_video(url1):
        return videos_match(url1, url2, is_deterministic)

    log.warning(f"Unknown URL format: {url1}")
    return True, ""


def is_image(url: str) -> bool:
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
    return url.lower().endswith(image_extensions)


def is_audio(url: str) -> bool:
    audio_extensions = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
    return url.lower().endswith(audio_extensions)


def is_video(url: str) -> bool:
    video_extensions = (".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm")
    return url.lower().endswith(video_extensions)


def extensions_match(url1: str, url2: str) -> bool:
    ext1 = Path(urlparse(url1).path).suffix
    ext2 = Path(urlparse(url2).path).suffix
    return ext1.lower() == ext2.lower()


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def images_match(url1: str, url2: str, is_deterministic: bool) -> tuple[bool, str]:
    with download(url1) as tmp1, download(url2) as tmp2:
        img1 = Image.open(tmp1)
        img2 = Image.open(tmp2)
        if img1.size != img2.size:
            return False, "Image sizes don't match"

        if is_deterministic:
            diff = sum(
                math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2)))
                for p1, p2 in zip(img1.getdata(), img2.getdata())  # pyright: ignore
            ) / (img1.width * img1.height)

            if diff > 8:  # arbitrary epsilon
                return False, "Images are not identical"
            return True, ""

        fuzzy_match = ai.boolean(
            "These two images have been generated by or modified by an AI model. Is it highly likely that those two predictions of the model had the same inputs?",
            files=[tmp1, tmp2],
        )
        if fuzzy_match:
            return True, ""
        return False, "Images are not similar"


def audios_match(url1: str, url2: str, is_deterministic: bool) -> tuple[bool, str]:
    # # TODO: is_deterministic branch
    # with download(url1) as tmp1, download(url2) as tmp2:
    #     fuzzy_match = ai.boolean(
    #         "Have these two audio files been generated by the same inputs to a generative AI model?",
    #         files=[tmp1, tmp2],
    #     )
    # if fuzzy_match:
    #     return True, ""
    # return False, "Audio files are not similar"

    # Not yet supported by claude
    assert url1
    assert url2
    assert is_deterministic in [True, False]
    return True, ""


def videos_match(url1: str, url2: str, is_deterministic: bool) -> tuple[bool, str]:
    # # TODO: is_deterministic branch
    # with download(url1) as tmp1, download(url2) as tmp2:
    #     fuzzy_match = ai.boolean(
    #         "Have these two videos been generated by the same inputs to a generative AI model?",
    #         files=[tmp1, tmp2],
    #     )
    # if fuzzy_match:
    #     return True, ""
    # return False, "Videos are not similar"

    # Not yet supported by claude
    assert url1
    assert url2
    assert is_deterministic in [True, False]
    return True, ""


@contextmanager
def download(url: str) -> Iterator[Path]:
    suffix = Path(url).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        response = requests.get(url)
        response.raise_for_status()
        tmp_file.write(response.content)
        tmp_file.flush()
        tmp_path = Path(tmp_file.name)

    try:
        yield tmp_path
    finally:
        tmp_path.unlink()


@contextmanager
def download_many(urls: List[str]) -> Iterator[List[Path]]:
    tmp_files: List[Path] = []
    try:
        for url in urls:
            suffix = Path(urlparse(url).path).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                response = requests.get(url)
                response.raise_for_status()
                tmp_file.write(response.content)
                tmp_file.flush()
                tmp_files.append(Path(tmp_file.name))
        yield tmp_files
    finally:
        for tmp_file in tmp_files:
            tmp_file.unlink(missing_ok=True)
