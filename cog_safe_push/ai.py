import base64
import functools
import json
import mimetypes
import subprocess
from pathlib import Path

import replicate

from . import log
from .exceptions import AIError

MAX_TOKENS = 65535
MODEL = "google/gemini-3-pro"


def async_retry(attempts=3):
    def decorator_retry(func):
        @functools.wraps(func)
        async def wrapper_retry(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    log.warning(f"Exception occurred: {e}")
                    if attempt < attempts:
                        log.warning(f"Retrying attempt {attempt}/{attempts}")
                    else:
                        log.warning(f"Giving up after {attempts} attempts")
                        raise
            return None

        return wrapper_retry

    return decorator_retry


@async_retry(3)
async def boolean(
    prompt: str, files: list[Path] | None = None, include_file_metadata: bool = False
) -> bool:
    system_prompt = "You are a boolean classifier. You must only respond with either YES or NO, and absolutely nothing else. Your response will be used in a programmatic context so it is critical that you only ever answer with either the string YES or the string NO."
    output = await call(
        system_prompt=system_prompt,
        prompt=prompt.strip(),
        files=files,
        include_file_metadata=include_file_metadata,
    )
    if output == "YES":
        return True
    if output == "NO":
        return False
    raise AIError(f"Failed to parse output as YES/NO: {output}")


@async_retry(3)
async def json_object(
    prompt: str,
    files: list[Path] | None = None,
    system_prompt: str = "",
) -> dict:
    if system_prompt:
        system_prompt = system_prompt.strip() + "\n\n"
    system_prompt += "You always respond with valid JSON, and nothing else (no backticks, etc.). Your outputs will be used in a programmatic context."
    output = await call(
        system_prompt=system_prompt,
        prompt=prompt.strip(),
        files=files,
    )

    if output.startswith("```json"):
        output = output[7:]
    elif output.startswith("```"):
        output = output[3:]
    if output.endswith("```"):
        output = output[:-3]
    output = output.strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise AIError(f"Failed to parse output as JSON: {output}")


async def call(
    system_prompt: str,
    prompt: str,
    files: list[Path] | None = None,
    include_file_metadata: bool = False,
) -> str:
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"

    input_params: dict = {
        "prompt": full_prompt,
        "max_output_tokens": MAX_TOKENS,
        "temperature": 1,
        "top_p": 0.95,
    }

    if files:
        if include_file_metadata:
            full_prompt += "\n\nMetadata for the attached file(s):\n"
            for path in files:
                full_prompt += "* " + file_info(path) + "\n"
            input_params["prompt"] = full_prompt

        images, audio_files = categorize_files(files)

        if images:
            input_params["images"] = [create_data_uri(img) for img in images]

        if audio_files:
            if len(audio_files) > 1:
                log.warning(
                    f"Only one audio file supported, ignoring {len(audio_files) - 1} additional audio files"
                )
            input_params["audio"] = create_data_uri(audio_files[0])

        log.vvv(f"Gemini prompt with {len(files)} files: {full_prompt}")
    else:
        log.vvv(f"Gemini prompt: {full_prompt}")

    output_parts = []
    for event in replicate.stream(MODEL, input=input_params):
        output_parts.append(str(event))

    output = "".join(output_parts)
    log.vvv(f"Gemini response: {output}")
    return output


def categorize_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    images = []
    audio_files = []

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    audio_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".oga", ".opus"}

    for path in files:
        ext = path.suffix.lower()
        if ext in image_extensions:
            images.append(path)
        elif ext in audio_extensions:
            audio_files.append(path)
        else:
            images.append(path)

    return images, audio_files


def create_data_uri(path: Path) -> str:
    with path.open("rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    mime_type, _ = mimetypes.guess_type(path, strict=False)
    if mime_type is None:
        mime_type = "application/octet-stream"
        log.v(f"Detected mime type {mime_type} for {path}")

    return f"data:{mime_type};base64,{encoded_string}"


def file_info(p: Path) -> str:
    result = subprocess.run(
        ["file", "-b", str(p)], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
