import base64
import functools
import json
import mimetypes
import subprocess
from pathlib import Path

import replicate

from . import log
from .exceptions import AIError

MAX_TOKENS = 8192
MODEL = "anthropic/claude-4.5-sonnet"


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
    input_params: dict = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_tokens": MAX_TOKENS,
    }

    if files:
        if include_file_metadata:
            prompt += "\n\nMetadata for the attached file(s):\n"
            for path in files:
                prompt += "* " + file_info(path) + "\n"
            input_params["prompt"] = prompt

        image_uris = create_image_data_uris(files)
        if image_uris:
            input_params["image"] = image_uris[0]
            if len(image_uris) > 1:
                log.warning(
                    f"Replicate Claude wrapper only supports one image, ignoring {len(image_uris) - 1} additional images"
                )

        log.vvv(f"Claude prompt with {len(files)} files: {prompt}")
    else:
        log.vvv(f"Claude prompt: {prompt}")

    output_parts = []
    for event in replicate.stream(MODEL, input=input_params):
        output_parts.append(str(event))

    output = "".join(output_parts)
    log.vvv(f"Claude response: {output}")
    return output


def create_image_data_uris(files: list[Path]) -> list[str]:
    uris = []
    for path in files:
        with path.open("rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()

        mime_type, _ = mimetypes.guess_type(path, strict=False)
        if mime_type is None:
            mime_type = "application/octet-stream"
            log.v(f"Detected mime type {mime_type} for {path}")

        data_uri = f"data:{mime_type};base64,{encoded_string}"
        uris.append(data_uri)

    return uris


def file_info(p: Path) -> str:
    result = subprocess.run(
        ["file", "-b", str(p)], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
