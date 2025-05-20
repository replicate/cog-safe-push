import base64
import functools
import json
import mimetypes
import os
import subprocess
from pathlib import Path
from typing import cast

import anthropic

from . import log
from .exceptions import AIError, ArgumentError


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
    system_prompt = "You only answer YES or NO, and absolutely nothing else. Your response will be used in a programmatic context so it's important that you only ever answer with either the string YES or the string NO."
    # system_prompt = "You are a helpful assistant"
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
async def json_object(prompt: str, files: list[Path] | None = None) -> dict:
    system_prompt = "You always respond with valid JSON, and nothing else (no backticks, etc.). Your outputs will be used in a programmatic context."
    output = await call(system_prompt=system_prompt, prompt=prompt.strip(), files=files)
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ArgumentError("ANTHROPIC_API_KEY is not defined")

    model = "claude-3-7-sonnet-latest"
    client = anthropic.AsyncAnthropic(api_key=api_key)

    try:
        if files:
            content = create_content_list(files)

            if include_file_metadata:
                prompt += "\n\nMetadata for the attached file(s):\n"
                for path in files:
                    prompt += "* " + file_info(path) + "\n"

            content.append({"type": "text", "text": prompt})

            log.vvv(f"Claude prompt with {len(files)} files: {prompt}")
        else:
            content = prompt
            log.vvv(f"Claude prompt: {prompt}")

        messages: list[anthropic.types.MessageParam] = [
            {"role": "user", "content": content}
        ]

        response = await client.messages.create(
            model=model,
            messages=messages,
            system=system_prompt,
            max_tokens=4096,
            stream=False,
            temperature=1.0,
        )
        content = cast("anthropic.types.TextBlock", response.content[0])

    finally:
        await client.close()

    output = content.text
    log.vvv(f"Claude response: {output}")
    return output


def create_content_list(
    files: list[Path],
) -> list[anthropic.types.ImageBlockParam | anthropic.types.TextBlockParam]:
    content = []
    for path in files:
        with path.open("rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()

        mime_type, _ = mimetypes.guess_type(path, strict=False)
        if mime_type is None:
            mime_type = "application/octet-stream"
            log.v(f"Detected mime type {mime_type} for {path}")

        content.append(
            {
                "type": "image",  # only image is supported
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": encoded_string,
                },
            }
        )

    return content


def file_info(p: Path) -> str:
    result = subprocess.run(
        ["file", "-b", str(p)], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
