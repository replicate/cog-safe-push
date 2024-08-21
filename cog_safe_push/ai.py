import base64
import mimetypes
from pathlib import Path
import os
import json
import anthropic

from .exceptions import AIError
from .retry import retry
from . import log


@retry(3)
def boolean(prompt: str, files: list[Path] | None = None) -> bool:
    system_prompt = "You only answer YES or NO, and absolutely nothing else. Your outputs will be used in a programmatic context so it's important that you only ever answer with either the string YES or the string NO."
    output = call(system_prompt=system_prompt, prompt=prompt.strip(), files=files)
    if output == "YES":
        return True
    if output == "NO":
        return False
    raise AIError(f"Failed to parse output as YES/NO: {output}")


@retry(3)
def json_object(prompt: str, files: list[Path] | None = None) -> dict:
    system_prompt = "You always respond with valid JSON, and nothing else (no backticks, etc.). Your outputs will be used in a programmatic context."
    output = call(system_prompt=system_prompt, prompt=prompt.strip(), files=files)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise AIError(f"Failed to parse output as JSON: {output}")


def call(system_prompt: str, prompt: str, files: list[Path] | None = None) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not defined")

    model = "claude-3-5-sonnet-20240620"
    client = anthropic.Anthropic(api_key=api_key)

    if files:
        content = create_content_list(files)
        content.append({"type": "text", "text": prompt})
        log.vvv(f"Claude prompt with {len(files)} files: {prompt}")
    else:
        content = prompt
        log.vvv(f"Claude prompt: {prompt}")

    messages = [{"role": "user", "content": content}]

    response = client.messages.create(
        model=model,
        messages=messages,
        system=system_prompt,
        max_tokens=4096,
        stream=False,
        temperature=0.9,
    )
    output = response.content[0].text
    log.vvv(f"Claude response: {output}")
    return output


def create_content_list(files: list[Path]):
    content = []
    for path in files:
        with path.open("rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

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
