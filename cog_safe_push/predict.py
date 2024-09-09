import json
import math
import random
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import replicate
import requests
from PIL import Image
from replicate.model import Model

from . import ai, log, schema
from .exceptions import (
    AIError,
    FuzzError,
    OutputsDontMatchError,
    PredictionFailedError,
    PredictionTimeoutError,
)


@dataclass
class SpecialInputValue:
    type: str


InputValueType = bool | int | float | str | SpecialInputValue

OMITTED_INPUT = SpecialInputValue("omit")


@dataclass
class WeightedInputValue:
    value: InputValueType
    weight_percent: float


def check_outputs_match(
    test_model: Model,
    model: Model,
    train: bool,
    train_destination: Model | None,
    timeout_seconds: float,
    inputs: dict[str, list[WeightedInputValue]],
    disabled_inputs: list[str],
):
    schemas = schema.get_schemas(model, train=train)
    predict_inputs, is_deterministic = make_predict_inputs(
        schemas,
        train=train,
        only_required=True,
        seed=1,
        fixed_inputs=first_input_per_key(inputs),
        disabled_inputs=disabled_inputs,
    )
    test_output = predict(
        model=test_model,
        train=train,
        train_destination=train_destination,
        inputs=predict_inputs,
        timeout_seconds=timeout_seconds,
    )
    output = predict(
        model=model,
        train=train,
        train_destination=train_destination,
        inputs=predict_inputs,
        timeout_seconds=timeout_seconds,
    )
    matches, error = outputs_match(test_output, output, is_deterministic)
    if not matches:
        raise OutputsDontMatchError(
            f"Outputs don't match:\n\ntest output:\n{test_output}\n\nmodel output:\n{output}\n\n{error}"
        )


def fuzz_model(
    model: Model,
    train: bool,
    train_destination: Model | None,
    timeout_seconds: float,
    max_iterations: int | None,
    inputs: dict[str, list[WeightedInputValue]],
    disabled_inputs: list[str],
):
    start_time = time.time()
    inputs_history = []
    successful_predictions = 0
    while True:
        schemas = schema.get_schemas(model, train=train)
        predict_inputs, _ = make_predict_inputs(
            schemas,
            train=train,
            only_required=False,
            seed=None,
            fixed_inputs=sample_inputs(inputs),
            disabled_inputs=disabled_inputs,
            inputs_history=inputs_history,
        )
        inputs_history.append(predict_inputs)
        predict_timeout = start_time + timeout_seconds - time.time()
        try:
            output = predict(
                model=model,
                train=train,
                train_destination=train_destination,
                inputs=predict_inputs,
                timeout_seconds=predict_timeout,
            )
        except PredictionTimeoutError:
            if not successful_predictions:
                log.warning(
                    f"No predictions succeeded in {timeout_seconds}, try increasing --fuzz-seconds"
                )
            return
        except PredictionFailedError as e:
            raise FuzzError(e)
        if not output:
            raise FuzzError("No output")
        successful_predictions += 1
        if max_iterations is not None and successful_predictions == max_iterations:
            return


def first_input_per_key(inputs: dict[str, list[WeightedInputValue]]) -> dict[str, Any]:
    return {key: values[0].value for key, values in inputs.items()}


def sample_inputs(inputs: dict[str, list[WeightedInputValue]]) -> dict[str, Any]:
    sampled_inputs = {}
    for key, values in inputs.items():
        total_weight = sum(v.weight_percent for v in values)
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        for value in values:
            cumulative_weight += value.weight_percent
            if random_value <= cumulative_weight:
                sampled_inputs[key] = value.value
                break
    return sampled_inputs


def make_predict_inputs(
    schemas: dict,
    train: bool,
    only_required: bool,
    seed: int | None,
    fixed_inputs: dict[str, Any],
    disabled_inputs: list[str],
    inputs_history: list[dict] | None = None,
    attempt=0,
) -> tuple[dict, bool]:
    input_name = "TrainingInput" if train else "Input"
    input_schema = schemas[input_name]
    properties = input_schema["properties"]
    required = input_schema.get("required", [])

    is_deterministic = False
    if "seed" in properties and seed is not None:
        is_deterministic = True
        del properties["seed"]

    omitted_inputs = [k for k, v in fixed_inputs.items() if v == OMITTED_INPUT]
    disabled_inputs = disabled_inputs + omitted_inputs
    fixed_inputs = {k: v for k, v in fixed_inputs.items() if k not in disabled_inputs}

    schemas_str = json.dumps(schemas, indent=2)
    prompt = (
        '''
Below is an example of an OpenAPI schema for a Cog model:

{
  "'''
        + input_name
        + '''": {
    "properties": {
      "my_bool": {
        "description": "A bool.",
        "title": "My Bool",
        "type": "boolean",
        "x-order": 3
      },
      "my_choice": {
        "allOf": [
          {
            "$ref": "#/components/schemas/my_choice"
          }
        ],
        "description": "A choice.",
        "x-order": 4
      },
      "my_constrained_int": {
        "description": "A constrained integer.",
        "maximum": 10,
        "minimum": 2,
        "title": "My Constrained Int",
        "type": "integer",
        "x-order": 5
      },
      "my_float": {
        "description": "A float.",
        "title": "My Float",
        "type": "number",
        "x-order": 2
      },
      "my_int": {
        "description": "An integer.",
        "title": "My Int",
        "type": "integer",
        "x-order": 1
      },
      "text": {
        "description": "Text that will be prepended by 'hello '.",
        "title": "Text",
        "type": "string",
        "x-order": 0
      }
    },
    "required": [
      "text",
      "my_int",
      "my_float",
      "my_bool",
      "my_choice",
      "my_constrained_int"
    ],
    "title": "'''
        + input_name
        + """",
    "type": "object"
  },
  "my_choice": {
    "description": "An enumeration.",
    "enum": [
      "foo",
      "bar",
      "baz"
    ],
    "title": "my_choice",
    "type": "string"
  }
}

A valid json payload for that input schema would be:

{
  "my_bool": true,
  "my_choice": "foo",
  "my_constrained_int": 9,
  "my_float": 3.14,
  "my_int": 10,
  "text": "world",
}

"""
        + f"""
Now, given the following OpenAPI schemas:

{schemas_str}

Generate a json payload for the {input_name} schema.

If inputs have format=uri, you should use one of the following media URLs:
Videos:
* https://storage.googleapis.com/cog-safe-push-public/harry-truman.webm
* https://storage.googleapis.com/cog-safe-push-public/mariner-launch.ogv
Images:
* https://storage.googleapis.com/cog-safe-push-public/skull.jpg
* https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg
* https://storage.googleapis.com/cog-safe-push-public/rolling-stones.jpg
* https://storage.googleapis.com/cog-safe-push-public/forest.png
* https://storage.googleapis.com/cog-safe-push-public/face.gif
Music audio:
* https://storage.googleapis.com/cog-safe-push-public/folk-music.mp3
* https://storage.googleapis.com/cog-safe-push-public/ocarina.ogg
* https://storage.googleapis.com/cog-safe-push-public/nu-style-kick.wav
Test audio:
* https://storage.googleapis.com/cog-safe-push-public/clap.ogg
* https://storage.googleapis.com/cog-safe-push-public/beeps.mp3
Long speech:
* https://storage.googleapis.com/cog-safe-push-public/chekhov-article.ogg
* https://storage.googleapis.com/cog-safe-push-public/momentos-spanish.ogg
Short speech:
* https://storage.googleapis.com/cog-safe-push-public/de-experiment-german-word.ogg
* https://storage.googleapis.com/cog-safe-push-public/de-ionendosis-german-word.ogg

    """
    )

    if fixed_inputs:
        fixed_inputs_str = json.dumps(fixed_inputs)
        prompt += f"The following key/values must be present in the payload if they exist in the schema: {fixed_inputs_str}\n"

    if disabled_inputs:
        disabled_inputs_str = json.dumps(disabled_inputs)
        prompt += f"The following keys must not be present in the payload: {disabled_inputs_str}\n"

    required_keys_str = ", ".join(required)
    if only_required:
        prompt += f"Only include the following required keys: {required_keys_str}"
    else:
        prompt += f"Include the following required keys (and preferably some optional keys too): {required_keys_str}"

    if inputs_history:
        inputs_history_str = "\n".join(["* " + json.dumps(i) for i in inputs_history])
        prompt += f"""

Return a new combination of inputs that you haven't used before. You have previously used these inputs:
{inputs_history_str}"""

    inputs = ai.json_object(prompt)
    if set(required) - set(inputs.keys()):
        max_attempts = 5
        if attempt == max_attempts:
            raise AIError(
                f"Failed to generate a json payload with the correct keys after {max_attempts} attempts, giving up"
            )
        return make_predict_inputs(
            schemas=schemas,
            train=train,
            only_required=only_required,
            seed=seed,
            fixed_inputs=fixed_inputs,
            disabled_inputs=disabled_inputs,
            attempt=attempt + 1,
        )

    if is_deterministic:
        inputs["seed"] = seed

    if fixed_inputs:
        for key, value in fixed_inputs.items():
            inputs[key] = value

    if disabled_inputs:
        for key in disabled_inputs:
            if key in inputs:
                del inputs[key]

    return inputs, is_deterministic


def predict(
    model: Model,
    train: bool,
    train_destination: Model | None,
    inputs: dict,
    timeout_seconds: float,
):
    log.vv(
        f"Running {'training' if train else 'prediction'} with inputs:\n{json.dumps(inputs, indent=2)}"
    )

    if train:
        assert train_destination
        version_ref = f"{model.owner}/{model.name}:{model.versions.list()[0].id}"
        prediction = replicate.trainings.create(
            version=version_ref,
            input=inputs,
            destination=f"{train_destination.owner}/{train_destination.name}",
        )
    else:
        prediction = replicate.predictions.create(
            version=model.versions.list()[0].id, input=inputs
        )

    start_time = time.time()
    while prediction.status not in ["succeeded", "failed", "canceled"]:
        time.sleep(0.5)
        if time.time() - start_time > timeout_seconds:
            raise PredictionTimeoutError()
        prediction.reload()

    if prediction.status == "failed":
        raise PredictionFailedError(prediction.error)

    log.vv(f"Got output: {truncate(prediction.output)}")

    return prediction.output


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
        return False, "Integers aren't identical"

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
            diff = math.sqrt(
                sum(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))
                    for p1, p2 in zip(img1.getdata(), img2.getdata())  # pyright: ignore
                )
            )

            if diff > 2:  # arbitrary epsilon
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
        tmp_path.unlink(missing_ok=True)


def truncate(s, max_length=500) -> str:
    s = str(s)
    if len(s) <= max_length:
        return s
    return s[:max_length] + "..."
