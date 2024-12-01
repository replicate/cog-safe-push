import asyncio
import json
import time
from typing import Any

import replicate
from replicate.exceptions import ReplicateError
from replicate.model import Model

from . import ai, log
from .exceptions import (
    AIError,
    PredictionFailedError,
    PredictionTimeoutError,
)


async def make_predict_inputs(
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

If inputs have format=uri, you should use one of the following media URLs (pick an appropriate URL for the the input, e.g. one of the image examples below if the input expects an image):
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

If the schema has default values for some of the inputs, feel free to either use the defaults or come up with new values.

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

Return a new combination of inputs that you haven't used before, ideally that's quite diverse from inputs you've used before. You have previously used these inputs:
{inputs_history_str}"""

    inputs = await ai.json_object(prompt)
    if set(required) - set(inputs.keys()):
        max_attempts = 5
        if attempt == max_attempts:
            raise AIError(
                f"Failed to generate a json payload with the correct keys after {max_attempts} attempts, giving up"
            )
        return await make_predict_inputs(
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


async def predict(
    model: Model,
    train: bool,
    train_destination: Model | None,
    inputs: dict,
    timeout_seconds: float,
):
    log.vv(
        f"Running {'training' if train else 'prediction'} with inputs:\n{json.dumps(inputs, indent=2)}"
    )

    start_time = time.time()

    if train:
        assert train_destination
        version_ref = f"{model.owner}/{model.name}:{model.versions.list()[0].id}"
        prediction = replicate.trainings.create(
            version=version_ref,
            input=inputs,
            destination=f"{train_destination.owner}/{train_destination.name}",
        )
    else:
        try:
            # await async_create doesn't seem to work here, throws
            # RuntimeError: Event loop is closed
            # But since we're async sleeping this should only block
            # a very short time
            prediction = replicate.predictions.create(
                version=model.versions.list()[0].id, input=inputs
            )
        except ReplicateError as e:
            if e.status == 404:
                # Assume it's an official model
                prediction = replicate.predictions.create(
                    model=model, input=inputs
                )
            else:
                raise

    log.vv(f"Prediction URL: https://replicate.com/p/{prediction.id}")

    while prediction.status not in ["succeeded", "failed", "canceled"]:
        await asyncio.sleep(0.5)
        if time.time() - start_time > timeout_seconds:
            raise PredictionTimeoutError()
        prediction.reload()

    if prediction.status == "failed":
        raise PredictionFailedError(prediction.error)

    duration = time.time() - start_time
    log.vv(f"Got output: {truncate(prediction.output)}  ({duration:.2f} sec)")

    return prediction.output


def truncate(s, max_length=500) -> str:
    s = str(s)
    if len(s) <= max_length:
        return s
    return s[:max_length] + "..."
