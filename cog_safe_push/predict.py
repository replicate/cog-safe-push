import asyncio
import json
import time
from typing import Any, cast

import httpx
import replicate
from replicate.exceptions import ReplicateError
from replicate.model import Model
from replicate.run import _has_output_iterator_array_type

from . import ai, log
from .exceptions import (
    AIError,
    PredictionTimeoutError,
)
from .utils import truncate


async def make_fuzz_system_prompt() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://multimedia-example-files.replicate.dev/index.txt"
        )
    multimedia_example_files = response.text
    return (
        """# Replicate model fuzzing inputs

Your task is to generate inputs for model fuzzing of a Replicate model.

Given a model input JSON schema, return a valid JSON payload for this model.

## Example

For example,

{
  "Input": {
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
    "title": "Input",
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

A valid JSON payload for that input schema would be:

{
  "my_bool": true,
  "my_choice": "foo",
  "my_constrained_int": 9,
  "my_float": 3.14,
  "my_int": 10,
  "text": "world",
}

The following is NOT a valid JSON payload:

{
  "my_bool": true,
  "my_choice": "foo",
  "my_constrained_int": 11,
  "my_float": 3.14,
  "my_int": 10,
  "text": "world",
}

...because my_constrained_int is greater than the maximum in the schema.

## Respect constraints

Be careful to respect constraints. For example:
* If there is a "maximum" or "minimum" constraint on a number input, your generated input value must not be below the minimum or above the maximum
* If there is an allOf constraint, your input values must be one of the valid enumeration values
* If the description of an input describes constraints, your generated input must respect those constraints
* etc.

## Multimedia file inputs

If an input have format=uri and you decide to populate that input, you should use one of the media URLs from the Multimedia example files section below.

Make sure you pick an appropriate URL for the the input, e.g. pick one of the image examples below if the input expects an image. Also make sure you respect any hints or documentation about file types.

IMPORTANT: DON'T USE SVG FILES UNLESS THE MODEL EXPLICITLY MENTINOS SVG SUPPORT!!! (most don't, but some svg-to-image, or vectorization models do).

"""
        + multimedia_example_files
    )


async def make_fuzz_inputs(
    schemas: dict,
    train: bool,
    only_required: bool,
    seed: int | None,
    fixed_inputs: dict[str, Any],
    disabled_inputs: list[str],
    fuzz_prompt: str | None,
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
    prompt = f"""Given the following OpenAPI schemas:

{schemas_str}

Generate a valid JSON payload for the {input_name} schema.

"""

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

Return a new combination of inputs that you haven't used before, ideally that's quite diverse from inputs you've used before -- but still make sure you respect the constraints in the input schema (respecting those constraints is very important!). You have previously used these inputs:
{inputs_history_str}"""

    if fuzz_prompt:
        prompt += f"""

# Additional instructions

You must follow these instructions: {fuzz_prompt}"""

    system_prompt = await make_fuzz_system_prompt()
    inputs = await ai.json_object(prompt, system_prompt=system_prompt, thinking=True)
    if set(required) - set(inputs.keys()):
        max_attempts = 5
        if attempt == max_attempts:
            raise AIError(
                f"Failed to generate a json payload with the correct keys after {max_attempts} attempts, giving up"
            )
        return await make_fuzz_inputs(
            schemas=schemas,
            train=train,
            only_required=only_required,
            seed=seed,
            fixed_inputs=fixed_inputs,
            disabled_inputs=disabled_inputs,
            fuzz_prompt=fuzz_prompt,
            inputs_history=inputs_history,
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

    # Filter out null values as Replicate API doesn't accept null for optional fields
    inputs = {k: v for k, v in inputs.items() if v is not None}

    return inputs, is_deterministic


async def predict(
    model: Model,
    train: bool,
    train_destination: Model | None,
    inputs: dict,
    timeout_seconds: float,
    prediction_index: int | None = None,
) -> tuple[Any | None, str | None, str]:
    prefix = f"[{prediction_index}] " if prediction_index is not None else ""
    log.vv(
        f"{prefix}Running {'training' if train else 'prediction'} with inputs:\n{json.dumps(inputs, indent=2)}"
    )

    start_time = time.time()
    version = model.versions.list()[0]

    if train:
        assert train_destination
        version_ref = f"{model.owner}/{model.name}:{version.id}"
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
            prediction = replicate.predictions.create(version=version.id, input=inputs)
        except ReplicateError as e:
            if e.status == 404:
                # Assume it's an official model
                prediction = replicate.predictions.create(model=model, input=inputs)
            else:
                raise

    prediction_url = f"https://replicate.com/p/{prediction.id}"
    log.v(f"{prefix}Prediction URL: {prediction_url}")

    while prediction.status not in ["succeeded", "failed", "canceled"]:
        await asyncio.sleep(0.5)
        if time.time() - start_time > timeout_seconds:
            raise PredictionTimeoutError()
        prediction.reload()

    duration = time.time() - start_time

    if prediction.status == "failed":
        log.v(f"{prefix}Got error: {prediction.error}  ({duration:.2f} sec)")
        return None, prediction.error, prediction_url

    output = prediction.output
    if _has_output_iterator_array_type(version):
        output = "".join(cast("list[str]", output))

    log.v(f"{prefix}Got output: {truncate(output)}  ({duration:.2f} sec)")

    return output, None, prediction_url
