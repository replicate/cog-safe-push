import copy

from replicate.exceptions import ReplicateError
from replicate.model import Model

from .exceptions import IncompatibleSchemaError, SchemaLintError


def lint(model: Model, train: bool):
    errors = []

    input_name = "TrainingInput" if train else "Input"
    schema = get_openapi_schema(model)
    properties = schema["components"]["schemas"][input_name]["properties"]
    for name, spec in properties.items():
        if spec.get("deprecated", False):
            continue
        description = spec.get("description")
        if not description:
            errors.append(f"{name}: Missing description")
            continue
        # if not description[0].isupper():
        #     errors.append(f"{name}: Description doesn't start with a capital letter")
        # if not description.endswith(('.', '?', '!')):
        #     errors.append(f"{name}: Description doesn't end with a period, question mark, or exclamation mark")

    if errors:
        raise SchemaLintError(
            "Schema failed linting: \n" + "\n".join(["* " + e for e in errors])
        )


def check_backwards_compatible(
    test_model_schemas: dict, model_schemas: dict, train: bool
):
    input_name = "TrainingInput" if train else "Input"
    output_name = "TrainingOutput" if train else "Output"

    test_inputs = test_model_schemas[input_name]
    inputs = model_schemas[input_name]

    errors = []
    for name, spec in inputs.items():
        if name not in test_inputs:
            errors.append(f"Missing input {name}")
            continue
        test_spec = test_inputs[name]
        if "type" in spec:
            input_type = spec["type"]
            test_input_type = test_spec.get("type")
            if input_type != test_input_type:
                errors.append(
                    f"{input_name} {name} has changed type from {input_type} to {test_input_type}"
                )
                continue

            if "minimum" in test_spec and "minimum" not in spec:
                errors.append(f"{input_name} {name} has added a minimum constraint")
            elif "minimum" in test_spec and "minimum" in spec:
                if test_spec["minimum"] > spec["minimum"]:
                    errors.append(f"{input_name} {name} has a higher minimum")

            if "maximum" in test_spec and "maximum" not in spec:
                errors.append(f"{input_name} {name} has added a maximum constraint")
            elif "maximum" in test_spec and "maximum" in spec:
                if test_spec["maximum"] < spec["maximum"]:
                    errors.append(f"{input_name} {name} has a lower maximum")

            if test_spec.get("format", "") != spec.get("format", ""):
                errors.append(f"{input_name} {name} has changed format")

            # We allow defaults to be changed

        elif "allOf" in spec:
            choice_schema = model_schemas[spec["allOf"][0]["$ref"].split("/")[-1]]
            test_choice_schema = test_model_schemas[
                spec["allOf"][0]["$ref"].split("/")[-1]
            ]
            choice_type = choice_schema["type"]
            test_choice_type = test_choice_schema["type"]
            if test_choice_type != choice_type:
                errors.append(
                    f"{input_name} {name} choices has changed type from {choice_type} to {test_choice_type}"
                )
                continue
            choices = set(choice_schema["enum"])
            test_choices = set(test_choice_schema["enum"])
            missing_choices = choices - test_choices
            if missing_choices:
                missing_choices_str = ", ".join([f"'{c}'" for c in missing_choices])
                errors.append(
                    f"{input_name} {name} is missing choices: {missing_choices_str}"
                )

    for name, spec in test_inputs.items():
        if name not in inputs and "default" not in spec:
            errors.append(f"{input_name} {name} is new and is required")

    output_schema = model_schemas[output_name]
    test_output_schema = test_model_schemas[output_name]

    if "type" not in test_output_schema:
        errors.append(f"'type' is not in test_output_schema: {test_output_schema}")

    if "type" not in output_schema:
        errors.append(f"'type' is not in output_schema: {output_schema}")

    if test_output_schema.get("type") != output_schema.get("type"):
        errors.append(f"{output_name} has changed type")

    if errors:
        raise IncompatibleSchemaError(
            "Schema is not backwards compatible: \n"
            + "\n".join(["* " + e for e in errors])
        )


def get_openapi_schema(model: Model) -> dict:
    try:
        schema = model.versions.list()[0].openapi_schema
    except ReplicateError as e:
        if e.status == 404:
            # Assume it's an official model
            assert model.latest_version
            schema = model.latest_version.openapi_schema
        else:
            raise

    return dereference_schema(schema)


def dereference_schema(schema: dict) -> dict:
    """
    Dereference a JSON schema by resolving all $ref references.
    """
    result = copy.deepcopy(schema)

    def resolve_ref(ref_path: str, root: dict) -> dict:
        parts = ref_path.lstrip("#/").split("/")
        current = root
        for part in parts:
            current = current[part]
        return current

    def dereference_object(obj, root):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                resolved = resolve_ref(ref_path, root)
                return dereference_object(resolved, root)
            return {k: dereference_object(v, root) for k, v in obj.items()}
        if isinstance(obj, list):
            return [dereference_object(item, root) for item in obj]
        return obj

    dereferenced = dereference_object(result, result)
    assert isinstance(dereferenced, dict)
    return dereferenced


def get_schemas(model, train: bool) -> dict:
    schemas = get_openapi_schema(model)["components"]["schemas"]
    unnecessary_keys = [
        "HTTPValidationError",
        "PredictionRequest",
        "PredictionResponse",
        "Status",
        "TrainingRequest",
        "TrainingResponse",
        "ValidationError",
        "WebhookEvent",
    ]

    if train:
        unnecessary_keys += ["Input", "Output"]
    else:
        unnecessary_keys += ["TrainingInput", "TrainingOutput"]

    for unnecessary_key in unnecessary_keys:
        if unnecessary_key in schemas:
            del schemas[unnecessary_key]
    return schemas
