import replicate

from .exceptions import IncompatibleSchema, SchemaLintError


def lint(model: replicate.model.Model):
    errors = []

    schema = model.versions.list()[0].openapi_schema
    properties = schema["components"]["schemas"]["Input"]["properties"]
    for name, spec in properties.items():
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


def check_backwards_compatible(test_model_schemas: dict, model_schemas: dict):
    test_inputs = test_model_schemas["Input"]
    inputs = model_schemas["Input"]

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
                errors.append(f"Input {name} has changed type from {input_type} to {test_input_type}")
                continue

            if "minimum" in test_spec and "minimum" not in spec:
                errors.append(f"Input {name} has added a minimum constraint")
            elif "minimum" in test_spec and "minimum" in spec:
                if test_spec["minimum"] > spec["minimum"]:
                    errors.append(f"Input {name} has a higher minimum")

            if "maximum" in test_spec and "maximum" not in spec:
                errors.append(f"Input {name} has added a maximum constraint")
            elif "maximum" in test_spec and "maximum" in spec:
                if test_spec["maximum"] < spec["maximum"]:
                    errors.append(f"Input {name} has a lower maximum")

            if test_spec.get("format", "") != spec.get("format", ""):
                errors.append(f"Input {name} has changed format")

            # We allow defaults to be changed

        elif "allOf" in spec:
            choice_schema = model_schemas[spec["allOf"][0]["$ref"].split("/")[-1]]
            test_choice_schema = test_model_schemas[spec["allOf"][0]["$ref"].split("/")[-1]]
            choice_type = choice_schema["type"]
            test_choice_type = test_choice_schema["type"]
            if test_choice_type != choice_type:
                errors.append(f"Input {name} choices has changed type from {choice_type} to {test_choice_type}")
                continue
            choices = set(choice_schema["enum"])
            test_choices = set(test_choice_schema["enum"])
            missing_choices = choices - test_choices
            if missing_choices:
                missing_choices_str = ', '.join([f"'{c}'" for c in missing_choices])
                errors.append(f"Input {name} is missing choices: {missing_choices_str}")

    for name, spec in test_inputs.items():
        if name not in inputs and "default" not in spec:
            errors.append(f"Input {name} is new and is required")

    output_schema = model_schemas["Output"]
    test_output_schema = test_model_schemas["Output"]

    if test_output_schema["type"] != output_schema["type"]:
        errors.append("Output has changed type")

    if errors:
        raise IncompatibleSchema(
            "Schema is not backwards compatible: \n" + "\n".join(["* " + e for e in errors])
        )


def get_schemas(model):
    schemas = model.versions.list()[0].openapi_schema["components"]["schemas"]
    for unnecessary_key in [
        "WebhookEvent",
        "HTTPValidationError",
        "PredictionRequest",
        "Status",
        "ValidationError",
    ]:
        if unnecessary_key in schemas:
            del schemas[unnecessary_key]
    return schemas
