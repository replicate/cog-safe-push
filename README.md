# cog-safe-push

Safely push a Cog model version by making sure it works and is backwards-compatible with previous versions.

> [!TIP]
> Check out our [guide to building a CI/CD pipeline for your model](https://replicate.com/docs/guides/continuous-model-deployment), which includes a step-by-step walkthrough of how to use this tool.

## Prerequisites

1. Set the `ANTHROPIC_API_KEY` and `REPLICATE_API_TOKEN` environment variables.
1. Install Cog and `cog login`
1. If you're running this from a cloned source, `pip install .` in the `cog-safe-push` directory.

## Installation

This package is not on PyPI yet, but you can install it directly from GitHub using pip:

```
pip install git+https://github.com/replicate/cog-safe-push.git
```

## Usage

To safely push a model to Replicate, run this inside your Cog directory:

```
$ cog-safe-push --test-hardware=<hardware> <username>/<model-name>
```

This will:
1. Lint the predict file with ruff
1. Create a private test model on Replicate, named `<username>/<model-name>-test` running `<hardware>`
1. Push the local Cog model to the test model on Replicate
1. Lint the model schema (making sure all inputs have descriptions, etc.)
1. If there is an existing version on the upstream `<username>/<model-name>` model, it will
   1. Make sure that the schema in the test version is backwards compatible with the existing upstream version
   1. Run predictions against both upstream and test versions and make sure the same inputs produce the same (or equivalent) outputs
1. Fuzz the test model for five minutes by throwing a bunch of different inputs at it and make sure it doesn't throw any errors

Both the creation of model inputs and comparison of model outputs is handled by Claude.

## Example GitHub Actions workflow

Create a new workflow file in `.github/workflows/cog-safe-push.yaml` and add the following:

```yaml
name: Cog Safe Push

on:
  workflow_dispatch:
    inputs:
      model:
        description: 'The name of the model to push in the format owner/model-name'
        type: string

jobs:
  cog-safe-push:
    # Tip: Create custom runners in your GitHub organization for faster builds
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"

    - name: Install Cog
      run: |
        sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
        sudo chmod +x /usr/local/bin/cog

    - name: cog login
      run: |
        echo ${{ secrets.COG_TOKEN }} | cog login --token-stdin

    - name: Install cog-safe-push
      run: |
        pip install git+https://github.com/replicate/cog-safe-push.git

    - name: Push selected models
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        REPLICATE_API_TOKEN: ${{ secrets.REPLICATE_API_TOKEN }}
      run: |
        cog-safe-push ${{ inputs.model }}
```

After pushing this workflow to the main branch, you can run it manually from the Actions tab.

### Full help text

<!-- generate: cog-safe-push --help -->

```text
# cog-safe-push --help

usage: cog-safe-push [-h] [--config CONFIG] [--help-config]
                     [--test-model TEST_MODEL] [--no-push]
                     [--test-hardware TEST_HARDWARE] [--no-compare-outputs]
                     [--predict-timeout PREDICT_TIMEOUT] [--fast-push]
                     [--test-case TEST_CASES]
                     [--fuzz-fixed-inputs FUZZ_FIXED_INPUTS]
                     [--fuzz-disabled-inputs FUZZ_DISABLED_INPUTS]
                     [--fuzz-iterations FUZZ_ITERATIONS]
                     [--fuzz-prompt FUZZ_PROMPT] [--parallel PARALLEL]
                     [--ignore-schema-compatibility] [-v]
                     [--push-official-model]
                     [model]

Safely push a Cog model, with tests

positional arguments:
  model                 Model in the format <owner>/<model-name>

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the YAML config file. If --config is not
                        passed, ./cog-safe-push.yaml will be used, if it
                        exists. Any arguments you pass in will override fields
                        on the predict configuration stanza.
  --help-config         Print a default cog-safe-push.yaml config to stdout.
  --test-model TEST_MODEL
                        Replicate model to test on, in the format
                        <username>/<model-name>. If omitted, <model>-test will
                        be used. The test model is created automatically if it
                        doesn't exist already
  --no-push             Only test the model, don't push it to <model>
  --test-hardware TEST_HARDWARE
                        Hardware to run the test model on. Only used when
                        creating the test model, if it doesn't already exist.
  --no-compare-outputs  Don't make predictions to compare that prediction
                        outputs match the current version
  --predict-timeout PREDICT_TIMEOUT
                        Timeout (in seconds) for predictions. Default: 300
  --fast-push           Use the --x-fast flag when doing cog push
  --test-case TEST_CASES
                        Inputs and expected output that will be used for
                        testing, you can provide multiple --test-case options
                        for multiple test cases. The first test case will be
                        used when comparing outputs to the current version.
                        Each --test-case is semicolon-separated key-value
                        pairs in the format
                        '<key1>=<value1>;<key2=value2>[<output-checker>]'.
                        <output-checker> can either be '==<exact-string-or-
                        url>' or '~=<ai-prompt>'. If you use '==<exact-string-
                        or-url>' then the output of the model must match
                        exactly the string or url you specify. If you use
                        '~=<ai-prompt>' then the AI will verify your output
                        based on <ai-prompt>. If you omit <output-checker>, it
                        will just verify that the prediction doesn't throw an
                        error.
  --fuzz-fixed-inputs FUZZ_FIXED_INPUTS
                        Inputs that should have fixed values during fuzzing.
                        All other non-disabled input values will be generated
                        by AI. If no test cases are specified, these will also
                        be used when comparing outputs to the current version.
                        Semicolon-separated key-value pairs in the format
                        '<key1>=<value1>;<key2=value2>' (etc.)
  --fuzz-disabled-inputs FUZZ_DISABLED_INPUTS
                        Don't pass values for these inputs during fuzzing.
                        Semicolon-separated keys in the format '<key1>;<key2>'
                        (etc.). If no test cases are specified, these will
                        also be disabled when comparing outputs to the current
                        version.
  --fuzz-iterations FUZZ_ITERATIONS
                        Maximum number of iterations to run fuzzing.
  --fuzz-prompt FUZZ_PROMPT
                        Additional prompting for the fuzz input generation
  --parallel PARALLEL   Number of parallel prediction threads.
  --ignore-schema-compatibility
                        Ignore schema compatibility checks when pushing the
                        model
  -v, --verbose         Increase verbosity level (max 3)
  --push-official-model
                        Push to the official model defined in the config
```

### Using a configuration file

You can use a configuration file instead of passing all arguments on the command line. If you create a file called `cog-safe-push.yaml` in your Cog directory, it will be used. Any command line arguments you pass will override the values in the config file.

<!-- generate: cog-safe-push --help-config -->

```yaml
# cog-safe-push --help-config

model: <model>
test_model: <test model, or empty to append '-test' to model>
test_hardware: <hardware, e.g. cpu>
predict:
  compare_outputs: true
  predict_timeout: 300
  test_cases:
  - inputs:
      <input1>: <value1>
    exact_string: <exact string match>
  - inputs:
      <input2>: <value2>
    match_url: <match output image against url>
  - inputs:
      <input3>: <value3>
    match_prompt: <match output using AI prompt, e.g. 'an image of a cat'>
  - inputs:
      <input4>: <value4>
    jq_query: <jq query to validate JSON output, e.g. ".status == \"success\" and
      .confidence > 0.8">
  - inputs:
      <input5>: <value5>
    error_contains: <assert that these inputs throws an error, and that the error
      message contains a string>
  fuzz:
    fixed_inputs: {}
    disabled_inputs: []
    iterations: 10
    prompt: <additional prompt for the language model when selecting fuzz inputs>
train:
  destination: <generated prediction model, e.g. andreasjansson/test-predict. leave
    blank to append '-dest' to the test model>
  destination_hardware: <hardware for the created prediction model, e.g. cpu>
  train_timeout: 300
  test_cases:
  - inputs:
      <input1>: <value1>
    exact_string: <exact string match>
  - inputs:
      <input2>: <value2>
    match_url: <match output image against url>
  - inputs:
      <input3>: <value3>
    match_prompt: <match output using AI prompt, e.g. 'an image of a cat'>
  - inputs:
      <input4>: <value4>
    jq_query: <jq query to validate JSON output, e.g. ".status == \"success\" and
      .confidence > 0.8">
  - inputs:
      <input5>: <value5>
    error_contains: <assert that these inputs throws an error, and that the error
      message contains a string>
  fuzz:
    fixed_inputs: {}
    disabled_inputs: []
    iterations: 10
    prompt: <additional prompt for the language model when selecting fuzz inputs>
deployment:
  owner: <owner>
  name: <name>
  hardware: <hardware>
parallel: 4
fast_push: false
use_cog_base_image: true
ignore_schema_compatibility: false
official_model: <official model, e.g. user/model>

# values between < and > should be edited
```

## Deployments

The tool can automatically create or update deployments for your model on Replicate. To use this feature:

1. Add deployment settings to your `cog.yaml`:

```yaml
deployment:
  name: my-model-deployment
  owner: your-username  # optional, defaults to model owner
  hardware: cpu  # or gpu-t4, gpu-a100, etc.
```

2. When you run `cog-safe-push`, it will:
   - Create a new deployment if one doesn't exist
   - Update the existing deployment with the new version if it does exist
   - Use appropriate instance scaling based on hardware:
     - CPU: 1-20 instances
     - GPU: 0-2 instances

The deployment will be created under the specified owner (or model owner if not specified) and will use the hardware configuration you provide.

## Nota bene

* This is alpha software. If you find a bug, please open an issue!
