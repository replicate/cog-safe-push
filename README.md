# cog-safe-push

Safely push a Cog model version by making sure it works and is backwards-compatible with previous versions.

## Prerequisites

1. Set the `ANTHROPIC_API_KEY` and `REPLICATE_API_TOKEN` environment variables.
1. Install Cog and `cog login`
1. If you're running this from a cloned source, `pip install .` in the `cog-safe-push` directory.

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

### Full help text

```
usage: cog-safe-push [-h] [--test-hardware TEST_HARDWARE]
                     [--test-model TEST_MODEL] [--test-only]
                     [-i INPUTS] [-x DISABLED_INPUTS]
                     [--no-compare-outputs]
                     [--fuzz-seconds FUZZ_SECONDS]
                     [--no-fuzz-user-inputs] [-v]
                     model

Safely push a Cog model, with tests

positional arguments:
  model                 Model in the format <owner>/<model-name>

options:
  -h, --help            show this help message and exit
  --test-hardware TEST_HARDWARE
                        Hardware to run the test model on. Only
                        used when creating the test model, if it
                        doesn't already exist.
  --test-model TEST_MODEL
                        Replicate model to test on, in the format
                        <username>/<model-name>. If omitted,
                        <model>-test will be used. The test model
                        is created automatically if it doesn't
                        exist already
  --test-only           Only test the model, don't push it to
                        <model>
  -i INPUTS, --input INPUTS
                        Input key-value pairs in the format
                        <key>=<value>. These will be used when
                        comparing outputs, as well as during
                        fuzzing (unless --no-fuzz-user-inputs is
                        specified)
  -x DISABLED_INPUTS, --disable-input DISABLED_INPUTS
                        Don't pass values to these inputs when
                        comparing outputs or fuzzing
  --no-compare-outputs  Don't make predictions to compare that
                        prediction outputs match
  --fuzz-seconds FUZZ_SECONDS
                        Number of seconds to run fuzzing. Set to
                        0 for no fuzzing
  --no-fuzz-user-inputs
                        Don't use -i/--input values when fuzzing,
                        instead use AI-generated values for every
                        input
  -v, --verbose         Increase verbosity level (max 3)

```

## Nota bene

* If you can't figure out the right name to use for `--test-hardware`, create the test model manually (setting hardware in the UI), leave `--test-hardware` blank, and set `--test-model=<test-username>/<test-model-name>` instead
* This is alpha software. If you find a bug, please open an issue!
