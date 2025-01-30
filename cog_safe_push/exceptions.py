class CogSafePushError(Exception):
    pass


class ArgumentError(CogSafePushError):
    pass


class CodeLintError(CogSafePushError):
    pass


class SchemaLintError(CogSafePushError):
    pass


class IncompatibleSchemaError(CogSafePushError):
    pass


class OutputsDontMatchError(CogSafePushError):
    pass


class FuzzError(CogSafePushError):
    pass


class PredictionTimeoutError(CogSafePushError):
    pass


class TestCaseFailedError(CogSafePushError):
    __test__ = False

    def __init__(self, message):
        super().__init__(f"Test case failed: {message}")


class AIError(Exception):
    pass
