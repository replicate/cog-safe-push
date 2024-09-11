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


class PredictionFailedError(CogSafePushError):
    pass


class TestCaseFailedError(CogSafePushError):
    pass


class AIError(Exception):
    pass
