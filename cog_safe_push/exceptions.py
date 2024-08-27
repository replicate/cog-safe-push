class CodeLintError(Exception):
    pass


class SchemaLintError(Exception):
    pass


class IncompatibleSchemaError(Exception):
    pass


class OutputsDontMatchError(Exception):
    pass


class FuzzError(Exception):
    pass


class PredictionTimeoutError(Exception):
    pass


class PredictionFailedError(Exception):
    pass


class AIError(Exception):
    pass
