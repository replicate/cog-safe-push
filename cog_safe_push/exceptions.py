class CodeLintError(Exception):
    pass


class SchemaLintError(Exception):
    pass


class IncompatibleSchema(Exception):
    pass


class OutputsDontMatch(Exception):
    pass


class FuzzError(Exception):
    pass


class PredictionTimeout(Exception):
    pass


class PredictionFailed(Exception):
    pass

class AIError(Exception):
    pass
