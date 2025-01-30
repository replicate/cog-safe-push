import sys

ERROR = 6
WARNING = 5
INFO = 4
VERBOSE1 = 3
VERBOSE2 = 2
VERBOSE3 = 1

level = INFO


def error(message):
    if level <= ERROR:
        print(message, file=sys.stderr)


def warning(message):
    if level <= WARNING:
        print(message, file=sys.stderr)


def info(message):
    if level <= INFO:
        print(message, file=sys.stderr)


def v(message):
    if level <= VERBOSE1:
        print(message, file=sys.stderr)


def vv(message):
    if level <= VERBOSE2:
        print(message, file=sys.stderr)


def vvv(message):
    if level <= VERBOSE3:
        print(message, file=sys.stderr)


def set_verbosity(verbosity):
    global level
    if verbosity <= 0:
        level = INFO
    if verbosity == 1:
        level = VERBOSE1
    if verbosity == 2:
        level = VERBOSE2
    if verbosity >= 3:
        level = VERBOSE3
