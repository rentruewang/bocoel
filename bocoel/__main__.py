import logging
import os

import fire
import structlog

from . import main


def logging_level() -> int:
    # The default level should be info.
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")

    try:
        return int(LOGGING_LEVEL)
    except:
        # Match the text to the supported logging levels below.
        pass

    match LOGGING_LEVEL.upper():
        case "DEBUG":
            return logging.DEBUG
        case "INFO":
            return logging.INFO
        case "WARNING":
            return logging.WARNING
        case "ERROR":
            return logging.ERROR
        case "CRITICAL":
            return logging.CRITICAL
        case _:
            raise ValueError(f"Unknown logging level {os.environ['LOGGING_LEVEL']}")


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging_level()),
    )

    # Not a class. Google's standard public functions are all capitalized.
    fire.Fire(main)
