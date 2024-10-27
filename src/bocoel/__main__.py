import logging
import os
from logging.config import dictConfig

import structlog


def logging_level() -> int:
    """
    Determines and returns the logging level as an integer based on the
    LOGGING_LEVEL environment variable. If the variable is not set or invalid,
    defaults to INFO level.
    """
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")

    try:
        return int(LOGGING_LEVEL)
    except ValueError:
        # Maps text values to standard logging levels if a string is provided.
        pass

    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(LOGGING_LEVEL.upper(), logging.INFO)


def configure_logging():
    """
    Configures logging settings for both standard logging and structlog.
    
    - Allows output to console and/or a log file, depending on environment variables.
    - Uses JSON formatting for structlog if logging to a file, to support structured logs.
    """
    log_file = os.environ.get("LOG_FILE", "app.log")
    use_file_logging = os.environ.get("USE_FILE_LOGGING", "false").lower() == "true"

    # Define a logging configuration that supports console and optional file logging
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
            "structlog": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": [
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.add_log_level,
                ],
            },
        },
        "handlers": {
            "console": {
                "level": logging_level(),
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
            "file": {
                "level": logging_level(),
                "class": "logging.FileHandler",
                "formatter": "structlog",
                "filename": log_file,
                "mode": "a",
            } if use_file_logging else None,
        },
        "root": {
            "level": logging_level(),
            "handlers": ["console"] + (["file"] if use_file_logging else []),
        },
    })

    # Configure structlog with additional processors for context and JSON output
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),  # Adds timestamp to each log
            structlog.processors.add_log_level,           # Adds log level to each entry
            structlog.processors.StackInfoRenderer(),     # Includes stack info for debugging
            structlog.processors.format_exc_info,         # Formats exceptions if present
            structlog.stdlib.filter_by_level,             # Filters by the configured logging level
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.JSONRenderer(),          # Renders log entries in JSON format
        ],
        context_class=dict,
        wrapper_class=structlog.make_filtering_bound_logger(logging_level()),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


if __name__ == "__main__":
    configure_logging()

    # Initialize the logger
    logger = structlog.get_logger()
    logger.info("Logging setup complete.", log_file=os.environ.get("LOG_FILE"))
