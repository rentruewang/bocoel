import fire
import structlog

from . import main

structlog.configure()

if __name__ == "__main__":
    # Not a class. Google's standard public functions are all capitalized.
    fire.Fire(main)
