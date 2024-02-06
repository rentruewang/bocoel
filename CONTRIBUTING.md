# Contributing

The goal here is to make contributing to `bocoel` as painless as possible.

Feel free to reach out and I'll try to respond as soon as possible!

## Development installation

First, clone and navigate into the project:

```bash
git clone https://github.com/rentruewang/bocoel
cd bocoel/
```

Alternatively, use ssh:
```bash
git clone git@github.com:rentruewang/bocoel
cd bocoel/
```

I'm using [PDM](https://pdm-project.org/latest/) in this project for dependency management.
To install all dependencies (including development dependencies) with `pdm`, run

```bash
pdm install -G:all
```

Alternatively, use of `pip` is also allowed (although might be less robust due to lack of version solving)

```bash
pip install -e .
```

Both commands perform an editable installation.

## Recommended development style

### Python code style

The code style in the project closely follows the recommended standard of python:

1. [PEP8](https://peps.python.org/pep-0008/)
2. Class imports are non-qualified (`from module.path import ClassName`), and do not use unqualified function names (however, upper case functions acting as classes are treated as classes, lower case classes are treated as functions).
3. All other imports are qualified.
4. TODO:

### Formatting

Use `autoflake`, `isort`, `black` for consistent formatting.

Prior to commiting, please run the following commands:

```bash
autoflake -i $(find -iname "*.py" ! -path '*/.venv/*' ! -name __init__.py) --remove-all-unused-imports
isort . --profile black
black .
```

### Typing

Be sure to run `mypy` prior to submitting! There can be issue with `mypy` not finding libraries. The command I use for checking is

```bash
mypy . --disable-error-code=import-untyped --disable-error-code=import-not-found
```

### Commit message

Add an emoji that best describes this commit a the start of the commit message.
This helps makes the project look good on GitHub.
