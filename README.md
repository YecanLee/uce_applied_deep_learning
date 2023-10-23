# pyproject-template
Python project template

## Things to modify when initialization

* Project folder name (currently "pyproject-template")
* package folder name (currently "todo")
* `setup.cfg`
  * `[metadata]` `name`
  * `[options]` `install_requires` (add your dependencies). For more information please refer to [setuptools](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html)
  * `[isort]` `known_first_party` (change to your package name), `know_third_party` (add your dependencies' names). If you don't wanna use isort you can ignore this step.

## Installation
Install your package in editable mode via `pip install -e .` (do no forget the ".").

## Optional
Install pre-commit hooks via `pre-commit install`.
