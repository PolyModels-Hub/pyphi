# Development Environment Setup

This guide walks through creating a reproducible development environment for the PyPhi refactor. The steps assume macOS (Darwin 24.6.0), but work on Linux and Windows with minor adjustments.

---

## 1. Prerequisites

- **Python**: Version 3.11 or 3.12. We recommend installing via [pyenv](https://github.com/pyenv/pyenv) or using your system Python if it meets the requirement.
- **Poetry**: 1.8+ for dependency and virtual environment management.
- **Build tools**: A C/C++ toolchain (Xcode Command Line Tools on macOS, `build-essential` on Linux) to compile scientific Python dependencies when wheels are unavailable.
- **Optional solvers**: IPOPT and/or GAMS for advanced Pyomo-backed workflows. These are not required to run the refactor itself but may be needed to exercise certain examples.

---

## 2. Clone the repository

```bash
# Choose your workspace directory
cd ~/Documents

# Clone if you have not already
git clone https://github.com/<your-org>/pyphi-project.git
cd pyphi-project/pyphi
```

> If this repo is already on your machine, pull the latest changes instead:
> ```bash
> git pull --rebase
> ```

---

## 3. Install Poetry (if needed)

Check whether Poetry is already installed:

```bash
poetry --version
```

If the command is not found, install it following the official recommendation:

```bash
curl -sSL https://install.python-poetry.org | python3 -
# Add Poetry to your PATH (if the installer instructs you to do so)
```

Restart your terminal or source your shell profile to pick up the new PATH.

---

## 4. Create the virtual environment

The project is configured to use Poetry’s `src/` layout and grouped dependencies (see `pyproject.toml`).

1. **Select the interpreter** (only needed once or when switching Python versions):
   ```bash
   poetry env use python3.11
   ```
   Replace `python3.11` with the absolute path to your desired interpreter if necessary.

2. **Install dependencies** (including dev tools for testing and linting):
   ```bash
   poetry install --with dev
   ```
   This command reads `pyproject.toml`, creates the virtual environment, and installs runtime + dev dependencies (`pytest`, `pytest-cov`, `ruff`, etc.).

3. **Validate the environment**:
   ```bash
   poetry check        # sanity-checks the pyproject configuration
   poetry env info     # shows where the virtualenv lives
   ```

---

## 5. Using the environment

You can either spawn an interactive shell or run commands ad hoc:

- **Interactive shell** (activates the venv for the current terminal session):
  ```bash
  poetry shell
  ```
  Exit with `exit` when finished.

- **Run individual commands** (preferred for scripts and CI):
  ```bash
  poetry run python -V
  poetry run pytest
  poetry run pytest --cov
  poetry run ruff check
  ```

Poetry automatically ensures the commands run inside the managed virtual environment.

---

## 6. Running tests during the refactor

The refactor plan calls for a test-first workflow. Useful invocations:

```bash
# Entire suite
poetry run pytest

# Single module test file (e.g., utilities)
poetry run pytest tests/test_utils.py -v

# Focus on one test function
poetry run pytest tests/test_utils.py -k "meancenterscale" -vv

# Collect coverage (threshold defined in pyproject.toml)
poetry run pytest --cov
```

> Pytest configuration lives in `pyproject.toml` (`[tool.pytest.ini_options]`). Default options are `-v --tb=short` and the test path is the `tests/` directory.

---

## 7. Linting and code hygiene

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting (and optional formatting). Examples:

```bash
# Lint the whole codebase
poetry run ruff check

# Auto-fix where possible
poetry run ruff check --fix

# Only check the new src layout once files exist
poetry run ruff check src/pyphi tests
```

> Ruff configuration (line length, source directories) is defined under `[tool.ruff]` in `pyproject.toml`.

---

## 8. Managing dependencies with Poetry

- **Add a runtime dependency**:
  ```bash
  poetry add <package>
  ```

- **Add a development-only dependency** (e.g., a testing plugin):
  ```bash
  poetry add --group dev <package>
  ```

- **Update the lock file** (after manual edits):
  ```bash
  poetry lock --no-update
  ```

- **Remove the virtual environment** (if you need a clean restart):
  ```bash
  poetry env list
  poetry env remove <venv-path-or-name>
  ```

---

## 9. Troubleshooting tips

| Problem | Resolution |
|---------|------------|
| `poetry env use` fails | Ensure the requested Python version is installed (`which python3.11`). |
| Packages fail to build | Install system build tools (`xcode-select --install` on macOS). |
| Solvers unavailable | IPOPT/GAMS are optional; warnings on import are expected without them. |
| Tests cannot import modules | Check that the `src/pyphi` structure exists and `packages = [{ include = "pyphi", from = "src" }]` is present in `pyproject.toml`. |
| Coverage below threshold | Improve tests or lower `fail_under` temporarily (not recommended) in `pyproject.toml`. |

---

## 10. Next steps

With the environment ready you can follow the refactor plan:

1. Create the `src/pyphi/` and `tests/` scaffolding (Phase 0).
2. Move utilities into module files using the TDD workflow (`pytest` before/after each move).
3. Keep `poetry run pytest` (and optionally `--cov`) in your loop to catch regressions early.
4. Update examples incrementally and document any new dependencies.

Happy refactoring! 🚀
