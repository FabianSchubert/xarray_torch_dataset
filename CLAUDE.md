# xarray_torch_dataset

Experimental development — this project is in an exploratory phase.

## Guidelines

- **No tests for now.** Don't write tests or set up test infrastructure unless explicitly asked.
- **Dependencies are added as we go.** Add them with `uv add <package>` when they are actually needed; don't pre-install things speculatively.
- Pre-commit hooks run `ruff check --fix`, `ruff format` and `mypy` (config in `pyproject.toml`). Run them manually with `uv run pre-commit run --all-files`.
- Use `uv` for everything: `uv run <command>` to run code, `uv add` / `uv remove` to manage dependencies.
