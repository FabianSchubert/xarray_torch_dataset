# xarray_torch_dataset

Experimental development — this project is in an exploratory phase.

See `STATUS.md` for the current project state and open design questions.

## Guidelines

- **Tests** live in `tests/` and cover the package in `src/` (not the examples). Run them with `uv run pytest`. Keep the suite small and in step with contract changes.
- **Dependencies are added as we go.** Add them with `uv add <package>` when they are actually needed; don't pre-install things speculatively.
- Pre-commit hooks run `ruff check --fix`, `ruff format`, `mypy` and `pytest` (config in `pyproject.toml`). Run them manually with `uv run pre-commit run --all-files`.
- Use `uv` for everything: `uv run <command>` to run code, `uv add` / `uv remove` to manage dependencies.
