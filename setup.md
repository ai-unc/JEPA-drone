# Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

From the repo root:

```bash
uv sync
uv run python -m skyjepa.train_jepa
```

`uv sync` creates `.venv` from `pyproject.toml` and `uv.lock`. The second command runs the training stub inside that environment.

The training stub runs once the environment, dataset, encoders, predictor, loss, and baseline are implemented.

Tensor layouts are in `docs/tensor_shapes.md`. The annotated notes are in `docs/SkyJEPA_annotations.pdf`.
