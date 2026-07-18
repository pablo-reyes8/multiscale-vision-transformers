# Contributing

Contributions are welcome through focused pull requests with tests and a clear
motivation.

## Development setup

```bash
git clone https://github.com/pablo-reyes8/multiscale-vision-transformers.git
cd multiscale-vision-transformers
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Quality checks

Run these before opening a pull request:

```bash
make lint
make test
make build
famous-vits validate-config --config configs/smoke_test.yaml
```

## Repository boundaries

- `famous_vits/` is the stable public library and should not import a generic
  `model` package outside the factory isolation boundary.
- `model_zoo/` contains architecture-specific research implementations.
- `configs/` contains versioned pipeline examples.
- `tests/` covers the public package contract.
- `artifacts/` contains small, intentional benchmark evidence—not datasets or
  generated checkpoints.

## Adding or changing a model

1. Keep the implementation within its architecture directory in `model_zoo/`.
2. Register a preset and aliases in `famous_vits/arena/presets.py` and
   `famous_vits/factory.py`.
3. Support `num_classes`, `in_chans` and `img_size` through `create_model`.
4. Add a small CPU-safe forward test and, when relevant, a checkpoint test.
5. Update `docs/library.md` and `CHANGELOG.md`.

## Pull requests

- Keep changes scoped and explain user-visible behavior.
- Add or update tests for every behavior change.
- Do not commit datasets, credentials, generated checkpoints or large outputs.
- Use conventional, imperative commit subjects such as `Add YAML arena runner`.
- Confirm that you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

