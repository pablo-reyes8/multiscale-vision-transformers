.PHONY: install lint format test build smoke docker-build

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check famous_vits tests vit_arena.py vit_arena_cli.py vit_arena_presets.py shared_dataset_zoo.py

format:
	ruff format famous_vits tests vit_arena.py vit_arena_cli.py vit_arena_presets.py shared_dataset_zoo.py
	ruff check --fix famous_vits tests vit_arena.py vit_arena_cli.py vit_arena_presets.py shared_dataset_zoo.py

test:
	pytest tests

build:
	python -m build

smoke:
	famous-vits validate-config --config configs/smoke_test.yaml
	famous-vits run --config configs/smoke_test.yaml

docker-build:
	docker build --tag famous-vits:local .

