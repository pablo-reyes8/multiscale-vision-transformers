# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned

- Publish curated pretrained checkpoints with reproducible training metadata.
- Add benchmark baselines for additional image resolutions.

## [0.1.0] - 2026-07-16

### Added

- Unified `famous_vits` registry and `create_model` API.
- High-level training, inference, checkpointing and explainability orchestrator.
- YAML pipelines for training, arena comparisons and analysis.
- Installable `famous-vits` and `vit-arena` command-line interfaces.
- Dataset-aware comparison arena and compatibility shims for legacy commands.
- Python package metadata, model-zoo packaging and isolated wheel smoke tests.
- CI, CodeQL, dependency review, Dependabot, Docker and release workflows.
- Community health files, citation metadata and security policy.

### Changed

- Moved architecture research projects under `model_zoo/`.
- Moved historical training logs under `artifacts/training_logs/`.
- Replaced the root container with a multi-stage, non-root runtime image.

### Fixed

- Fixed MaxViT drop-path construction that referenced an undefined helper.
- Prevented import collisions between the legacy `model.*` namespaces.

[Unreleased]: https://github.com/pablo-reyes8/multiscale-vision-transformers/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pablo-reyes8/multiscale-vision-transformers/releases/tag/v0.1.0

