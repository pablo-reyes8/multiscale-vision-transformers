# Releasing

1. Ensure `main` is green and the working tree is clean.
2. Move relevant entries from `Unreleased` in `CHANGELOG.md` into a versioned
   section using `YYYY-MM-DD`.
3. Update the version in `pyproject.toml`, `famous_vits/__init__.py` and
   `CITATION.cff`.
4. Run `make lint`, `make test`, `make build` and the installed-wheel smoke test.
5. Create and push an annotated tag:

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

The release workflow builds the wheel and source archive, then creates a GitHub
release with generated notes and attached distributions.

