"""Backward-compatible executable shim for the packaged arena CLI."""

from famous_vits.arena.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
