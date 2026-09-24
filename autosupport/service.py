"""The UI seam (CLAUDE.md). Returns Pydantic objects, never formatted strings —
cli.py is a pure renderer over what this module returns. This is the only module
cli.py is allowed to call. Built starting at CP3 (docs/checkpoints.md)."""
