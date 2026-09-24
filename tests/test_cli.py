from typer.testing import CliRunner

from autosupport.cli import app

runner = CliRunner()

COMMANDS = ["ingest", "new", "resume", "show", "list", "memory", "eval", "demo"]


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    for command in COMMANDS:
        assert command in result.output, f"{command!r} missing from --help output"
