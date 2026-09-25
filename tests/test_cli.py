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


def test_prompted_input_drops_a_byte_order_mark():
    # PowerShell prefixes piped input with one; kept, it would make "C-1" a different customer.
    import typer

    from autosupport.cli import _ask

    probe = typer.Typer()

    @probe.command()
    def go() -> None:
        typer.echo(f"[{_ask('Customer ID', False)}]")

    result = runner.invoke(probe, [], input="\ufeffC-2663  \n")
    assert "[C-2663]" in result.output
