from pydantic import ValidationError

from autosupport.config import Settings


def test_missing_anthropic_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    try:
        Settings(_env_file=None)
        assert False, "expected ValidationError"
    except ValidationError as exc:
        assert any(err["loc"][0] == "ANTHROPIC_API_KEY" for err in exc.errors())


def test_tracing_without_langsmith_key_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    try:
        Settings(_env_file=None)
        assert False, "expected ValidationError"
    except ValidationError:
        pass


def test_derived_paths_sit_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("AUTOSUPPORT_DATA_DIR", str(tmp_path))

    s = Settings(_env_file=None)

    assert s.data_dir == tmp_path
    assert s.sqlite_path == tmp_path / "autosupport.sqlite"
    assert s.checkpoint_path == tmp_path / "checkpoints.sqlite"
    assert s.chroma_dir == tmp_path / "chroma"


def test_blank_anthropic_key_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    try:
        Settings(_env_file=None)
        assert False, "expected ValidationError"
    except ValidationError:
        pass


def test_blank_langsmith_key_with_tracing_on_raises(monkeypatch):
    # A blank `LANGSMITH_API_KEY=` (present but empty, as shipped in .env.example) must be
    # treated the same as unset, not as a valid empty secret.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    try:
        Settings(_env_file=None)
        assert False, "expected ValidationError"
    except ValidationError:
        pass


def test_relative_data_dir_anchors_to_repo_root(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("AUTOSUPPORT_DATA_DIR", "./data")

    s = Settings(_env_file=None)

    assert s.data_dir.is_absolute()
    assert s.data_dir.name == "data"


def test_model_string_requires_provider_prefix(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("AUTOSUPPORT_MAIN_MODEL", "claude-sonnet-5")  # missing "anthropic:"

    try:
        Settings(_env_file=None)
        assert False, "expected ValidationError"
    except ValidationError:
        pass
