import pytest

from sentinel.config import AgentConfig
from sentinel.memory import LongTermMemory
from sentinel.perception import WorkspaceWatcher, snapshot
from sentinel.tools.builtin import default_registry
from sentinel.tools.registry import RunState, ToolContext, ToolError, validate_input


@pytest.fixture
def ctx(tmp_path):
    cfg = AgentConfig(workspace=tmp_path, approval="auto")
    return ToolContext(cfg, LongTermMemory(tmp_path / ".sentinel" / "m.jsonl"), RunState("g"), lambda q: None)


@pytest.fixture
def tools():
    return default_registry()


def run(tools, ctx, name, **args):
    tool = tools.get(name)
    validate_input(tool, args)
    return tool.handler(args, ctx)


def test_write_read_edit_search(tools, ctx):
    assert "Created" in run(tools, ctx, "write_file", path="pkg/a.py", content="x = 1\ny = 2\n")
    assert "x = 1" in run(tools, ctx, "read_file", path="pkg/a.py")
    run(tools, ctx, "edit_file", path="pkg/a.py", old_text="y = 2", new_text="y = 3")
    assert "pkg/a.py:2: y = 3" in run(tools, ctx, "search_files", pattern=r"y = \d", glob="*.py")
    with pytest.raises(ToolError):
        run(tools, ctx, "edit_file", path="pkg/a.py", old_text="nope", new_text="z")


def test_read_file_paging(tools, ctx):
    run(tools, ctx, "write_file", path="big.txt", content="\n".join(f"line {i}" for i in range(1, 51)))
    out = run(tools, ctx, "read_file", path="big.txt", start_line=10, max_lines=5)
    assert "line 10" in out and "line 14" in out and "line 15" not in out.split("more lines")[0]
    assert "start_line=15" in out


def test_run_python_and_command(tools, ctx):
    assert "42" in run(tools, ctx, "run_python", code="print(6 * 7)")
    out = run(tools, ctx, "run_command", command="python -c \"import sys; sys.exit(3)\"")
    assert "[exit code 3]" in out
    with pytest.raises(ToolError, match="guardrail"):
        run(tools, ctx, "run_command", command="sudo rm -rf /")


def test_child_processes_do_not_see_secrets(tools, ctx, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    out = run(tools, ctx, "run_python", code="import os; print(os.environ.get('ANTHROPIC_API_KEY'))")
    assert "sk-test-secret" not in out and "None" in out


def test_fetch_url_rejects_internal_hosts(tools, ctx):
    with pytest.raises(ToolError, match="private"):
        run(tools, ctx, "fetch_url", url="http://127.0.0.1:8080/admin")
    with pytest.raises(ToolError):
        run(tools, ctx, "fetch_url", url="file:///etc/passwd")


def test_input_validation(tools):
    with pytest.raises(ToolError, match="Missing"):
        validate_input(tools.get("read_file"), {})
    with pytest.raises(ToolError, match="Unknown"):
        validate_input(tools.get("read_file"), {"path": "a", "bogus": 1})
    with pytest.raises(ToolError, match="type"):
        validate_input(tools.get("read_file"), {"path": 5})


def test_plan_and_memory_tools(tools, ctx):
    out = run(tools, ctx, "update_plan", steps=[{"step": "inspect", "status": "done"},
                                                 {"step": "fix", "status": "in_progress"}])
    assert "[x] inspect" in out and "[>] fix" in out
    run(tools, ctx, "remember", text="tests live in tests/", kind="fact")
    assert "tests live in tests/" in run(tools, ctx, "recall", query="where are tests")
    assert "No human" in run(tools, ctx, "ask_human", question="which db?")


def test_perception_snapshot_and_change_detection(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    snap = snapshot(tmp_path)
    assert "a.txt" in snap and "Platform" in snap
    watcher = WorkspaceWatcher(tmp_path)
    assert watcher.changes() is None
    (tmp_path / "b.txt").write_text("new")
    (tmp_path / "a.txt").unlink()
    change = watcher.changes()
    assert "added: b.txt" in change and "deleted: a.txt" in change
