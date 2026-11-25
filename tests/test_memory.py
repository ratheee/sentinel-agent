from sentinel.memory import LongTermMemory


def test_add_search_and_persist(tmp_path):
    path = tmp_path / "mem.jsonl"
    mem = LongTermMemory(path)
    mem.add("Run the test suite with python -m pytest -q", kind="lesson", tags=["testing"])
    mem.add("The inventory module stores prices in cents", kind="fact")
    mem.add("Deploys go through the staging branch", kind="fact")

    hits = mem.search("how do I run the tests?")
    assert hits and "pytest" in hits[0].text

    reloaded = LongTermMemory(path)
    assert len(reloaded.all()) == 3
    assert reloaded.search("prices cents")[0].text.startswith("The inventory")


def test_duplicates_are_not_stored_twice(tmp_path):
    mem = LongTermMemory(tmp_path / "mem.jsonl")
    a = mem.add("Use UTF-8 everywhere")
    b = mem.add("use utf-8 everywhere")
    assert a.id == b.id and len(mem.all()) == 1


def test_kind_filter_and_clear(tmp_path):
    mem = LongTermMemory(tmp_path / "mem.jsonl")
    mem.add("Goal: fix tests -> success", kind="episode")
    mem.add("fix tests by updating fixtures", kind="lesson")
    assert [e.kind for e in mem.search("fix tests", kinds=("lesson",))] == ["lesson"]
    assert mem.clear() == 2 and mem.all() == []


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "mem.jsonl"
    path.write_text('{"text": "good", "kind": "fact"}\nnot json\n', encoding="utf-8")
    assert [e.text for e in LongTermMemory(path).all()] == ["good"]
