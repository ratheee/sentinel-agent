"""Long-term memory: a small persistent store of facts, lessons and episodes.

Short-term memory is the conversation itself (plus the plan held in
``RunState``). Long-term memory survives across runs so the agent can reuse
what it learned: project conventions it discovered, approaches that failed,
and a one-line record of every past goal and its outcome.

Retrieval is lexical (TF-IDF-style overlap with a recency tie-break). It needs
no embedding service, which keeps the agent runnable with one API key.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with into not but you your i we our they them".split()
)
KINDS = ("fact", "lesson", "episode")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class MemoryEntry:
    text: str
    kind: str = "fact"
    tags: list[str] = field(default_factory=list)
    goal: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: float = field(default_factory=time.time)

    def render(self) -> str:
        tag_text = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"({self.kind}{tag_text}) {self.text}"


class LongTermMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: list[MemoryEntry] | None = None

    # -- storage -------------------------------------------------------------

    def _load(self) -> list[MemoryEntry]:
        if self._entries is None:
            self._entries = []
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            self._entries.append(MemoryEntry(**json.loads(line)))
                        except (json.JSONDecodeError, TypeError):
                            continue  # skip a corrupt line rather than lose the store
        return self._entries

    def add(self, text: str, kind: str = "fact", tags: list[str] | None = None, goal: str = "") -> MemoryEntry:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        text = text.strip()
        entries = self._load()
        for existing in entries:  # don't store the same thing twice
            if existing.kind == kind and existing.text.lower() == text.lower():
                return existing
        entry = MemoryEntry(text=text, kind=kind, tags=list(tags or []), goal=goal)
        entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def all(self) -> list[MemoryEntry]:
        return list(self._load())

    def clear(self) -> int:
        count = len(self._load())
        self._entries = []
        if self.path.exists():
            self.path.unlink()
        return count

    # -- retrieval -----------------------------------------------------------

    def search(self, query: str, k: int = 6, kinds: tuple[str, ...] | None = None) -> list[MemoryEntry]:
        entries = [e for e in self._load() if kinds is None or e.kind in kinds]
        q_tokens = set(_tokens(query))
        if not entries or not q_tokens:
            return []

        docs = [set(_tokens(e.text + " " + " ".join(e.tags) + " " + e.goal)) for e in entries]
        n = len(docs)
        df: dict[str, int] = {}
        for doc in docs:
            for tok in doc:
                df[tok] = df.get(tok, 0) + 1

        newest = max(e.created for e in entries)
        scored = []
        for entry, doc in zip(entries, docs):
            overlap = q_tokens & doc
            if not overlap:
                continue
            score = sum(math.log(1 + n / df[t]) for t in overlap)
            score += 0.05 / (1 + (newest - entry.created) / 86400)  # recency tie-break
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]
