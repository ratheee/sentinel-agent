"""Filesystem tools, confined to the workspace."""

from __future__ import annotations

import fnmatch
import re

from ..guardrails import resolve_path
from ..perception import iter_files, tree
from .registry import Risk, Tool, ToolContext, ToolError, schema

MAX_READ_BYTES = 2_000_000


def _list_files(args: dict, ctx: ToolContext) -> str:
    target = resolve_path(ctx.workspace, args.get("path", "."))
    if not target.exists():
        raise ToolError(f"'{args.get('path', '.')}' does not exist.")
    if target.is_file():
        return f"{target.name} is a file ({target.stat().st_size} B)."
    return tree(target, max_depth=int(args.get("depth", 2)), max_entries=300)


def _read_file(args: dict, ctx: ToolContext) -> str:
    path = resolve_path(ctx.workspace, args["path"])
    if not path.is_file():
        raise ToolError(f"'{args['path']}' is not a file.")
    if path.stat().st_size > MAX_READ_BYTES:
        raise ToolError("File is larger than 2 MB; read it with run_command or run_python instead.")
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ToolError("File looks binary; inspect it with run_python instead.")
    lines = raw.decode("utf-8", errors="replace").splitlines()
    start = max(int(args.get("start_line", 1)), 1)
    count = int(args.get("max_lines", 400))
    chunk = lines[start - 1:start - 1 + count]
    body = "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(chunk))
    remaining = len(lines) - (start - 1 + len(chunk))
    if remaining > 0:
        body += f"\n... {remaining} more lines (use start_line={start + len(chunk)})"
    return body or "(empty file)"


def _write_file(args: dict, ctx: ToolContext) -> str:
    path = resolve_path(ctx.workspace, args["path"], for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    path.write_text(args["content"], encoding="utf-8", newline="")
    return f"{'Overwrote' if existed else 'Created'} {args['path']} ({len(args['content'])} chars)."


def _edit_file(args: dict, ctx: ToolContext) -> str:
    path = resolve_path(ctx.workspace, args["path"], for_write=True)
    if not path.is_file():
        raise ToolError(f"'{args['path']}' does not exist; use write_file to create it.")
    text = path.read_text(encoding="utf-8")
    old, new = args["old_text"], args["new_text"]
    occurrences = text.count(old)
    if occurrences == 0:
        raise ToolError("old_text was not found. Re-read the file and copy the exact text, including whitespace.")
    if occurrences > 1 and not args.get("replace_all", False):
        raise ToolError(f"old_text appears {occurrences} times; add surrounding context or set replace_all.")
    path.write_text(text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1),
                    encoding="utf-8", newline="")
    return f"Edited {args['path']} ({occurrences if args.get('replace_all') else 1} replacement(s))."


def _search_files(args: dict, ctx: ToolContext) -> str:
    try:
        pattern = re.compile(args["pattern"], 0 if args.get("case_sensitive") else re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"Invalid regex: {exc}") from exc
    glob = args.get("glob", "*")
    hits: list[str] = []
    for path in iter_files(ctx.workspace):
        rel = path.relative_to(ctx.workspace).as_posix()
        if not fnmatch.fnmatch(path.name, glob) and not fnmatch.fnmatch(rel, glob):
            continue
        try:
            if path.stat().st_size > MAX_READ_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                if len(hits) >= 100:
                    return "\n".join(hits) + "\n... (stopped at 100 matches)"
    return "\n".join(hits) if hits else "No matches."


TOOLS = [
    Tool(
        "list_files",
        "List the directory tree under a workspace path (ignores .git, caches, virtualenvs).",
        schema({"path": {"type": "string", "description": "Directory relative to the workspace. Default '.'"},
                "depth": {"type": "integer", "description": "Max depth, default 2"}}),
        _list_files, Risk.READ,
    ),
    Tool(
        "read_file",
        "Read a text file with line numbers. Use start_line/max_lines to page through large files.",
        schema({"path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-based, default 1"},
                "max_lines": {"type": "integer", "description": "Default 400"}}, ["path"]),
        _read_file, Risk.READ,
    ),
    Tool(
        "search_files",
        "Regex search across workspace text files. Returns file:line: text matches.",
        schema({"pattern": {"type": "string", "description": "Python regular expression"},
                "glob": {"type": "string", "description": "Filename filter such as '*.py'. Default '*'"},
                "case_sensitive": {"type": "boolean"}}, ["pattern"]),
        _search_files, Risk.READ,
    ),
    Tool(
        "write_file",
        "Create or overwrite a file in the workspace with the given content. Parent directories are created.",
        schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        _write_file, Risk.WRITE,
    ),
    Tool(
        "edit_file",
        "Replace exact text in an existing file. old_text must match exactly and be unique unless replace_all is true.",
        schema({"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
                "replace_all": {"type": "boolean"}}, ["path", "old_text", "new_text"]),
        _edit_file, Risk.WRITE,
    ),
]
