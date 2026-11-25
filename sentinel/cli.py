"""Command-line entry point.

    sentinel run "goal" [options]     run one goal to completion
    sentinel chat [options]           give goals one after another (memory carries over)
    sentinel memory list|search|clear inspect the agent's long-term memory
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import DEFAULT_MODEL, AgentConfig


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides real environment variables."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PROVIDERS = ["auto", "anthropic", "gemini", "groq", "ollama", "openai-compatible"]


def _detect_provider() -> str:
    """Pick a provider from whichever API key is configured (free ones after Anthropic)."""
    explicit = os.environ.get("SENTINEL_PROVIDER")
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "anthropic"


def _add_agent_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("-w", "--workspace", default=".", help="directory the agent works in (default: current)")
    p.add_argument("--provider", default="auto", choices=PROVIDERS,
                   help="LLM provider. auto (default) picks from the API key you have set. "
                        "gemini, groq and ollama are free.")
    p.add_argument("--base-url", default=None, help="endpoint for --provider openai-compatible")
    p.add_argument("--model", default=os.environ.get("SENTINEL_MODEL"),
                   help=f"model ID (default: {DEFAULT_MODEL} for anthropic, auto-picked for free providers)")
    p.add_argument("--effort", default=os.environ.get("SENTINEL_EFFORT", "high"),
                   choices=["low", "medium", "high", "xhigh", "max"], help="reasoning depth (default: high)")
    p.add_argument("--approval", default="risky", choices=["auto", "risky", "all"],
                   help="auto: fully autonomous; risky: confirm shell/code execution (default); "
                        "all: confirm every side effect")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--budget", type=float, default=3.0, help="max spend in USD for one goal (0 = no limit)")
    p.add_argument("--no-web", action="store_true", help="disable the web search tool")
    p.add_argument("--no-fallback", action="store_true", help="disable server-side refusal fallback")
    p.add_argument("-q", "--quiet", action="store_true", help="hide reasoning summaries and tool output")


def _config(args: argparse.Namespace, provider: str) -> AgentConfig:
    workspace = Path(args.workspace)
    if not workspace.is_dir():
        raise SystemExit(f"workspace '{workspace}' is not a directory")
    anthropic_provider = provider == "anthropic"
    return AgentConfig(
        workspace=workspace, model=args.model or DEFAULT_MODEL, effort=args.effort, approval=args.approval,
        max_steps=args.max_steps,
        max_cost_usd=args.budget if anthropic_provider else 0.0,  # free providers: nothing to cap
        web_search=anthropic_provider and not args.no_web,         # web_search is an Anthropic server tool
        refusal_fallback=not args.no_fallback,
    )


def _make_llm(config: AgentConfig, provider: str, base_url: str | None):
    if provider == "anthropic":
        from .llm import LLM

        return LLM(config)
    from .openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(config, provider, base_url=base_url)


def _run_goal(agent, goal: str) -> int:
    import anthropic

    from .llm import LLMAuthError

    no_credentials = ("Authentication failed or no credentials found: set ANTHROPIC_API_KEY "
                      "(or put it in a .env file), or use a free provider, e.g. GEMINI_API_KEY with "
                      "--provider gemini. See README > Installation.")
    try:
        result = agent.run(goal)
    except anthropic.AuthenticationError:
        print(no_credentials, file=sys.stderr)
        return 2
    except LLMAuthError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except TypeError as exc:
        # The SDK raises TypeError at request time when no credential source is configured at all.
        if "authentication" not in str(exc).lower():
            raise
        print(no_credentials, file=sys.stderr)
        return 2
    except anthropic.NotFoundError as exc:
        print(f"Model or endpoint not found ({exc}). Check --model.", file=sys.stderr)
        return 2
    except anthropic.PermissionDeniedError as exc:
        print(f"Permission denied by the API: {exc}", file=sys.stderr)
        return 2
    return 0 if result.status == "success" else 1


def main(argv: list[str] | None = None) -> int:
    _load_dotenv(Path.cwd() / ".env")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="sentinel", description="Autonomous goal-driven AI agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="pursue one goal autonomously")
    run_p.add_argument("goal", nargs="+", help="the goal, in plain language")
    _add_agent_options(run_p)

    chat_p = sub.add_parser("chat", help="enter goals one after another")
    _add_agent_options(chat_p)

    mem_p = sub.add_parser("memory", help="inspect long-term memory")
    mem_p.add_argument("action", choices=["list", "search", "clear"])
    mem_p.add_argument("query", nargs="*")
    mem_p.add_argument("-w", "--workspace", default=".")

    args = parser.parse_args(argv)

    if args.command == "memory":
        from .memory import LongTermMemory

        store = LongTermMemory(AgentConfig(workspace=Path(args.workspace)).state_dir / "memory.jsonl")
        if args.action == "clear":
            print(f"Removed {store.clear()} memories.")
        else:
            entries = store.search(" ".join(args.query), k=20) if args.action == "search" else store.all()
            for e in entries:
                print(f"{e.id}  {e.render()}")
            if not entries:
                print("(no memories)")
        return 0

    from .agent import Agent
    from .reporting import ConsoleReporter

    from .llm import LLMError

    provider = _detect_provider() if args.provider == "auto" else args.provider
    config = _config(args, provider)
    try:
        llm = _make_llm(config, provider, args.base_url)
    except LLMError as exc:  # missing key, unreachable endpoint, no usable model
        print(str(exc), file=sys.stderr)
        return 2
    agent = Agent(config, llm=llm, reporter=ConsoleReporter(verbose=not args.quiet))

    if args.command == "run":
        return _run_goal(agent, " ".join(args.goal))

    print("Sentinel chat: type a goal and press Enter (Ctrl+C or 'exit' to quit).")
    while True:
        try:
            goal = input("\ngoal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if goal.lower() in ("exit", "quit"):
            return 0
        if goal:
            try:
                _run_goal(agent, goal)
            except KeyboardInterrupt:
                print("\n(run interrupted)")


if __name__ == "__main__":
    raise SystemExit(main())
