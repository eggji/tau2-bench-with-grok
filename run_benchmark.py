"""Convenience wrapper for running Tau2 benchmarks with Grok (xAI) models.

This script simply forwards into the core `tau2` runtime (RunConfig + run_domain)
but tweaks the default LLM choices so that you can run Grok benchmarks without
re-implementing the evaluation stack.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

# Ensure we can import from ./src
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from tau2.cli import add_run_args  # type: ignore  # noqa: E402
from tau2.data_model.simulation import RunConfig  # type: ignore  # noqa: E402
from tau2.run import run_domain  # type: ignore  # noqa: E402

DEFAULT_GROK_MODEL = os.environ.get("TAU2_GROK_MODEL", "grok-4-fast-reasoning")


def _flag_was_provided(argv: Sequence[str], flag: str) -> bool:
    normalized = flag.strip()
    for arg in argv:
        if arg == normalized or arg.startswith(f"{normalized}="):
            return True
    return False


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tau2 benchmark with Grok via LiteLLM"
    )
    add_run_args(parser)
    parser.add_argument(
        "--grok-model",
        default=DEFAULT_GROK_MODEL,
        help="Default Grok model to use when --agent-llm/--user-llm are not provided.",
    )
    parser.add_argument(
        "--no-auto-grok",
        action="store_true",
        help="Disable automatic substitution of Grok model for agent/user.",
    )
    return parser.parse_args(args=argv)


def maybe_apply_grok_defaults(args: argparse.Namespace, argv: Sequence[str]) -> None:
    if args.no_auto_grok:
        return
    if not _flag_was_provided(argv, "--agent-llm"):
        args.agent_llm = args.grok_model
    if not _flag_was_provided(argv, "--user-llm"):
        args.user_llm = args.grok_model


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        domain=args.domain,
        task_set_name=args.task_set_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        agent=args.agent,
        llm_agent=args.agent_llm,
        llm_args_agent=args.agent_llm_args,
        user=args.user,
        llm_user=args.user_llm,
        llm_args_user=args.user_llm_args,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        save_to=args.save_to,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    args = parse_args(argv)
    maybe_apply_grok_defaults(args, argv)
    config = build_run_config(args)
    print(
        f"Running Tau2 benchmark: domain={config.domain}, agent_llm={config.llm_agent}, "
        f"user_llm={config.llm_user}, num_trials={config.num_trials}, num_tasks={config.num_tasks or 'all'}"
    )
    run_domain(config)


if __name__ == "__main__":
    main()
