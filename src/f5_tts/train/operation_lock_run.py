"""Run a child process while holding a shared VisualNovel operation lock."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from f5_tts.train.datasets.visualnovel_control import operation_lock


def run_locked(lock_root: str | Path, operation: str, owner: str, command: list[str]) -> int:
    if not command:
        raise ValueError("locked runner requires a child command")
    child: subprocess.Popen | None = None

    def forward(signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    with operation_lock(lock_root, operation, owner):
        previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        try:
            child = subprocess.Popen(command)
            return child.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-root", required=True)
    parser.add_argument("--operation", required=True, choices=("transfer", "training", "heavy"))
    parser.add_argument("--owner", default=f"pid-{os.getpid()}")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_locked(args.lock_root, args.operation, args.owner, command)


if __name__ == "__main__":
    sys.exit(main())
