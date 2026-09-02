#!/usr/bin/env python
"""World-size equivalence smoke for the LR schedule, against the real Accelerate wrapper.

The unit tests simulate `AcceleratedScheduler.step()`; this runs the genuine
article under a genuine distributed launch, so the two together cover both
"we modelled Accelerate correctly" and "Accelerate behaves as modelled".

It deliberately builds the schedule with the *production* helpers
(`resolve_scheduler_contract` + `build_scheduler`) rather than a local copy, so a
future edit to the trainer that breaks the units fails here too.

No dataset, no model weights, no checkpoints: a one-parameter tensor is enough to
own an optimizer, and the schedule is what is under test.

    accelerate launch --num_processes 1 tools/smoke_scheduler_world_size.py \
        --global-warmup 100 --global-total 5000 --updates 120 --out trace_gpu1.json
    accelerate launch --multi_gpu --num_processes 8 tools/smoke_scheduler_world_size.py \
        --global-warmup 100 --global-total 5000 --updates 120 --out trace_gpu8.json

Then compare the two traces with --compare.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from f5_tts.model.trainer import build_scheduler, resolve_scheduler_contract, resolve_split_batches


def run_trace(args) -> dict:
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    parameter = torch.nn.Parameter(torch.zeros(1, device=accelerator.device))
    optimizer = torch.optim.AdamW([parameter], lr=args.learning_rate)

    contract = resolve_scheduler_contract(
        global_warmup_updates=args.global_warmup,
        global_total_updates=args.global_total,
        num_processes=accelerator.num_processes,
        split_batches=resolve_split_batches(accelerator),
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_updates=contract["warmup_updates"],
        total_updates=contract["total_updates"],
    )

    # prepare() is the point of the exercise: it wraps the scheduler in
    # AcceleratedScheduler, which is what actually decides how far the LR moves
    # per optimizer step.
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    trace = []
    for update in range(1, args.updates + 1):
        # A real gradient so the optimizer step is genuine rather than a no-op.
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        trace.append({"global_update": update, "lr": float(scheduler.get_last_lr()[0])})

    result = {
        "num_processes": int(accelerator.num_processes),
        "mixed_precision": str(accelerator.mixed_precision),
        "contract": contract,
        "trace": trace,
    }

    if accelerator.is_main_process:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.out}: num_processes={result['num_processes']} contract={contract}")
    accelerator.wait_for_everyone()
    return result


def load_trace(path: str) -> dict:
    """Load either a scheduler-smoke JSON trace or a trainer JSONL lr_trace.

    The two smokes emit different shapes -- the standalone one writes a single
    JSON object, the trainer appends JSONL -- but comparing them against each
    other is the whole point, so both are normalised here.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "trace" in data:
            return data
    except json.JSONDecodeError:
        pass

    contract = None
    trace = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("record") == "scheduler_contract":
            contract = {key: value for key, value in record.items() if key != "record"}
        elif "global_update" in record:
            trace.append({"global_update": record["global_update"], "lr": record["lr"]})
    if not trace:
        raise ValueError(f"no LR records found in {path}")
    if contract is None:
        raise ValueError(f"no scheduler_contract record found in {path}")
    return {"num_processes": contract["num_processes"], "contract": contract, "trace": trace}


def compare(paths: list[str], anchors: list[int], tolerance: float) -> int:
    traces = [(path, load_trace(path)) for path in paths]

    print("=== contracts ===")
    for path, data in traces:
        contract = data["contract"]
        print(
            f"{path}: num_processes={contract['num_processes']} multiplier={contract['scheduler_step_multiplier']} "
            f"inner_warmup={contract['warmup_updates']} inner_total={contract['total_updates']}"
        )

    lookup = []
    for path, data in traces:
        lookup.append((path, {entry["global_update"]: entry["lr"] for entry in data["trace"]}))

    reference_path, reference = lookup[0]
    failures = []
    print(f"\n=== LR at anchors (reference: {reference_path}) ===")
    header = "update".ljust(10) + "".join(Path(path).name.ljust(26) for path, _ in lookup)
    print(header)
    for anchor in anchors:
        if anchor not in reference:
            continue
        row = str(anchor).ljust(10)
        for path, table in lookup:
            row += f"{table.get(anchor, float('nan')):.12e}".ljust(26)
        print(row)
        for path, table in lookup[1:]:
            if anchor not in table:
                failures.append(f"{path} has no global_update {anchor}")
                continue
            delta = abs(table[anchor] - reference[anchor])
            if delta > tolerance:
                failures.append(f"{path} differs at update {anchor}: delta={delta:.3e} > {tolerance:.1e}")

    common = set(reference)
    for _, table in lookup[1:]:
        common &= set(table)
    for update in sorted(common):
        for path, table in lookup[1:]:
            if abs(table[update] - reference[update]) > tolerance:
                failures.append(f"{path} differs at update {update}")
                break

    print()
    if failures:
        for failure in sorted(set(failures))[:20]:
            print(f"FAIL: {failure}")
        print(f"\nWORLD-SIZE EQUIVALENCE: FAILED ({len(set(failures))} differences)")
        return 1
    print(f"WORLD-SIZE EQUIVALENCE: OK (all {len(common)} common updates agree within {tolerance:.1e})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--global-warmup", type=int, default=100)
    parser.add_argument("--global-total", type=int, default=5000)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--out", default="scheduler_trace.json")
    parser.add_argument("--compare", nargs="+", help="compare two or more trace files instead of running")
    parser.add_argument("--anchors", type=int, nargs="+", default=[1, 50, 100, 101, 120])
    parser.add_argument("--tolerance", type=float, default=1e-15)
    args = parser.parse_args(argv)

    if args.compare:
        return compare(args.compare, args.anchors, args.tolerance)
    run_trace(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
