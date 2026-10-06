# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

"""Compare a results directory against the reference scores.

    python compare.py results/ [--per-task]

A suite passes when its headline number is within 0.5 points of the reference. JEVBench splits
and ImaJEV runs are small (48-173 items), so they pass within two items instead.
"""
import argparse
import json
import os

TOLERANCE = 0.005
SMALL_SET_ITEMS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--per-task", action="store_true", help="also list tasks that moved by more than 1 point")
    args = ap.parse_args()
    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_scores.json")))

    print(f"{'benchmark':<26}{'reference':>10}{'ours':>10}{'delta':>9}  status")
    failed = 0
    for suite, expected in ref.items():
        path = os.path.join(args.results, suite, "summary.json")
        if not os.path.exists(path):
            print(f"{suite:<26}{'':>10}{'':>10}{'':>9}  missing")
            continue
        got = json.load(open(path))
        if suite == "jevbench":
            rows = [(f"jevbench/{s}", expected["tasks"][s], got["splits"][s]["accuracy"],
                     SMALL_SET_ITEMS / expected["n"][s] + 1e-9) for s in expected["tasks"] if s in got["splits"]]
        elif suite == "imajev":
            rows = [(f"imajev/{run}", expected["runs"][run], got["runs"][run]["accuracy"],
                     SMALL_SET_ITEMS / expected["n"][run] + 1e-9) for run in expected["runs"] if run in got["runs"]]
        else:
            rows = [(suite, expected["score"], got["score"], TOLERANCE)]
            if got["num_tasks"] != expected["num_tasks"]:
                rows[0] = (f"{suite} ({got['num_tasks']}/{expected['num_tasks']} tasks)",) + rows[0][1:]
        for name, want, have, tol in rows:
            ok = abs(have - want) <= tol
            failed += not ok
            print(f"{name:<26}{want:>10.4f}{have:>10.4f}{100 * (have - want):>+9.2f}  {'ok' if ok else 'FAIL'}")
        if args.per_task and "tasks" in expected and suite != "jevbench":
            for task, want in expected["tasks"].items():
                have = got["tasks"].get(task)
                if have is None:
                    print(f"    {task}: missing")
                elif abs(have - want) > 0.01:
                    print(f"    {task}: {want:.4f} -> {have:.4f} ({100 * (have - want):+.2f})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
