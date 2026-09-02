#!/usr/bin/env python3
"""Explicit CALVIN GTFuture oracle evaluation entry point.

This runner is intentionally separate from visual-only GenFuture/NoFuture use.
It grants access to held-out demonstration frames and labels every output as an
oracle result; it must not be used for training or deployment claims.
"""

import os
import sys


os.environ["CALVIN_ALLOW_GT_ORACLE"] = "1"
os.environ["CALVIN_FUTURE_MODE"] = "gt"

from calvin_infer_llm_future_bc_rl import main  # noqa: E402


def _option_value(name, default):
    prefix = name + "="
    for index, value in enumerate(sys.argv[1:], start=1):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return default


def _force_gt_future():
    rewritten = [sys.argv[0]]
    skip_next = False
    for value in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if value == "--future_mode":
            skip_next = True
            continue
        if value.startswith("--future_mode="):
            continue
        rewritten.append(value)
    rewritten.extend(["--future_mode", "gt"])
    sys.argv[:] = rewritten


if __name__ == "__main__":
    if _option_value("--eval_mode", "single") == "table4":
        raise ValueError("Table 4 is a generated-future diagnostic; use the visual inference runner")
    _force_gt_future()
    main()
