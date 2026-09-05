"""Per-episode result records and strict aggregation into table rows.

Two rules this module enforces:

1. Evaluation writes one record per episode and nothing else. Means and
   standard deviations are computed here, never typed into the paper by hand.
2. Aggregation refuses to emit a row unless the expected tasks, seeds and
   episode counts are all present. A silently short run producing a plausible
   number is the failure mode this exists to prevent.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:  # package context (scripts.common)
    from . import paper_protocol as P
except ImportError:  # flat context (directory on sys.path)
    import paper_protocol as P


def git_commit(default: str = "unknown") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return default


@dataclass
class EpisodeRecord:
    """One evaluation episode. Written as a JSONL line."""

    benchmark: str          # "calvin" | "robocasa" | "real_robot"
    task: str
    method: str             # "nofuture" | "gen" | "gt" | "uniform" | "full" ...
    seed: int
    episode: int
    success: bool

    # Perturbation actually applied at evaluation time.
    perturbation: str = "none"      # "none" | "shift" | "windowed" | "rate"
    shift: Optional[int] = None
    rate: Optional[float] = None

    steps: Optional[int] = None
    termination: Optional[str] = None   # "success" | "timeout" | "broken"

    # Gate diagnostics, episode means. Feeds Table VII directly.
    alpha_mean: Optional[float] = None
    weights_mean: Optional[Sequence[float]] = None

    # Provenance.
    checkpoint: Optional[str] = None
    future_file: Optional[str] = None
    label_source: str = "auto"      # "auto" | "manual" for hardware trials
    protocol: str = field(default_factory=P.describe)
    commit: str = field(default_factory=git_commit)

    def validate(self) -> None:
        if self.perturbation == "shift" or self.perturbation == "windowed":
            if self.shift is None:
                raise ValueError("perturbation={} requires a shift".format(self.perturbation))
            if self.shift not in P.GLOBAL_SHIFTS:
                raise ValueError("shift {} outside the reported set".format(self.shift))
        if self.perturbation == "rate":
            if self.rate is None:
                raise ValueError("perturbation='rate' requires a rate")
            if self.rate not in P.RATE_WARPS:
                raise ValueError("rate {} outside the reported set".format(self.rate))
        if self.perturbation == "none" and (self.shift not in (None, 0) or self.rate not in (None, 1.0)):
            raise ValueError("unperturbed record carries a perturbation value")
        if self.weights_mean is not None:
            w = list(self.weights_mean)
            if len(w) != len(P.LOCAL_SHIFTS):
                raise ValueError("expected {} weights".format(len(P.LOCAL_SHIFTS)))
            if not math.isclose(sum(w), 1.0, abs_tol=1e-5):
                raise ValueError("temporal weights do not sum to 1: {}".format(sum(w)))
            if any(x < -1e-6 or x > 1.0 + 1e-6 for x in w):
                raise ValueError("temporal weight outside [0, 1]")
        if self.alpha_mean is not None and not (-1e-6 <= self.alpha_mean <= 1.0 + 1e-6):
            raise ValueError("alpha outside [0, 1]: {}".format(self.alpha_mean))


class ResultWriter:
    """Append-only JSONL writer. One file per (benchmark, method, condition)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: EpisodeRecord) -> None:
        record.validate()
        assert self._fh is not None, "use ResultWriter as a context manager"
        self._fh.write(json.dumps(asdict(record)) + "\n")
        self._fh.flush()


def load_records(path) -> list:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_dir(directory, pattern: str = "*.jsonl") -> list:
    out = []
    for p in sorted(Path(directory).glob(pattern)):
        out.extend(load_records(p))
    return out


class IncompleteRun(RuntimeError):
    """Raised instead of reporting a number computed from a partial run."""


def _select(records, **criteria):
    out = []
    for r in records:
        if all(r.get(k) == v for k, v in criteria.items()):
            out.append(r)
    return out


def seed_success_rates(
    records,
    tasks: Sequence[str],
    seeds: Sequence[int] = P.SEEDS,
    episodes_per_task_seed: int = P.NUM_EVAL_EPISODES,
    strict: bool = True,
    **criteria,
) -> dict:
    """Task-balanced success rate per seed.

    Returns {seed: rate}. Each seed's rate is the mean over tasks of that
    task's success rate, matching the paper's "task-balanced benchmark
    aggregates". With strict=True, missing tasks or short episode counts raise
    rather than producing a number computed from whatever happened to finish.
    """
    per_seed = {}
    for seed in seeds:
        task_rates = []
        for task in tasks:
            rows = _select(records, task=task, seed=seed, **criteria)
            if strict and len(rows) != episodes_per_task_seed:
                raise IncompleteRun(
                    "task={} seed={} {}: found {} episodes, expected {}".format(
                        task, seed, criteria, len(rows), episodes_per_task_seed
                    )
                )
            if not rows:
                raise IncompleteRun(
                    "no episodes for task={} seed={} {}".format(task, seed, criteria)
                )
            task_rates.append(sum(bool(r["success"]) for r in rows) / len(rows))
        per_seed[seed] = sum(task_rates) / len(task_rates)
    return per_seed


def mean_std(values) -> tuple:
    vals = list(values)
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var)


def cell(records, tasks, strict: bool = True, **criteria) -> str:
    """One "mean +- sd" table cell, in percent."""
    rates = seed_success_rates(records, tasks, strict=strict, **criteria)
    mean, sd = mean_std(rates.values())
    return "${:.1f}\\!\\pm\\!{:.1f}$".format(100 * mean, 100 * sd)


def avg_shifted(records, tasks, strict: bool = True, **criteria) -> float:
    """The "Avg. shifted" column: unweighted mean over the six nonzero shifts.

    The uniform weighting is a choice, not a property of the data. It gives a
    +-6 condition the same mass as a +-2 one; state that assumption wherever
    this number appears.
    """
    means = []
    for s in P.NONZERO_GLOBAL_SHIFTS:
        rates = seed_success_rates(
            records, tasks, strict=strict, perturbation="shift", shift=s, **criteria
        )
        means.append(sum(rates.values()) / len(rates))
    return 100 * sum(means) / len(means)


def latex_row(label: str, cells: Iterable[str], bold: bool = False) -> str:
    body = " & ".join(cells)
    name = "\\textbf{{{}}}".format(label) if bold else label
    return "{} & {}\\\\".format(name, body)
