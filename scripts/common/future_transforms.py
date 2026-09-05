"""Temporal transformations applied to future clips.

Single source of truth for every temporal perturbation used by the paper, so
that training, inference and evaluation cannot drift apart.

Correspondence to the paper:

    shift_future      Eq. (4)   clipped integer shift, repeats boundary frames
    make_null_future  Eq. (6)   first future frame repeated across the horizon
    distinct_window   Eq. (5)   contiguous window from a longer source rollout
    rate_warp         Sec. III-B  monotone temporal-rate mismatch

All functions take and return uint8 arrays of shape (T, H, W, C) and never pad
silently: if a source sequence is too short for the requested transform they
raise rather than repeat frames, because boundary repetition is a confound the
paper explicitly controls for.
"""

from __future__ import annotations

import math

import numpy as np

try:  # package context (scripts.common)
    from .paper_protocol import RATE_WARP_ROUNDING
except ImportError:  # flat context (directory on sys.path)
    from paper_protocol import RATE_WARP_ROUNDING

# Local candidate offsets constructed by RAFC at every step (Sec. III-C).
RAFC_SHIFTS = (-2, 0, 2)

# Global evaluation-only perturbations (Sec. III-B).
EVAL_SHIFTS = (-6, -4, -2, 0, 2, 4, 6)
EVAL_RATES = (0.75, 1.25)

# Eq. (5) geometry: 16-frame window taken from a 28-frame source rollout,
# centred so that s = 0 selects frames 7..22 (1-indexed).
WINDOW_LENGTH = 16
WINDOW_BASE_OFFSET = 6
SOURCE_ROLLOUT_LENGTH = 28


def _as_frames(frames) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.uint8)
    if arr.ndim != 4:
        raise ValueError(
            "expected frames of shape (T, H, W, C), got {}".format(arr.shape)
        )
    if arr.shape[0] == 0:
        raise ValueError("empty frame sequence")
    return arr


def _round_index(x: float, rounding: str) -> int:
    """Map a fractional source position to a frame index.

    The paper writes round(...) without fixing the tie rule, and the two
    plausible rules disagree at exact halves, which rho = 0.75 hits. numpy.rint
    and Python's round() use banker's rounding (round(0.5) == 0, round(1.5) == 1
    under numpy, round(2.5) == 2); floor(x + 0.5) rounds half away from zero.
    Different frames get selected, so this is a reproducibility knob, not a
    style choice: it must match whatever produced the reported warp numbers.
    """
    if rounding == "rint":
        return int(np.rint(x))
    if rounding == "half_up":
        return int(math.floor(x + 0.5))
    raise ValueError("unknown rounding rule: {}".format(rounding))


def shift_future(frames, shift: int) -> np.ndarray:
    """Eq. (4): v^(s)_i = v_clip(i+s, 1, T).

    Out-of-range indices repeat the boundary frame, which also reduces temporal
    diversity at large |shift|. That side effect is the reason distinct_window
    exists as a control.
    """
    arr = _as_frames(frames)
    t = arr.shape[0]
    idx = np.arange(t, dtype=np.int64) + int(shift)
    idx = np.clip(idx, 0, t - 1)
    return arr[idx]


def make_null_future(frames) -> np.ndarray:
    """Eq. (6): repeat the first future frame across the horizon.

    Note this is a static clip, not a no-future condition: the BC policy was
    never trained on frozen video, so the null branch is off-distribution by
    construction. alpha -> 0 does not recover the NoFuture policy.
    """
    arr = _as_frames(frames)
    return np.repeat(arr[:1], repeats=arr.shape[0], axis=0)


def distinct_window(
    source,
    shift: int,
    window_length: int = WINDOW_LENGTH,
    base_offset: int = WINDOW_BASE_OFFSET,
) -> np.ndarray:
    """Eq. (5): v^win(s)_i = source[i + base_offset + shift], i = 1..window_length.

    Every returned window holds `window_length` distinct source frames, so the
    imposed global perturbation changes represented phase without clipping,
    padding or boundary repetition.
    """
    arr = _as_frames(source)
    start = int(base_offset) + int(shift)
    stop = start + int(window_length)
    if start < 0 or stop > arr.shape[0]:
        raise ValueError(
            "distinct_window(shift={}) needs source frames [{}, {}) but the "
            "source rollout has {} frames; render a longer rollout rather than "
            "padding".format(shift, start, stop, arr.shape[0])
        )
    return arr[start:stop]


def rate_warp(
    source,
    rate: float,
    length: int = WINDOW_LENGTH,
    rounding: str = RATE_WARP_ROUNDING,
) -> np.ndarray:
    """Monotone temporal-rate mismatch: j_i = round(1 + rate * (i - 1)).

    Models the deployment failure the paper is motivated by, where the physical
    robot falls progressively behind (rate < 1) or ahead of (rate > 1) the
    fixed-rate generated clip. Unlike a fixed shift, the phase error grows with
    the horizon.

    Evaluation-only: never used to construct RAFC candidates or training inputs.
    """
    arr = _as_frames(source)
    if rate <= 0:
        raise ValueError("rate must be positive, got {}".format(rate))
    idx = np.array(
        [_round_index(float(rate) * k, rounding) for k in range(int(length))],
        dtype=np.int64,
    )
    needed = int(idx[-1]) + 1
    if needed > arr.shape[0]:
        raise ValueError(
            "rate_warp(rate={}, length={}) needs {} source frames but only {} "
            "are available; render a longer rollout rather than padding".format(
                rate, length, needed, arr.shape[0]
            )
        )
    return arr[idx]


def required_source_length(
    rate: float, length: int = WINDOW_LENGTH, rounding: str = RATE_WARP_ROUNDING
) -> int:
    """Minimum source rollout length for a given warp, for dataset generation."""
    return _round_index(float(rate) * (int(length) - 1), rounding) + 1


def warp_indices(rate: float, length: int = WINDOW_LENGTH, rounding: str = RATE_WARP_ROUNDING):
    """The literal source indices a warp selects.

    Print this for both rates and compare against the run that produced the
    reported warp numbers before trusting the public code to reproduce them.
    """
    return [_round_index(float(rate) * k, rounding) for k in range(int(length))]


def make_rafc_candidates(frames, shifts=RAFC_SHIFTS) -> np.ndarray:
    """Stack [null, shift(s_0), shift(s_1), ...] as RAFC consumes them.

    Constructed after any global perturbation has been applied, matching the
    evaluation protocol in Sec. III-C.
    """
    arr = _as_frames(frames)
    out = [make_null_future(arr)]
    out.extend(shift_future(arr, s) for s in shifts)
    return np.stack(out, axis=0)


def apply_global_perturbation(
    frames,
    mode: str,
    shift: int = 0,
    rate: float = 1.0,
    source=None,
) -> np.ndarray:
    """Single dispatch point for the evaluation-time perturbation.

    mode:
        "none"      pass the clip through unchanged
        "shift"     Eq. (4) clipped shift of the 16-frame clip
        "windowed"  Eq. (5) distinct-frame window, requires `source`
        "rate"      monotone rate warp, requires `source`
    """
    mode = str(mode).strip().lower()
    if mode in ("none", "identity", ""):
        return _as_frames(frames)
    if mode == "shift":
        return shift_future(frames, shift)
    if mode == "windowed":
        if source is None:
            raise ValueError("mode='windowed' requires the long source rollout")
        return distinct_window(source, shift)
    if mode == "rate":
        if source is None:
            raise ValueError("mode='rate' requires the long source rollout")
        return rate_warp(source, rate, length=_as_frames(frames).shape[0])
    raise ValueError("unknown perturbation mode: {}".format(mode))
