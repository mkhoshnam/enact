"""Tests that pin each transform to a property the paper claims in text.

Each test names the claim it protects, so a future refactor that breaks one
also tells you which sentence of the paper became false.
"""

import numpy as np
import pytest

from future_transforms import (
    EVAL_RATES,
    EVAL_SHIFTS,
    SOURCE_ROLLOUT_LENGTH,
    WINDOW_LENGTH,
    apply_global_perturbation,
    distinct_window,
    make_null_future,
    make_rafc_candidates,
    rate_warp,
    required_source_length,
    shift_future,
)


def make_source(n=SOURCE_ROLLOUT_LENGTH, h=4, w=4, c=3):
    """Frames whose pixel value encodes their source index, so identity of a
    returned frame can be recovered exactly."""
    frames = np.zeros((n, h, w, c), dtype=np.uint8)
    for i in range(n):
        frames[i] = i
    return frames


def frame_ids(frames):
    return [int(f[0, 0, 0]) for f in frames]


# --- Eq. (4): clipped shift -------------------------------------------------

def test_shift_zero_is_identity():
    clip = make_source(WINDOW_LENGTH)
    assert frame_ids(shift_future(clip, 0)) == frame_ids(clip)


def test_shift_repeats_boundary_frames():
    """Paper: 'out-of-range indices repeat the boundary frame'."""
    clip = make_source(WINDOW_LENGTH)
    assert frame_ids(shift_future(clip, -6))[:6] == [0] * 6
    assert frame_ids(shift_future(clip, 6))[-6:] == [WINDOW_LENGTH - 1] * 6


def test_shift_loses_temporal_diversity_at_large_offsets():
    """Paper: 'At large shifts, boundary repetition additionally reduces
    temporal diversity.' This is the confound distinct_window controls for."""
    clip = make_source(WINDOW_LENGTH)
    assert len(set(frame_ids(shift_future(clip, 6)))) < WINDOW_LENGTH


# --- Eq. (5): distinct-frame source window ----------------------------------

def test_window_has_no_repeated_frames_at_every_offset():
    """Paper: 'Even at s = +-6, each globally shifted source window contains 16
    distinct frames.' This is the central control; if it fails, Table I's
    windowed rows no longer mean what the caption says."""
    source = make_source()
    for s in EVAL_SHIFTS:
        ids = frame_ids(distinct_window(source, s))
        assert len(ids) == WINDOW_LENGTH
        assert len(set(ids)) == WINDOW_LENGTH


def test_window_spans_frames_1_to_16_through_13_to_28():
    """Paper: 'the windows span frames 1-16 through 13-28' (1-indexed)."""
    source = make_source()
    assert frame_ids(distinct_window(source, -6))[0] == 0
    assert frame_ids(distinct_window(source, -6))[-1] == 15
    assert frame_ids(distinct_window(source, 6))[0] == 12
    assert frame_ids(distinct_window(source, 6))[-1] == 27


def test_window_is_contiguous_and_monotone():
    source = make_source()
    for s in EVAL_SHIFTS:
        ids = frame_ids(distinct_window(source, s))
        assert ids == list(range(ids[0], ids[0] + WINDOW_LENGTH))


def test_window_raises_instead_of_padding():
    with pytest.raises(ValueError):
        distinct_window(make_source(20), 6)


# --- Eq. (6): null future ---------------------------------------------------

def test_null_future_repeats_first_frame_and_preserves_length():
    clip = make_source(WINDOW_LENGTH)
    null = make_null_future(clip)
    assert null.shape == clip.shape
    assert frame_ids(null) == [0] * WINDOW_LENGTH


def test_null_is_derived_from_the_same_clip_as_the_candidates():
    """Paper: 'The null and shifted candidates share the same task command and
    first frame, so the gate cannot identify the perturbation from a different
    task or initial scene.'

    Read literally that sentence is false: the local +2 candidate begins at
    source frame 2, not frame 0. What actually holds is that the null branch is
    built from the same clip's first frame as the unshifted candidate, so no
    candidate carries a different task or initial scene. Only the negative
    shifts coincide with the null on frame 0, and then only through clipping.
    """
    clip = make_source(WINDOW_LENGTH)
    null, cand_minus, cand_zero, cand_plus = make_rafc_candidates(clip)
    assert frame_ids(null) == [0] * WINDOW_LENGTH
    assert int(cand_zero[0][0, 0, 0]) == int(null[0][0, 0, 0])
    assert int(cand_minus[0][0, 0, 0]) == 0
    assert int(cand_plus[0][0, 0, 0]) == 2


# --- rate warp --------------------------------------------------------------

def test_rate_warp_is_monotone_non_decreasing():
    source = make_source(40)
    for rate in EVAL_RATES:
        ids = frame_ids(rate_warp(source, rate))
        assert all(b >= a for a, b in zip(ids, ids[1:]))


def test_rate_warp_phase_error_grows_with_horizon():
    """The property that distinguishes a warp from a fixed shift: the gap to
    the unwarped clip is zero at the first frame and grows monotonically."""
    source = make_source(40)
    for rate in EVAL_RATES:
        ids = frame_ids(rate_warp(source, rate))
        errors = [abs(v - i) for i, v in enumerate(ids)]
        assert errors[0] == 0
        assert errors[-1] > errors[len(errors) // 2] >= 0


def test_rate_warp_unit_rate_is_identity():
    source = make_source(40)
    assert frame_ids(rate_warp(source, 1.0)) == list(range(WINDOW_LENGTH))


def test_rate_warp_requires_enough_source_frames():
    """Paper: 'with enough source frames to avoid padding'. rho = 1.25 over 16
    output frames reaches source index 19, so a 16-frame clip is not enough."""
    assert required_source_length(1.25) == 20
    assert required_source_length(0.75) == 12
    with pytest.raises(ValueError):
        rate_warp(make_source(WINDOW_LENGTH), 1.25)


def test_rate_warp_tie_breaking_is_pinned():
    """0.75 hits exact halves, where the two rounding rules diverge (they agree
    at 1.5 but not at 4.5). The convention lives in paper_protocol so it can be
    set to whatever produced the reported warp numbers."""
    ids = frame_ids(rate_warp(make_source(40), 0.75))
    assert ids[:5] == [0, 1, 2, 2, 3]


# --- dispatch ---------------------------------------------------------------

def test_apply_global_perturbation_matches_direct_calls():
    source = make_source()
    clip = distinct_window(source, 0)
    assert frame_ids(apply_global_perturbation(clip, "shift", shift=4)) == frame_ids(
        shift_future(clip, 4)
    )
    assert frame_ids(
        apply_global_perturbation(clip, "windowed", shift=4, source=source)
    ) == frame_ids(distinct_window(source, 4))
    assert frame_ids(
        apply_global_perturbation(clip, "rate", rate=0.75, source=make_source(40))
    ) == frame_ids(rate_warp(make_source(40), 0.75))


def test_rafc_candidates_built_after_global_perturbation():
    """Paper: 'At evaluation, RAFC constructs its local candidates after the
    global perturbation is applied.' Candidate 2 (local +2) under a global -6
    must have effective shift -4, which is the mechanism the Results section
    uses to explain the +-6 numbers."""
    source = make_source()
    perturbed = distinct_window(source, -6)
    candidates = make_rafc_candidates(perturbed)
    local_plus_two = frame_ids(candidates[3])
    assert local_plus_two[0] == 2
