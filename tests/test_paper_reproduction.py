"""Tests for the protocol constants, ablation specs and result aggregation."""

import json
import math

import numpy as np
import pytest

import paper_protocol as P
import result_io as R
from future_transforms import warp_indices
from rafc_variants import (
    RAFCVariant,
    blend,
    center_index,
    get_spec,
    resolve_gate,
)


# --- protocol ---------------------------------------------------------------

def test_protocol_matches_the_manuscript():
    assert P.SEEDS == (42, 43, 44)
    assert P.MAX_EPISODE_STEPS == 200
    assert P.NUM_EVAL_EPISODES == 300
    assert P.TOTAL_ENV_STEPS == 140_000
    assert P.FUTURE_FRAMES == 16 and P.FUTURE_BINS == 4
    assert len(P.CALVIN_TASKS) == 8 and len(P.ROBOCASA_TASKS) == 5
    assert len(P.NONZERO_GLOBAL_SHIFTS) == 6


def test_window_geometry_fits_the_source_rollout():
    for s in P.GLOBAL_SHIFTS:
        start = P.WINDOW_BASE_OFFSET + s
        assert start >= 0
        assert start + P.FUTURE_FRAMES <= P.SOURCE_ROLLOUT_FRAMES


# --- ablation variants ------------------------------------------------------

def test_five_variants_cover_table_six():
    assert {v.value for v in RAFCVariant} == {
        "forced",
        "shiftaug",
        "uniform",
        "no_null",
        "full",
    }


def test_only_full_rafc_uses_the_null_branch():
    """Table VI isolates the null fallback as the last +2.6 points, so exactly
    one variant may have it."""
    with_null = [v for v in RAFCVariant if get_spec(v).uses_null]
    assert with_null == [RAFCVariant.FULL]


def test_only_shiftaug_uses_episode_level_augmentation():
    aug = [v for v in RAFCVariant if get_spec(v).episode_shift_augmentation]
    assert aug == [RAFCVariant.SHIFTAUG]


def test_forced_and_shiftaug_use_a_single_future():
    """Paper: ShiftAug 'uses one future without RAFC or a null branch'."""
    for v in (RAFCVariant.FORCED, RAFCVariant.SHIFTAUG):
        spec = get_spec(v)
        assert not spec.uses_candidates
        assert spec.num_futures_per_step == 1


def test_rafc_family_costs_four_forward_passes():
    """The latency multiplier that has to be reported next to the control rate."""
    assert get_spec(RAFCVariant.FULL).num_futures_per_step == 4
    assert get_spec(RAFCVariant.UNIFORM).num_futures_per_step == 3
    assert get_spec(RAFCVariant.NO_NULL).num_futures_per_step == 3


def test_uniform_weights_are_exactly_one_third():
    alpha, weights = resolve_gate(RAFCVariant.UNIFORM, batch_size=2)
    assert np.allclose(weights, 1.0 / 3.0)
    assert np.allclose(alpha, 1.0)


def test_no_null_forces_alpha_one_but_learns_weights():
    learned = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
    alpha, weights = resolve_gate(
        RAFCVariant.NO_NULL, gate_weights=learned, batch_size=1
    )
    assert np.allclose(alpha, 1.0)
    assert np.allclose(weights, learned)


def test_full_rafc_passes_both_learned_quantities_through():
    a = np.array([[0.42]], dtype=np.float32)
    w = np.array([[0.1, 0.6, 0.3]], dtype=np.float32)
    alpha, weights = resolve_gate(RAFCVariant.FULL, gate_alpha=a, gate_weights=w)
    assert np.allclose(alpha, a) and np.allclose(weights, w)


def test_single_future_variants_put_all_mass_on_the_unshifted_candidate():
    _, weights = resolve_gate(RAFCVariant.FORCED, batch_size=1)
    assert weights[0, center_index()] == 1.0
    assert weights.sum() == 1.0


def test_learned_variant_without_gate_output_raises():
    with pytest.raises(ValueError):
        resolve_gate(RAFCVariant.FULL, batch_size=1)


def test_blend_reduces_to_the_candidate_mix_when_alpha_is_one():
    """Variants without a null branch must be unaffected by whatever is passed
    as the null value."""
    cand = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    alpha, weights = resolve_gate(RAFCVariant.UNIFORM, batch_size=1)
    out_zeros = blend(alpha, weights, np.zeros((1, 2), np.float32), cand)
    out_junk = blend(alpha, weights, np.full((1, 2), 99.0, np.float32), cand)
    assert np.allclose(out_zeros, out_junk)
    assert np.allclose(out_zeros, cand.mean(axis=1))


def test_blend_returns_the_null_value_when_alpha_is_zero():
    cand = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    null = np.full((1, 2), 7.0, dtype=np.float32)
    out = blend(np.zeros((1, 1), np.float32), np.full((1, 3), 1 / 3, np.float32), null, cand)
    assert np.allclose(out, null)


# --- warp reproducibility ---------------------------------------------------

def test_warp_indices_are_printable_for_provenance_checking():
    """These exact index lists are what must be compared against the run that
    produced the reported warp numbers. Under banker's rounding, 0.75 * 2 = 1.5
    maps to frame 2 and 0.75 * 6 = 4.5 maps to frame 4."""
    idx = warp_indices(0.75, rounding="rint")
    assert idx[:8] == [0, 1, 2, 2, 3, 4, 4, 5]
    assert warp_indices(1.0, rounding="rint") == list(range(16))
    assert warp_indices(1.25, rounding="rint")[-1] == 19


def test_rounding_rules_actually_disagree():
    """If they agreed, the convention would not matter and the knob could go."""
    assert warp_indices(0.75, rounding="rint") != warp_indices(0.75, rounding="half_up")


# --- result records ---------------------------------------------------------

def make_record(**kw):
    base = dict(
        benchmark="calvin",
        task="open_drawer",
        method="full",
        seed=42,
        episode=0,
        success=True,
    )
    base.update(kw)
    return R.EpisodeRecord(**base)


def test_record_rejects_out_of_protocol_shift():
    with pytest.raises(ValueError):
        make_record(perturbation="shift", shift=3).validate()


def test_record_rejects_unnormalized_temporal_weights():
    with pytest.raises(ValueError):
        make_record(weights_mean=[0.5, 0.5, 0.5]).validate()


def test_record_rejects_alpha_outside_unit_interval():
    with pytest.raises(ValueError):
        make_record(alpha_mean=1.4).validate()


def test_record_accepts_a_well_formed_gate_row():
    make_record(alpha_mean=0.79, weights_mean=[0.17, 0.60, 0.23]).validate()


def test_record_carries_protocol_and_commit_provenance():
    r = make_record()
    assert "horizon=200" in r.protocol
    assert "episodes=300" in r.protocol
    assert isinstance(r.commit, str) and r.commit


# --- aggregation ------------------------------------------------------------

def synthetic(tasks, seeds, n, success_fn, **extra):
    out = []
    for t in tasks:
        for s in seeds:
            for e in range(n):
                row = dict(
                    benchmark="calvin",
                    task=t,
                    method="full",
                    seed=s,
                    episode=e,
                    success=success_fn(t, s, e),
                )
                row.update(extra)
                out.append(row)
    return out


def test_aggregation_refuses_a_short_run():
    """The central guard: a run that lost episodes must raise rather than
    report a plausible-looking number."""
    tasks = P.CALVIN_TASKS[:2]
    recs = synthetic(tasks, P.SEEDS, 299, lambda t, s, e: e % 2 == 0)
    with pytest.raises(R.IncompleteRun):
        R.seed_success_rates(recs, tasks, episodes_per_task_seed=300)


def test_aggregation_refuses_a_missing_seed():
    tasks = P.CALVIN_TASKS[:2]
    recs = synthetic(tasks, (42, 43), 10, lambda t, s, e: True)
    with pytest.raises(R.IncompleteRun):
        R.seed_success_rates(recs, tasks, episodes_per_task_seed=10)


def test_task_balanced_rate_weights_tasks_equally():
    """A task with more episodes must not dominate the aggregate."""
    tasks = ("a", "b")
    recs = []
    for e in range(10):
        recs.append(dict(benchmark="calvin", task="a", method="m", seed=42,
                         episode=e, success=True))
        recs.append(dict(benchmark="calvin", task="b", method="m", seed=42,
                         episode=e, success=e < 5))
    rates = R.seed_success_rates(recs, tasks, seeds=(42,), episodes_per_task_seed=10)
    assert math.isclose(rates[42], 0.75)


def test_mean_std_uses_sample_standard_deviation():
    mean, sd = R.mean_std([0.6, 0.7, 0.8])
    assert math.isclose(mean, 0.7)
    assert math.isclose(sd, 0.1, abs_tol=1e-9)


def test_cell_formats_percentages():
    tasks = ("a",)
    recs = []
    for seed, rate in zip(P.SEEDS, (0.6, 0.7, 0.8)):
        for e in range(10):
            recs.append(dict(benchmark="calvin", task="a", method="m", seed=seed,
                             episode=e, success=e < int(rate * 10)))
    out = R.cell(recs, tasks, episodes_per_task_seed=10)
    assert out.startswith("$70.0") and "10.0$" in out


def test_writer_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    with R.ResultWriter(path) as w:
        w.write(make_record(episode=0))
        w.write(make_record(episode=1, success=False))
    rows = R.load_records(path)
    assert len(rows) == 2
    assert [r["success"] for r in rows] == [True, False]


def test_writer_rejects_an_invalid_record_before_writing(tmp_path):
    path = tmp_path / "out.jsonl"
    with R.ResultWriter(path) as w:
        with pytest.raises(ValueError):
            w.write(make_record(alpha_mean=2.0))
    assert not path.exists() or path.read_text() == ""
