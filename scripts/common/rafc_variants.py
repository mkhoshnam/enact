"""The five conditions in Table VI, expressed as one gate specification.

Rather than forking the training script per ablation, each variant is a small
declarative spec that tells the existing TD3 loop three things:

    how many futures to build per step,
    whether the null branch exists,
    how alpha and the temporal weights are produced.

The mixing arithmetic is identical in every case, so a bug in the blend cannot
affect one row of Table VI and not another.

Mapping to the paper (Table VI, Avg. shifted):

    forced     51.8   one future, no candidates, no gate
    shiftaug   58.2   one future, per-episode offset from S, no gate
    uniform    62.2   three candidates, fixed 1/3 weights, alpha = 1
    no_null    66.6   three candidates, learned weights, alpha = 1
    full       69.2   three candidates, learned weights, learned alpha

The gap between `uniform` and `full` is what separates candidate ensembling
from learned reliability; keep both runnable or that claim is unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

try:  # package context (scripts.common)
    from .paper_protocol import LOCAL_SHIFTS
except ImportError:  # flat context (directory on sys.path)
    from paper_protocol import LOCAL_SHIFTS


class RAFCVariant(str, Enum):
    FORCED = "forced"
    SHIFTAUG = "shiftaug"
    UNIFORM = "uniform"
    NO_NULL = "no_null"
    FULL = "full"


@dataclass(frozen=True)
class VariantSpec:
    """What the training loop needs to know about a variant."""

    variant: RAFCVariant

    # Build the {-2, 0, +2} candidate set at every step?
    uses_candidates: bool

    # Build the frozen-first-frame null branch and mix it in?
    uses_null: bool

    # Are alpha / weights produced by the learned gate MLP?
    learned_alpha: bool
    learned_weights: bool

    # Sample one fixed offset per episode during RL (ShiftAug only).
    episode_shift_augmentation: bool

    @property
    def num_futures_per_step(self) -> int:
        """Frozen-BC forward passes needed per control step.

        This is the latency multiplier: 1 for forced/shiftaug, 4 for the RAFC
        family (null + three candidates). Report it alongside control rate.
        """
        n = len(LOCAL_SHIFTS) if self.uses_candidates else 1
        return n + (1 if self.uses_null else 0)

    @property
    def trains_gate_parameters(self) -> bool:
        return self.learned_alpha or self.learned_weights


_SPECS = {
    RAFCVariant.FORCED: VariantSpec(
        variant=RAFCVariant.FORCED,
        uses_candidates=False,
        uses_null=False,
        learned_alpha=False,
        learned_weights=False,
        episode_shift_augmentation=False,
    ),
    RAFCVariant.SHIFTAUG: VariantSpec(
        variant=RAFCVariant.SHIFTAUG,
        uses_candidates=False,
        uses_null=False,
        learned_alpha=False,
        learned_weights=False,
        episode_shift_augmentation=True,
    ),
    RAFCVariant.UNIFORM: VariantSpec(
        variant=RAFCVariant.UNIFORM,
        uses_candidates=True,
        uses_null=False,
        learned_alpha=False,
        learned_weights=False,
        episode_shift_augmentation=False,
    ),
    RAFCVariant.NO_NULL: VariantSpec(
        variant=RAFCVariant.NO_NULL,
        uses_candidates=True,
        uses_null=False,
        learned_alpha=False,
        learned_weights=True,
        episode_shift_augmentation=False,
    ),
    RAFCVariant.FULL: VariantSpec(
        variant=RAFCVariant.FULL,
        uses_candidates=True,
        uses_null=True,
        learned_alpha=True,
        learned_weights=True,
        episode_shift_augmentation=False,
    ),
}


def get_spec(variant) -> VariantSpec:
    return _SPECS[RAFCVariant(variant)]


def center_index(shifts: Sequence[int] = LOCAL_SHIFTS) -> int:
    """Index of the unshifted candidate."""
    return list(shifts).index(0)


def resolve_gate(variant, gate_alpha=None, gate_weights=None, batch_size=1, backend=None):
    """Return (alpha, weights) for the blend, given raw gate outputs.

    `gate_alpha` and `gate_weights` are the learned gate's outputs and are only
    consulted when the spec says the variant learns them. Everything else is
    forced to a constant here, so the training loop has no per-variant
    branching in the blend itself.

    `backend` supplies `full(shape, value)` and `stack`; pass a small torch
    shim in the training script. Defaults to numpy.
    """
    spec = get_spec(variant)
    if backend is None:
        import numpy as np

        backend = _NumpyBackend(np)

    n = len(LOCAL_SHIFTS)

    if spec.learned_weights:
        if gate_weights is None:
            raise ValueError(
                "variant {} learns temporal weights but none were "
                "supplied".format(spec.variant.value)
            )
        weights = gate_weights
    elif spec.uses_candidates:
        # Uniform averaging over the same candidates the gate would see.
        weights = backend.full((batch_size, n), 1.0 / n)
    else:
        # Single future: all mass on the unshifted candidate.
        weights = backend.onehot(batch_size, n, center_index())

    if spec.learned_alpha:
        if gate_alpha is None:
            raise ValueError(
                "variant {} learns alpha but none was supplied".format(
                    spec.variant.value
                )
            )
        alpha = gate_alpha
    else:
        # No null branch means full trust in the future by construction.
        alpha = backend.full((batch_size, 1), 1.0)

    return alpha, weights


def blend(alpha, weights, value_null, value_candidates, backend=None):
    """Eqs. (9)-(11), applied to any of f, g or a_BC.

    value_candidates has shape (batch, num_shifts, dim); value_null (batch, dim).
    When alpha == 1 the null term vanishes, so variants without a null branch
    may pass zeros for value_null.
    """
    if backend is None:
        import numpy as np

        backend = _NumpyBackend(np)
    mixed = backend.weighted_sum(weights, value_candidates)
    return (1.0 - alpha) * value_null + alpha * mixed


class _NumpyBackend:
    def __init__(self, np):
        self.np = np

    def full(self, shape, value):
        return self.np.full(shape, value, dtype=self.np.float32)

    def onehot(self, batch, n, idx):
        out = self.np.zeros((batch, n), dtype=self.np.float32)
        out[:, idx] = 1.0
        return out

    def weighted_sum(self, weights, candidates):
        return (weights[..., None] * candidates).sum(axis=1)


class TorchBackend:
    """Drop-in backend for the training script."""

    def __init__(self, torch, device=None, dtype=None):
        self.torch = torch
        self.device = device
        self.dtype = dtype

    def full(self, shape, value):
        return self.torch.full(shape, value, device=self.device, dtype=self.dtype)

    def onehot(self, batch, n, idx):
        out = self.torch.zeros((batch, n), device=self.device, dtype=self.dtype)
        out[:, idx] = 1.0
        return out

    def weighted_sum(self, weights, candidates):
        return (weights.unsqueeze(-1) * candidates).sum(dim=1)
