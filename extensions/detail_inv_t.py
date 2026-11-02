"""Timestep-dependent loss weighting for rectified-flow models, implemented outside sd-scripts."""

from __future__ import annotations

import torch


SUPPORTED_LOSS_WEIGHTINGS = ("none", "detail_inv_t")
_EPSILON = 1e-4


def validate_detail_inv_t_settings(
    loss_weighting: str,
    detail_inv_t_min: float,
    detail_inv_t_max: float,
    weight_cap_ratio: float,
    upstream_weighting_scheme: str,
) -> None:
    if loss_weighting not in SUPPORTED_LOSS_WEIGHTINGS:
        raise ValueError(
            f"Unsupported loss_weighting: {loss_weighting!r}. "
            f"Expected one of: {', '.join(SUPPORTED_LOSS_WEIGHTINGS)}"
        )

    if not 1.0 <= detail_inv_t_min <= 20.0:
        raise ValueError("detail_inv_t_min must be between 1.0 and 20.0")
    if not 0.1 <= detail_inv_t_max <= 50.0:
        raise ValueError("detail_inv_t_max must be between 0.1 and 50.0")
    if detail_inv_t_min > detail_inv_t_max:
        raise ValueError("detail_inv_t_min must not be greater than detail_inv_t_max")
    if not 0.0 <= weight_cap_ratio <= 50.0:
        raise ValueError("weight_cap_ratio must be between 0.0 and 50.0")

    if loss_weighting == "detail_inv_t" and upstream_weighting_scheme in ("sigma_sqrt", "cosmap"):
        raise ValueError(
            "loss_weighting='detail_inv_t' cannot be combined with "
            f"weighting_scheme={upstream_weighting_scheme!r}; use 'uniform', 'none', "
            "'logit_normal', or 'mode'"
        )


def compute_detail_inv_t_weights(
    timesteps: torch.Tensor,
    detail_inv_t_min: float = 1.0,
    detail_inv_t_max: float = 5.0,
    weight_cap_ratio: float = 0.0,
) -> torch.Tensor:
    """Return per-sample weights for normalized rectified-flow timesteps."""
    t = timesteps.detach().float().reshape(-1).clamp(_EPSILON, 1.0 - _EPSILON)
    weights = t.reciprocal().clamp(min=float(detail_inv_t_min), max=float(detail_inv_t_max))

    if weight_cap_ratio > 1.0 and weights.numel() > 0:
        minimum = weights.min().clamp_min(_EPSILON)
        weights = weights.clamp(max=minimum * float(weight_cap_ratio))

    return weights
