"""Reusable trainer integration for detail_inv_t loss weighting."""

from __future__ import annotations

import argparse
from typing import Callable

from detail_inv_t import (
    SUPPORTED_LOSS_WEIGHTINGS,
    compute_detail_inv_t_weights,
    validate_detail_inv_t_settings,
)


EXTERNAL_DETAIL_INV_T_OPTIONS = (
    "--loss_weighting",
    "--detail_inv_t_min",
    "--detail_inv_t_max",
    "--weight_cap_ratio",
)


def add_detail_inv_t_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    conflicts = [option for option in EXTERNAL_DETAIL_INV_T_OPTIONS if option in parser._option_string_actions]
    if conflicts:
        raise RuntimeError(
            "The installed sd-scripts now defines external loss options: "
            f"{', '.join(conflicts)}. Review the upstream implementation before using this extension."
        )

    parser.add_argument(
        "--loss_weighting",
        type=str,
        default="none",
        choices=SUPPORTED_LOSS_WEIGHTINGS,
        help="External loss weighting applied after the per-sample base loss.",
    )
    parser.add_argument(
        "--detail_inv_t_min",
        type=float,
        default=1.0,
        help="Lower clamp bound for detail_inv_t weights (default: 1.0).",
    )
    parser.add_argument(
        "--detail_inv_t_max",
        type=float,
        default=5.0,
        help="Upper clamp bound for detail_inv_t weights (default: 5.0).",
    )
    parser.add_argument(
        "--weight_cap_ratio",
        type=float,
        default=0.0,
        help="Maximum within-batch weight ratio; 0 disables the cap (default: 0).",
    )
    return parser


def validate_detail_inv_t_args(args: argparse.Namespace) -> None:
    validate_detail_inv_t_settings(
        loss_weighting=str(args.loss_weighting or "none").lower(),
        detail_inv_t_min=float(args.detail_inv_t_min),
        detail_inv_t_max=float(args.detail_inv_t_max),
        weight_cap_ratio=float(args.weight_cap_ratio),
        upstream_weighting_scheme=str(args.weighting_scheme or "uniform").lower(),
    )


class DetailInvTTrainerMixin:
    """Apply detail_inv_t after the per-sample loss. Set the scale when timesteps are not in [0, 1]."""

    detail_inv_t_timestep_scale = 1.0

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        loss = super().post_process_loss(loss, args, timesteps, noise_scheduler)
        if str(args.loss_weighting or "none").lower() != "detail_inv_t":
            return loss

        weights = compute_detail_inv_t_weights(
            timesteps / self.detail_inv_t_timestep_scale,
            detail_inv_t_min=float(args.detail_inv_t_min),
            detail_inv_t_max=float(args.detail_inv_t_max),
            weight_cap_ratio=float(args.weight_cap_ratio),
        ).to(device=loss.device, dtype=loss.dtype)

        if loss.ndim != 1 or loss.shape[0] != weights.shape[0]:
            raise RuntimeError(
                "detail_inv_t expected one loss value per timestep, got "
                f"loss shape {tuple(loss.shape)} and timestep shape {tuple(timesteps.shape)}"
            )
        return loss * weights

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_loss_weighting"] = args.loss_weighting
        metadata["ss_detail_inv_t_min"] = args.detail_inv_t_min
        metadata["ss_detail_inv_t_max"] = args.detail_inv_t_max
        metadata["ss_weight_cap_ratio"] = args.weight_cap_ratio


def show_timesteps_with_detail_inv_t(
    args: argparse.Namespace,
    show_timesteps: Callable[[argparse.Namespace], None],
    weighting_module: object,
    weighting_name: str,
) -> None:
    """Show the upstream timestep preview with its weighting function swapped for detail_inv_t."""
    if str(args.loss_weighting or "none").lower() != "detail_inv_t":
        show_timesteps(args)
        return

    original_compute = getattr(weighting_module, weighting_name)

    def compute_external_weighting(_weighting_scheme, sigmas):
        weights = compute_detail_inv_t_weights(
            sigmas,
            detail_inv_t_min=float(args.detail_inv_t_min),
            detail_inv_t_max=float(args.detail_inv_t_max),
            weight_cap_ratio=float(args.weight_cap_ratio),
        )
        return weights.reshape(sigmas.shape)

    setattr(weighting_module, weighting_name, compute_external_weighting)
    try:
        show_timesteps(args)
    finally:
        setattr(weighting_module, weighting_name, original_compute)
