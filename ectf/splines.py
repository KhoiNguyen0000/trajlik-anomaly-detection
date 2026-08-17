import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)


def rational_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    unnormalized_derivatives: Tensor,
    *,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Elementwise monotonic rational-quadratic spline with linear tails.

    The first dimensions of every parameter tensor must match ``inputs``;
    the final dimension enumerates spline bins (or internal derivatives).
    Returned log determinants have the same shape as ``inputs``.
    """

    if tail_bound <= 0:
        raise ValueError("tail_bound must be positive")
    if inputs.shape != unnormalized_widths.shape[:-1]:
        raise ValueError("Spline parameter shape does not match inputs")
    if unnormalized_widths.shape != unnormalized_heights.shape:
        raise ValueError("Spline widths and heights must have equal shapes")

    num_bins = unnormalized_widths.shape[-1]
    if num_bins < 2:
        raise ValueError("A spline requires at least two bins")
    if unnormalized_derivatives.shape != (*inputs.shape, num_bins - 1):
        raise ValueError("Expected one derivative for each internal knot")
    if min_bin_width * num_bins >= 2 * tail_bound:
        raise ValueError("min_bin_width is too large for tail_bound")
    if min_bin_height * num_bins >= 2 * tail_bound:
        raise ValueError("min_bin_height is too large for tail_bound")

    interval = 2.0 * tail_bound
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (interval - min_bin_width * num_bins) * widths
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), (1, 0))
    cumwidths = cumwidths - tail_bound
    cumwidths[..., 0] = -tail_bound
    cumwidths[..., -1] = tail_bound

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (
        interval - min_bin_height * num_bins
    ) * heights
    cumheights = F.pad(torch.cumsum(heights, dim=-1), (1, 0))
    cumheights = cumheights - tail_bound
    cumheights[..., 0] = -tail_bound
    cumheights[..., -1] = tail_bound

    boundary_parameter = math.log(math.expm1(1.0 - min_derivative))
    derivatives = F.pad(
        unnormalized_derivatives,
        (1, 1),
        value=boundary_parameter,
    )
    derivatives = min_derivative + F.softplus(derivatives)

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    safe_inputs = inputs.clamp(
        min=-tail_bound + torch.finfo(inputs.dtype).eps,
        max=tail_bound - torch.finfo(inputs.dtype).eps,
    )
    knots = cumheights if inverse else cumwidths
    bin_indices = torch.sum(
        safe_inputs.unsqueeze(-1) >= knots[..., 1:-1],
        dim=-1,
    )

    input_cumwidths = _gather(cumwidths, bin_indices)
    input_bin_widths = _gather(widths, bin_indices)
    input_cumheights = _gather(cumheights, bin_indices)
    input_bin_heights = _gather(heights, bin_indices)
    input_delta = input_bin_heights / input_bin_widths
    input_derivatives = _gather(derivatives, bin_indices)
    input_derivatives_plus_one = _gather(
        derivatives,
        bin_indices + 1,
    )

    if inverse:
        shifted = safe_inputs - input_cumheights
        derivative_sum = (
            input_derivatives
            + input_derivatives_plus_one
            - 2.0 * input_delta
        )
        a = shifted * derivative_sum + input_bin_heights * (
            input_delta - input_derivatives
        )
        b = input_bin_heights * input_derivatives - shifted * derivative_sum
        c = -input_delta * shifted
        discriminant = (b.square() - 4.0 * a * c).clamp_min(0.0)
        denominator = -b - torch.sqrt(discriminant)
        quadratic_root = 2.0 * c / denominator
        linear_root = -c / b
        theta = torch.where(a.abs() < 1e-7, linear_root, quadratic_root)
        theta = theta.clamp(0.0, 1.0)
        inside_outputs = input_cumwidths + theta * input_bin_widths
    else:
        theta = (safe_inputs - input_cumwidths) / input_bin_widths

    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = input_delta + (
        input_derivatives
        + input_derivatives_plus_one
        - 2.0 * input_delta
    ) * theta_one_minus_theta

    if not inverse:
        numerator = input_bin_heights * (
            input_delta * theta.square()
            + input_derivatives * theta_one_minus_theta
        )
        inside_outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.square() * (
        input_derivatives_plus_one * theta.square()
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta).square()
    )
    inside_logabsdet = torch.log(derivative_numerator) - 2.0 * torch.log(
        denominator
    )
    if inverse:
        inside_logabsdet = -inside_logabsdet

    outputs = torch.where(inside, inside_outputs, inputs)
    logabsdet = torch.where(
        inside,
        inside_logabsdet,
        torch.zeros_like(inputs),
    )
    return outputs, logabsdet
