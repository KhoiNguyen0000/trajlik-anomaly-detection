from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def _prepare_z0(value: Tensor) -> tuple[Tensor, bool]:
    if not isinstance(value, Tensor):
        raise TypeError("z_0 must be a torch.Tensor")

    if value.ndim == 4:
        return value, False

    if value.ndim == 3:
        return value.unsqueeze(0), True

    raise ValueError(
        "z_0 must have shape [B,C,H,W] or [C,H,W], "
        f"got {tuple(value.shape)}"
    )


def _prepare_sequence(
    value: Tensor | Sequence[Tensor],
    name: str,
    unbatched: bool,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.ndim == 5:
            return value

        if value.ndim == 4 and unbatched:
            return value.unsqueeze(0)

        raise ValueError(
            f"{name} must have shape [B,S,C,H,W]"
            f" or [S,C,H,W], got {tuple(value.shape)}"
        )

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a Tensor or a sequence of Tensors"
        )

    if not value:
        raise ValueError(f"{name} cannot be empty")

    if not all(isinstance(step, Tensor) for step in value):
        raise TypeError(f"Every item in {name} must be a torch.Tensor")

    expected_ndim = 3 if unbatched else 4
    if any(step.ndim != expected_ndim for step in value):
        raise ValueError(
            f"Every item in {name} must have {expected_ndim} dimensions"
        )

    stack_dim = 0 if unbatched else 1
    stacked = torch.stack(tuple(value), dim=stack_dim)
    return stacked.unsqueeze(0) if unbatched else stacked


def _validate_shapes(states: Tensor, epsilons: Tensor) -> None:
    if states.ndim != 5:
        raise ValueError(
            "states must have shape [B,S+1,C,H,W], "
            f"got {tuple(states.shape)}"
        )

    if epsilons.ndim != 5:
        raise ValueError(
            "epsilons must have shape [B,S,C,H,W], "
            f"got {tuple(epsilons.shape)}"
        )

    expected_eps_shape = (
        states.shape[0],
        states.shape[1] - 1,
        *states.shape[2:],
    )
    if tuple(epsilons.shape) != expected_eps_shape:
        raise ValueError(
            "epsilons shape is inconsistent with states: expected "
            f"{expected_eps_shape}, got {tuple(epsilons.shape)}"
        )


def _validate_deltas(deltas: Tensor, computed_deltas: Tensor) -> None:
    if deltas.shape != computed_deltas.shape:
        raise ValueError(
            "deltas shape is inconsistent with states: expected "
            f"{tuple(computed_deltas.shape)}, got {tuple(deltas.shape)}"
        )

    low_precision = {
        torch.float16,
        torch.bfloat16,
    }
    uses_low_precision = (
        deltas.dtype in low_precision
        or computed_deltas.dtype in low_precision
    )
    atol = 5e-3 if uses_low_precision else 1e-5
    rtol = 5e-3 if uses_low_precision else 1e-4

    if not torch.allclose(
        deltas.float(),
        computed_deltas.float(),
        atol=atol,
        rtol=rtol,
    ):
        raise ValueError(
            "deltas are inconsistent with consecutive states. "
            "Use an unprojected cache or ensure z and delta use "
            "the same projection."
        )


def build_trajectory_batch(output: Mapping[str, object]) -> dict[str, Tensor]:
    """Convert online Module 0 or cached output to one batched contract.

    Accepted raw keys are ``z_0``, ``z_seq``, ``eps_seq`` and optional
    ``delta_z_seq``. Canonical ``states`` and ``epsilons`` inputs are also
    accepted, making the function idempotent for existing DCTE callers.
    """

    if "states" in output or "epsilons" in output:
        if "states" not in output or "epsilons" not in output:
            raise KeyError("Canonical input requires both states and epsilons")

        states = output["states"]
        epsilons = output["epsilons"]
        if not isinstance(states, Tensor) or not isinstance(epsilons, Tensor):
            raise TypeError("states and epsilons must be torch.Tensors")

        if states.ndim == 4 and epsilons.ndim == 4:
            states = states.unsqueeze(0)
            epsilons = epsilons.unsqueeze(0)

        _validate_shapes(states, epsilons)
        z0 = states[:, 0]
        supplied_deltas = output.get("deltas")
    else:
        required_keys = {"z_0", "z_seq", "eps_seq"}
        missing_keys = required_keys.difference(output)
        if missing_keys:
            raise KeyError(
                "Trajectory input is missing keys: "
                f"{sorted(missing_keys)}"
            )

        z0, unbatched = _prepare_z0(output["z_0"])
        z_sequence = _prepare_sequence(
            output["z_seq"],
            "z_seq",
            unbatched,
        )
        epsilons = _prepare_sequence(
            output["eps_seq"],
            "eps_seq",
            unbatched,
        )
        states = torch.cat((z0.unsqueeze(1), z_sequence), dim=1)
        _validate_shapes(states, epsilons)

        raw_deltas = output.get("delta_z_seq")
        supplied_deltas = (
            _prepare_sequence(raw_deltas, "delta_z_seq", unbatched)
            if raw_deltas is not None
            else None
        )

    computed_deltas = states[:, 1:] - states[:, :-1]
    if supplied_deltas is not None:
        if not isinstance(supplied_deltas, Tensor):
            raise TypeError("deltas must be a torch.Tensor")
        _validate_deltas(supplied_deltas, computed_deltas)
        deltas = supplied_deltas
    else:
        deltas = computed_deltas

    a_end_coarse = torch.linalg.vector_norm(
        states[:, -1].float(),
        ord=2,
        dim=1,
    )

    return {
        "z0": z0.float(),
        "states": states.float(),
        "epsilons": epsilons.float(),
        "deltas": deltas.float(),
        "a_end_coarse": a_end_coarse,
    }
