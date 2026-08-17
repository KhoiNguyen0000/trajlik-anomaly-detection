from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def normal_train_calibration_split(
    num_images: int,
    calibration_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Return deterministic, disjoint indices for normal-only fit/calibration."""

    if num_images < 2:
        raise ValueError("At least two normal images are required")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    calibration_size = max(1, round(num_images * calibration_fraction))
    calibration_size = min(calibration_size, num_images - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_images, generator=generator)
    calibration_indices = permutation[:calibration_size].sort().values
    training_indices = permutation[calibration_size:].sort().values
    return training_indices, calibration_indices


def normal_kfold_indices(
    num_images: int,
    num_folds: int = 5,
    seed: int = 42,
) -> list[tuple[Tensor, Tensor]]:
    """Build normal-only cross-fitting folds without consulting labels."""

    if num_folds < 2 or num_images < num_folds:
        raise ValueError("num_images must be at least num_folds >= 2")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_images, generator=generator)
    folds = torch.tensor_split(permutation, num_folds)
    result = []
    for fold_index, calibration_indices in enumerate(folds):
        training_indices = torch.cat(
            [fold for index, fold in enumerate(folds) if index != fold_index]
        )
        result.append(
            (
                training_indices.sort().values,
                calibration_indices.sort().values,
            )
        )
    return result


class EmpiricalTailCalibrator(nn.Module):
    """Normal-only empirical upper-tail transform from Equations (21)-(25)."""

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.register_buffer("endpoint_reference", torch.empty(0))
        self.register_buffer("path_reference", torch.empty(0))

    @property
    def fitted(self) -> bool:
        return self.endpoint_reference.numel() > 0 and self.path_reference.numel() > 0

    @staticmethod
    def _validated_reference(values: Tensor, name: str) -> Tensor:
        if not isinstance(values, Tensor):
            raise TypeError(f"{name} scores must be a torch.Tensor")
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            raise ValueError(f"{name} calibration scores cannot be empty")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} calibration scores must be finite")
        return values.sort().values

    def fit(self, endpoint_scores: Tensor, path_scores: Tensor):
        """Fit only score marginals; anomaly labels are intentionally not accepted."""

        self.endpoint_reference = self._validated_reference(
            endpoint_scores,
            "endpoint",
        )
        self.path_reference = self._validated_reference(path_scores, "path")
        return self

    def _tail(self, values: Tensor, reference: Tensor) -> Tensor:
        if reference.numel() == 0:
            raise RuntimeError("EmpiricalTailCalibrator must be fit before scoring")
        values_float = values.float()
        ranks = torch.searchsorted(
            reference.to(values_float.device),
            values_float.reshape(-1),
            right=True,
        ).reshape(values_float.shape)
        empirical_cdf = (1.0 + ranks.float()) / (reference.numel() + 2.0)
        return -torch.log(1.0 - empirical_cdf + self.epsilon)

    def endpoint_tail(self, endpoint_scores: Tensor) -> Tensor:
        return self._tail(endpoint_scores, self.endpoint_reference)

    def path_tail(self, path_scores: Tensor) -> Tensor:
        return self._tail(path_scores, self.path_reference)

    def forward(
        self,
        endpoint_scores: Tensor,
        path_scores: Tensor,
        *,
        output_size: tuple[int, int] | None = None,
        lambda_path: float = 1.0,
    ) -> dict[str, Tensor]:
        if endpoint_scores.ndim != 3:
            raise ValueError("endpoint_scores must have shape [B,H,W]")
        if path_scores.ndim == 2:
            if path_scores.shape != (
                endpoint_scores.shape[0],
                endpoint_scores.shape[1] * endpoint_scores.shape[2],
            ):
                raise ValueError("Flat path scores do not match endpoint grid")
            path_scores = path_scores.reshape_as(endpoint_scores)
        if path_scores.shape != endpoint_scores.shape:
            raise ValueError("path_scores and endpoint_scores must share a grid")

        endpoint_tail = self.endpoint_tail(endpoint_scores)
        path_tail = self.path_tail(path_scores)
        coarse_map = endpoint_tail + float(lambda_path) * path_tail
        image_score = coarse_map.amax(dim=(-2, -1)) - coarse_map.amin(
            dim=(-2, -1)
        )

        if output_size is None:
            pixel_map = coarse_map
        else:
            if len(output_size) != 2 or min(output_size) <= 0:
                raise ValueError("output_size must contain two positive dimensions")
            pixel_map = F.interpolate(
                coarse_map.unsqueeze(1),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return {
            "endpoint_tail": endpoint_tail,
            "path_tail": path_tail,
            "coarse_map": coarse_map,
            "pixel_map": pixel_map,
            "image_score": image_score,
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
        epsilon: float = 1e-6,
    ):
        calibrator = cls(epsilon=epsilon)
        calibrator.endpoint_reference = state_dict["endpoint_reference"].clone()
        calibrator.path_reference = state_dict["path_reference"].clone()
        return calibrator
