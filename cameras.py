"""Bounded camera corrections learned from fit frames and saved for fit rendering."""

from __future__ import annotations

import torch
from torch import nn


class TrainCameraOffsets(nn.Module):
    """Bounded per-fit-camera CV-frame SE(3) offsets.

    The first fit camera anchors the coordinate frame. Held-out views use
    their original calibration.
    """

    def __init__(self, nfit: int, object_radius: float, device: str | torch.device = "cpu"):
        super().__init__()
        self.nfit = int(nfit)
        self.object_radius = float(object_radius)
        self.raw = nn.Parameter(torch.zeros(self.nfit, 6, device=device))
        active = torch.ones(self.nfit, 1, device=device)
        active[0] = 0.0
        self.register_buffer("active", active)

    @staticmethod
    def _skew(vector: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(vector[..., 0])
        x, y, z = vector.unbind(-1)
        return torch.stack(
            (
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros,
            ),
            dim=-1,
        ).reshape(*vector.shape[:-1], 3, 3)

    def _delta_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        rotation = 0.03 * raw[..., :3].tanh()
        translation = 0.03 * self.object_radius * raw[..., 3:].tanh()
        rotation = torch.linalg.matrix_exp(self._skew(rotation))
        upper = torch.cat((rotation, translation[..., None]), dim=-1)
        bottom = torch.zeros(
            *raw.shape[:-1], 1, 4, device=raw.device, dtype=raw.dtype
        )
        bottom[..., 3] = 1.0
        return torch.cat((upper, bottom), dim=-2)

    def deltas(self) -> torch.Tensor:
        return self._delta_from_raw(self.raw * self.active)

    def correct(self, sample: dict[str, object], local_fit_index: int) -> dict[str, object]:
        """Return a shallow corrected sample; all non-camera data is shared."""
        corrected = dict(sample)
        index = int(local_fit_index)
        viewmat = self._delta_from_raw(self.raw[index] * self.active[index]) @ sample["viewmat"]
        corrected["viewmat"] = viewmat
        flip = torch.diag(
            viewmat.new_tensor((1.0, -1.0, -1.0, 1.0))
        )
        corrected["c2w"] = torch.linalg.inv(viewmat) @ flip
        return corrected

    def rms(self) -> tuple[float, float]:
        """Return bounded rotation and translation RMS for training logs."""
        bounded = self.raw.detach() * self.active
        rotation = 0.03 * bounded[:, :3].tanh()
        translation = 0.03 * self.object_radius * bounded[:, 3:].tanh()
        return (
            float(rotation.square().mean().sqrt().item()),
            float(translation.square().mean().sqrt().item()),
        )
