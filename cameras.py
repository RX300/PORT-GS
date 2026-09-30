"""Bounded camera corrections learned from fit frames and saved for fit rendering."""

from __future__ import annotations

import torch
from torch import nn


def correspondence_errors(checkpoint, correspondences, dataset):
    """CPU geometric audit on fixed training-image matches in checkpoint pixels.

    The matches must already be selected; they never enter the camera optimizer.
    Platform filtering is an image heuristic, not a semantic or 3D annotation.
    """
    import cv2
    import numpy as np
    fit_indices = {index:local for local,index in enumerate(checkpoint['fit_indices'])}
    offsets = load_camera_offsets(checkpoint, checkpoint['radius'])

    def residual(first, second, p, q):
        relative = second['viewmat'].numpy() @ np.linalg.inv(first['viewmat'].numpy())
        t = relative[:3, 3]
        cross = np.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]])
        fundamental = np.linalg.inv(second['K'].numpy()).T @ cross @ relative[:3, :3] @ np.linalg.inv(first['K'].numpy())
        p, q = np.c_[p, np.ones(len(p))], np.c_[q, np.ones(len(q))]
        fp, fq = p @ fundamental.T, q @ fundamental
        return np.abs((q*fp).sum(-1))/np.sqrt((fp[:, :2]**2).sum(-1)+(fq[:, :2]**2).sum(-1))

    def platform(sample, points):
        image, alpha = sample['image'].numpy(), sample['alpha'].numpy()[..., 0]
        y = np.where(alpha > .9)[0]
        pixel = np.rint(points).astype(int)
        color = cv2.blur(image, (5, 5))[pixel[:, 1], pixel[:, 0]]
        return (points[:, 1] > y.min()+.65*(y.max()-y.min())) & (color[:, 0] < 1.5*color[:, 1])

    rows = []
    with torch.no_grad():
        for key in correspondences:
            i, j = map(int, key.split('_'))
            if i not in fit_indices or j not in fit_indices:
                raise ValueError('Camera correspondence audit requires fit-frame matches')
            first, second = dataset[i], dataset[j]
            p, q = correspondences[key][:, 0], correspondences[key][:, 1]
            original = residual(first, second, p, q)
            board = platform(first, p) & platform(second, q)
            if offsets is not None:
                first = offsets.correct(first, fit_indices[i])
                second = offsets.correct(second, fit_indices[j])
            corrected = residual(first, second, p, q)
            row = {'frames':[i, j], 'matches':len(p), 'platform_matches':int(board.sum()),
                   'original_median_px':float(np.median(original)),
                   'corrected_median_px':float(np.median(corrected)),
                   'original_p90_px':float(np.quantile(original, .9)),
                   'corrected_p90_px':float(np.quantile(corrected, .9))}
            if board.any():
                row.update(platform_original_median_px=float(np.median(original[board])),
                           platform_corrected_median_px=float(np.median(corrected[board])))
            rows.append(row)
    result = {'rows':rows, 'optimized_poses':offsets is not None}
    for label, count, minimum, prefix in [('all', 'matches', 20, ''), ('platform', 'platform_matches', 8, 'platform_')]:
        selected = [row for row in rows if row[count] >= minimum]
        result[label] = {'eligible_pairs':len(selected), **{
            key:float(np.median([row[prefix+key] for row in selected])) if selected else None
            for key in ['original_median_px', 'corrected_median_px']}}
    if offsets is not None:
        rotation, translation = offsets.rms()
        result['pose_components_rms'] = {'rotation_rad':rotation, 'translation_world':translation}
    return result


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

    def regularization(self, local_fit_index: int) -> torch.Tensor:
        return 0.001 * self.raw[int(local_fit_index)].square().mean()

    def optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=lr)


class TrainCameraRotations(nn.Module):
    """Per-fit-camera camera-frame rotations that keep every optical center fixed.

    ``viewmat' = [dR | 0] @ viewmat`` rotates a camera about its own center, so
    calibrated positions keep defining the world frame: there is no anchor
    camera whose own calibration error would move the reconstruction. Every fit
    camera is corrected, including a rotation shared by all of them, which is
    observable when cameras surround the object. Rotations are radians in the
    OpenCV camera frame. Rows receive sparse gradients, so a lazy optimizer
    updates only the sampled camera. Held-out views keep original calibration.
    """

    def __init__(self, nfit: int, object_radius: float | None = None,
                 device: str | torch.device = "cpu", regularization_weight: float = 1.0):
        super().__init__()
        self.nfit = int(nfit)
        self.regularization_weight = float(regularization_weight)
        self.rotation = nn.Embedding(self.nfit, 3, sparse=True, device=device)
        nn.init.zeros_(self.rotation.weight)

    def _rotation(self, local_fit_index: int) -> torch.Tensor:
        index = torch.tensor([int(local_fit_index)], device=self.rotation.weight.device)
        vector = self.rotation(index)[0]
        return torch.linalg.matrix_exp(TrainCameraOffsets._skew(vector))

    def correct(self, sample: dict[str, object], local_fit_index: int) -> dict[str, object]:
        """Return a shallow corrected sample; the camera center is unchanged."""
        corrected = dict(sample)
        rotation = self._rotation(local_fit_index).to(sample["viewmat"].dtype)
        viewmat = torch.cat((rotation @ sample["viewmat"][:3], sample["viewmat"][3:]), dim=0)
        corrected["viewmat"] = viewmat
        flip = torch.diag(viewmat.new_tensor((1.0, -1.0, -1.0, 1.0)))
        corrected["c2w"] = torch.linalg.inv(viewmat) @ flip
        return corrected

    def rms(self) -> tuple[float, float]:
        """Rotation RMS in radians; translations are fixed at zero."""
        return float(self.rotation.weight.detach().square().mean().sqrt().item()), 0.0

    def regularization(self, local_fit_index: int) -> torch.Tensor:
        # Weak tie-breaker toward the given calibration, in squared degrees.
        index = torch.tensor([int(local_fit_index)], device=self.rotation.weight.device)
        degrees = self.rotation(index)[0] * (180.0 / torch.pi)
        return 1e-4 * self.regularization_weight * degrees.square().sum()

    def optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.SparseAdam(list(self.parameters()), lr=lr)

    @torch.no_grad()
    def set_translation_gauge(self, viewmats: torch.Tensor, intrinsics: torch.Tensor, center: torch.Tensor) -> None:
        """Precompute the rotations that re-aim each camera at a translated scene.

        With fixed centers and free per-camera rotations, translating the whole
        scene by ``d`` is compensated to first order at the object center by
        rotating camera i by ``C_i d`` (minimum-norm, image of the center fixed).
        These three directions are a gauge of the joint problem, so the image
        loss alone does not pin them and they drift with gradient noise.
        """
        rotation, translation = viewmats[:, :3, :3], viewmats[:, :3, 3]
        point = (rotation @ center[:, None]).squeeze(-1) + translation
        depth = point[:, 2:3]
        focal = intrinsics[:, [0, 1], [0, 1]]
        zeros = torch.zeros_like(depth)
        projection = torch.stack((torch.cat((focal[:, :1] / depth, zeros, -focal[:, :1] * point[:, :1] / depth**2), -1),
                                  torch.cat((zeros, focal[:, 1:] / depth, -focal[:, 1:] * point[:, 1:2] / depth**2), -1)), 1)
        scene = projection @ rotation                                      # image motion per scene translation
        camera = -projection @ TrainCameraOffsets._skew(point)            # image motion per camera rotation
        self.gauge = -torch.linalg.pinv(camera) @ scene                   # [N,3,3] rotation per scene translation
        self.gauge_normal = torch.linalg.inv((self.gauge.transpose(1, 2) @ self.gauge).sum(0))

    @torch.no_grad()
    def remove_translation_gauge(self) -> torch.Tensor:
        """Project the corrections off the translation gauge; return the compensating scene shift.

        Removing ``C_i d`` from every camera and translating the scene by ``-d``
        leaves the rendered object center unchanged in all fit views while fixing
        the reconstruction's position to the calibrated frame.
        """
        rotations = self.rotation.weight
        shift = self.gauge_normal @ (self.gauge.transpose(1, 2) @ rotations[..., None]).sum(0).squeeze(-1)
        rotations -= self.gauge @ shift
        return -shift


class TrainLightOffsets(nn.Module):
    """Per-fit-frame world-space point-light position offsets, in object-radius units.

    Rows receive sparse gradients (lazy per-frame updates). Held-out views keep
    their calibrated light positions.
    """

    def __init__(self, nfit: int, object_radius: float, device: str | torch.device = "cpu"):
        super().__init__()
        self.nfit = int(nfit)
        self.object_radius = float(object_radius)
        self.offset = nn.Embedding(self.nfit, 3, sparse=True, device=device)
        nn.init.zeros_(self.offset.weight)

    def _offset(self, local_fit_index: int) -> torch.Tensor:
        index = torch.tensor([int(local_fit_index)], device=self.offset.weight.device)
        return self.offset(index)[0]

    def correct(self, sample: dict[str, object], local_fit_index: int) -> dict[str, object]:
        corrected = dict(sample)
        corrected["light_pos"] = sample["light_pos"] + self.object_radius * self._offset(local_fit_index).to(
            sample["light_pos"].dtype)
        return corrected

    def rms(self) -> float:
        return float((self.object_radius * self.offset.weight.detach()).square().sum(-1).mean().sqrt().item())

    def regularization(self, local_fit_index: int) -> torch.Tensor:
        return 1e-3 * self._offset(local_fit_index).square().sum()

    def optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.SparseAdam(list(self.parameters()), lr=lr)


CAMERA_MODES = {"anchor": TrainCameraOffsets, "rotation": TrainCameraRotations}


def build_camera_offsets(mode: str, nfit: int, object_radius: float) -> nn.Module:
    if mode not in CAMERA_MODES:
        raise ValueError(f"Unknown camera correction mode: {mode}")
    return CAMERA_MODES[mode](nfit, object_radius)


def load_camera_offsets(checkpoint: dict, object_radius: float) -> nn.Module | None:
    """Restore saved fit-camera corrections; checkpoints before modes are anchor SE(3)."""
    if checkpoint.get("camera_offsets") is None:
        return None
    mode = checkpoint["config"].get("camera_mode", "anchor")
    offsets = build_camera_offsets(mode, len(checkpoint["fit_indices"]), object_radius)
    offsets.load_state_dict(checkpoint["camera_offsets"])
    return offsets
