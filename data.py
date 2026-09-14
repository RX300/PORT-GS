"""Training data loading for PORT-GS.

The loader keeps image decoding and resizing in ``__getitem__``. Metadata is
read once when the dataset is constructed, while each sample is decoded on
demand and returned as CPU tensors.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# OpenCV's EXR reader is disabled by default in this environment. Enabling it
# here changes only this process and keeps EXR values in their stored domain.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

__all__ = ["SceneDataset", "split_train_lights"]

_LOGGER = logging.getLogger(__name__)
_OPENGL_TO_OPENCV_CAMERA = torch.diag(torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float32))


def _frame_path(scene_path: Path, frame: dict[str, Any]) -> Path:
    """Resolve a frame's path relative to its scene directory."""

    if "file_paths" in frame:
        relative = Path(frame["file_paths"][0])
        extension = ".png"
    else:
        relative = Path(frame["file_path"])
        extension = str(frame["file_ext"])
    if relative.suffix.lower() != extension.lower():
        relative = Path(f"{relative}{extension}")
    return scene_path / relative


def _decode_image(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one PNG/EXR image as RGB and a separate optional alpha channel."""

    suffix = path.suffix.lower()
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not decode image: {path}")

    if suffix == ".png":
        if raw.dtype == np.uint8:
            divisor = 255.0
        elif raw.dtype == np.uint16:
            divisor = 65535.0
        else:
            raise TypeError(f"PNG must decode to uint8 or uint16, got {raw.dtype}")
    elif suffix == ".exr":
        divisor = 1.0
    else:
        raise ValueError(f"PORT-GS training data supports PNG and EXR, got {path}")

    if raw.ndim == 2:
        raw = raw[..., None]

    channels = raw.shape[-1]
    if channels == 1:
        rgb = np.repeat(raw, 3, axis=-1)
        alpha = None
    elif channels == 2:
        # OpenCV uses two channels for grayscale + alpha.
        rgb = np.repeat(raw[..., :1], 3, axis=-1)
        alpha = raw[..., 1:2]
    elif channels == 3:
        rgb = raw[..., [2, 1, 0]]
        alpha = None
    elif channels == 4:
        # OpenCV returns BGR(A); the contract exposes RGB(A).
        rgb = raw[..., [2, 1, 0]]
        alpha = raw[..., 3:4]
    else:
        raise ValueError(f"unsupported channel count {channels} in {path}")

    image = rgb.astype(np.float32, copy=False) / divisor
    if alpha is not None:
        alpha = alpha.astype(np.float32, copy=False) / divisor
    return image, alpha


def _resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an HWC float array without changing its numeric color space."""

    if array.shape[0] == height and array.shape[1] == width:
        return array
    resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    if array.shape[-1] == 1 and resized.ndim == 2:
        resized = resized[..., None]
    return resized.astype(np.float32, copy=False)


def _camera_intrinsics(
    metadata: dict[str, Any],
    width: int,
    height: int,
    frame: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build the unscaled 3x3 K matrix from a scene's metadata."""

    if frame is not None and "fl_x" in frame:
        fx, fy = frame["fl_x"], frame["fl_y"]
        cx, cy = frame["cx"], frame["cy"]
    elif metadata.get("camera_intrinsics") is not None:
        cx, cy, fx, fy = metadata["camera_intrinsics"]
    else:
        angle_x = float(metadata["camera_angle_x"])
        fx = 0.5 * width / math.tan(0.5 * angle_x)
        fy = fx
        cx = 0.5 * width
        cy = 0.5 * height

    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _frame_light_position(frame: dict[str, Any]) -> Sequence[float]:
    """Return coordinates in the fixed angular-partition basis, not renderer axes.

    SSS partitions retain their original 180-degree X rotation so correcting
    the renderer's light convention never changes the held-out frame set.
    """

    if "pl_pos" in frame:
        return frame["pl_pos"]
    position = list(frame["light_positions"][0])
    position[1] *= -1.0
    position[2] *= -1.0
    return position


class SceneDataset(Dataset):
    """On-demand loader for one supported PORT-GS scene.

    Args:
        scene_path: Scene directory containing the selected transforms JSON and
            its referenced images.
        split: Metadata split to read, for example ``"train"``, ``"valid"`` or
            ``"test"``. The caller must choose the split explicitly.
        resolution: Target length of the resized image's longest edge.
        unit_light_intensity: Explicit positive scalar expanded to RGB for
            SSS-GS frames, whose metadata contains light positions but no intensity.

    ``frames`` is the raw JSON frame list. No image is decoded during
    construction; ``__getitem__`` returns one sample as CPU tensors.
    """

    def __init__(
        self,
        scene_path: str | Path,
        split: str = "train",
        resolution: int = 256,
        unit_light_intensity: float | None = None,
    ) -> None:
        self.scene_path = Path(scene_path)
        self.split = split
        self.resolution = int(resolution)
        self._is_sss = self.scene_path.parent.name == "Synthetic_SSS-GS"
        self.unit_light_intensity = (
            None if unit_light_intensity is None else float(unit_light_intensity)
        )
        if self._is_sss and (
            self.unit_light_intensity is None
            or not math.isfinite(self.unit_light_intensity)
            or self.unit_light_intensity <= 0.0
        ):
            raise ValueError("Synthetic_SSS-GS requires an explicit positive unit_light_intensity")
        metadata_split = ("val" if self._is_sss else "valid") if split == "val" else split
        metadata_path = self.scene_path / f"transforms_{metadata_split}.json"
        self.metadata_path = metadata_path
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: dict[str, Any] = json.load(handle)
        self.frames: list[dict[str, Any]] = self.metadata["frames"]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        frame = self.frames[index]
        image_path = _frame_path(self.scene_path, frame)
        image, alpha = _decode_image(image_path)

        source_height, source_width = image.shape[:2]
        scale = self.resolution / max(source_height, source_width)
        height = int(round(source_height * scale))
        width = int(round(source_width * scale))
        if alpha is not None:
            image = _resize(image * alpha, height, width)
            alpha = _resize(alpha, height, width)
            image = np.divide(
                image,
                alpha,
                out=np.zeros_like(image),
                where=alpha > 0.0,
            )
        else:
            image = _resize(image, height, width)

        K = _camera_intrinsics(self.metadata, source_width, source_height, frame)
        K[0] *= np.float32(width / source_width)
        K[1] *= np.float32(height / source_height)

        c2w = torch.from_numpy(np.asarray(frame["transform_matrix"], dtype=np.float32).copy())
        viewmat = torch.linalg.inv(c2w @ _OPENGL_TO_OPENCV_CAMERA)

        if self._is_sss:
            # Official reader and CameraDataset each flip Y/Z: the net result
            # is the raw world position, also stored in the original anno.json.
            light_pos = torch.tensor(frame["light_positions"][0], dtype=torch.float32)
            light_intensity = torch.full((3,), self.unit_light_intensity, dtype=torch.float32)
            name = Path(frame["file_paths"][0]).stem
        else:
            light_pos = torch.tensor(frame["pl_pos"], dtype=torch.float32)
            light_intensity = torch.tensor(frame["pl_intensity"], dtype=torch.float32)
            name = Path(frame["file_path"]).stem

        return {
            "is_hdr": image_path.suffix.lower() == ".exr",
            "image": torch.from_numpy(np.asarray(image, dtype=np.float32)),
            "alpha": (
                None if alpha is None else torch.from_numpy(np.asarray(alpha, dtype=np.float32))
            ),
            "c2w": c2w,
            "viewmat": viewmat,
            "K": torch.from_numpy(K),
            "light_pos": light_pos,
            "light_intensity": light_intensity,
            "name": name,
            "frame_index": index % len(self.frames),
        }


def _light_angle_group(position: Sequence[float]) -> tuple[int, int]:
    """Return a fixed 30-degree elevation/azimuth group for a light position."""

    x, y, z = (float(value) for value in position)
    radius = math.sqrt(x * x + y * y + z * z)
    if radius == 0.0:
        raise ValueError("light position at the scene origin has no direction")

    azimuth = math.atan2(z, x)
    elevation = math.asin(y / radius)
    azimuth_bin = min(11, int((azimuth + math.pi) / (2.0 * math.pi / 12.0)))
    elevation_bin = min(5, int((elevation + 0.5 * math.pi) / (math.pi / 6.0)))
    return elevation_bin, azimuth_bin


def split_train_lights(
    frames: Sequence[dict[str, Any]], fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    """Split frame indices by deterministic light-angle groups.

    Directions are measured from the metadata scene origin and quantized into
    30-degree elevation and azimuth bins. Whole bins are selected for the
    validation set in stable key order; individual images are never sampled at
    random. The function returns ``(fit_indices, val_indices)`` in original
    frame order. A six-decimal light-position audit is logged so callers can
    record whether a repeated position appears in more than one angular bin.
    SSS-GS bin coordinates retain a fixed 180-degree X rotation; rendering uses
    raw world coordinates. This fixed bin basis preserves the established
    held-out frames. Real/GS3 ``pl_pos`` is kept as stored.
    """

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if len(frames) < 2:
        raise ValueError("at least two training frames are required")

    frame_groups: list[tuple[int, int]] = []
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    same_position_groups: dict[tuple[float, float, float], set[tuple[int, int]]] = defaultdict(set)
    for index, frame in enumerate(frames):
        light_position = _frame_light_position(frame)
        group = _light_angle_group(light_position)
        frame_groups.append(group)
        groups[group].append(index)
        position = tuple(round(float(value), 6) for value in light_position)
        same_position_groups[position].add(group)

    ordered_groups = sorted(groups)
    if len(ordered_groups) < 2:
        raise ValueError("at least two light-angle groups are required")
    target = max(1, min(len(frames) - 1, int(round(len(frames) * fraction))))
    remaining = set(ordered_groups)
    val_groups: list[tuple[int, int]] = []
    val_count = 0
    while remaining and val_count < target and len(val_groups) < len(ordered_groups) - 1:
        group = min(
            remaining,
            key=lambda candidate: (
                abs(val_count + len(groups[candidate]) - target),
                candidate,
            ),
        )
        remaining.remove(group)
        val_groups.append(group)
        val_count += len(groups[group])

    val_group_set = set(val_groups)
    val_indices = [index for index, group in enumerate(frame_groups) if group in val_group_set]
    fit_indices = [index for index, group in enumerate(frame_groups) if group not in val_group_set]

    crossing_positions = sum(len(group_set) > 1 for group_set in same_position_groups.values())
    _LOGGER.info(
        "split_train_lights: frames=%d angle_groups=%d target=%d val=%d "
        "same_light_cross_group=%s crossing_positions=%d",
        len(frames),
        len(ordered_groups),
        target,
        len(val_indices),
        bool(crossing_positions),
        crossing_positions,
    )
    return fit_indices, val_indices
