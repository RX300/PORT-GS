"""Surface supervision in OpenCV camera coordinates, with frozen image priors."""
from pathlib import Path

import torch
from torch.nn import functional as F


def depth_normals(depth, intrinsics):
    """Differentiate camera-Z depth at pixel centers; orient normals toward camera."""
    h, w = depth.shape[:2]
    y, x = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype) + .5,
                          torch.arange(w, device=depth.device, dtype=depth.dtype) + .5, indexing="ij")
    rays = torch.stack((x, y, torch.ones_like(x)), -1) @ torch.linalg.inv(intrinsics).T
    points = rays * depth
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = F.normalize(torch.linalg.cross(dy, dx, dim=-1), dim=-1)
    return F.pad(normals.permute(2, 0, 1), (1, 1, 1, 1)).permute(1, 2, 0)


class SurfacePriors:
    def __init__(self, directory, dataset, fit_indices, need_normal=True, need_depth=True):
        self.targets = {}
        for kind, enabled in (("normal", need_normal), ("depth", need_depth)):
            if not enabled:
                continue
            record = torch.load(Path(directory) / f"{kind}.pt", map_location="cpu", weights_only=False)
            if record["scene"] != str(dataset.scene_path.resolve()) or record["resolution"] != dataset.resolution:
                raise ValueError("Surface prior scene/resolution differs from training")
            if kind == "normal" and (record["coordinates"] != "opencv_camera"
                                     or record["normal_conversion"] != "negate_xyz"):
                raise ValueError("Normal targets must use OpenCV camera coordinates")
            if kind == "depth" and record["depth_type"] != "relative_camera_z":
                raise ValueError("Depth targets must use relative camera-Z depth")
            for index in fit_indices:
                if record["frames"][index] != dataset.frames[index]:
                    raise ValueError(f"Surface prior metadata differs for frame {index}")
                tensor = record["predictions"][index]
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite {kind} prior at frame {index}")
            self.targets[kind] = record["predictions"]

    def frame(self, index, device):
        return {kind: values[index].to(device=device, dtype=torch.float32)
                for kind, values in self.targets.items()}


def surface_losses(info, alpha, sample, targets, radius, normal_weight,
                   depth_weight, consistency_weight, distortion_weight):
    depth = info["surface_depth"]
    normals = F.normalize(info["surface_normals"] @ sample["viewmat"][:3, :3].T, dim=-1)
    derived = depth_normals(depth, sample["K"])
    mask = (alpha[..., 0].detach() > .05) & (depth[..., 0].detach() > 0)
    if sample["alpha"] is not None:
        mask &= sample["alpha"][..., 0] > .9
    # Only supervise derivatives whose entire stencil belongs to the foreground.
    interior = F.avg_pool2d(mask.float()[None, None], 3, stride=1, padding=1)[0, 0] == 1
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    weight = interior.float()
    count = weight.sum().clamp_min(1)

    def angular(a, b):
        return ((1 - (a * b).sum(-1).clamp(-1, 1)) * weight).sum() / count

    terms = {"surface_consistency": consistency_weight * angular(normals, derived),
             "surface_distortion": distortion_weight *
             (info["surface_distortion"][..., 0] * weight).sum() / count / radius**2}
    if normal_weight:
        target = F.normalize(targets["normal"], dim=-1)
        terms["normal_prior"] = normal_weight * .5 * (angular(normals, target) + angular(derived, target))
    if depth_weight:
        prior = targets["depth"][..., 0]
        valid = mask & torch.isfinite(prior) & (prior > 0)
        weights = valid.float()
        denom = weights.sum().clamp_min(1)
        residual = depth[..., 0].clamp_min(1e-6).log() - prior.clamp_min(1e-6).log()
        # DA3 monocular depth is up to scale. Eliminate log-scale, not scene shape.
        residual = residual - (residual * weights).sum() / denom
        terms["depth_prior"] = depth_weight * (
            F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=.1)
            * weights).sum() / denom
    return terms
