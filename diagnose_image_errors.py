"""GPU image-error measurements on existing real-scene fit/validation frames."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from gsplat import rasterization
from torch.nn import functional as F

from cameras import TrainCameraOffsets
from data import SceneDataset
from evaluate import load_model, to_device, render_observation, target_image
from renderer import visibility_hint


def block_energy(image, block):
    return F.avg_pool2d(image.permute(2, 0, 1)[None], block).square().mean()


def boundary(mask):
    eroded = -F.max_pool2d(-mask[None, None].float(), 3, 1, 1)[0, 0]
    return mask & (eroded == 0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    g, t, checkpoint = load_model(args.checkpoint)
    cfg = checkpoint["config"]
    dataset = SceneDataset(
        cfg["scene"], "train", cfg["resolution"], unit_light_intensity=cfg["unit_light_intensity"]
    )
    fit = checkpoint["fit_indices"]
    selected = [("fit", i) for i in fit[:: max(1, len(fit) // 32)][:32]]
    selected += [("validation", i) for i in checkpoint["val_indices"]]
    show = set(selected[:2]) | {("validation", index) for index in checkpoint["val_indices"][:2]}
    camera_offsets = None
    if checkpoint["camera_offsets"] is not None:
        camera_offsets = TrainCameraOffsets(len(fit), g.radius).cuda()
        camera_offsets.load_state_dict(checkpoint["camera_offsets"])
    camera_indices = {index: local for local, index in enumerate(fit)}
    rows = []
    for split, index in selected:
        sample = to_device(dataset[index], "cuda")
        if split == "fit" and camera_offsets is not None:
            sample = camera_offsets.correct(sample, camera_indices[index])
        pred, alpha, info = render_observation(
            g,
            t,
            sample,
            cfg["background"],
            checkpoint["step"] >= cfg["shadow_start"],
            checkpoint["step"] >= cfg["port_start"],
            cfg["display_gamma"],
            cfg["shadow_mode"],
        )
        pred = (pred.clamp(0, 1) * 255).round() / 255
        target = (
            target_image(sample, cfg["background"], cfg["display_gamma"]).clamp(0, 1) * 255
        ).round() / 255
        error = pred - target
        error2 = error.square().mean(-1)
        gt_mask = sample["alpha"][..., 0] > 0.5
        pred_mask = alpha[..., 0] > 0.5
        mask = gt_mask[None, None].float()
        inner = -F.max_pool2d(-mask, 5, 1, 2)[0, 0]
        outer = F.max_pool2d(mask, 5, 1, 2)[0, 0]
        regions = {"interior": inner, "boundary": outer - inner, "background": 1 - outer}
        distances = torch.cdist(
            torch.stack(torch.where(boundary(gt_mask)), -1).float(),
            torch.stack(torch.where(boundary(pred_mask)), -1).float(),
        )
        edge_dist = torch.cat((distances.amin(0), distances.amin(1)))
        visibility = visibility_hint(g, sample["light_pos"], mode=cfg["shadow_mode"])
        radius = info["radii"][0].amax(-1).float()
        attributes = torch.stack((visibility, radius), -1)
        rendered, _, _ = rasterization(
            **g.raster_inputs(),
            colors=attributes,
            viewmats=sample["viewmat"][None],
            Ks=sample["K"][None],
            width=pred.shape[1],
            height=pred.shape[0],
            packed=False,
        )
        maps = rendered[0] / alpha.clamp_min(1e-8)
        overlap = gt_mask & pred_mask
        mse = error2.mean()
        row = {
            "split": split,
            "frame_index": index,
            "PSNR": float(-10 * mse.log10()),
            "MSE": float(mse),
            "error_share": {
                key: float((error2 * value).sum() / error2.sum()) for key, value in regions.items()
            },
            "block_error_fraction": {
                str(block): float(block_energy(error, block) / mse) for block in [2, 4, 8, 16]
            },
            "fine_detail_energy_ratio": float(
                (block_energy(pred, 1) - block_energy(pred, 2))
                / (block_energy(target, 1) - block_energy(target, 2))
            ),
            "interior_rgb_bias": ((error * inner[..., None]).sum((0, 1)) / inner.sum()).tolist(),
            "silhouette_distance_mean_px": float(edge_dist.mean()),
            "silhouette_distance_p90_px": float(edge_dist.quantile(0.9)),
            "silhouette_iou": float((gt_mask & pred_mask).sum() / (gt_mask | pred_mask).sum()),
            "mean_visible_support_radius_px": float(maps[..., 1][overlap].mean()),
            "shadow_condition_bins": {},
        }
        for name, low, high in [("deep", 0, 0.1), ("partial", 0.1, 0.5), ("lit", 0.5, 1.01)]:
            selection = overlap & (maps[..., 0] >= low) & (maps[..., 0] < high)
            row["shadow_condition_bins"][name] = {
                "pixels": int(selection.sum()),
                "squared_error": float(error2[selection].sum()),
                "target_luminance_sum": float(target.mean(-1)[selection].sum()),
                "pred_luminance_sum": float(pred.mean(-1)[selection].sum()),
            }
        rows.append(row)
        if (split, index) in show:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            panels = [
                (target, "GT"),
                (pred, f'Prediction {row["PSNR"]:.2f} dB'),
                (error.abs().mean(-1), "Absolute RGB error"),
                (
                    torch.stack(
                        (sample["alpha"][..., 0], alpha[..., 0], torch.zeros_like(alpha[..., 0])),
                        -1,
                    ),
                    "Alpha: red GT, green prediction",
                ),
                (maps[..., 0], "Camera-weighted shadow visibility"),
                (maps[..., 1], "Camera-weighted support radius (px)"),
            ]
            for ax, (values, title) in zip(axes.flat, panels):
                kwargs = {"vmin": 0, "vmax": 1} if title.endswith("visibility") else {}
                if title.startswith("Absolute"):
                    kwargs = {"vmin": 0, "vmax": 0.3}
                if title.endswith("(px)"):
                    kwargs = {"vmin": 0, "vmax": 80}
                artist = ax.imshow(values.cpu().numpy(), **kwargs)
                ax.set_title(title)
                ax.axis("off")
                if values.ndim == 2:
                    fig.colorbar(artist, ax=ax, fraction=0.035)
            camera_label = "saved fit camera" if split == "fit" and camera_offsets is not None else "original camera"
            fig.suptitle(f"{split} frame {index} | {camera_label}")
            fig.tight_layout()
            fig.savefig(output / f"{split}_{index:04}.png", dpi=120)
            plt.close(fig)
    result = {
        "checkpoint": args.checkpoint,
        "rows": rows,
        "notes": [
            "Saved training camera offsets apply to fit frames; held-out frames use original cameras.",
            "Predictions use the saved observation model and training-stage schedule.",
            "Block energy is squared error after non-overlapping area averaging; differences between levels are orthogonal block-detail energy.",
            "Support radius is camera-alpha-compositing weighted; shadow bins use the model visibility estimate.",
            "Silhouette distances use threshold .5 contours in pixels.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frames": len(rows), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
