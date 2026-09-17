"""Evaluate saved models using their training and held-out camera contracts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from gaussians import Gaussians
from methods import build_transport
from renderer import render


def ssim(x, y):
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11)
    mx, my = F.conv2d(x, window, padding=5, groups=3), F.conv2d(y, window, padding=5, groups=3)
    vx = F.conv2d(x * x, window, padding=5, groups=3) - mx * mx
    vy = F.conv2d(y * y, window, padding=5, groups=3) - my * my
    cov = F.conv2d(x * y, window, padding=5, groups=3) - mx * my
    return (
        ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2))
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


def to_device(sample, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


def observation_image(linear, gamma=1.0, alpha=None, background=0.0):
    """Encode HDR RGB or encode PNG foreground before alpha composition."""
    if alpha is not None:
        foreground = (linear - background * (1 - alpha)).clamp_min(0) / alpha.clamp_min(1e-8)
    else:
        foreground = linear
    encoded = foreground if gamma == 1.0 else torch.where(
        foreground > 0,
        foreground.clamp_min(1e-8).pow(1 / gamma),
        torch.zeros_like(foreground),
    )
    return encoded if alpha is None else encoded * alpha + background * (1 - alpha)


def target_image(sample, background, display_gamma=1.0):
    image, alpha = sample["image"], sample["alpha"]
    target = image if alpha is None else image * alpha + background * (1 - alpha)
    return observation_image(target, display_gamma) if sample["is_hdr"] else target


def render_observation(
    gaussians,
    transport,
    sample,
    background,
    shadow,
    port_active,
    display_gamma=1.0,
    shadow_mode="depth",
    absgrad=False,
):
    """Render and apply the same observation model in training and evaluation."""
    linear, alpha, info = render(
        gaussians,
        transport,
        sample,
        background,
        shadow,
        port_active,
        shadow_mode=shadow_mode,
        absgrad=absgrad,
    )
    predicted = observation_image(
        linear,
        display_gamma,
        alpha=None if sample["is_hdr"] else alpha,
        background=background,
    )
    return predicted, alpha, info


def save_pair(path, predicted, target):
    pair = torch.cat((target, predicted), dim=1).detach().clamp(0, 1).cpu().numpy()
    Image.fromarray(np.rint(pair * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def evaluate_samples(
    gaussians,
    transport,
    samples,
    background,
    shadow,
    port_active,
    output=None,
    perceptual=False,
    display_gamma=1.0,
    quantize=False,
    shadow_mode="depth",
):
    transport.eval()
    lpips_model = None
    if perceptual:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg").to(gaussians.center.device).eval()
    results = []
    for idx, sample in enumerate(samples):
        predicted, alpha, _ = render_observation(
            gaussians,
            transport,
            sample,
            background,
            shadow,
            port_active,
            display_gamma,
            shadow_mode,
        )
        target = target_image(sample, background, display_gamma)
        # Unit data range; clamp only for explicitly reported display-domain metrics.
        x, y = predicted.clamp(0, 1), target.clamp(0, 1)
        if quantize:
            x, y = (x * 255).round() / 255, (y * 255).round() / 255
        item = {
            "frame_index": sample["frame_index"],
            "name": sample["name"],
            "PSNR": (-10 * torch.log10((x - y).square().mean())).item(),
            "SSIM": ssim(x, y).item(),
            "raw_MSE": (predicted - target).square().mean().item(),
        }
        if sample["alpha"] is not None:
            item["alpha_L1"] = (alpha - sample["alpha"]).abs().mean().item()
        if lpips_model is not None:
            item["LPIPS"] = lpips_model(
                x.permute(2, 0, 1)[None] * 2 - 1, y.permute(2, 0, 1)[None] * 2 - 1
            ).item()
        results.append(item)
        if output and idx < 4:
            save_pair(Path(output) / f"pair_{idx:03}.png", x, y)
    metrics = {
        key: sum(row[key] for row in results) / len(results)
        for key in results[0]
        if key not in ["frame_index", "name"]
    }
    transport.train()
    return metrics, results


def load_model(path):
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    state = checkpoint["gaussians"]
    count = len(state["params.means"])
    gaussians = Gaussians(
        count,
        state["center"],
        checkpoint["radius"],
        config["feature_dim"],
    )
    gaussians.load_state_dict(state)
    transport = build_transport(
        config, checkpoint["transport"]["light_scale"].item()
    ).cuda()
    transport.load_state_dict(checkpoint["transport"])
    return gaussians, transport, checkpoint


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--split", choices=["validation", "fit", "test", "train"], default="validation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Deterministic frame subsample for diagnostics; 0 evaluates every frame",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()
    from data import SceneDataset

    gaussians, transport, checkpoint = load_model(args.checkpoint)
    config = checkpoint["config"]
    dataset = SceneDataset(
        config["scene"],
        "train" if args.split in ["validation", "fit"] else args.split,
        config["resolution"],
        unit_light_intensity=config["unit_light_intensity"],
    )
    indices = (
        checkpoint["val_indices"]
        if args.split == "validation"
        else checkpoint["fit_indices"] if args.split == "fit" else list(range(len(dataset)))
    )
    if not len(indices):
        raise ValueError(
            "This checkpoint has no held-out validation frames; select an existing split explicitly"
        )
    if args.limit:
        indices = indices[:: max(1, len(indices) // args.limit)][: args.limit]
    camera_offsets = None
    fit_camera_indices = {}
    if args.split in ["fit", "train"] and checkpoint["camera_offsets"] is not None:
        from cameras import TrainCameraOffsets

        camera_offsets = TrainCameraOffsets(len(checkpoint["fit_indices"]), gaussians.radius).cuda()
        camera_offsets.load_state_dict(checkpoint["camera_offsets"])
        fit_camera_indices = {index: local for local, index in enumerate(checkpoint["fit_indices"])}

    def evaluation_samples():
        for index in indices:
            sample = to_device(dataset[index], "cuda")
            if index in fit_camera_indices:
                sample = camera_offsets.correct(sample, fit_camera_indices[index])
            yield sample

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metrics, rows = evaluate_samples(
        gaussians,
        transport,
        evaluation_samples(),
        config["background"],
        checkpoint["step"] >= config["shadow_start"],
        checkpoint["step"] >= config["port_start"],
        output,
        args.lpips,
        config["display_gamma"],
        quantize=True,
        shadow_mode=config["shadow_mode"],
    )
    report = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "views": rows,
        "protocol": "uint8-quantized observation RGB, unit PSNR, zero-padded 11x11 SSIM sigma1.5; LPIPS standard [-1,1]; raw_MSE is unclipped observation MSE; original held-out camera calibration",
        "resolution": config["resolution"],
        "background": config["background"],
        "evaluated_frames": len(indices),
        "limit": args.limit,
        "display_gamma": config["display_gamma"],
        "camera_protocol": "saved training offsets on fit frames; original calibration on held-out frames",
        "corrected_fit_frames": sum(index in fit_camera_indices for index in indices),
        "observation_protocol": "PNG foreground gamma then alpha composition; HDR full-image gamma",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
