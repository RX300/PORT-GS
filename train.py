"""One GPU per run, training-only light holdout, explicit checkpoints."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from gsplat.strategy.ops import remove

from data import SceneDataset, split_train_lights
from gaussians import Gaussians, camera_bounds
from transport import Transport
from refinement import Refinement
from evaluate import to_device, target_image, ssim, evaluate_samples, save_pair, render_observation


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--points", type=int, default=20000)
    p.add_argument("--max-points", type=int, default=400000)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--width", type=int, default=128)
    p.add_argument(
        "--rank", type=int, default=512,
        help="number of learned spatial exchange nodes (anchors)",
    )
    p.add_argument("--port-start", type=int, default=5000)
    p.add_argument("--shadow-start", type=int, default=1500)
    p.add_argument("--refine-stop", type=int, default=15000)
    p.add_argument("--validate-every", type=int, default=10000)
    p.add_argument("--val-limit", type=int, default=100000)
    p.add_argument("--background", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint")
    initialization.add_argument("--init-geometry")
    p.add_argument("--freeze-geometry", action="store_true")
    p.add_argument("--unit-light-intensity", type=float)
    p.add_argument("--mask-weight", type=float, default=0.05)
    p.add_argument("--init-radius", type=float)
    p.add_argument("--display-gamma", type=float, default=2.2)
    p.add_argument("--opacity-cap", type=float, default=0.99)
    p.add_argument("--min-scale", type=float, default=1e-4)
    p.add_argument("--max-scale", type=float, default=0.1)
    p.add_argument("--position-scale", choices=["camera", "object"], default="camera")
    p.add_argument("--densification-scale", choices=["camera", "object"], default="object")
    p.add_argument("--position-decay", type=float, default=0.01)
    p.add_argument("--optimize-cameras", action="store_true")
    p.add_argument("--camera-start", type=int, default=1000)
    p.add_argument("--camera-lr", type=float, default=0.0003)
    p.add_argument("--shadow-mode", choices=["depth", "deep"], default="deep")
    p.add_argument("--absgrad", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--fit-all",
        action="store_true",
        help="Final fit on all official training frames; no validation-based checkpoint selection",
    )
    return p.parse_args()


def main():
    args = arguments()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args)
    config["sss_light_axes"] = "world"
    dataset = SceneDataset(
        args.scene, "train", args.resolution, unit_light_intensity=args.unit_light_intensity
    )
    assert dataset.metadata_path.name == "transforms_train.json"
    config["metadata_path"] = str(dataset.metadata_path)
    saved = (
        torch.load(args.init_checkpoint, map_location="cuda", weights_only=False)
        if args.init_checkpoint
        else None
    )
    if args.fit_all:
        fit_indices, val_indices = list(range(len(dataset))), []
    elif saved is not None:
        fit_indices, val_indices = saved["fit_indices"], saved["val_indices"]
        config["fit_all"] = len(val_indices) == 0
    else:
        fit_indices, val_indices = split_train_lights(dataset.frames)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "split.json").write_text(
        json.dumps({"fit": fit_indices, "validation": val_indices}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "loading_train_only",
                "metadata_path": str(dataset.metadata_path),
                "total": len(dataset),
                "fit": len(fit_indices),
                "validation": len(val_indices),
            }
        ),
        flush=True,
    )
    # Keep the training images on the single assigned GPU.
    samples = [to_device(dataset[i], "cuda") for i in range(len(dataset))]
    bound_samples = [samples[i] for i in fit_indices[:: max(1, len(fit_indices) // 32)]]
    center, radius = camera_bounds(bound_samples)
    if args.init_radius is not None:
        center = torch.zeros(3, device="cuda")
        radius = args.init_radius
    irradiances = torch.stack(
        [
            samples[i]["light_intensity"] / (samples[i]["light_pos"] - center).square().sum()
            for i in fit_indices
        ]
    )
    light_scale = irradiances.median().item()
    gaussians = Gaussians(args.points, center, radius, args.feature_dim)
    transport = Transport(
        feature_dim=args.feature_dim,
        width=args.width,
        rank=args.rank,
        light_scale=light_scale,
    ).cuda()
    if args.init_checkpoint:
        if dataset.scene_path.parent.name == "Synthetic_SSS-GS":
            assert (
                saved.get("config", {}).get("sss_light_axes") == "world"
            ), "Invalid legacy SSS initialization"
        gaussians = Gaussians(
            len(saved["gaussians"]["params.means"]),
            saved["gaussians"]["center"],
            saved["radius"],
            args.feature_dim,
        )
        gaussians.load_state_dict(saved["gaussians"])
        transport.load_state_dict(saved["transport"])
    elif args.init_geometry:
        geometry = torch.load(args.init_geometry, map_location="cuda", weights_only=False)
        if dataset.scene_path.parent.name == "Synthetic_SSS-GS":
            assert (
                geometry.get("config", {}).get("sss_light_axes") == "world"
            ), "SSS geometry must document the corrected training-data contract"
        assert geometry["fit_indices"] == fit_indices and geometry["val_indices"] == val_indices
        state_geometry = geometry["gaussians"]
        gaussians = Gaussians(
            len(state_geometry["params.means"]),
            state_geometry["center"],
            geometry["radius"],
            args.feature_dim,
        )
        with torch.no_grad():
            for key in ["means", "scales", "quats", "opacities"]:
                gaussians.params[key].copy_(state_geometry["params." + key])
    gaussians.project_geometry(args.opacity_cap, args.min_scale, args.max_scale)
    if args.freeze_geometry:
        for key in ["means", "scales", "quats", "opacities"]:
            gaussians.params[key].requires_grad_(False)
    camera_origins = torch.stack([samples[i]["c2w"][:3, 3] for i in fit_indices])
    camera_extent = 1.1 * (camera_origins - camera_origins.mean(0)).norm(dim=-1).max().item()
    position_scale = camera_extent if args.position_scale == "camera" else gaussians.radius
    density_scale = camera_extent if args.densification_scale == "camera" else gaussians.radius
    optimizers = gaussians.optimizers(position_scale)
    network_optimizer = torch.optim.Adam(transport.parameters(), lr=0.001, eps=1e-15)
    camera_offsets = None
    if args.optimize_cameras:
        from cameras import TrainCameraOffsets

        camera_offsets = TrainCameraOffsets(len(fit_indices), gaussians.radius).cuda()
        if args.init_checkpoint and saved["camera_offsets"] is not None:
            assert saved["fit_indices"] == fit_indices
            camera_offsets.load_state_dict(saved["camera_offsets"])
        camera_optimizer = torch.optim.Adam(camera_offsets.parameters(), lr=args.camera_lr)
        camera_indices = {index: local for local, index in enumerate(fit_indices)}
    strategy = Refinement(
        refine_stop_iter=args.refine_stop,
        refine_scale2d_stop_iter=args.refine_stop,
        grow_scale2d=0.03,
        # Split broad footprints; opacity and world size determine pruning.
        prune_scale2d=float("inf"),
        grow_grad2d=0.0008 if args.absgrad else 0.0002,
        reset_every=3000,
        absgrad=args.absgrad,
        max_points=args.max_points,
    )
    if not args.freeze_geometry:
        strategy.check_sanity(gaussians.params, optimizers)
    state = strategy.initialize_state(density_scale)
    excess = len(gaussians.params["means"]) - args.max_points
    if excess > 0:
        remove_mask = torch.zeros(len(gaussians.params["means"]), device="cuda", dtype=torch.bool)
        remove_mask[gaussians.params["opacities"].detach().argsort()[:excess]] = True
        remove(gaussians.params, optimizers, state, remove_mask)
    print(
        json.dumps(
            {
                "event": "initialized",
                "center": gaussians.center.tolist(),
                "radius": gaussians.radius,
                "light_scale": transport.light_scale.item(),
                "points": len(gaussians.params["means"]),
                "opacity_max": gaussians.params["opacities"].sigmoid().max().item(),
                "min_scale_ratio": gaussians.params["scales"].min().exp().item() / gaussians.radius,
                "camera_extent": camera_extent,
                "position_scale": position_scale,
                "densification_scale": density_scale,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )
    validation = [
        samples[i]
        for i in val_indices[:: max(1, len(val_indices) // args.val_limit)][: args.val_limit]
    ]
    history = (output / "history.jsonl").open("a", buffering=1)
    start = time.monotonic()
    best = -math.inf

    def checkpoint(step, path):
        torch.save(
            {
                "config": config,
                "step": step,
                "gaussians": gaussians.state_dict(),
                "transport": transport.state_dict(),
                "radius": gaussians.radius,
                "val_indices": val_indices,
                "fit_indices": fit_indices,
                "camera_offsets": None if camera_offsets is None else camera_offsets.state_dict(),
            },
            path,
        )

    for step in range(1, args.steps + 1):
        sample_index = random.choice(fit_indices)
        sample = samples[sample_index]
        camera_active = camera_offsets is not None and step >= args.camera_start
        if camera_active:
            camera_optimizer.zero_grad(set_to_none=True)
            sample = camera_offsets.correct(sample, camera_indices[sample_index])
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        network_optimizer.zero_grad(set_to_none=True)
        decay = 0.1 ** (step / args.steps)
        optimizers["means"].param_groups[0]["lr"] = (
            1.6e-4 * position_scale * args.position_decay ** (step / args.steps)
        )
        network_optimizer.param_groups[0]["lr"] = 0.001 * (0.2 + 0.8 * decay)
        predicted, alpha, info = render_observation(
            gaussians,
            transport,
            sample,
            args.background,
            step >= args.shadow_start,
            step >= args.port_start,
            args.display_gamma,
            args.shadow_mode,
            absgrad=args.absgrad,
        )
        target = target_image(sample, args.background, args.display_gamma)
        l1 = (predicted - target).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(predicted.clamp(0, 1), target.clamp(0, 1)))
        if sample["alpha"] is not None:
            loss = loss + args.mask_weight * (alpha - sample["alpha"]).abs().mean()
        loss = loss + 1e-5 * gaussians.params["features"].square().mean()
        if camera_active:
            loss = loss + 0.001 * camera_offsets.raw[camera_indices[sample_index]].square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {step}")
        if not args.freeze_geometry:
            strategy.step_pre_backward(gaussians.params, optimizers, state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        network_optimizer.step()
        if camera_active:
            camera_optimizer.step()
        if not args.freeze_geometry:
            event = strategy.step_post_backward(
                gaussians.params, optimizers, state, step, info, packed=False
            )
            if event is not None:
                history.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
        gaussians.project_geometry(args.opacity_cap, args.min_scale, args.max_scale)
        if step % 100 == 0 or step == 1:
            row = {
                "step": step,
                "frame_index": sample_index,
                "loss": loss.item(),
                "l1": l1.item(),
                "points": len(gaussians.params["means"]),
                "seconds": round(time.monotonic() - start, 2),
                "memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
            }
            history.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            if not validation:
                checkpoint(step, output / "last.pt")
                continue
            metrics, _ = evaluate_samples(
                gaussians,
                transport,
                validation,
                args.background,
                step >= args.shadow_start,
                step >= args.port_start,
                display_gamma=args.display_gamma,
                shadow_mode=args.shadow_mode,
            )
            row = {
                "step": step,
                "validation": metrics,
                "seconds": round(time.monotonic() - start, 2),
            }
            if camera_offsets is not None:
                rotation_rms, translation_rms = camera_offsets.rms()
                row.update(camera_rotation_rms=rotation_rms, camera_translation_rms=translation_rms)
            history.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            checkpoint(step, output / "last.pt")
            if metrics["PSNR"] > best:
                best = metrics["PSNR"]
                checkpoint(step, output / "best.pt")
            val = validation[0]
            with torch.no_grad():
                preview, _, _ = render_observation(
                    gaussians,
                    transport,
                    val,
                    args.background,
                    step >= args.shadow_start,
                    step >= args.port_start,
                    args.display_gamma,
                    args.shadow_mode,
                )
                save_pair(
                    output / "validation_pair.png",
                    preview,
                    target_image(val, args.background, args.display_gamma),
                )
    history.close()
    print(
        json.dumps(
            {
                "event": "complete",
                "best_validation_psnr": best if validation else None,
                "seconds": time.monotonic() - start,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
