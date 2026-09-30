"""Generate train-only surface seeds or StableNormal/DA3 targets.

Run each predictor in its own inference environment; training only reads tensors.
"""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from data import SceneDataset, split_train_lights
from evaluate import target_image

ROOT = Path(__file__).resolve().parent


def depth_context_batches(dataset, fit_indices, targets, views):
    """Share spatially spread training anchors across camera-conditioned batches."""
    if views == 1:
        return [([index], [index]) for index in targets]
    if views < 4 or views > len(fit_indices):
        raise ValueError("Multi-view depth requires 4 or more views, within the training split")
    origins = torch.tensor([dataset.frames[i]['transform_matrix'] for i in fit_indices])[:, :3, 3]
    selected = [0]
    distance = (origins-origins[0]).square().sum(-1)
    for _ in range(views//2-1):
        selected.append(int(distance.argmax()))
        distance = torch.minimum(distance, (origins-origins[selected[-1]]).square().sum(-1))
    anchors = [fit_indices[i] for i in selected]
    batch_size = views-len(anchors)
    return [(list(dict.fromkeys(anchors+targets[start:start+batch_size])), targets[start:start+batch_size])
            for start in range(0, len(targets), batch_size)]


def stable_normal(weights, source, device):
    sys.path.insert(0, str(source))
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
    from stablenormal.pipeline_stablenormal import StableNormalPipeline
    from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
    options = dict(variant="fp16", torch_dtype=torch.float16, safety_checker=None,
                   local_files_only=True, trust_remote_code=True)
    initial = YOSONormalsPipeline.from_pretrained(str(weights / "yoso-normal-v0-3"), **options).to(device)
    pipe = StableNormalPipeline.from_pretrained(
        str(weights / "stable-normal-v0-1"), **options,
        scheduler=HEURI_DDIMScheduler(prediction_type="sample", beta_start=.00085,
                                     beta_end=.012, beta_schedule="scaled_linear"))
    pipe.x_start_pipeline = initial
    pipe.to(device)
    pipe.prior.to(device, torch.float16)
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["normal", "depth"], required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--processing-resolution", type=int, default=512)
    parser.add_argument("--unit-light-intensity", type=float)
    parser.add_argument("--background", type=float, default=0.)
    parser.add_argument("--display-gamma", type=float, default=2.2)
    parser.add_argument("--fit-all", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Explicit small subset for integration checks")
    parser.add_argument("--weights", type=Path, default=ROOT / "third_party/weights")
    parser.add_argument("--normal-source", type=Path, default=ROOT / "third_party/StableNormal")
    parser.add_argument("--depth-model", default="DA3-LARGE")
    parser.add_argument("--depth-views", type=int, default=1,
                        help="Maximum camera-conditioned views per DA3 batch; 1 uses independent monocular inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.kind}.pt"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite supervision: {destination}")
    dataset = SceneDataset(args.scene, "train", args.resolution, unit_light_intensity=args.unit_light_intensity)
    teacher_dataset = SceneDataset(args.scene, "train", max(args.resolution, args.processing_resolution),
                                    unit_light_intensity=args.unit_light_intensity)
    fit_indices = list(range(len(dataset))) if args.fit_all else split_train_lights(dataset.frames)[0]
    indices = fit_indices[:args.limit] if args.limit else fit_indices
    if args.kind == 'normal' and args.depth_views != 1:
        parser.error('--depth-views applies only to depth')
    if args.kind == "normal":
        model = stable_normal(args.weights, args.normal_source, args.device)
        source = args.normal_source
        model_name = "Stable-X/stable-normal-v0-1 + Stable-X/yoso-normal-v0-3"
    else:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(str(args.weights / args.depth_model)).to(args.device).eval()
        source = ROOT / "third_party/Depth-Anything-3"
        model_name = "depth-anything/" + args.depth_model
    predictions = {}
    contexts = {}
    batches = depth_context_batches(dataset, fit_indices, indices, args.depth_views) if args.kind == 'depth' else [([i], [i]) for i in indices]
    with torch.inference_mode():
        for context_indices, output_indices in batches:
            samples = [teacher_dataset[i] for i in context_indices]
            rgbs = [target_image(sample, args.background, args.display_gamma).clamp(0, 1) for sample in samples]
            images = [Image.fromarray((rgb.numpy()*255).round().astype(np.uint8)) for rgb in rgbs]
            if args.kind == "normal":
                result = model(images[0], processing_resolution=args.processing_resolution,
                               match_input_resolution=True).prediction[0]
                value = torch.from_numpy(np.asarray(result).copy()).float()
                # Author's 2DGS integration negates all three raw prediction axes:
                # hugoycj/2d-gaussian-splatting-great-again/utils/camera_utils.py.
                value = F.normalize(-value, dim=-1)
                values = [value]
            else:
                cameras = {} if args.depth_views == 1 else dict(
                    extrinsics=torch.stack([sample['viewmat'] for sample in samples]).numpy(),
                    intrinsics=torch.stack([sample['K'] for sample in samples]).numpy(),
                    align_to_input_ext_scale=True, ref_view_strategy='first')
                result = model.inference(images, process_res=args.processing_resolution,
                                         process_res_method="upper_bound_resize", **cameras)
                values = [torch.from_numpy(result.depth[context_indices.index(i)].copy()).float()[..., None]
                          for i in output_indices]
            for index, value in zip(output_indices, values):
                h, w = rgbs[context_indices.index(index)].shape[:2]
                ratio = args.resolution / max(h, w)
                shape = (round(h * ratio), round(w * ratio))
                value = F.interpolate(value.permute(2, 0, 1)[None], size=shape,
                                      mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                if args.kind == "normal":
                    value = F.normalize(value, dim=-1)
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite {args.kind} output for frame {index}")
                predictions[index] = value.cpu()
                contexts[index] = context_indices
                print(f"{args.kind}: {len(predictions)}/{len(indices)} frame={index}", flush=True)
    record = dict(scene=str(dataset.scene_path.resolve()), resolution=args.resolution,
                  frames={i: dataset.frames[i] for i in indices}, predictions=predictions,
                  model=model_name, source_revision=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
                  coordinates="opencv_camera", depth_type="relative_camera_z",
                  normal_conversion="negate_xyz",
                  processing_resolution=args.processing_resolution, background=args.background,
                  display_gamma=args.display_gamma, seed=args.seed)
    if args.kind == 'depth':
        record.update(depth_views=args.depth_views, contexts=contexts,
                      context_frames={i:dataset.frames[i] for ids in contexts.values() for i in ids},
                      aligned_to_input_camera_scale=args.depth_views>1)
    torch.save(record, destination)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
