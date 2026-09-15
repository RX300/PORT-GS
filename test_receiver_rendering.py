"""Audit continuous receiver projection and real Cat rendering derivatives on GPU."""

import json
from pathlib import Path

import torch
from gsplat import rasterization
from torch.nn import functional as F

from data import SceneDataset
from evaluate import observation_image, target_image, to_device
from gaussians import Gaussians
from renderer import render, visibility_hint
from transport import Transport


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent
    checkpoint_path = root / "runs/research_20260912/cat_r1_s0/last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    state = checkpoint["gaussians"]
    gaussians = Gaussians(len(state["params.means"]), state["center"], checkpoint["radius"], 32)
    gaussians.load_state_dict(state)
    model = Transport(
        light_scale=checkpoint["transport"]["light_scale"].item(),
        hash_encoding=json.loads((root / "configs/hashgrid.json").read_text()),
    ).cuda()
    dataset = SceneDataset(checkpoint["config"]["scene"], "train", 512)
    sample = to_device(dataset[0], "cuda")
    target = target_image(sample, 0.0, 2.2).detach()
    capture = {}

    def capture_receivers(module, arguments, output):
        capture["receivers"] = {key: value.detach() for key, value in arguments[0].items()}
        capture["foreground"] = output.detach()

    hook = model.register_forward_hook(capture_receivers)
    with torch.no_grad():
        linear, alpha, _ = render(gaussians, model, sample, 0.0, True, "deep")
        visibility = visibility_hint(gaussians, sample["light_pos"], mode="deep")
        attributes = torch.cat((gaussians.params["base"], gaussians.params["features"], visibility[:, None]), -1)
        h, w = sample["image"].shape[:2]
        raw, raw_alpha, _ = rasterization(
            **gaussians.raster_inputs(), colors=attributes,
            viewmats=sample["viewmat"][None], Ks=sample["K"][None], width=w, height=h,
            backgrounds=attributes.new_zeros((1, attributes.shape[-1])),
            render_mode="RGB+D", packed=False,
        )
        covered = alpha[..., 0] > 0
        y, x = torch.where(covered)
        expected_pixels = torch.stack((x + 0.5, y + 0.5), -1)
        world = capture["receivers"]["means"]
        viewmat = sample["viewmat"]
        camera = world @ viewmat[:3, :3].T + viewmat[:3, 3]
        projected = camera @ sample["K"].T
        projected = projected[:, :2] / projected[:, 2:]
        # Native RGB+D returns unnormalized weighted camera depth, independently
        # confirming that the RGB+ED receiver depth is already alpha-normalized.
        expected_depth = raw[0, ..., -1][covered] / raw_alpha[0, ..., 0][covered]
        receiver_attributes = torch.cat((capture["receivers"]["base"],
                                          capture["receivers"]["features"],
                                          capture["receivers"]["visibility"][:, None]), -1)
        expected_attributes = raw[0, ..., :-1][covered] / raw_alpha[0][covered]
        predicted = observation_image(linear, 2.2, alpha=alpha, background=0.0)
        encoded_reference = torch.zeros_like(predicted)
        encoded_reference[covered] = capture["foreground"].pow(1 / 2.2) * alpha[covered]
        identities = {
            "covered_pixels": int(covered.sum()),
            "projection_max_pixels": float((projected - expected_pixels).abs().max()),
            "depth_max_relative": float((camera[:, 2] - expected_depth).abs().max() / expected_depth.abs().max()),
            "attribute_max_absolute": float((receiver_attributes - expected_attributes).abs().max()),
            "alpha_max_absolute": float((alpha - raw_alpha[0]).abs().max()),
            "png_encoding_max_absolute": float((predicted - encoded_reference).abs().max()),
        }
    hook.remove()
    raw_mask = sample["alpha"][..., 0]
    interior = F.avg_pool2d((raw_mask > .99).float()[None, None], 9, 1, 4)[0, 0] > .999
    edge = (raw_mask > .05) & (raw_mask < .95)
    masks = {"interior": interior, "transparent_edge": edge}
    derivatives = {}
    native_absgrad = {}
    for region, mask in masks.items():
        weights = (target * mask[..., None]).detach()
        gaussians.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        linear, alpha, info = render(gaussians, model, sample, 0.0, True, "deep", absgrad=True)
        info["means2d"].retain_grad()
        prediction = observation_image(linear, 2.2, alpha=alpha, background=0.0)
        loss = (prediction.double() * weights.double()).sum()
        loss.backward()
        native_absgrad[region] = {
            "shape": list(info["means2d"].absgrad.shape),
            "finite": bool(torch.isfinite(info["means2d"].absgrad).all()),
            "nonzero": bool((info["means2d"].absgrad > 0).any()),
            "region_pixels": int(mask.sum()),
        }
        selections = {}
        for parameter in ["means", "opacities"]:
            gradient = gaussians.params[parameter].grad
            index = int(gradient.abs().argmax())
            selections[parameter] = (index, float(gradient.flatten()[index]))
        del linear, alpha, prediction, loss, info
        derivatives[region] = {}
        for parameter, (index, analytic) in selections.items():
            value = gaussians.params[parameter].flatten()
            original = value[index].detach().clone()
            epsilons = ([1e-4, 3e-5, 1e-5] if parameter == "opacities"
                        else [gaussians.radius * factor for factor in [1e-4, 3e-5, 1e-5]])
            rows = []
            with torch.no_grad():
                for epsilon in epsilons:
                    values = []
                    for sign in [1, -1]:
                        value[index] = original + sign * epsilon
                        linear, alpha, _ = render(gaussians, model, sample, 0.0, True, "deep")
                        prediction = observation_image(linear, 2.2, alpha=alpha, background=0.0)
                        values.append(float((prediction.double() * weights.double()).sum()))
                    value[index] = original
                    finite = (values[0] - values[1]) / (2 * epsilon)
                    rows.append({"epsilon": epsilon, "finite_difference": finite,
                                 "relative_error": abs(finite - analytic) / abs(analytic)})
            derivatives[region][parameter] = {"flat_parameter_index": index, "autograd": analytic,
                                              "finite_differences": rows}
    report = {
        "checkpoint": str(checkpoint_path), "train_frame": 0, "points": len(state["params.means"]),
        "K": sample["K"].tolist(), "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": "0",
        "tf32_enabled": False, "geometry_and_images": "Actual saved r1 Cat geometry/base/features and actual train image",
        "material": "Residual HashGrid RGB decoder initialization, seed 0",
        "render": "512px; full actual geometry; native deep shadows and continuous receiver shading",
        "identities": identities, "native_absgrad": native_absgrad, "derivatives": derivatives,
        "thresholds": {"projection_pixels": .005, "depth_relative": 1e-5,
                       "attributes_absolute": 1e-5, "png_absolute": 1e-5,
                       "best_finite_difference_relative": .05},
        "derivative_scope": "Largest actual nonzero coordinate/logit gradient per GT-defined region; native thresholded rasterization is piecewise differentiable",
    }
    report["passed"] = (
        identities["projection_max_pixels"] < .005 and identities["depth_max_relative"] < 1e-5
        and identities["attribute_max_absolute"] < 1e-5 and identities["png_encoding_max_absolute"] < 1e-5
        and all(row["finite"] and row["nonzero"] and row["shape"] == [1, len(state["params.means"]), 2]
                for row in native_absgrad.values())
        and all(abs(item["autograd"]) > 1e-8 and min(row["relative_error"] for row in item["finite_differences"]) < .05
                for group in derivatives.values() for item in group.values())
    )
    destination = root / "runs/residual_hashgrid_preflight/receiver_rendering_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["passed"], "Receiver projection/gradient audit failed; inspect measured residuals"


if __name__ == "__main__":
    main()
