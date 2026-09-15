"""Audit direct HashGrid queries on actual Cat geometry and training lights."""

import io
import json
from pathlib import Path

import torch

from data import SceneDataset
from evaluate import to_device
from gaussians import Gaussians
from renderer import visibility_hint
from transport import ResidualBlock, Transport


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent
    source = root / "runs/research_20260912/cat_r1_s0/last.pt"
    checkpoint = torch.load(source, map_location="cuda", weights_only=False)
    state = checkpoint["gaussians"]
    gaussians = Gaussians(len(state["params.means"]), state["center"], checkpoint["radius"], 32)
    gaussians.load_state_dict(state)
    dataset = SceneDataset(checkpoint["config"]["scene"], "train", 512)
    sample = to_device(dataset[checkpoint["fit_indices"][0]], "cuda")
    indices = torch.linspace(0, len(gaussians.params["means"]) - 1, 2048, device="cuda").long()
    with torch.no_grad():
        visibility = visibility_hint(gaussians, sample["light_pos"], mode="deep")[indices]
    receivers = {
        name: gaussians.params[name][indices].detach().requires_grad_()
        for name in ("means", "base", "features")
    }
    receivers["visibility"] = visibility.detach().requires_grad_()
    light = sample["light_pos"].detach().clone().requires_grad_()
    eye = sample["c2w"][:3, 3].detach().clone().requires_grad_()
    intensity = sample["light_intensity"]
    encoding = json.loads((root / "configs/hashgrid.json").read_text())
    model = Transport(hash_encoding=encoding,
                      light_scale=checkpoint["transport"]["light_scale"].item()).cuda()

    def query(points, illumination=intensity):
        return model(points, gaussians.center, gaussians.radius, eye, light, illumination)

    predicted = query(receivers)
    with torch.no_grad():
        # A direct query must be independent of what other receivers share its batch.
        chunks = torch.cat([query({key: value[i:i + 173] for key, value in receivers.items()})
                            for i in range(0, len(indices), 173)])
        torch.testing.assert_close(chunks, predicted, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(query(receivers, intensity * 2), predicted * 2, atol=1e-6, rtol=1e-6)
        assert torch.count_nonzero(query(receivers, torch.zeros_like(intensity))) == 0
        occluded = query({**receivers, "visibility": torch.zeros_like(visibility)})
        assert torch.isfinite(occluded).all() and (occluded > 0).any()
    before = model.spatial_encoding.params.detach().clone()
    predicted.square().mean().backward()
    gradients = {
        name: {"finite": bool(torch.isfinite(gradient).all()), "max_abs": gradient.abs().max().item()}
        for name, gradient in {
            "hash_table": model.spatial_encoding.params.grad,
            "decoder": model.decoder[0].weight.grad,
            "query_position": receivers["means"].grad,
            "material": receivers["features"].grad,
            "visibility": receivers["visibility"].grad,
            "light_position": light.grad,
            "view_position": eye.grad,
        }.items()
    }
    # Check both affine layers in each actual residual branch receive gradients.
    for index, block in enumerate(module for module in model.decoder if isinstance(module, ResidualBlock)):
        for layer in (0, 2):
            gradient = block.branch[layer].weight.grad
            gradients[f"residual_{index}_layer_{layer}"] = {
                "finite": bool(torch.isfinite(gradient).all()), "max_abs": gradient.abs().max().item(),
            }
    assert all(row["finite"] and row["max_abs"] > 0 for row in gradients.values())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-15)
    optimizer.step()
    update = (model.spatial_encoding.params.detach() - before).abs().max().item()
    assert update > 0

    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = Transport(hash_encoding=encoding).cuda()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(
            restored(receivers, gaussians.center, gaussians.radius, eye, light, intensity),
            query(receivers), atol=0, rtol=0,
        )
    report = {
        "checkpoint_geometry": str(source), "train_frame": sample["frame_index"],
        "queries": len(indices), "representation": "residual_hashgrid_rgb",
        "decoder_input_dim": model.decoder[0].in_features,
        "hash_encoding": encoding, "native_gradients": gradients, "hash_update_max": update,
        "checks": ["batch-independent queries", "linear intensity scaling", "zero light gives zero RGB",
                   "occlusion is not a hard zero gate", "native HashGrid optimization", "checkpoint roundtrip"],
        "passed": True,
    }
    destination = root / "runs/residual_hashgrid_preflight/query_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
