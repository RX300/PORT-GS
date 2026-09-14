"""GPU regression of conservative exchange on a recorded Cat reconstruction."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data import SceneDataset
from transport import Transport, exchange_irradiance as continuous_exchange, quadrature_mass


def exchange_irradiance(incident, partition, logits, mass):
    return continuous_exchange(incident, partition, logits, mass, incident, partition, logits)


def relative_error(actual, expected):
    return float((actual - expected).abs().max() / expected.abs().max())


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    source = root / "runs/research_20260912/cat_r1_s0/last.pt"
    archive = root / "runs/research_20260912/source_before.tar"
    checkpoint = torch.load(source, map_location="cuda", weights_only=False)
    state = checkpoint["gaussians"]
    indices = torch.linspace(0, len(state["params.means"]) - 1, 2048, device="cuda").long()
    params = {
        key.removeprefix("params."): value[indices].double()
        for key, value in state.items() if key.startswith("params.")
    }
    gaussians = SimpleNamespace(
        params=params, center=state["center"].double(), radius=checkpoint["radius"]
    )
    dataset = SceneDataset(checkpoint["config"]["scene"], "train", resolution=512)
    frame_indices = [checkpoint["fit_indices"][0], checkpoint["fit_indices"][len(checkpoint["fit_indices"]) // 2]]
    frames = [dataset[index] for index in frame_indices]
    incident = [
        frame["light_intensity"].cuda().double()[None]
        / (frame["light_pos"].cuda().double() - params["means"]).square().sum(-1, keepdim=True)
        for frame in frames
    ]
    hash_config = json.loads((root / "configs/hashgrid.json").read_text())
    model = Transport(feature_dim=params["features"].shape[-1], hash_encoding=hash_config).cuda()
    mass = quadrature_mass(params)
    xyz = ((params["means"] - gaussians.center) / gaussians.radius).float().detach().requires_grad_()
    native_partition = model.partition(xyz)
    # Float64 renormalization isolates the operator audit from float32 encoding rounding.
    partition = native_partition.double().log_softmax(-1)
    # Use actual learned feature channels to exercise spatially varying RGB
    # exchange fractions. A constant initialized head could conceal a missing
    # source-side exchange factor in an otherwise incorrect implementation.
    logits = model.exchange(params["features"].float()).double() + params["features"][:, :3]
    output = exchange_irradiance(incident[0], partition, logits, mass)
    constant = incident[0].mean(0).expand_as(incident[0])
    constant_output = exchange_irradiance(constant, partition, logits, mass)
    sums = exchange_irradiance(incident[0] + incident[1], partition, logits, mass)
    separate = output + exchange_irradiance(incident[1], partition, logits, mass)
    measures = {
        "constant_preservation_relative_error": relative_error(constant_output, constant),
        "weighted_conservation_relative_error": relative_error(
            (mass[:, None] * output).sum(0), (mass[:, None] * incident[0]).sum(0)
        ),
        "irradiance_additivity_relative_error": relative_error(sums, separate),
    }

    # Build the actual 256-point operator explicitly, independently of the
    # implementation's log-domain pooling, and check its weighted transpose.
    small_mass = mass[:256] / mass[:256].sum()
    small_partition = partition[:256].exp()
    small_fraction = logits[:256].sigmoid()
    denominator = torch.einsum("n,nc,nr->rc", small_mass, small_fraction, small_partition)
    kernel = torch.einsum("ir,jr,rc,ic,jc,j->cij", small_partition, small_partition,
                          denominator.reciprocal(), small_fraction, small_fraction, small_mass)
    kernel = kernel + torch.diag_embed((1 - small_fraction).T)
    weighted_kernel = kernel * small_mass[None, :, None]
    measures["detailed_balance_relative_error"] = relative_error(
        weighted_kernel, weighted_kernel.transpose(-1, -2)
    )
    explicit = torch.einsum("cij,jc->ic", kernel, incident[0][:256])
    implicit = exchange_irradiance(incident[0][:256], partition[:256], logits[:256], small_mass)
    measures["explicit_operator_relative_error"] = relative_error(implicit, explicit)

    query_indices = torch.arange(257, 1025, device="cuda")
    query_full = continuous_exchange(
        incident[0], partition, logits, mass,
        incident[0][query_indices], partition[query_indices], logits[query_indices],
    )
    query_chunks = torch.cat([
        continuous_exchange(incident[0], partition, logits, mass,
                            incident[0][chunk], partition[chunk], logits[chunk])
        for chunk in query_indices.split(173)
    ])
    measures["source_node_query_relative_error"] = relative_error(query_full, output[query_indices])
    measures["query_chunk_relative_error"] = relative_error(query_chunks, query_full)

    # A second real light field supplies a nonconstant scalar probe, so the
    # conservation identity cannot make this gradient audit tautological.
    probe = incident[1] / incident[1].square().mean().sqrt()
    logit_direction = params["features"][:, :3]
    logit_direction = logit_direction / logit_direction.square().mean().sqrt()

    def objective(log_partition, exchange_logits):
        exchanged = exchange_irradiance(incident[0], log_partition, exchange_logits, mass)
        return (mass[:, None] * probe * exchanged).sum()

    loss = objective(partition, logits)
    logits_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    epsilon = 1e-4
    finite_logit = (
        objective(partition.detach(), logits.detach() + epsilon * logit_direction)
        - objective(partition.detach(), logits.detach() - epsilon * logit_direction)
    ) / (2 * epsilon)
    analytic_logit = (logits_gradient * logit_direction).sum()
    # Exercise native HashGrid backward through actual exchange and verify Adam
    # updates the native table, projection head, and geometry input gradients.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-15)
    before = model.spatial_encoding.params.detach().clone()
    loss.backward()
    native_gradients = {
        name: {"finite": bool(torch.isfinite(gradient).all()), "max_abs": float(gradient.abs().max())}
        for name, gradient in {
            "hash_table": model.spatial_encoding.params.grad,
            "partition_head": model.partition_head.weight.grad,
            "xyz": xyz.grad,
        }.items()
    }
    optimizer.step()
    table_update = float((model.spatial_encoding.params.detach() - before).abs().max())
    gradients = {
        "feature_conditioned_exchange_logits": {
            "autograd": float(analytic_logit), "finite_difference": float(finite_logit),
            "relative_error": relative_error(analytic_logit, finite_logit),
        },
    }
    # Also check the arithmetic used by production rather than relying solely
    # on the double-precision mathematical reference.
    float_output = exchange_irradiance(
        incident[0].float(), partition.detach().float(), logits.detach().float(), mass.float()
    )
    measures["float32_vs_float64_relative_error"] = relative_error(float_output.double(), output.detach())
    torch.backends.cuda.matmul.allow_tf32 = True
    tf32_output = exchange_irradiance(
        incident[0].float(), partition.detach().float(), logits.detach().float(), mass.float()
    )
    measures["training_tf32_vs_float64_relative_error"] = relative_error(tf32_output.double(), output.detach())
    report = {
        "checkpoint": str(source), "archive": str(archive), "archive_bytes": archive.stat().st_size,
        "dataset": str(dataset.metadata_path), "frame_indices": frame_indices,
        "source_points": len(state["params.means"]), "audited_points": len(indices),
        "selection": "2048 uniformly spaced checkpoint point indices; explicit operator uses first 256",
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": "0",
        "arithmetic": "CUDA float64 reference, CUDA float32, and training-enabled TF32 comparison",
        "illumination": "Two actual train-frame point lights; direct irradiance before visibility",
        "model": "NVIDIA HashGrid partition, seed 0; actual saved Gaussian feature channels",
        "hash_encoding": hash_config, "native_gradients": native_gradients, "hash_table_update_max": table_update,
        "measurements": measures, "gradient_epsilon": epsilon, "gradients": gradients,
        "thresholds": {"identities_relative": 1e-10, "float32_relative": 2e-6,
                       "training_tf32_relative": 2e-3, "gradient_relative": 1e-4},
    }
    report["passed"] = (
        all(value < 1e-10 for key, value in measures.items()
            if key not in ["float32_vs_float64_relative_error", "training_tf32_vs_float64_relative_error"])
        and measures["float32_vs_float64_relative_error"] < 2e-6
        and measures["training_tf32_vs_float64_relative_error"] < 2e-3
        and all(value["relative_error"] < 1e-4 and abs(value["autograd"]) > 1e-10
                for value in gradients.values())
        and all(item["finite"] and item["max_abs"] > 0 for item in native_gradients.values())
        and table_update > 0
    )
    destination = root / "runs/hashgrid_preflight/operator_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["passed"], "Conservative transport regression failed; inspect measured residuals"


if __name__ == "__main__":
    main()
