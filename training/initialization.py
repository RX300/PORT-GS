"""Frame selection, GPU samples and the initial Gaussian scene with its transport.

Call order matters: the global torch RNG is consumed by the fresh Gaussians and
the transport even when a checkpoint or imported geometry later replaces them.
"""

import json

import torch

from data import split_train_lights
from evaluate import to_device, target_image, neutral_peak_mask
from gaussians import Gaussians, camera_bounds
from methods import build_transport


def select_frames(args, config, dataset, saved):
    if args.fit_all:
        return list(range(len(dataset))), []
    if saved is not None:
        config["fit_all"] = len(saved["val_indices"]) == 0
        return saved["fit_indices"], saved["val_indices"]
    return split_train_lights(dataset.frames)


def load_surface_priors(args, config, dataset, fit_indices):
    if not (config["geometry"] == "2dgs" and (args.normal_weight or args.depth_weight)
            and not (args.sdf_volume_only or args.radiance_residual)):
        return None
    from surface import SurfacePriors
    return SurfacePriors(args.surface_priors, dataset, fit_indices,
                         bool(args.normal_weight), bool(args.depth_weight))


def load_training_samples(args, dataset, fit_indices):
    """All train frames on the assigned GPU, plus GT-only peak and peak-context masks."""
    samples = [to_device(dataset[i], "cuda") for i in range(len(dataset))]
    peak_masks = {i: neutral_peak_mask(target_image(samples[i], args.background, args.display_gamma),
                                       samples[i]['alpha'])
                  for i in fit_indices} if args.highlight_weight or args.sdf_volume_weight or args.radiance_residual else {}
    peak_context_masks = {
        i: (torch.nn.functional.max_pool2d(mask[None, None].float(), 11, 1, 5)[0, 0] > 0)
           & ~mask & (samples[i]['alpha'][..., 0] > .9)
        for i, mask in peak_masks.items()} if args.sdf_volume_peak_context_fraction or args.residual_context_fraction else {}
    return samples, peak_masks, peak_context_masks


def scene_normalization(args, samples, fit_indices):
    """Object center/radius from fit cameras and the median fit irradiance used as light scale."""
    bound_samples = [samples[i] for i in fit_indices[:: max(1, len(fit_indices) // 32)]]
    center, radius = camera_bounds(bound_samples)
    if args.init_radius is not None:
        center = torch.zeros(3, device="cuda")
        radius = args.init_radius
    irradiances = torch.stack([
        samples[i]["light_intensity"] / (samples[i]["light_pos"] - center).square().sum()
        for i in fit_indices
    ])
    return center, radius, irradiances.median().item()


def _import_geometry(args, config, dataset, output, fit_indices, val_indices):
    """Return (state, source checkpoint) for --init-geometry; GGGS is converted to world 3DGS."""
    if args.init_geometry_format == 'gggs':
        from gggs_reconstruction import author_modules
        author_modules()  # The source capture serializes the author's mode enum.
    geometry = torch.load(args.init_geometry, map_location="cuda", weights_only=False)
    if dataset.scene_path.parent.name == "Synthetic_SSS-GS":
        assert (
            geometry.get("config", {}).get("sss_light_axes") == "world"
        ), "SSS geometry must document the corrected training-data contract"
    if args.init_geometry_format != 'gggs':
        assert geometry["fit_indices"] == fit_indices and geometry["val_indices"] == val_indices
        return geometry["gaussians"], geometry
    from gggs_reconstruction import relighting_state
    if geometry['config']['scene'] != str(dataset.scene_path.resolve()) or geometry['config']['resolution'] != args.resolution:
        raise ValueError('GGGS source scene/resolution differs')
    if not set(geometry['fit_indices']).issubset(fit_indices) or (not args.fit_all and geometry['val_indices'] != val_indices):
        raise ValueError('GGGS source has seen held-out material frames')
    state = relighting_state(geometry)
    config.update(source_geometry_step=geometry['step'],
                  source_geometry_fit_indices=geometry['fit_indices'], source_geometry_val_indices=geometry['val_indices'],
                  source_geometry_config=geometry['config'],
                  geometry_handoff='GGGS filtered world covariance and compensated opacity; '
                                   'gsplat EWA/expected-center-depth/deferred transport',
                  geometry_frozen=args.freeze_geometry)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return state, geometry


def build_scene(args, config, dataset, saved, output, fit_indices, val_indices, center, radius, light_scale):
    """Return (gaussians, transport, source scene-gauge shift) ready for optimizer construction."""
    gaussians = Gaussians(args.points, center, radius, args.feature_dim, geometry=config["geometry"])
    transport = build_transport(config, light_scale).cuda()
    source_shift = None
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
            geometry=config["geometry"],
        )
        gaussians.load_state_dict(saved["gaussians"])
        transport_state = saved["transport"]
        if args.reset_material:
            transport_state = {key: value for key, value in transport_state.items() if not key.startswith('decoder.')}
            transport_state.update({'decoder.' + key: value for key, value in transport.decoder.state_dict().items()})
        transport.load_state_dict(transport_state)
        source_shift = saved.get("scene_gauge_shift")
    elif args.init_geometry:
        state, geometry = _import_geometry(args, config, dataset, output, fit_indices, val_indices)
        if state["params.scales"].shape[-1] != gaussians.params["scales"].shape[-1]:
            raise ValueError("--init-geometry requires matching 3DGS/2DGS geometry; no implicit flattening")
        gaussians = Gaussians(
            len(state["params.means"]),
            state["center"],
            geometry["radius"],
            args.feature_dim,
            geometry=config["geometry"],
        )
        with torch.no_grad():
            for key in ["means", "scales", "quats", "opacities"]:
                gaussians.params[key].copy_(state["params." + key])
        if args.init_geometry_format != 'gggs':
            source_shift = geometry.get("scene_gauge_shift")
    gaussians.surface_depth = args.surface_depth
    if not (args.sdf_volume_only or args.radiance_residual or args.init_geometry_format == 'gggs'):
        gaussians.project_geometry(args.opacity_cap, args.min_scale, args.max_scale)
    if config["representation"] == "neural_material" and (not args.init_checkpoint or args.reset_material):
        transport.initialize_material(gaussians, reset_normal=not args.init_checkpoint)
    if args.freeze_geometry:
        for key in ["means", "scales", "quats", "opacities"]:
            gaussians.params[key].requires_grad_(False)
    return gaussians, transport, source_shift
