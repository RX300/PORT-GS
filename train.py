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
from methods import build_transport, add_method_arguments, resolve_config
from refinement import Refinement
from evaluate import to_device, target_image, observation_image, ssim, evaluate_samples, save_pair, render_observation, neutral_peak_mask


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr-decay-steps", type=int,
                   help="Finish LR decay at this step, then hold it fixed; defaults to --steps")
    p.add_argument("--save-steps", type=int, nargs="*", default=[],
                   help="Explicit intermediate checkpoints, without evaluating or selecting them during training")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--points", type=int, default=20000)
    p.add_argument("--max-points", type=int, default=400000)
    add_method_arguments(p)
    p.add_argument("--port-start", type=int, default=5000)
    p.add_argument("--shadow-start", type=int, default=5000)
    p.add_argument("--refine-stop", type=int, default=25000)
    p.add_argument("--validate-every", type=int, default=0,
                   help="Periodic validation/checkpoint interval; 0 disables periodic validation")
    p.add_argument("--val-limit", type=int, default=100000)
    p.add_argument("--background", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint")
    initialization.add_argument("--init-geometry")
    p.add_argument('--init-geometry-format', choices=['port', 'gggs'], default='port',
                   help='GGGS imports filtered world-space geometry into the default gsplat renderer')
    p.add_argument("--reset-material", action="store_true",
                   help="Reinitialize material response/codes while retaining fitted shading normals and transport")
    p.add_argument("--freeze-geometry", action="store_true")
    p.add_argument("--unit-light-intensity", type=float)
    p.add_argument("--mask-weight", type=float, default=0.05)
    p.add_argument("--highlight-weight", type=float, default=0.,
                   help="Extra RGB and local-contrast loss on training-only neutral bright peaks")
    p.add_argument('--radiance-residual', action='store_true',
                   help='Fit only a light-conditioned radiance residual on a frozen neural-material checkpoint')
    p.add_argument('--residual-normal', choices=['geometry', 'material'], default='geometry')
    p.add_argument('--residual-interaction', choices=['none', 'add', 'multiply'], default='none',
                   help='Spatial-direction interaction in the radiance residual first hidden layer')
    p.add_argument('--residual-angular-bank', choices=['none', 'wide', 'narrow'], default='none',
                   help='Movable appearance-center spherical Gaussian features in the residual head')
    p.add_argument('--residual-rays', type=int, default=2048)
    p.add_argument('--residual-peak-fraction', type=float, default=0.)
    p.add_argument('--residual-context-fraction', type=float, default=0.)
    p.add_argument('--residual-paired-context', action='store_true',
                   help='Pair GT peak anchors with their own 11px context, using fixed quarter-budget quotas')
    p.add_argument('--residual-pair-weight', type=float, default=0.,
                   help='Weight of RGB pair-difference L1; requires --residual-paired-context when positive')
    p.add_argument('--residual-frames', type=int, nargs='+',
                   help='Ordered frame indices for this residual stage only; must be unique saved fit frames. '
                        'Omitting this option, including on resume, samples all saved fit frames. '
                        'Source splits and fitted camera mappings are unchanged.')
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
    p.add_argument("--camera-mode", choices=["anchor", "rotation"], default="anchor",
                   help="anchor: bounded SE(3) with the first fit camera fixed; rotation: all fit cameras "
                        "rotate about their fixed calibrated centers")
    p.add_argument("--camera-lr-final", type=float,
                   help="Exponentially decay the camera learning rate from --camera-lr at --camera-start "
                        "to this value at the final step; omitted keeps it constant")
    p.add_argument("--camera-gauge", choices=["none", "translation"], default="none",
                   help="translation: after each camera step, remove the part of the rotation corrections that "
                        "is equivalent to translating the whole scene and translate the scene instead")
    p.add_argument("--optimize-lights", action="store_true",
                   help="Fit per-frame point-light position offsets on fit frames; held-out views keep calibration")
    p.add_argument("--light-start", type=int, default=10000)
    p.add_argument("--light-lr", type=float, default=0.001, help="Offset learning rate in object-radius units")
    p.add_argument("--light-lr-final", type=float, default=0.00001)
    p.add_argument("--optimize-light-scale", action="store_true",
                   help="Fit one positive scene-wide irradiance normalization from training images")
    p.add_argument("--shadow-mode", choices=["depth", "deep"], default="deep")
    p.add_argument("--absgrad", action=argparse.BooleanOptionalAction, default=True)
    surface = p.add_argument_group("2D Gaussian surface supervision")
    surface.add_argument("--surface-priors", help="Directory containing normal.pt and depth.pt")
    surface.add_argument("--normal-weight", type=float, default=0.05)
    surface.add_argument("--depth-weight", type=float, default=0.05)
    surface.add_argument("--surface-consistency-weight", type=float, default=0.01)
    surface.add_argument("--distortion-weight", type=float, default=0.01)
    surface.add_argument("--surface-start", type=int, default=1000)
    surface.add_argument("--geometry-warmup-steps", type=int, default=0,
                         help="First N steps use only 2DGS RGB and surface losses; relighting starts at N+1")
    surface.add_argument("--sdf", action="store_true", help="Train an auxiliary SDF with bidirectional 2DGS surface supervision")
    surface.add_argument("--sdf-shading", action="store_true",
                        help="Use a fitted SDF's gradient normals in neural-material shading, including at inference")
    surface.add_argument("--sdf-warmup-steps", type=int, default=500)
    surface.add_argument("--sdf-start", type=int, default=1,
                        help="First training iteration that fits the SDF and enables mutual geometry supervision")
    surface.add_argument('--sdf-volume-weight', type=float, default=0.,
                        help='Independent light-conditioned SDF RGB/alpha supervision; 0 keeps point supervision')
    surface.add_argument('--sdf-volume-rays', type=int, default=512)
    surface.add_argument('--sdf-volume-peak-fraction', type=float, default=0.,
                        help='Fraction of SDF rays sampled from training GT neutral peaks; reweights the RGB objective')
    surface.add_argument('--sdf-volume-peak-context-fraction', type=float, default=0.,
                        help='SDF ray quota in the 11px GT peak neighborhood, excluding peaks; reweights the objective')
    surface.add_argument('--sdf-volume-hint-encoding', action='store_true',
                        help='Zero-initialized four-band encoding residual of existing detached log highlight hints')
    surface.add_argument('--sdf-volume-samples', type=int, default=64)
    surface.add_argument('--sdf-volume-fixed-sharpness', type=float,
                        help='Set and freeze positive CDF sharpness after loading weights; omitted means learn it')
    surface.add_argument('--sdf-volume-warmup', type=int, default=500,
                        help='Fit radiance on detached geometry before photometric and ray mutual supervision')
    surface.add_argument('--sdf-volume-only', action='store_true',
                        help='Fit the SDF branch against a completely fixed Gaussian checkpoint')
    surface.add_argument('--sdf-detail', action='store_true', help='Zero-initialized multiresolution SDF residual')
    surface.add_argument('--sdf-volume-detail', action='store_true', help='Position-grid residual in the SDF radiance head')
    surface.add_argument("--sdf-samples", type=int, default=1024)
    surface.add_argument('--sdf-lr', type=float, default=.001)
    surface.add_argument("--sdf-weight", type=float, default=.05)
    surface.add_argument("--sdf-normal-weight", type=float, default=.01)
    surface.add_argument("--sdf-primitive-weight", type=float, default=0.,
                        help="SDF zero-set loss on approximately visible, high-opacity Gaussian centers")
    surface.add_argument("--freeze-sdf", action="store_true",
                        help="Keep an already fitted SDF fixed during Gaussian geometry refinement")
    surface.add_argument("--normal-field", action="store_true",
                        help="Add a position-only continuous residual to neural-material shading normal codes")
    surface.add_argument("--surface-depth", choices=['center', 'intersection'], default='center',
                        help="2DGS receiver/SDF depth: native center Z or per-ray surfel intersections")
    p.add_argument(
        "--fit-all",
        action="store_true",
        help="Final fit on all official training frames; no validation-based checkpoint selection",
    )
    args = p.parse_args()
    if args.init_geometry_format == 'gggs' and (not args.init_geometry
            or args.representation not in ('directional_port_v1','neural_material')
            or (args.optimize_cameras and args.freeze_geometry) or args.sdf):
        p.error('GGGS import requires directional/neural-material relighting and --init-geometry; '
                'camera fitting additionally requires trainable geometry')
    if args.camera_lr_final is not None and (not args.optimize_cameras or args.camera_lr_final <= 0):
        p.error('--camera-lr-final requires --optimize-cameras and a positive value')
    if args.camera_gauge != 'none' and (not args.optimize_cameras or args.camera_mode != 'rotation'):
        p.error('--camera-gauge requires --optimize-cameras with --camera-mode rotation')
    if args.optimize_lights and (args.light_lr <= 0 or args.light_lr_final <= 0 or args.light_start < 1
                                 or args.radiance_residual or args.sdf_volume_only):
        p.error('Light offsets need positive rates/start and a trainable relighting stage')
    if args.init_geometry_format == 'gggs' and args.representation == 'neural_material' and (
            args.normal_weight or args.depth_weight or args.surface_consistency_weight or args.distortion_weight):
        p.error('Material on 3D GGGS requires explicit zero 2D surface/prior weights; no implicit surfel losses')
    if args.representation == "neural_material" and args.material_model == 'neural' and not args.init_checkpoint and not args.material_decoder:
        p.error("fresh neural_material training requires --material-decoder")
    if args.reset_material and (args.representation != 'neural_material' or not args.init_checkpoint
                               or (args.material_model == 'neural' and not args.material_decoder)):
        p.error('--reset-material requires neural_material and --init-checkpoint; neural also requires --material-decoder')
    if args.radiance_residual:
        if args.representation != 'neural_material' or not args.init_checkpoint:
            p.error('--radiance-residual requires neural_material and --init-checkpoint')
        if (args.sdf or args.sdf_shading or args.sdf_volume_weight or args.sdf_volume_only or
            args.freeze_sdf or args.normal_field or args.geometry_warmup_steps or
            args.reset_material or args.optimize_light_scale or args.highlight_weight):
            p.error('Radiance-residual-only training excludes SDF/volume/normal-field, geometry warmup, '
                    'material reset, light-scale optimization and highlight losses')
        args.freeze_geometry = True
        args.camera_lr = 0.
        args.camera_start = 1
    if (args.residual_rays <= 0 or not 0 <= args.residual_peak_fraction < 1 or
        not 0 <= args.residual_context_fraction < 1 or
        args.residual_peak_fraction+args.residual_context_fraction >= 1 or
        (args.residual_context_fraction and not args.residual_peak_fraction)):
        p.error('Residual rays must be positive; peak/context quotas must be nonnegative, sum <1, '
                'and context requires a positive peak quota')
    if (args.residual_peak_fraction or args.residual_context_fraction) and not args.radiance_residual:
        p.error('Residual sampling quotas require --radiance-residual')
    if not math.isfinite(args.residual_pair_weight) or args.residual_pair_weight < 0:
        p.error('--residual-pair-weight must be finite and nonnegative')
    if args.residual_pair_weight and not args.residual_paired_context:
        p.error('Positive --residual-pair-weight requires --residual-paired-context')
    if args.residual_paired_context:
        if not args.radiance_residual:
            p.error('--residual-paired-context requires --radiance-residual')
        if args.residual_peak_fraction != .25 or args.residual_context_fraction != .25:
            p.error('--residual-paired-context requires peak/context fractions of .25 each')
        if args.residual_rays % 4:
            p.error('--residual-paired-context requires --residual-rays divisible by 4')
    if args.residual_frames is not None and not args.radiance_residual:
        p.error('--residual-frames requires --radiance-residual')
    if args.residual_interaction != 'none' and not args.radiance_residual:
        p.error('--residual-interaction requires --radiance-residual')
    if args.residual_angular_bank != 'none' and (
        not args.radiance_residual or args.residual_interaction != 'multiply' or args.residual_normal != 'geometry'
    ):
        p.error('--residual-angular-bank requires --radiance-residual, --residual-interaction multiply '
                'and --residual-normal geometry')
    if args.sdf_volume_weight:
        if not args.sdf or args.sdf_shading or args.sdf_primitive_weight or args.geometry_warmup_steps:
            p.error('SDF volume supervision requires --sdf without SDF shading, primitive loss or RGB geometry warmup')
        if args.sdf_volume_weight < 0 or args.sdf_volume_rays <= 0 or args.sdf_volume_samples < 4 or args.sdf_volume_warmup < 0:
            p.error('SDF volume weight/rays must be positive, samples >=4 and warmup >=0')
    if args.sdf_volume_only:
        if not args.sdf_volume_weight or not args.init_checkpoint or args.sdf_start != 1 or args.optimize_light_scale:
            p.error('SDF-volume-only requires a volume weight, checkpoint, sdf-start=1 and fixed light scale')
        if args.reset_material:
            p.error('SDF-volume-only preserves the Gaussian teacher material')
        args.freeze_geometry = True
        args.camera_lr = 0.
        args.camera_start = 1
    if not 0 <= args.sdf_volume_peak_fraction < 1 or (args.sdf_volume_peak_fraction and not args.sdf_volume_weight):
        p.error('SDF peak-ray fraction must be in [0,1) and requires SDF volume supervision')
    if (args.sdf_volume_peak_context_fraction < 0 or
        args.sdf_volume_peak_fraction+args.sdf_volume_peak_context_fraction >= 1 or
        (args.sdf_volume_peak_context_fraction and not args.sdf_volume_peak_fraction)):
        p.error('SDF context quota requires a positive peak quota, and their sum must be <1')
    if args.sdf_volume_hint_encoding and not args.sdf_volume_weight:
        p.error('SDF hint encoding requires volume supervision')
    if args.sdf_volume_fixed_sharpness is not None and (
        not math.isfinite(args.sdf_volume_fixed_sharpness) or args.sdf_volume_fixed_sharpness <= 0 or
        not args.sdf_volume_weight > 0):
        p.error('Fixed SDF volume sharpness must be positive and finite, and requires positive volume weight')
    if args.sdf_detail and not args.sdf:
        p.error('--sdf-detail requires --sdf')
    if args.sdf_volume_detail and not args.sdf_volume_weight:
        p.error('--sdf-volume-detail requires --sdf-volume-weight')
    return args


def residual_frame_pool(fit_indices, residual_frames, enabled):
    """Select stage sampling frames without changing source fit membership."""
    if residual_frames is None:
        return list(fit_indices)
    if not enabled:
        raise ValueError('Residual frame selection requires --radiance-residual')
    if len(residual_frames) != len(set(residual_frames)):
        raise ValueError('--residual-frames must contain unique indices')
    if not residual_frames or not set(residual_frames).issubset(fit_indices):
        raise ValueError('--residual-frames must contain saved fit frame indices')
    return list(residual_frames)


def residual_optimization_stats(residual):
    """Post-update weight/pre-update gradient RMS; None for an unqueried group."""
    groups = {
        'interaction_spatial': residual.interaction_spatial.parameters(),
        'interaction_angular': residual.interaction_angular.parameters(),
        'interaction_projection': residual.interaction_projection.parameters(),
        'grid_table': [residual.detail.grid.table],
        'detail_projection': residual.detail.projection.parameters(),
        'network_first': residual.network[0].parameters(),
        'network_last': residual.network[-1].parameters(),
    }
    if residual.angular_bank != 'none':
        groups.update({
            'angular_center_first': residual.center_network[0].parameters(),
            'angular_center_last': residual.center_network[2].parameters(),
            'angular_projection': residual.angular_projection.parameters(),
        })
    stats = {}
    for name, parameters in groups.items():
        parameters = list(parameters)
        count = sum(parameter.numel() for parameter in parameters)
        stats[name+'_parameter_rms'] = (
            sum(parameter.detach().square().sum() for parameter in parameters)/count).sqrt().item()
        stats[name+'_gradient_rms'] = (None if any(parameter.grad is None for parameter in parameters) else
            (sum(parameter.grad.detach().square().sum() for parameter in parameters)/count).sqrt().item())
    if residual.angular_bank != 'none':
        gradient = residual.angular_projection.weight.grad
        for index in range(residual.angular_projection.in_features):
            stats[f'angular_kernel_{index}_projection_gradient_rms'] = (
                None if gradient is None else gradient[:, index].detach().square().mean().sqrt().item())
    return stats


def set_training_stage(gaussians, transport, optimizers, step, warmup_steps):
    geometry_only = step <= warmup_steps
    transport.requires_grad_(not geometry_only)
    gaussians.params["features"].requires_grad_(not geometry_only)
    if warmup_steps and step == warmup_steps + 1:
        # Warmup colors encode observed lighting, not material reflectance.
        with torch.no_grad():
            gaussians.params["base"].fill_(-1.5)
        optimizers["base"].state.clear()
    return geometry_only


def main():
    args = arguments()
    decay_steps = args.steps if args.lr_decay_steps is None else args.lr_decay_steps
    if decay_steps <= 0 or any(step <= 0 or step >= args.steps for step in args.save_steps):
        raise ValueError("LR decay steps must be positive; saved steps must lie strictly inside training")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = resolve_config(vars(args))
    saved = (
        torch.load(args.init_checkpoint, map_location="cuda", weights_only=False)
        if args.init_checkpoint else None
    )
    if saved is not None:
        if (saved['config'].get('residual_angular_bank', 'none') != 'none' and
            not saved['config'].get('radiance_residual', False)):
            raise ValueError('Radiance residual angular bank requires an enabled source configuration')
        if (saved['config'].get('radiance_residual', False) or 'radiance_residual' in saved) and not args.radiance_residual:
            raise ValueError('This checkpoint contains a radiance residual; explicitly enable --radiance-residual')
        if args.radiance_residual:
            if (saved['config'].get('sdf_shading', False) or saved['config'].get('normal_field', False) or
                saved['step'] <= saved['config'].get('geometry_warmup_steps', 0)):
                raise ValueError('Radiance residual requires a relighting source without SDF shading or normal fields')
            if 'radiance_residual' in saved and not saved['config'].get('radiance_residual', False):
                raise ValueError('Radiance residual weights require an enabled source configuration')
            if saved['config'].get('radiance_residual', False):
                if saved['config']['residual_normal'] != args.residual_normal:
                    raise ValueError('Resuming a radiance residual requires its saved --residual-normal')
                if saved['config'].get('residual_interaction', 'none') != args.residual_interaction:
                    raise ValueError('Resuming a radiance residual requires its saved --residual-interaction')
                if saved['config'].get('residual_angular_bank', 'none') != args.residual_angular_bank:
                    raise ValueError('Resuming a radiance residual requires its saved --residual-angular-bank')
                if saved['config'].get('residual_paired_context', False) != args.residual_paired_context:
                    raise ValueError('Resuming a radiance residual requires its saved --residual-paired-context')
                if saved['config'].get('residual_pair_weight', 0.) != args.residual_pair_weight:
                    raise ValueError('Resuming a radiance residual requires its saved --residual-pair-weight')
                if 'radiance_residual' not in saved or 'residual_steps' not in saved:
                    raise ValueError('Enabled radiance residual source is missing its state or step count')
            for key in ['surface_depth', 'shadow_mode', 'background', 'display_gamma', 'unit_light_intensity']:
                value = saved['config'].get(key, 'center') if key == 'surface_depth' else saved['config'][key]
                setattr(args, key, value)
                config[key] = value
    if args.surface_depth == 'intersection' and config['geometry'] != '2dgs':
        raise ValueError('Intersection surface depth requires 2DGS geometry')
    if args.geometry_warmup_steps and (config["geometry"] != "2dgs" or args.freeze_geometry):
        raise ValueError("Geometry warmup requires trainable 2DGS geometry")
    if args.sdf and (config["geometry"] != "2dgs" or (args.freeze_geometry and not args.sdf_volume_only)):
        raise ValueError("SDF supervision requires trainable 2DGS geometry, except in volume-only mode")
    if args.sdf_shading and (not args.sdf or args.representation != 'neural_material' or args.geometry_warmup_steps):
        raise ValueError("SDF shading requires --sdf, neural_material and no RGB geometry warmup")
    if args.sdf and (args.sdf_samples <= 0 or args.sdf_lr <= 0 or args.sdf_warmup_steps < 0 or args.sdf_start < 1):
        raise ValueError("SDF samples/start/lr must be positive and warmup must be nonnegative")
    if args.sdf_primitive_weight and not args.sdf:
        raise ValueError("Primitive surface supervision requires --sdf")
    if args.freeze_sdf and not args.sdf:
        raise ValueError("Frozen SDF guidance requires --sdf")
    if args.normal_field and (args.representation != 'neural_material' or args.geometry_warmup_steps):
        raise ValueError("Normal residual fields require neural_material without RGB geometry warmup")
    if config["geometry"] == "2dgs":
        if args.shadow_mode != "deep":
            raise ValueError("2D Gaussian rendering uses the disk deep-shadow approximation")
        if (args.normal_weight or args.depth_weight) and not args.surface_priors and not (args.sdf_volume_only or args.radiance_residual):
            raise ValueError("2D Gaussian surface supervision requires --surface-priors; set both prior weights to 0 for geometry-only ablation")
    config["sss_light_axes"] = "world"
    dataset = SceneDataset(
        args.scene, "train", args.resolution, unit_light_intensity=args.unit_light_intensity
    )
    assert dataset.metadata_path.name == "transforms_train.json"
    config["metadata_path"] = str(dataset.metadata_path)
    if (args.sdf_shading or args.freeze_sdf) and (saved is None or not saved['config'].get('sdf', False)
                             or saved['sdf_steps'] <= args.sdf_warmup_steps):
        raise ValueError("SDF shading/frozen guidance requires a fitted SDF checkpoint beyond warmup")
    if saved is not None:
        saved_method = saved["config"]["representation"]
        if saved_method != args.representation:
            raise ValueError("--init-checkpoint requires the same method; use --init-geometry for another method")
        if args.representation == 'neural_material' and saved['config'].get('material_model', 'neural') != args.material_model and not args.reset_material:
            raise ValueError('Changing material model requires --reset-material to initialize its parameter meanings')
        if args.sdf_volume_only or args.radiance_residual:
            args.optimize_cameras = saved['camera_offsets'] is not None
            config['optimize_cameras'] = args.optimize_cameras
            for key in ['port_start', 'shadow_start']:
                value = 1 if saved['step'] >= saved['config'][key] else args.steps+1
                setattr(args, key, value)
                config[key] = value
    if args.fit_all:
        fit_indices, val_indices = list(range(len(dataset))), []
    elif saved is not None:
        fit_indices, val_indices = saved["fit_indices"], saved["val_indices"]
        config["fit_all"] = len(val_indices) == 0
    else:
        fit_indices, val_indices = split_train_lights(dataset.frames)
    residual_sample_indices = residual_frame_pool(
        saved['fit_indices'] if args.radiance_residual else fit_indices,
        args.residual_frames, args.radiance_residual)
    residual_frame_counts = {index:0 for index in residual_sample_indices}
    priors = None
    if config["geometry"] == "2dgs" and (args.normal_weight or args.depth_weight) and not (args.sdf_volume_only or args.radiance_residual):
        from surface import SurfacePriors
        priors = SurfacePriors(args.surface_priors, dataset, fit_indices,
                               bool(args.normal_weight), bool(args.depth_weight))
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
    peak_masks = {i:neutral_peak_mask(target_image(samples[i],args.background,args.display_gamma),samples[i]['alpha'])
                  for i in fit_indices} if args.highlight_weight or args.sdf_volume_weight or args.radiance_residual else {}
    peak_context_masks = {
        i: (torch.nn.functional.max_pool2d(mask[None,None].float(),11,1,5)[0,0] > 0)
           & ~mask & (samples[i]['alpha'][...,0] > .9)
        for i,mask in peak_masks.items()} if args.sdf_volume_peak_context_fraction or args.residual_context_fraction else {}
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
    gaussians = Gaussians(args.points, center, radius, args.feature_dim, geometry=config["geometry"])
    transport = build_transport(config, light_scale).cuda()
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
    elif args.init_geometry:
        if args.init_geometry_format == 'gggs':
            from gggs_reconstruction import author_modules
            author_modules()  # The source capture serializes the author's mode enum.
        geometry = torch.load(args.init_geometry, map_location="cuda", weights_only=False)
        if dataset.scene_path.parent.name == "Synthetic_SSS-GS":
            assert (
                geometry.get("config", {}).get("sss_light_axes") == "world"
            ), "SSS geometry must document the corrected training-data contract"
        if args.init_geometry_format == 'gggs':
            from gggs_reconstruction import relighting_state
            if geometry['config']['scene'] != str(dataset.scene_path.resolve()) or geometry['config']['resolution'] != args.resolution:
                raise ValueError('GGGS source scene/resolution differs')
            if not set(geometry['fit_indices']).issubset(fit_indices) or (not args.fit_all and geometry['val_indices'] != val_indices):
                raise ValueError('GGGS source has seen held-out material frames')
            state_geometry = relighting_state(geometry)
            config.update(source_geometry_step=geometry['step'],
                source_geometry_fit_indices=geometry['fit_indices'],source_geometry_val_indices=geometry['val_indices'],
                source_geometry_config=geometry['config'],
                geometry_handoff='GGGS filtered world covariance and compensated opacity; gsplat EWA/expected-center-depth/deferred transport',
                geometry_frozen=args.freeze_geometry)
            (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        else:
            assert geometry["fit_indices"] == fit_indices and geometry["val_indices"] == val_indices
            state_geometry = geometry["gaussians"]
        if state_geometry["params.scales"].shape[-1] != gaussians.params["scales"].shape[-1]:
            raise ValueError("--init-geometry requires matching 3DGS/2DGS geometry; no implicit flattening")
        gaussians = Gaussians(
            len(state_geometry["params.means"]),
            state_geometry["center"],
            geometry["radius"],
            args.feature_dim,
            geometry=config["geometry"],
        )
        with torch.no_grad():
            for key in ["means", "scales", "quats", "opacities"]:
                gaussians.params[key].copy_(state_geometry["params." + key])
    gaussians.surface_depth = args.surface_depth
    if not (args.sdf_volume_only or args.radiance_residual or args.init_geometry_format == 'gggs'):
        gaussians.project_geometry(args.opacity_cap, args.min_scale, args.max_scale)
    if config["representation"] == "neural_material" and (not args.init_checkpoint or args.reset_material):
        transport.initialize_material(gaussians, reset_normal=not args.init_checkpoint)
    if args.freeze_geometry:
        for key in ["means", "scales", "quats", "opacities"]:
            gaussians.params[key].requires_grad_(False)
    camera_origins = torch.stack([samples[i]["c2w"][:3, 3] for i in fit_indices])
    camera_extent = 1.1 * (camera_origins - camera_origins.mean(0)).norm(dim=-1).max().item()
    position_scale = camera_extent if args.position_scale == "camera" else gaussians.radius
    density_scale = camera_extent if args.densification_scale == "camera" else gaussians.radius
    optimizers = gaussians.optimizers(position_scale)
    network_parameters = list(transport.parameters())
    network_optimizer = (torch.optim.Adam(network_parameters, lr=0.001, eps=1e-15)
                         if network_parameters else None)
    radiance_residual = None
    residual_steps = 0
    if args.radiance_residual:
        from materials.radiance_residual import RadianceResidual
        from sdf import silhouette_rays
        from sdf_volume import select_rays
        if args.residual_paired_context:
            from residual_sampling import (build_residual_pair_pool, select_residual_pairs,
                                           residual_pair_loss, residual_pair_sample_stats)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            radiance_residual = RadianceResidual(gaussians.center, gaussians.radius,
                                                normal_source=args.residual_normal,
                                                interaction=args.residual_interaction,
                                                angular_bank=args.residual_angular_bank)
        if saved['config'].get('radiance_residual', False):
            expected_scales = (radiance_residual.angular_scales.clone()
                               if args.residual_angular_bank != 'none' else None)
            radiance_residual.load_state_dict(saved['radiance_residual'], strict=True)
            if expected_scales is not None and not torch.equal(radiance_residual.angular_scales, expected_scales):
                raise ValueError('Radiance residual angular scales disagree with the saved angular-bank configuration')
            residual_steps = saved['residual_steps']
        else:
            torch.save({'config': config, 'radiance_residual': radiance_residual.state_dict(),
                        'residual_steps': 0}, output / 'residual_initial.pt')
        residual_optimizer = torch.optim.Adam(radiance_residual.parameters(), lr=.001, eps=1e-15)
        residual_generator = torch.Generator(device='cuda').manual_seed(args.seed+3)
    if args.optimize_light_scale:
        # Keep the checkpoint/render contract: save the fitted positive scale
        # in the existing buffer. Only training needs its log parameter.
        initial_light_scale = transport.light_scale.detach().clone()
        log_light_scale = torch.nn.Parameter(initial_light_scale.log())
        light_scale_optimizer = torch.optim.Adam([log_light_scale], lr=.001)
    sdf = None
    sdf_steps = 0
    if args.sdf:
        from sdf import SurfaceSDF, sample_surface, visible_primitives
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            sdf = SurfaceSDF(gaussians.center, gaussians.radius, detail=args.sdf_detail)
        if saved is not None and saved["config"].get("sdf", False):
            if saved['config'].get('sdf_detail', False) and not args.sdf_detail:
                raise ValueError('Restoring this SDF requires --sdf-detail')
            state_sdf = saved['sdf']
            if args.sdf_detail and not saved['config'].get('sdf_detail', False):
                # Explicit expansion preserves old keys and initializes only
                # the new zero-output residual; normal reload remains strict.
                state_sdf = {**sdf.state_dict(), **state_sdf}
            sdf.load_state_dict(state_sdf)
            sdf_steps = saved["sdf_steps"]
        if args.freeze_sdf:
            sdf.requires_grad_(False)
        else:
            sdf_optimizer = torch.optim.Adam(sdf.parameters(), lr=args.sdf_lr)
        sdf_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
        primitive_generator = torch.Generator(device="cuda").manual_seed(args.seed + 2)
    shading_field = sdf if args.sdf_shading else None
    volume = None
    volume_steps = 0
    if args.sdf_volume_weight:
        from sdf_volume import SDFRadiance, select_rays, render_rays, ray_geometry_losses
        from sdf import silhouette_rays
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            volume = SDFRadiance('cuda', detail=args.sdf_volume_detail, hint_encoding=args.sdf_volume_hint_encoding)
        if saved is not None and saved['config'].get('sdf_volume_weight', 0):
            if saved['config'].get('sdf_volume_detail', False) and not args.sdf_volume_detail:
                raise ValueError('Restoring this radiance head requires --sdf-volume-detail')
            if saved['config'].get('sdf_volume_hint_encoding', False) and not args.sdf_volume_hint_encoding:
                raise ValueError('Restoring this radiance head requires --sdf-volume-hint-encoding')
            state_volume = saved['sdf_volume']
            if ((args.sdf_volume_detail and not saved['config'].get('sdf_volume_detail', False)) or
                (args.sdf_volume_hint_encoding and not saved['config'].get('sdf_volume_hint_encoding', False))):
                state_volume = {**volume.state_dict(), **state_volume}
            volume.load_state_dict(state_volume)
            volume_steps = saved['sdf_volume_steps']
        if args.sdf_volume_fixed_sharpness is not None:
            with torch.no_grad():
                volume.log_sharpness.fill_(math.log(args.sdf_volume_fixed_sharpness))
            volume.log_sharpness.requires_grad_(False)
        volume_optimizer = torch.optim.Adam(volume.parameters(), lr=.001)
        volume_generator = torch.Generator(device='cuda').manual_seed(args.seed+3)
    normal_field = None
    if args.normal_field:
        from materials.normal_field import NormalResidualField
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            normal_field = NormalResidualField(gaussians.center, gaussians.radius)
        if saved is not None and saved['config'].get('normal_field', False):
            normal_field.load_state_dict(saved['normal_field'])
        normal_optimizer = torch.optim.Adam(normal_field.parameters(), lr=.001)
    camera_offsets = None
    if args.optimize_cameras:
        from cameras import build_camera_offsets

        if args.init_checkpoint and saved["camera_offsets"] is not None:
            saved_mode = saved["config"].get("camera_mode", "anchor")
            if (args.sdf_volume_only or args.radiance_residual) and saved_mode != args.camera_mode:
                args.camera_mode = config["camera_mode"] = saved_mode
            if saved_mode != args.camera_mode:
                raise ValueError("Resuming saved camera corrections requires their --camera-mode")
        camera_offsets = build_camera_offsets(args.camera_mode, len(fit_indices), gaussians.radius).cuda()
        if args.init_checkpoint and saved["camera_offsets"] is not None:
            assert saved["fit_indices"] == fit_indices
            camera_offsets.load_state_dict(saved["camera_offsets"])
        camera_optimizer = camera_offsets.optimizer(args.camera_lr)
        camera_indices = {index: local for local, index in enumerate(fit_indices)}
        if args.camera_gauge == 'translation':
            camera_offsets.set_translation_gauge(torch.stack([samples[i]["viewmat"] for i in fit_indices]),
                                                 torch.stack([samples[i]["K"] for i in fit_indices]),
                                                 gaussians.center)
    light_offsets = None
    if args.optimize_lights:
        from cameras import TrainLightOffsets

        light_offsets = TrainLightOffsets(len(fit_indices), gaussians.radius).cuda()
        if args.init_checkpoint and saved.get("light_offsets") is not None:
            assert saved["fit_indices"] == fit_indices
            light_offsets.load_state_dict(saved["light_offsets"])
        light_optimizer = light_offsets.optimizer(args.light_lr)
        light_indices = {index: local for local, index in enumerate(fit_indices)}
    scene_gauge_shift = torch.zeros(3, device="cuda")
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
        # gsplat 1.5.3 attaches absolute 2DGS gradients to means2d, while
        # signed densification gradients live on gradient_2dgs.
        key_for_gradient="gradient_2dgs" if config["geometry"] == "2dgs" and not args.absgrad else "means2d",
    )
    if not args.freeze_geometry:
        strategy.check_sanity(gaussians.params, optimizers)
    state = strategy.initialize_state(density_scale)
    excess = len(gaussians.params["means"]) - args.max_points
    if excess > 0:
        if args.sdf_volume_only or args.radiance_residual or (args.init_geometry_format == 'gggs' and args.freeze_geometry):
            raise ValueError('A fixed Gaussian teacher requires max-points >= checkpoint point count')
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
    volume_peak_rays_total = 0
    volume_context_rays_total = 0
    volume_rays_total = 0
    residual_peak_rays_total = 0
    residual_context_rays_total = 0
    residual_rays_total = 0
    if args.residual_paired_context:
        residual_pair_pools = {}
        residual_pair_eligibility = {}
        residual_pair_coverage = {name: {} for name in ('1_to_4', '5_to_16', 'gt16')}
        residual_pair_totals = {
            key: 0 for key in ('nominal_pairs', 'sampled_pairs', 'supported_pairs', 'excluded_pairs',
                              'fallback_steps', 'draws', 'pair_peak_draws', 'pair_ring_draws',
                              'foreground_draws', 'valid_draws')}
        residual_pair_totals['component_buckets'] = {
            name: {'anchor_draws': 0, 'sampled_components': 0} for name in residual_pair_coverage}

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
                "light_offsets": None if light_offsets is None else light_offsets.state_dict(),
                "scene_gauge_shift": scene_gauge_shift.tolist(),
                **({}),
                **({"sdf": sdf.state_dict(), "sdf_steps": sdf_steps} if sdf is not None else {}),
                **({'sdf_volume':volume.state_dict(), 'sdf_volume_steps':volume_steps} if volume is not None else {}),
                **({'normal_field':normal_field.state_dict()} if normal_field is not None else {}),
                **({'radiance_residual':radiance_residual.state_dict(), 'residual_steps':residual_steps,
                    'residual_sample_indices':residual_sample_indices, 'residual_frame_counts':residual_frame_counts}
                   if radiance_residual is not None else {}),
                **({'residual_pair_totals': residual_pair_totals,
                    'residual_pair_eligibility': residual_pair_eligibility,
                    'residual_pair_coverage': residual_pair_coverage}
                   if args.residual_paired_context else {}),
            },
            path,
        )
        if args.residual_paired_context:
            (output / 'residual_pair_audit.json').write_text(json.dumps({
                'stage_steps': step, 'residual_pair_totals': residual_pair_totals,
                'residual_pair_eligibility': residual_pair_eligibility,
                'residual_pair_coverage': residual_pair_coverage}, indent=2) + '\n')

    for step in range(1, args.steps + 1):
        geometry_only = set_training_stage(gaussians, transport, optimizers, step, args.geometry_warmup_steps)
        if args.sdf_volume_only or args.radiance_residual:
            gaussians.requires_grad_(False)
            transport.requires_grad_(False)
            if normal_field is not None:
                normal_field.requires_grad_(False)
            if camera_offsets is not None:
                camera_offsets.requires_grad_(False)
        if args.geometry_warmup_steps and step == args.geometry_warmup_steps + 1:
            event = {"event": "relighting_start", "step": step, "warmup_rgb_reset": True}
            history.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
        sample_index = (random.choice(residual_sample_indices))
        sample = samples[sample_index]
        camera_active = not geometry_only and camera_offsets is not None and step >= args.camera_start
        if camera_active:
            camera_optimizer.zero_grad(set_to_none=True)
            if args.camera_lr_final is not None:
                progress = (step - args.camera_start) / max(1, args.steps - args.camera_start)
                camera_optimizer.param_groups[0]["lr"] = (
                    args.camera_lr * (args.camera_lr_final / args.camera_lr) ** progress)
            sample = camera_offsets.correct(sample, camera_indices[sample_index])
        light_active = not geometry_only and light_offsets is not None and step >= args.light_start
        if light_active:
            light_optimizer.zero_grad(set_to_none=True)
            progress = (step - args.light_start) / max(1, args.steps - args.light_start)
            light_optimizer.param_groups[0]["lr"] = args.light_lr * (args.light_lr_final / args.light_lr) ** progress
            sample = light_offsets.correct(sample, light_indices[sample_index])
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if network_optimizer is not None:
            network_optimizer.zero_grad(set_to_none=True)
        if args.optimize_light_scale and not geometry_only:
            light_scale_optimizer.zero_grad(set_to_none=True)
            transport.light_scale = log_light_scale.exp()
        if normal_field is not None:
            normal_optimizer.zero_grad(set_to_none=True)
        scheduled_step = min(step, decay_steps)
        relight_step = max(0, scheduled_step - args.geometry_warmup_steps)
        decay = 0.1 ** (relight_step / max(1, decay_steps - args.geometry_warmup_steps))
        optimizers["means"].param_groups[0]["lr"] = (
            1.6e-4 * position_scale * args.position_decay ** (scheduled_step / decay_steps)
        )
        network_lr = 0.001 * (0.2 + 0.8 * decay)
        if network_optimizer is not None:
            network_optimizer.param_groups[0]["lr"] = network_lr
        if normal_field is not None:
            normal_optimizer.param_groups[0]['lr'] = network_lr
        residual_indices = None
        if radiance_residual is not None:
            residual_optimizer.zero_grad(set_to_none=True)
            residual_optimizer.param_groups[0]['lr'] = network_lr
            rays = silhouette_rays(sample, radiance_residual)
            if args.residual_paired_context:
                if sample_index not in residual_pair_pools:
                    # Corrected cameras and source geometry stay fixed in this stage;
                    # cache GT candidates only, not receivers or render outputs.
                    pool = build_residual_pair_pool(rays, peak_masks[sample_index])
                    residual_pair_pools[sample_index] = pool
                    residual_pair_eligibility[sample_index] = {
                        **pool['stats'], 'valid_pixels': len(rays['valid']),
                        'foreground_pixels': int((rays['target'][rays['valid']] > .9).sum())}
                    history.write(json.dumps({'event': 'residual_pair_eligibility',
                        'frame_index': sample_index, **residual_pair_eligibility[sample_index]}) + '\n')
                residual_indices, residual_pairs = select_residual_pairs(
                    rays, args.residual_rays, residual_generator, residual_pair_pools[sample_index])
            else:
                residual_indices = select_rays(rays, args.residual_rays, residual_generator,
                    peak_masks.get(sample_index), args.residual_peak_fraction,
                    peak_context_masks.get(sample_index), args.residual_context_fraction)
            residual_peak_rays_total += int(peak_masks[sample_index].reshape(-1)[residual_indices].sum())
            if args.residual_context_fraction:
                residual_context_rays_total += int(peak_context_masks[sample_index].reshape(-1)[residual_indices].sum())
            residual_rays_total += len(residual_indices)
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
            geometry_only=geometry_only,
            surface_field=shading_field,
            normal_field=normal_field,
            radiance_residual=radiance_residual,
            residual_indices=residual_indices,
        )
        target = target_image(sample, args.background, args.display_gamma)
        if radiance_residual is not None:
            l1 = (predicted.reshape(-1, 3)[residual_indices] - target.reshape(-1, 3)[residual_indices]).abs().mean()
            loss_terms = {'residual_rgb': l1}
            if args.residual_paired_context:
                pair_loss, pair_support = residual_pair_loss(predicted, target, alpha, residual_pairs, args.residual_rays // 4)
                loss_terms['residual_pair'] = args.residual_pair_weight * pair_loss
                sampled_pairs = len(residual_pairs)
                supported_pairs = int(pair_support.sum())
                remaining_draws = len(residual_indices) - 2 * sampled_pairs
                foreground_draws = remaining_draws // 2 if residual_pair_eligibility[sample_index]['foreground_pixels'] else 0
                pair_counts = {'nominal_pairs': args.residual_rays // 4, 'sampled_pairs': sampled_pairs, 'supported_pairs': supported_pairs, 'excluded_pairs': sampled_pairs - supported_pairs, 'fallback_steps': int(sampled_pairs == 0), 'draws': len(residual_indices), 'pair_peak_draws': sampled_pairs, 'pair_ring_draws': sampled_pairs, 'foreground_draws': foreground_draws, 'valid_draws': remaining_draws - foreground_draws}
                for key, value in pair_counts.items():
                    residual_pair_totals[key] += value
                pair_sample_stats = residual_pair_sample_stats(residual_pair_pools[sample_index], residual_pairs)
                for name, bucket in pair_sample_stats['component_buckets'].items():
                    covered_components = residual_pair_coverage[name].setdefault(sample_index, [])
                    previous_count = len(covered_components)
                    covered_components[:] = sorted(set(covered_components).union(bucket['component_ids']))
                    totals = residual_pair_totals['component_buckets'][name]
                    totals['anchor_draws'] += bucket['anchor_draws']
                    totals['sampled_components'] += len(covered_components) - previous_count
                residual_pair_stats = {**pair_counts, **pair_sample_stats, 'excluded_fraction': (sampled_pairs - supported_pairs) / sampled_pairs if sampled_pairs else 0.0, 'pair_loss': pair_loss.item(), 'weighted_pair_loss': loss_terms['residual_pair'].item()}
        else:
            l1 = (predicted - target).abs().mean()
            loss_terms = {'l1': 0.8 * l1, 'ssim': 0.2 * (1 - ssim(predicted.clamp(0, 1), target.clamp(0, 1))), 'mask': l1.new_zeros(()), 'feature_reg': l1.new_zeros(()) if geometry_only else 1e-05 * gaussians.params['features'].square().mean(), 'camera_reg': l1.new_zeros(())}
        if sample['alpha'] is not None and radiance_residual is None:
            loss_terms['mask'] = args.mask_weight * (alpha - sample['alpha']).abs().mean()
        if normal_field is not None:
            loss_terms['normal_field_reg'] = 1e-4*info['normal_residual'].square().mean()
        if args.highlight_weight:
            mask=peak_masks[sample_index]
            count=mask.sum().clamp_min(1)
            p_neutral,t_neutral=predicted.amin(-1),target.amin(-1)
            def contrast(values):
                return values-torch.nn.functional.avg_pool2d(values[None,None],11,1,5)[0,0]
            error=(predicted-target).abs().mean(-1)+.5*(contrast(p_neutral)-contrast(t_neutral)).abs()
            loss_terms['highlight']=args.highlight_weight*(error*mask).sum()/count
        if camera_active and radiance_residual is None:
            loss_terms["camera_reg"] = camera_offsets.regularization(camera_indices[sample_index])
        if light_active:
            loss_terms["light_reg"] = light_offsets.regularization(light_indices[sample_index])
        if config['geometry'] == '2dgs' and step >= args.surface_start and (not (args.sdf_volume_only or args.radiance_residual)):
            from surface import surface_losses
            targets = priors.frame(sample_index, alpha.device) if priors is not None else {}
            loss_terms.update(surface_losses(info, alpha, sample, targets, gaussians.radius, args.normal_weight, args.depth_weight, args.surface_consistency_weight, args.distortion_weight))
        sdf_active = args.sdf_shading and not args.freeze_sdf
        volume_active = volume is not None and step >= args.sdf_start
        volume_warmup = volume_steps < args.sdf_volume_warmup
        primitive_candidates = 0
        if sdf is not None and not args.freeze_sdf:
            sdf_optimizer.zero_grad(set_to_none=True)
        if sdf is not None and step >= args.sdf_start and (volume is None or (volume_warmup and not args.sdf_volume_only)):
            surface_samples = sample_surface(info, alpha, sample, sdf, args.sdf_samples, sdf_generator)
            if surface_samples is not None:
                points, normals, toward_camera = surface_samples
                if not args.freeze_sdf:
                    loss_terms.update(sdf.fit_losses(points, normals, toward_camera))
                ramp = min(1., max(0., (sdf_steps-args.sdf_warmup_steps) / max(1, args.sdf_warmup_steps)))
                if ramp > 0 and volume is None:
                    loss_terms.update(sdf.geometry_losses(points, normals, ramp*args.sdf_weight,
                                                          ramp*args.sdf_normal_weight))
                    if args.sdf_primitive_weight:
                        selected = visible_primitives(gaussians.params['means'],
                            gaussians.params['opacities'].sigmoid(), info, alpha, sample, gaussians.radius)
                        primitive_candidates = len(selected)
                        if len(selected):
                            selected = selected[torch.randint(len(selected), (args.sdf_samples,),
                                device=selected.device, generator=primitive_generator)]
                            loss_terms['sdf_primitive'] = ramp*args.sdf_primitive_weight*sdf.primitive_loss(
                                gaussians.params['means'][selected])
                sdf_active = not args.freeze_sdf
        if volume_active:
            volume_optimizer.zero_grad(set_to_none=True)
            volume_optimizer.param_groups[0]['lr'] = network_lr
            rays = silhouette_rays(sample, sdf)
            ray_indices = select_rays(rays, args.sdf_volume_rays, volume_generator,
                                     peak_masks.get(sample_index), args.sdf_volume_peak_fraction,
                                     peak_context_masks.get(sample_index), args.sdf_volume_peak_context_fraction)
            volume_peak_rays_total += int(peak_masks[sample_index].reshape(-1)[ray_indices].sum())
            if args.sdf_volume_peak_context_fraction:
                volume_context_rays_total += int(peak_context_masks[sample_index].reshape(-1)[ray_indices].sum())
            volume_rays_total += len(ray_indices)
            volume_result = render_rays(sdf, volume, sample, rays, ray_indices,
                transport.light_scale.detach(), samples=args.sdf_volume_samples,
                background=args.background, detach_field=volume_warmup or args.freeze_sdf)
            volume_pred = observation_image(volume_result['linear'], args.display_gamma,
                alpha=None if sample['is_hdr'] else volume_result['alpha'], background=args.background)
            volume_target = target.reshape(-1, 3)[ray_indices]
            loss_terms['sdf_volume_rgb'] = args.sdf_volume_weight*(volume_pred-volume_target).abs().mean()
            loss_terms['sdf_volume_mask'] = args.sdf_volume_weight*.1*(
                volume_result['alpha'][:, 0]-rays['target'][ray_indices]).abs().mean()
            if not volume_warmup:
                if not args.freeze_sdf:
                    loss_terms['sdf_volume_eikonal'] = .1*volume_result['eikonal']
                    sdf_active = True
                ramp = min(1., (volume_steps-args.sdf_volume_warmup+1)/max(1, args.sdf_volume_warmup))
                loss_terms.update(ray_geometry_losses(volume_result, ray_indices, info, alpha,
                    sample, gaussians.radius, ramp*args.sdf_weight, ramp*args.sdf_normal_weight))
            if args.sdf_volume_only:
                l1 = (volume_pred-volume_target).abs().mean()
                loss_terms = {key:value for key,value in loss_terms.items()
                              if key.startswith('sdf_') and not key.endswith('_to_gs')}
        loss = sum(loss_terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at {step}")
        if not args.freeze_geometry:
            strategy.step_pre_backward(gaussians.params, optimizers, state, step, info)
        loss.backward()
        if sdf_active:
            sdf_optimizer.step()
            sdf_steps += 1
        if volume_active:
            volume_optimizer.step()
            volume_steps += 1
        if radiance_residual is not None:
            residual_optimizer.step()
            residual_steps += 1
            residual_frame_counts[sample_index] += 1
        else:
            for optimizer in optimizers.values():
                optimizer.step()
        if not geometry_only and radiance_residual is None and network_optimizer is not None:
            network_optimizer.step()
        if normal_field is not None:
            normal_optimizer.step()
        if camera_active and radiance_residual is None:
            camera_optimizer.step()
            if args.camera_gauge == 'translation':
                shift = camera_offsets.remove_translation_gauge()
                with torch.no_grad():
                    gaussians.params["means"] += shift
                scene_gauge_shift += shift
        if light_active:
            light_optimizer.step()
        if args.optimize_light_scale and not geometry_only:
            light_scale_optimizer.step()
            transport.light_scale = log_light_scale.detach().exp()
        if not args.freeze_geometry:
            event = strategy.step_post_backward(gaussians.params, optimizers, state, step, info, packed=False)
            if event is not None:
                history.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
        if not (args.sdf_volume_only or args.radiance_residual or (args.init_geometry_format == 'gggs' and args.freeze_geometry)):
            gaussians.project_geometry(args.opacity_cap, args.min_scale, args.max_scale)
        if step % 100 == 0 or step == 1 or step == args.steps:
            row = {
                "step": step,
                "stage": ("gaussian_residual" if radiance_residual is not None else
                          "sdf_volume" if args.sdf_volume_only else "geometry" if geometry_only else "relighting"),
                "frame_index": sample_index,
                "loss": loss.item(),
                "l1": l1.item(),
                "loss_terms": {name: value.item() for name, value in loss_terms.items()},
                "points": len(gaussians.params["means"]),
                "seconds": round(time.monotonic() - start, 2),
                "memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
                **({}),
                **({'light_scale':float(transport.light_scale),
                    'relative_light_gain':float(initial_light_scale/transport.light_scale)}
                   if args.optimize_light_scale else {}),
                **({'camera_rotation_rms': camera_offsets.rms()[0],
                    'camera_lr': camera_optimizer.param_groups[0]['lr']}
                   if camera_offsets is not None else {}),
                **({'scene_gauge_shift': scene_gauge_shift.tolist()} if args.camera_gauge != 'none' else {}),
                **({'light_offset_rms': light_offsets.rms(),
                    'light_lr': light_optimizer.param_groups[0]['lr']}
                   if light_offsets is not None else {}),
                **({"sdf_steps": sdf_steps} if sdf is not None else {}),
                **({'sdf_volume_steps':volume_steps, 'sdf_volume_sharpness':float(volume.log_sharpness.exp()),
                    'sdf_volume_peak_rays_total':volume_peak_rays_total, 'sdf_volume_rays_total':volume_rays_total,
                    'sdf_volume_context_rays_total':volume_context_rays_total}
                   if volume is not None else {}),
                **({"sdf_primitive_candidates": primitive_candidates} if args.sdf_primitive_weight else {}),
                **({'residual_steps': residual_steps, 'residual_rays_total': residual_rays_total,
                    'residual_frame_counts': residual_frame_counts,
                    'residual_peak_rays_total': residual_peak_rays_total,
                    'residual_context_rays_total': residual_context_rays_total,
                    'residual_lr': residual_optimizer.param_groups[0]['lr'],
                    'residual_stats': {key: value.item() if isinstance(value, torch.Tensor) else value
                                       for key, value in info['residual_stats'].items()}}
                   if radiance_residual is not None else {}),
                **({'residual_optimization_stats': residual_optimization_stats(radiance_residual)}
                   if args.residual_interaction != 'none' else {}),
                **({'residual_pair_stats': residual_pair_stats,
                    'residual_pair_totals': residual_pair_totals}
                   if args.residual_paired_context else {}),
            }
            history.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step in args.save_steps:
            checkpoint(step, output / f"step_{step:06d}.pt")
        if step == args.steps or (args.validate_every > 0 and step % args.validate_every == 0):
            if not validation or args.validate_every == 0:
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
                geometry_only=geometry_only,
                surface_field=shading_field,
                normal_field=normal_field,
                radiance_residual=radiance_residual,
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
                    geometry_only=geometry_only,
                    surface_field=shading_field,
                    normal_field=normal_field,
                    radiance_residual=radiance_residual,
                )
                save_pair(
                    output / "validation_pair.png",
                    preview,
                    target_image(val, args.background, args.display_gamma),
                )
    history.close()
    from plot_loss import plot_loss

    loss_plot = plot_loss(output / "history.jsonl")
    print(
        json.dumps(
            {
                "event": "complete",
                "loss_plot": str(loss_plot),
                "best_validation_psnr": best if math.isfinite(best) else None,
                "seconds": time.monotonic() - start,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
