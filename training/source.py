"""Source-checkpoint loading and checks that need the resolved method geometry."""

import torch


def load_source(path):
    return torch.load(path, map_location="cuda", weights_only=False) if path else None


def inherit_residual_source(args, config, saved):
    """Check residual options against the source; residual stages copy its observation contract."""
    if saved is None:
        return
    if (saved['config'].get('residual_angular_bank', 'none') != 'none' and
        not saved['config'].get('radiance_residual', False)):
        raise ValueError('Radiance residual angular bank requires an enabled source configuration')
    if (saved['config'].get('radiance_residual', False) or 'radiance_residual' in saved) and not args.radiance_residual:
        raise ValueError('This checkpoint contains a radiance residual; explicitly enable --radiance-residual')
    if not args.radiance_residual:
        return
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


def check_geometry_options(args, config):
    """Options whose validity depends on the resolved 3DGS/2DGS geometry."""
    if args.surface_depth == 'intersection' and config['geometry'] != '2dgs':
        raise ValueError('Intersection surface depth requires 2DGS geometry')
    if args.initialization == "hull" and config["geometry"] != "3dgs":
        raise ValueError("Hull initialization requires 3DGS geometry")
    if args.geometry_warmup_steps and (args.freeze_geometry or
            (config["geometry"] != "2dgs" and args.representation != "light_atlas")):
        raise ValueError("Geometry warmup requires trainable 2DGS or light_atlas geometry")
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
        if ((args.normal_weight or args.depth_weight) and not args.surface_priors
                and not (args.sdf_volume_only or args.radiance_residual)):
            raise ValueError("2D Gaussian surface supervision requires --surface-priors; "
                             "set both prior weights to 0 for geometry-only ablation")


def check_source_compatibility(args, config, saved):
    """Method, material and SDF agreement; frozen stages inherit the source schedule."""
    if (args.sdf_shading or args.freeze_sdf) and (saved is None or not saved['config'].get('sdf', False)
                                                 or saved['sdf_steps'] <= args.sdf_warmup_steps):
        raise ValueError("SDF shading/frozen guidance requires a fitted SDF checkpoint beyond warmup")
    if saved is None:
        return
    if saved["config"]["representation"] != args.representation:
        raise ValueError("--init-checkpoint requires the same method; use --init-geometry for another method")
    config['source_checkpoint_step'] = saved['step']
    config['training_steps_before'] = saved['config'].get('training_steps_before', 0) + saved['step']
    if (args.representation == 'light_atlas' and
            saved['config'].get('normal_model', 'covariance') != args.normal_model):
        raise ValueError('Changing LiSA normal parameter meanings requires fresh material via --init-geometry')
    if (args.representation == 'neural_material' and not args.reset_material and
            saved['config'].get('material_model', 'neural') != args.material_model):
        raise ValueError('Changing material model requires --reset-material to initialize its parameter meanings')
    fixed_lisa = args.representation == 'light_atlas' and args.freeze_geometry
    if fixed_lisa:
        if saved['step'] <= saved['config'].get('geometry_warmup_steps', 0):
            raise ValueError('Frozen LiSA appearance requires completed geometry warmup')
        if saved.get('light_offsets') is not None or args.optimize_lights or args.optimize_light_scale:
            raise ValueError('Frozen LiSA appearance requires fixed source light calibration')
        for key in ['camera_mode', 'camera_gauge']:
            setattr(args, key, saved['config'][key])
            config[key] = saved['config'][key]
        args.camera_start = config['camera_start'] = 1
        args.camera_lr = config['camera_lr'] = 0.
        args.camera_lr_final = config['camera_lr_final'] = None
    if args.sdf_volume_only or args.radiance_residual or fixed_lisa:
        args.optimize_cameras = saved['camera_offsets'] is not None
        config['optimize_cameras'] = args.optimize_cameras
        for key in ['port_start', 'shadow_start']:
            value = 1 if saved['step'] >= saved['config'][key] else args.steps+1
            setattr(args, key, value)
            config[key] = value
