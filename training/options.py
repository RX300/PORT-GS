"""Training command line: parser construction and option-combination checks.

Checks here depend only on the command line. Checks that need the resolved
method geometry or a source checkpoint live in training.source.
"""

import argparse
import math

from methods import add_method_arguments


def build_parser(argv=None):
    """Return the train.py parser; ``argv`` selects the method's option group."""
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
    add_method_arguments(p, argv)
    p.add_argument("--port-start", type=int, default=5000)
    p.add_argument("--shadow-start", type=int, default=5000)
    p.add_argument("--refine-stop", type=int, default=25000)
    p.add_argument("--opacity-reset-every", type=int, default=3000,
                   help="Opacity reset interval before --refine-stop; 0 disables resets. Large-Gaussian pruning "
                        "keeps its fixed 3000-step start")
    p.add_argument("--validate-every", type=int, default=0,
                   help="Validation/checkpoint interval; 0 disables all validation, including the final one, "
                        "and writes only last.pt")
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
    return p


def validate_arguments(parser, args):
    """Reject inconsistent option combinations and apply frozen-stage overrides."""
    if args.val_limit <= 0:
        parser.error('--val-limit must be positive')
    if args.opacity_reset_every < 0:
        parser.error('--opacity-reset-every must be nonnegative')
    if args.init_geometry_format == 'gggs' and (not args.init_geometry
            or args.representation not in ('directional_port_v1','neural_material')
            or (args.optimize_cameras and args.freeze_geometry) or args.sdf):
        parser.error('GGGS import requires directional/neural-material relighting and --init-geometry; '
                'camera fitting additionally requires trainable geometry')
    if args.camera_lr_final is not None and (not args.optimize_cameras or args.camera_lr_final <= 0):
        parser.error('--camera-lr-final requires --optimize-cameras and a positive value')
    if args.camera_gauge != 'none' and (not args.optimize_cameras or args.camera_mode != 'rotation'):
        parser.error('--camera-gauge requires --optimize-cameras with --camera-mode rotation')
    if args.optimize_lights and (args.light_lr <= 0 or args.light_lr_final <= 0 or args.light_start < 1
                                 or args.radiance_residual or args.sdf_volume_only):
        parser.error('Light offsets need positive rates/start and a trainable relighting stage')
    if args.init_geometry_format == 'gggs' and args.representation == 'neural_material' and (
            args.normal_weight or args.depth_weight or args.surface_consistency_weight or args.distortion_weight):
        parser.error('Material on 3D GGGS requires explicit zero 2D surface/prior weights; no implicit surfel losses')
    if args.representation == "neural_material" and args.material_model == 'neural' and not args.init_checkpoint and not args.material_decoder:
        parser.error("fresh neural_material training requires --material-decoder")
    if args.reset_material and (args.representation != 'neural_material' or not args.init_checkpoint
                               or (args.material_model == 'neural' and not args.material_decoder)):
        parser.error('--reset-material requires neural_material and --init-checkpoint; neural also requires --material-decoder')
    if args.radiance_residual:
        if args.representation != 'neural_material' or not args.init_checkpoint:
            parser.error('--radiance-residual requires neural_material and --init-checkpoint')
        if (args.sdf or args.sdf_shading or args.sdf_volume_weight or args.sdf_volume_only or
            args.freeze_sdf or args.normal_field or args.geometry_warmup_steps or
            args.reset_material or args.optimize_light_scale or args.highlight_weight):
            parser.error('Radiance-residual-only training excludes SDF/volume/normal-field, geometry warmup, '
                    'material reset, light-scale optimization and highlight losses')
        args.freeze_geometry = True
        args.camera_lr = 0.
        args.camera_start = 1
    if (args.residual_rays <= 0 or not 0 <= args.residual_peak_fraction < 1 or
        not 0 <= args.residual_context_fraction < 1 or
        args.residual_peak_fraction+args.residual_context_fraction >= 1 or
        (args.residual_context_fraction and not args.residual_peak_fraction)):
        parser.error('Residual rays must be positive; peak/context quotas must be nonnegative, sum <1, '
                'and context requires a positive peak quota')
    if (args.residual_peak_fraction or args.residual_context_fraction) and not args.radiance_residual:
        parser.error('Residual sampling quotas require --radiance-residual')
    if not math.isfinite(args.residual_pair_weight) or args.residual_pair_weight < 0:
        parser.error('--residual-pair-weight must be finite and nonnegative')
    if args.residual_pair_weight and not args.residual_paired_context:
        parser.error('Positive --residual-pair-weight requires --residual-paired-context')
    if args.residual_paired_context:
        if not args.radiance_residual:
            parser.error('--residual-paired-context requires --radiance-residual')
        if args.residual_peak_fraction != .25 or args.residual_context_fraction != .25:
            parser.error('--residual-paired-context requires peak/context fractions of .25 each')
        if args.residual_rays % 4:
            parser.error('--residual-paired-context requires --residual-rays divisible by 4')
    if args.residual_frames is not None and not args.radiance_residual:
        parser.error('--residual-frames requires --radiance-residual')
    if args.residual_interaction != 'none' and not args.radiance_residual:
        parser.error('--residual-interaction requires --radiance-residual')
    if args.residual_angular_bank != 'none' and (
        not args.radiance_residual or args.residual_interaction != 'multiply' or args.residual_normal != 'geometry'
    ):
        parser.error('--residual-angular-bank requires --radiance-residual, --residual-interaction multiply '
                'and --residual-normal geometry')
    if args.sdf_volume_weight:
        if not args.sdf or args.sdf_shading or args.sdf_primitive_weight or args.geometry_warmup_steps:
            parser.error('SDF volume supervision requires --sdf without SDF shading, primitive loss or RGB geometry warmup')
        if args.sdf_volume_weight < 0 or args.sdf_volume_rays <= 0 or args.sdf_volume_samples < 4 or args.sdf_volume_warmup < 0:
            parser.error('SDF volume weight/rays must be positive, samples >=4 and warmup >=0')
    if args.sdf_volume_only:
        if not args.sdf_volume_weight or not args.init_checkpoint or args.sdf_start != 1 or args.optimize_light_scale:
            parser.error('SDF-volume-only requires a volume weight, checkpoint, sdf-start=1 and fixed light scale')
        if args.reset_material:
            parser.error('SDF-volume-only preserves the Gaussian teacher material')
        args.freeze_geometry = True
        args.camera_lr = 0.
        args.camera_start = 1
    if not 0 <= args.sdf_volume_peak_fraction < 1 or (args.sdf_volume_peak_fraction and not args.sdf_volume_weight):
        parser.error('SDF peak-ray fraction must be in [0,1) and requires SDF volume supervision')
    if (args.sdf_volume_peak_context_fraction < 0 or
        args.sdf_volume_peak_fraction+args.sdf_volume_peak_context_fraction >= 1 or
        (args.sdf_volume_peak_context_fraction and not args.sdf_volume_peak_fraction)):
        parser.error('SDF context quota requires a positive peak quota, and their sum must be <1')
    if args.sdf_volume_hint_encoding and not args.sdf_volume_weight:
        parser.error('SDF hint encoding requires volume supervision')
    if args.sdf_volume_fixed_sharpness is not None and (
        not math.isfinite(args.sdf_volume_fixed_sharpness) or args.sdf_volume_fixed_sharpness <= 0 or
        not args.sdf_volume_weight > 0):
        parser.error('Fixed SDF volume sharpness must be positive and finite, and requires positive volume weight')
    if args.sdf_detail and not args.sdf:
        parser.error('--sdf-detail requires --sdf')
    if args.sdf_volume_detail and not args.sdf_volume_weight:
        parser.error('--sdf-volume-detail requires --sdf-volume-weight')
    return args


def parse_arguments(argv=None):
    parser = build_parser(argv)
    return validate_arguments(parser, parser.parse_args(argv))
