> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

# Author 2DGS reconstruction stage

This is a reconstruction baseline, not another relighting method. The user requested geometry first on 2026-09-26. The author GaussianModel, optimizer groups, SH3 schedule, densification/pruning and CUDA rasterizer are loaded from an unmodified project-local clone:

- https://github.com/hbb1/2d-gaussian-splatting
- root revision f3e3b9fa67bbd1c75e05167ff37391d8dab2a678
- diff-surfel-rasterization e0ed0207b3e0669960cfad70852200a4a5847f61
- simple-knn f155ec04131cb579f53443a06879d37115f4612f

`native_reconstruction.py` owns the NRHints camera/data adapter and the separate train/evaluate path. Existing `train.py`, `evaluate.py`, `configs/validation.json`, `launch_validation.sh` remain the entrypoints. `native_2dgs` is registered as a reconstruction choice, outside the relighting METHODS dictionary.

## Faithfulness and explicit adaptations

Uses author SH RGB evaluation before splat composition, degree0→3 every1000 steps, .8L1+.2DSSIM, normal consistency .05 after7000, author default distortion weight0, median depth for bounded scenes, clone/split/prune every100 from500 to15000, opacity cull .05/reset every3000, original optimizer settings and camera-extent position LR. No PORT, BRDF, light inputs, SDF, pretrained geometry, neural material, camera fitting or original points3d.ply.

The dataset is not an author COLMAP scene. Initial positions come from **fit-only** mask occupancy carving (192 grid,95% consensus,2px padding,40000 sampled points), without any distance field. Author kNN scales and random rotations are retained; occupancy-derived normals are not used. Initial RGB is constant. This is a documented initialization adaptation, not a claim that every original data/preprocessing choice is reproduced.

The adapter preserves full K, including Cat's off-center principal point. Native pixel indices correspond to data pixel centers i+.5. Projection offsets are 2cx/W−1,2cy/H−1 with native ndc2pix; depth normals use the same calibrated rays. Source model and CUDA remain unchanged. The original viewer, original dataset loader and Open3D extraction dependencies are not imported.

The existing environment's simple_knn binary required unavailable GLIBC2.34. Both author CUDA extensions were therefore built **in place under this project clone**, not installed into or substituted in the shared Conda environment. PyTorch/CUDA were not upgraded. Licenses remain in the upstream directories and per-run source archives.

## Outputs and handoff

`last.pt` retains the author capture/optimizer plus a canonical geometry-only payload (`means`, 2D `scales`, `quats`, `opacities`, scene center/radius and exact fit/validation indices). Existing `--init-geometry` can use it to initialize a later 2DGS relighting stage with fresh material parameters and the same split. This interface does not assert the geometry has passed review.

`surfels.ply` is the author Gaussian PLY. Evaluations export four predetermined RGB/normal/depth/clay panels, selected-view depth/normal buffers, and oriented surface points. The exported surface points are not a watertight reconstructed mesh. No TSDF or SDF supervision is used.

Silhouette IoU and normal/depth agreement are proxies. Cross-view tests reproject each selected view into its nearest fit camera in both directions, report raw 1%-radius agreement, and separately report depths classified as occluded before reporting remaining errors. These are **self-consistency**, not ground-truth geometric accuracy. The clay view uses fixed world illumination and ignores learned SH colors and captured lamps.

The dataset contains moving lights, whereas native SH models only view dependence. A sharper RGB fit can bake lighting into appearance or geometry. Geometry review must therefore gate the next relighting stage; RGB PSNR alone is insufficient.

## Mask-supervised geometry profile

After the unmasked control produced opaque black sheets, `--native-mask-weight .2` adds L1 between rendered alpha and supplied alpha. `--native-distortion-weight 1000` enables the author's DTU geometry preset. Defaults preserve the unmasked control (both weights0). Author model and CUDA are still unchanged; the additional mask objective is clearly an adaptation. Both objective settings and the frozen source are recorded per run.

## Coordinate normalization

`--native-coordinate-scale camera_extent` optionally trains with `world = scale*local + center`, with scale equal to author camera extent. All geometry, camera translations, kNN scales, spatial LR, and clone/split/prune world thresholds use the same similarity. Rays and SH directions are preserved; native inverse-depth distortion is scale dependent, so this intentionally changes its effective regularization scale and numerical precision. The default `world` setting preserves the preceding controls.

`capture` and its optimizer are in training coordinates recorded in `training_coordinate_transform`; `gaussians` and `surfels.ply` are always in original-world coordinates. `restore_native_model` loads appearance and copies canonical world geometry for inference. It must not be used to resume the training-coordinate optimizer. Existing geometry-only relighting initialization consumes `gaussians` and keeps original lamp/camera coordinates.

## Reading the diagnostic exports

`self_normal_angular_error` is mean `1-cos(theta)` between normalized rendered and depth-derived normals on a3x3 eroded valid foreground, not degrees/radians and not GT-normal error. `valid_surface_fraction` is valid pixels divided by the full image, not foreground recall. The additional review reports foreground recall and background opaque fraction separately.

`surface_points.ply` samples only four predetermined views, using their evaluation alpha masks plus a2*radius crop. It is a foreground-filtered diagnostic cloud, not the full model, reference geometry or a training initializer. Raw `surfels.ply`, raw-alpha normal/depth/clay images and mask error expose background geometry instead of hiding it with this export filter.


## 2026-09-27 topology correction

`native_topology.py` supplies a local subclass selected by `--native-topology corrected` (new default); upstream files and CUDA stay unchanged. `author` remains available for exact historical behavior. Densification preserves screen-radius history on surviving old points. Exact clones inherit their parent's history so cloning cannot evade the size threshold. Displaced split children begin with zero/unobserved radius and acquire their own observations. The author clone/split geometry, opacity pruning and Adam moment handling are retained. Every topology event and cumulative clone/split/screen/world/opacity removal count is logged and included in the final checkpoint.

`--native-split-extent object` rescales only `percent_dense` from.01 to `.01*object_radius/camera_extent`; therefore the actual split sigma threshold is1% of the existing camera-derived object-bound radius. Position LR, world-size culling and global coordinate normalization retain camera extent. This is an explicit optional scale adaptation, tested independently of the topology repair. The CLI depth-ratio default now matches the author's generic0; existing saved configurations explicitly recording1 are unchanged.
