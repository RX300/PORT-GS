# Training-camera refinement and geometric audit

`cameras.py` supplies bounded per-fit-view SE(3) corrections. The first fit camera
anchors the coordinate frame. Rotation components are limited to 0.03 radians;
translation components to 0.03 object radius. The module saves raw parameters and
its active-camera mask in the scene checkpoint.

`train.py --optimize-cameras --camera-start 1` optimizes those corrections together
with scene parameters. Fit rendering restores the saved corrections. Test and
validation rendering use the original input cameras; test GT does not calibrate
poses. A corrected-fit PSNR gain is therefore not itself a held-out quality gain.

`diagnose_image_errors.py <checkpoint> --output <fresh path> --camera-matches <bank.npz>`
runs a CPU-only geometric audit. NPZ keys are `frame1_frame2`, and each value has
shape N×2×2 for fixed pixel correspondences at the checkpoint's image resolution.
All referenced frames must belong to the saved fit split. Correspondences are
selected before the audit and are not rematched to suit the refined cameras.

The audit constructs fundamental matrices from original and corrected K/poses and
reports per-pair Sampson distances. The main summary requires at least 20 matches;
a platform-oriented image stratum requires at least eight. That stratum uses the
lower 35% of foreground image extent and local mean R < 1.5G; it is a heuristic,
not semantic ground truth. Repeated/planar textures and incomplete view coverage
limit interpretation. Pose RMS values are per-component, in radians/world units.

The initial Pixiu audit exactly reproduces the independent baseline measurements
(1.704978 px across 14 pairs; 1.775818 px across 18 platform pairs). Real GPU checks
cover pose updates, the fixed gauge, alpha-to-pose gradients, checkpoint reload,
and CLI corrected-fit rendering in `runs/intersection_camera_smoke`.

## 2026-09-30: calibration modes, translation gauge and light offsets

The [calibration audit](../../experiments/calibration_audit_20260930.md) found 4-6 px
per-frame error in the official Real_NRHints training poses. `--camera-mode` selects the
fit-camera correction:

- `anchor` (default, all earlier checkpoints): the SE(3) module above. Its first fit camera
  is fixed, so the whole reconstruction inherits that single camera's pose error.
- `rotation` (`TrainCameraRotations`): every fit camera gets a camera-frame rotation
  `viewmat' = [dR|0] @ viewmat`, which keeps each calibrated optical center exactly fixed.
  There is no anchor camera. Rotations are stored in radians (`rotation.weight`, one
  `nn.Embedding` row per fit view) and receive sparse gradients; `SparseAdam` updates only
  the sampled view, so each view's step size does not depend on how rarely it is sampled.
  Regularizer: `1e-4 * |dR in degrees|^2` for the sampled view (tie-breaker only).
  `--camera-lr-final` decays the rate exponentially from `--camera-lr` at `--camera-start`.

With fixed centers and free per-view rotations, translating the whole scene by `d` can be
compensated by re-aiming every camera (`C_i d`, the minimum-norm rotation that keeps the
object center's image fixed). These three directions are a gauge of the joint problem, so
the image loss does not pin them and gradient noise drifts the reconstruction frame (the
same mechanism moved GS3/SSD-GS by 5-9 px on the official test views).
`--camera-gauge translation` projects the corrections off this gauge after every camera step,
`w <- w - C d*` with `d* = argmin sum_i |w_i - C_i d|^2`, and translates all Gaussian means by
`-d*`, which leaves every fit view's object-center image unchanged to first order. Shared
rotations that are not explainable by a translation (for example a common yaw error of all
cameras around the object) remain free, because they are observable. The accumulated scene
shift is logged (`scene_gauge_shift`) and saved.

`--optimize-lights` adds `TrainLightOffsets`: one world-space light-position offset per fit
frame (object-radius units, sparse rows, `1e-3 |offset|^2` regularizer), active from
`--light-start` with an exponentially decayed rate. Fit evaluation restores saved camera
and light corrections; validation and official test always use the original calibration.

`load_camera_offsets(checkpoint, radius)` restores either mode (missing `camera_mode`
means `anchor`). Unit tests: `test_methods.CameraCalibrationTests` (fixed centers, sparse
single-row updates, gauge removal preserving center images and idempotence, light offsets,
sub-pixel shift recovery for the secondary aligned metric).

## 2026-09-30: training package and frozen stages

Training uses `training/pose.py`: `CameraFit` owns the offsets, optimizer, rate decay and
gauge projection; `LightFit` and `LightScaleFit` own the light corrections. In frozen
stages (`--radiance-residual`, `--sdf-volume-only`) saved corrections are applied without an
optimizer; SparseAdam rejects the former zero rate, which made rotation-mode sources fail.

The translation gauge moves only the Gaussian means. Lights stay at calibrated world
positions and `gaussians.center` (the transport's normalization origin) is not moved, so the
projection is a first-order camera-geometry symmetry, not an exact relighting symmetry.
R2b accumulated 0.018 (Cat, radius 0.50) and 0.040 (Pixiu, radius 1.41) world units.
`scene_gauge_shift` is now inherited from `--init-checkpoint` instead of restarting at zero.
