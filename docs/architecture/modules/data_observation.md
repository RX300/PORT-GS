# Data and observation

The [data contract](../../data_contract.md) defines metadata, explicit splits,
camera axes, intrinsic scaling, image values and point-light conventions.
Camera-bound calculations operate on uploaded cameras, intrinsics and image
sizes; decoded training samples remain on the selected GPU.

The second-round renderer first rasterizes base/features/visibility and expected
camera-Z with `RGB+ED`. Attribute channels are alpha-weighted and normalized by
coverage; the final depth channel already represents expected depth. Pixel
centers `(x+0.5,y+0.5)` and `K` reconstruct camera points, transformed to world
coordinates for receiver shading. This uses a mixed-depth approximation on rays
with multiple contributors. Shaded foreground is linearly alpha-composited.
The default material configuration has 36 attribute channels plus expected depth.
`channel_chunk=attributes.shape[-1]+1` renders all 37 channels in one pass.
The dependency's default chunk size 32 creates multiple passes and overwrites
`means2d.absgrad`, omitting part of the attribute-gradient contribution used for
geometry growth. A single pass supplies the complete absolute projected gradient.
The receiver rendering audit passed its defined identity and derivative checks;
its exact derivative scope is recorded in
[receiver rendering audit](../../../runs/research_20260912/receiver_rendering_audit.json).
The accepted full-fit phase retains this observation and camera protocol.

`render_observation` provides one optimization/evaluation/preview transform.
For PNG, the renderer's alpha-composited RGB is converted back to foreground,
encoded with the configured gamma, then composited with predicted alpha and the
chosen background. PNG GT is already encoded; its RGB is composited with GT
alpha. This encoded-foreground composition is selected automatically by input
format. HDR/EXR prediction and GT both receive gamma on their complete composite.
The observation transform is a modeling choice, not a claim of calibrated
physical radiance.

Current-checkpoint initialization inherits fit/validation indices. `--fit-all`
uses the complete official train split. Optional training-camera offsets are
saved and restored during fit/train evaluation only for their fitted frames.
Validation and test use original metadata calibration. Evaluations record
`camera_protocol` and `corrected_fit_frames` so this distinction is visible.

Training validation uses floating-point observations. Explicit final evaluation
clips and quantizes RGB to uint8 for PSNR/SSIM/standard LPIPS, while also recording
unclipped observation MSE. Standard LPIPS receives values in [-1, 1]. Quantized
final evaluation and unquantized training validation are reported separately.
