# Shadow visibility and optimization

`renderer.visibility_hint` estimates point-light visibility from the current
Gaussians. Depth shadows use an expected-depth map. Deep shadows distribute
source alpha contributions across 64 depth bins at 256px and query the cumulative
transmittance at receivers; lights within the fitted support use cube faces.

The historical September 11 round-two shadow repair propagates gradients through source projection, opacity,
depth-bin weights and receiver coordinates. Every forward chooses a fitted
sampling grid, but focal/near/far and the self-exclusion bias stay fixed for that
backward. gsplat has no K derivatives, so detaching only the lookup focal would
be inconsistent; both raster and lookup use the same fixed focal.

The opacity finite-difference audit on a real Cat fit frame converges at smaller
steps. All tested geometry gradients are finite, and forward RGB/alpha matches
the prior rendering. Thresholded rasterization remains piecewise differentiable.
Cube backward has static review but is not covered by the Cat exterior-light
numerical check. The September 11 experiment evaluated that shadow repair. The accepted
September 12 round-two change concerns pixel-receiver shading; it inherits this
approximate visibility path. Neither experiment establishes physically exact
visibility or removes real-camera calibration uncertainty.
