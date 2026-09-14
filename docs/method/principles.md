# Continuous receivers over conservative Gaussian-source exchange

The accepted second-round representation separates the source integral from the location where
material response is evaluated. Every Gaussian remains a source with geometry,
opacity, base appearance and learned features. The camera rasterizes base,
features, direct-light visibility and expected camera-Z depth. Covered pixels
recover alpha-normalized attributes and a world-space receiver point from depth,
pixel center and camera intrinsics.

## Source integral and receiver query

For each RGB channel, Gaussian source `j` supplies visible inverse-square
point-light irradiance `E_j`, quadrature `m_j > 0`, material-conditioned exchange
fraction `a_j`, and softmax spatial partition `f_jr`. Pool over all sources:

```text
z_r = sum_j m_j a_j f_jr
u_r = sum_j m_j a_j f_jr E_j / z_r.
```

At receiver `x`, interpolate material features through rasterization, predict its
exchange fraction `a(x)`, and evaluate the same anchor partition `f_r(x)`:

```text
E'(x) = (1-a(x)) E(x) + a(x) sum_r f_r(x) u_r
L(x) = softplus(base(x) + angular_material_network(x)) * E'(x).
```

The network consumes features, encoded light/view/half-vector directions,
light-view dot product and an eight-band encoding of normalized world position
(51 spatial channels). Foreground radiance is shaded per covered pixel, then
linearly alpha-composited before the shared observation transform. Spatial
encoding is intended to represent detail within projected Gaussian support;
GPU receiver/gradient audits passed, and fixed-last validation supports a
perceptual/detail gain with lower PSNR than round one. The acceptance tradeoff is
recorded in [results](../experiments/results.md).

## Exact operator properties and their scope

When receivers are the original source nodes with the same `m`, `a` and `f`,

```text
T_ij = (1-a_i) delta_ij + a_i a_j m_j sum_r f_ir f_jr / z_r.
```

This positive row-normalized operator preserves constant irradiance and
`sum_i m_i E_i`, and satisfies detailed balance `m_i T_ij = m_j T_ji`.
Arbitrary pixel queries are a continuous extension of the same source integral.
They have their own query fractions and partitions; their rendered image has
no discrete source-mass conservation or reversibility guarantee. Both queries
remain linear in supplied light intensity before the observation transform.

Quadrature is detached opacity-times-projected-area, a representational measure.
The angular material response, approximate visibility and alpha composition add
separate modeling assumptions. Expected depth merges contributors along a ray,
so the reconstructed receiver may lie between surfaces. RGB pooling compresses
incoming direction, and the response uses the original point-light direction.
Anchor connectivity approximates source-to-receiver transport. These limits
remain explicit in the candidate's physical interpretation.

The source operator's analytical review is in
[review_20260912.md](../project/review_20260912.md). First-round results and the
motivation for receiver shading are in [results](../experiments/results.md).
Pixel/deferred shading has established related work; this candidate's complete
transport formulation and empirical value require evidence before any novelty
claim. See [external research](../research/related_work_20260912.md).
