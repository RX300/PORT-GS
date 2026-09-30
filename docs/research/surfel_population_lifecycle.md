# Fixed storage versus active surfel support

## Completed full-fit initialization/center-support diagnostic, 2026-09-23

While the separate5000-step cutoff recovery study was running, a CPU-only
diagnostic inspected the original intersection initial/25k/100k states with
their same100000 point identities and all506fit masks. No recovery checkpoint,
validation image, fitted camera correction or optimizer was used. This differs
from the older SDF64-view/high-opacity4096-sample hull diagnostic: here the
question is the new foundation's initial sampling density and point survival.

The existing integration entrypoint's `--surfel-initial-support-run` projects
each point CENTER with saved training cameras at native512, tests alpha>.9,
and reports square mask dilation0/2/8pixels. Behind-near-plane and out-of-image
centers fail that view. This is not a ray/disk visibility test, a visual-hull
mesh reconstruction, or true geometry accuracy; broad surfels can intersect
foreground even when their centers project outside it. Calibration/mask errors
can also violate consistency for genuine surface points. All three dilations
are retained rather than choosing the favorable one.

| Initial100000 centers | No dilation | 2px dilation | 8px dilation |
| --- | ---: | ---: | ---: |
| Inside at least95% of506fit masks | 2387 | 2683 | 3441 |
| Inside at least99% | 1939 | 2179 | 2863 |
| Inside every fit mask | 1204 | 1378 | 1859 |

Only2.683% of initial centers meet the95%-view criterion with2px tolerance.
At25k,2778 of all stored centers meet it, but only346 are opacity>=1/255;
the other2432 are below the cutoff. At100k,2805 meet it, of which280 are
threshold-eligible and2525 are below. Thus initial geometry sampling is sparse
in a mask-consistent volume, and transparency also removes many centers within
that volume. This does not prove they are on the surface or should all survive.

The eligible population's median mask consistency at2px improves from.079051
at25k to.970356 at100k even as its count falls5047→511. Rejection therefore
cannot simply be labelled wholly harmful: later survivors' centers are more
consistent with the foreground masks. This strengthens the need to examine
initial allocation and replenishment together with opacity gradient access,
without declaring the pending cutoff-only quality study successful in advance.

First execution completed in6.046s without CUDA. Evidence:
`runs/surface_opacity_smoke/initial_support/{source.tar,report.json,counts.pt}`
and `initial_support_first.log`. `counts.pt` stores per-point/per-state hit
counts and in-view counts for all declared dilations; no dataset or source
checkpoint was modified. The helper was added after recovery launch and is
separately archived; no in-use production rendering/training source changed.

## Observed scheduled 25k states, 2026-09-23

The ongoing `intersection_reflectance_pilot` remains fixed at 100k updates per arm.
This note does not select a checkpoint, change its budget, or evaluate an image.
CPU inspection read the completed, immutable `step_025000.pt` files only after
each live history had advanced beyond that checkpoint. All state tensors examined
were finite; CUDA was not initialized by the diagnostic.

Evidence: `runs/intersection_reflectance_pilot/parameter_snapshot_025000.json`.

| Uniform primitive statistic at25k | Intersection | Aggregate |
| --- | ---: | ---: |
| Stored primitives | 100000 | 100000 |
| Opacity at least1/255 | 5047 | 6160 |
| Initial light scale / fitted scale | .106150 | 1.266533 |
| Mean diffuse RGB, threshold-eligible subset | .095938 | .108120 |
| Mean F0 RGB, threshold-eligible subset | .310121 | .247885 |
| Mean core GGX alpha, threshold-eligible subset | .039049 | .056265 |
| Core alpha below.01, threshold-eligible subset | 61.6604% | 60.3409% |
| Median plane-normal change, threshold-eligible subset | 26.8776° | 24.4282° |
| Median center movement / initial radius, threshold-eligible subset | .039159 | .027481 |
| Median tangent scale / radius, threshold-eligible subset | .032683 | .031671 |

These are **not visibility-weighted** statistics. Threshold eligibility does not
prove a primitive is visible, accurately placed, or responsible for a GT peak.
Plane-normal change ignores normal sign because the current disks are two-sided;
it measures change from random initialization, not error against true geometry.
Narrow roughness parameters have been learned, but their alignment, amplitudes and
generalization remain unknown until the fixed terminal image evaluations.

## Consequence of the implemented support rule

`surface_fragments.py` retains a fragment only when
`alpha = opacity * exp(-rho/2) >= 1/255`, with nonnegative `rho`.
Thus `alpha <= opacity`: every primitive below1/255 is unable to contribute a
retained fragment at this snapshot, regardless of camera or light. Fixed tensor
length100000 therefore must not be described as100000 active surface samples.

The current surface branch has no native densification, opacity reset or
relocation. For a below-threshold primitive the rendering losses provide no
direct opacity gradient through a retained fragment; Adam momentum and the
stored parameter count are distinct from renewed photometric support. The
substantial population reduction is a plausible capacity/optimization limitation,
**not a demonstrated explanation of any quality result**. It can coexist with
useful selection of surface-adjacent samples from a random volume.

The light scalar and F0 also differ considerably between modes. This inverse
problem does not identify absolute reflectance or illumination. `kd/light_scale`
and `F0/light_scale` in the JSON are coefficient proxies, not radiance: angles,
roughness, distance, composition weights and visibility still matter.

## Relevant primary research and installed implementation

3DGS-MCMC relocates low-opacity samples, adjusts opacity/scales and introduces
position noise. Its relocation is approximate; the derivation matches the center
and integrals of one-dimensional slices rather than every pixel. This is a
relevant population-management idea, not evidence of relighting or tiny-highlight
recovery in our renderer. [Original paper, §3.4 and Appendix A](https://arxiv.org/html/2404.09591v2#S3.SS4).

The installed `ssd-gs` environment already contains `gsplat/strategy/mcmc.py`,
`gsplat/strategy/ops.py`, and `gsplat/relocation.py`; no environment change was made.
Direct local inspection found:

- The installed relocation frontend asserts `scales.shape == (N,3)`. Our stored
  tangent scales have shape(N,2).
- Position noise uses the full covariance via `quat_scale_to_covar_preci`, also
  requiring three scales. Adding a unit third scale would introduce normal-axis
  variance into a zero-thickness disk; it is not a justified adapter.
- The installed binomial table has size51 and copy ratios are clamped to that
  bound. A future relocation must inspect multiplicities instead of assuming the
  actual number of copies always equals the formula's count.
- Opacities of relocated copies are clamped from below. In a thresholded renderer,
  center correction, finite support, new cutoff losses and clamp effects require
  an explicit check; they cannot be called exactly image preserving.

The current online docs describe strategy APIs that have evolved beyond this
installed version; they are context, not the executable contract.
[Official gsplat strategy documentation](https://docs.gsplat.studio/main/apis/strategy.html).

## Completed analytic relocation experiment

The existing integration entrypoint now provides the opt-in CPU diagnostic
`--surfel-relocation-diagnostic`. Its first execution completed successfully in
`runs/intersection_reflectance_smoke/relocation_analytic/`; the sibling
`relocation_analytic_first.log`, `config.json`, `source.tar`, `report.json` and
`comparison.png` retain the command output, assumptions and measured results.
No checkpoint or dataset was read and CUDA was not initialized.

This is a constructed single-plane experiment, not a run of the original MCMC
implementation. At native512, focal480, eye/light height3, tangent sigma.2 and
opacity.95, coincident copies receive the paper's center/one-dimensional-integral
opacity and scale correction. The current finite rules, rho<=9 and individual
fragment alpha>=1/255, are applied. The unchanged local shader uses F0=.04,
core GGX alpha=.003, haze alpha=.08, mixture.999 and zero diffuse. Moving the
point light places a narrow peak at each specified offset from the disk center.

Independent numerical quadrature matches the untruncated one-dimensional
integral for 1, 2, 4 and 16 copies within the declared 1e-10 relative tolerance;
the center matches within 1e-12. These checks do not impose an RGB outcome.

| Peak offset / original sigma | Original peak alpha | Same-pixel radiance, 2 copies / original | 4 copies / original | 16 copies / original |
| --- | ---: | ---: | ---: | ---: |
| 0 | .950000 | 1.000000 | 1.000000 | 1.000000 |
| 1.5 | .308420 | .952349 | .920930 | .894498 |
| 2.5 | .041740 | .457070 | 0 | 0 |
| 2.875 | .015236 | 0 | 0 | 0 |

Four copies shrink tangent scales to .772804 of the original, reduce finite
support from28917 to17281 pixels and reduce summed image alpha by9.2433%.
For the2.5-sigma case the reference peak loses support completely; even the
new image maximum is only.001862 of the old maximum. Full-frame RGB MAE
normalized by the reference maximum is nevertheless only2.55e-5: averaging over
mostly dark pixels can hide a destroyed small peak. All measurements are in
unclamped linear radiance, and the visualization uses one shared reference-peak
display normalization within each row. The constructed lights/materials do not
measure real-image error, display-space metrics or physical GT accuracy.

This counterexample rules out calling this **naive surfel adaptation** pointwise
or highlight preserving. It does not refute the paper's reconstruction results,
show that relocation caused the current pilot's behavior (it has no relocation),
or establish a better training strategy. Its scope is the instantaneous copy
transition, before optimization, displacement, material changes or occlusion.

## Implications for a later decision

### Pointwise transmittance partition: analytic candidate only

The same CPU entrypoint's new `--surfel-partition-diagnostic` completed its first
execution in `runs/intersection_reflectance_smoke/transmittance_partition/`.
The previous experiment and its source remain untouched. This follow-up tests a
direct algebraic construction, with no claim of literature novelty or adoption:
for a parent alpha profile a(x), define child profiles

```
a_j(x) = 1 - (1 - a(x)) ** w_j,
w_j > 0, sum_j w_j = 1.
```

Then the product of child transmissions is exactly 1-a(x). If all children
initially have the parent's geometry, material and illumination response, and
are contiguous at the parent's depth in ray order, their weighted color sum is
also identical for any spatially varying BRDF. Foreground layers, background
layers and background color are consequently preserved. This is an instantaneous
identity, not a statement about training after the child parameters separate.

This construction **changes the opacity profile family**. In general a child
cannot be written as a new scalar opacity times a Gaussian. The candidate keeps
the parent Gaussian support/cutoff before taking fractional transmittance powers;
applying the current1/255 cutoff to each child's final alpha destroys the identity.
It would therefore require a renderer/profile change, not just a densification
callback or passing two scales into the installed MCMC operator.

Measured CPU-double results for2/4/16 coincident children:

- Pointwise composited alpha maximum error at most1.11e-16; all28917 original
  supported pixels remain. Each of the four original narrow peak amplitudes is
  preserved, and full linear RGB passes rtol1e-12/atol1e-12.
- A separate scalar layered compositor with unequal shares[.1,.2,.3,.4] gives
  RGB maximum error5.55e-17 and identical shared-parent derivatives. These are
  derivatives of the tied construction, not independent child optimizer updates.
- Restoring postpartition1/255 cutoff loses the2.875-sigma peak with4 children
  and both outer peaks with16. Support becomes26429/17665 pixels respectively.
- Deliberately interleaving a different-material layer among children changes
  RGB by.000954752. Stable numeric primitive IDs alone do not guarantee that
  children stay contiguous around other exactly coplanar primitives.

This is a bounded successful algebra/implementation check, not a successful
population strategy. Identical children can remain symmetric without controlled
perturbation; separating them changes the image. Repeated splits, parameter and
optimizer-state semantics, contributions below the old alpha threshold,
independent geometry gradients, tie ordering, memory/runtime and training quality
remain unresolved. No production profile, relocation, sampler, budget or current
checkpoint was modified. The fixed two-arm pilot still supplies the next actual
scene-quality evidence.

### Current experiment decision

Keep the current controlled experiment unchanged and retain all terminal gates.
Do not paste the volumetric MCMC strategy into the running surfel model.
If terminal evidence warrants a population-management study, first define
thin-surface relocation/noise and optimizer-moment behavior, then check finite
support, layered occlusion and narrow spatially varying BRDF responses. Matching
an opacity integral alone does not bound RGB error near a sharp highlight.

There is also a separate exact-support acceleration opportunity: a primitive
whose maximum opacity is below1/255 can be excluded from candidate tiles because
it cannot pass the existing fragment test. This follows from the current formula,
not from MCMC. It is not implemented or timed, and it would not restore capacity.
No production files or shared dependencies were modified for this investigation.
