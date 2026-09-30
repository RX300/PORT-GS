# Tilted finite-opacity pixel study completed (2026-09-24 JST)

The separate CPU-only `--surface-pixel-opacity` diagnostic completed its first
execution in10.729s under `runs/surface_pixel_opacity_analytic/`. Production
training/rendering stayed unchanged during the ongoing initialization pilot.
No checkpoint, real scene image, GPU or validation data was used.

This extends the previous broad front-facing plane, rather than rerunning it:
2 tilts(0/60deg) ×2 tangent sigmas(.0125/.1 world units) ×2 GGX alphas(.001/.003)
×2 mirror offsets(0/2.5 sigma), each with9 horizontal subpixel light phases.
The9x9 crop uses actual perspective ray/plane intersections, point-light GGX,
Gaussian opacity.95 and finite rho<=9, cutoff0. Reference is64x64 midpoint
quadrature per pixel, independently checked against32x32 in EVERY phase.
All144 phase checks pass the predeclared relativeL1<=.005 limit; worst.000360982.
The phases vary horizontally with the vertical mirror location centered; these
are constructed stress cases, not an average camera-phase distribution.

A new analytic prototype computes GGX slope-PDF first moments as well as mass:
for p(s)=a²/[pi(a²+|s|²)²],

    integral_P s*p(s) ds = -a²/(2*pi) * integral_boundary n/(a²+|s|²) dl.

The existing edge rational integral gives this moment. Two independent SciPy
polygon quadratures and vertex/roughness gradient checks pass. The conditional
slope centroid is mapped through the local affine footprint to a subpixel
receiving point; smooth BRDF factors and Gaussian opacity are evaluated there.
This remains an approximation: the footprint is affine and support clipping
is not integrated into the slope mass. No positivity/centroid clamp was used.
All tested float64 masses/RGB are nonnegative/finite, and all conditional
centroids remain inside their pixels. Earlier float32 mass failures remain
unresolved and are NOT erased by this narrower double-precision result.

| Worst group relative radiance L1 across the same16 groups | Error |
| --- | ---: |
| Center point sampling | 1.344580 |
| Analytic polygon mass, center prefactors | .252124 |
| Analytic polygon mass, conditional-centroid prefactors | .069041 |
| Direct4x4 quadrature | .018333 |

The worst row for each method need not be the same group. Full per-group and
per-phase maxima are preserved, not only this summary. In the dense/small
sigma=.0125,60deg,alpha=.001,2.5sigma-offset case, center/polygon-center/
polygon-centroid/4x4 errors are.778124/.252124/.069041/.010545. Thus opacity
variation matters even after integrating the sharp NDF. Conditional centroids
help but do not make the small-surfel filter exact or production ready.
For sigma=.1, centroid errors range.000233–.004157; this difference from the
small-sigma cases must not be hidden by only reporting broad surfaces.

Artifacts: config.json, source.tar, report.json, radiance.pt, errors.png and the
sibling surface_pixel_opacity_analytic_first.log. Root viewed errors.png.
The scalar mass/first-moment formulas are checked; the COMPLETE filtered shader
and its training gradients, two-lobe/diffuse mixtures, inter-surfel transmission,
visibility boundaries, real sensor response and GPU cost remain unvalidated.
This experiment does not show real GT highlight recovery, change the current
10000-step protocol, or select a production pixel filter.

The following sections retain the earlier broad-plane experiments and limits.

# Pixel footprint and narrow local reflectance

## Completed synthetic integration diagnostic, 2026-09-23

The current pilot evaluates the local BRDF on pixel-center rays. This note tests
the difference between that point value and a box-filtered pixel integral; it
does not change the live pilot or assume the real capture uses a box sensor PSF.

First execution completed successfully through
`test_method_integration.py --surface-pixel-footprint`, using the existing
ssd-gs environment, eight CPU threads and `CUDA_VISIBLE_DEVICES=''`. Evidence:
`runs/intersection_reflectance_smoke/pixel_footprint_analytic/` contains config,
source archive, saved radiance arrays, report and phase curves. The sibling
`pixel_footprint_analytic_first.log` preserves the complete first output. CUDA
was not initialized; no scene image, validation frame or checkpoint was used.

A known plane is shaded by the current GGX implementation: camera/light height3,
native512 camera focal480, opacity.95, tangent sigma1, zero diffuse and F0=.04.
The core-only lobe uses alpha .001/.003/.01. Moving the light translates its
mirror point through17 offsets spanning one pixel. We render the same9x9-pixel
crop with1/2/4/8/16/32/64 samples per pixel axis. Linear radiance is integrated
before any observation encoding; support boundaries lie outside the crop.
The64x64 midpoint reference is numerical, not exact: the predeclared32-to-64
relative image-L1 convergence limit is.005, and all three cases pass.

| GGX alpha | 32-to-64 reference relative L1 | Center relative L1 | 2x2 relative L1 | 4x4 relative L1 | 8x8 relative L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| .001 | .00019881 | .470717 | .056123 | .016927 | .004180 |
| .003 | .00003224 | .048398 | .011170 | .002750 | .000678 |
| .01 | .00000357 | .004918 | .001219 | .000303 | .0000749 |

For alpha.001 the center-sampled peak reaches2.4067 times the same-phase
reference peak; maximum crop-energy relative error is71.13%. With4x4 samples,
maximum crop-energy error falls below.0096%, although relative image-L1 remains
1.69%. Energy matching alone does not certify the pixel distribution. For the
other roughnesses, reference crop energy itself varies slightly with phase
because the crop is finite and the exact light/BRDF geometry changes; errors
are always computed against the corresponding phase, not a constant-energy
assumption.

This is evidence that point sampling can distort a representable tiny highlight
under a specified image-formation model. It is not evidence that antialiasing
restores missing measured peaks, identifies the camera PSF, improves geometry,
or improves the current model's validation quality. Supersampling would also
increase work and would require correct geometry, alpha/occlusion and observation
integration. No production supersampling or learned-model inference was added.

## Local analytic slope-domain prototype

The follow-up `--surface-pixel-filter-reference` reuses the completed radiance
arrays above; it does not rerun the64x64 reference. Its first execution completed
in `runs/intersection_reflectance_smoke/pixel_polygon_filter/`, with separate
source/config/report/filtered arrays/phase curves and a sibling first log.
Only the existing integration-test entrypoint implements this prototype.

For isotropic GGX, write the slope density as

```
p_a(s) = a² / [pi (a² + |s|²)²].
div_s [s / (a² + |s|²)] = 2 a² / (a² + |s|²)².
```

The divergence theorem converts a polygon integral of this density into one
analytic integral per edge. For a counterclockwise edge p+t*v, t in[0,1], let
A=dot(v,v), B=2dot(p,v), c=cross(p,v), and d=2sqrt(A*a²+c²). Its contribution is

```
(1 / (2*pi)) * (2*c/d) * atan2(2*A*d, d² + B*(2*A+B)).
```

Summing the edges gives the exact slope-density mass in real arithmetic. This
is a direct derivation, not a literature-novelty claim. The synthetic filter
maps a box pixel through the center Jacobian of half-vector slopes
s=(wi_xy+wo_xy)/(wi_z+wo_z). That gives a parallelogram of area abs(det(J)).
The filter replaces the center slope density by polygon_mass/area, keeping
other factors of the center-shaded radiance fixed. The relation
D(h)=p_a(s)*(1+|s|²)² explains the ratio used in the prototype.

**Exact polygon density integration does not imply exact full-BRDF pixel
integration.** Half-vector linearization and center Fresnel, masking, geometry,
irradiance, opacity and slope-to-NDF factors are approximations. This experiment
uses a smooth plane near normal incidence, one lobe, and no visibility/support
boundary. Independent multi-layer coverage cannot generally be integrated by
averaging each alpha and then composing; the correlated visibility still matters.

Three skewed/off-center polygon cases agree with independent SciPy quadrature
within rtol1e-8/atol1e-12. All three vertex/roughness finite-difference gradient
checks pass at eps1e-7, rtol1e-4/atol1e-5. These are density-integral derivatives,
not full renderer geometry/material training checks.

| GGX alpha | Prototype relative image L1 vs64x64 reference | Maximum phase crop-energy relative error |
| --- | ---: | ---: |
| .001 | .00006370 | .00000220 |
| .003 | .00000863 | .00000208 |
| .01 | .000000907 | .000000913 |

All double-precision RGB values in this scene are finite and nonnegative. The
errors approach the numerical-reference convergence scale, so they should not
be presented as an exact error against the continuous pixel integral.

### Preserved numerical failure and remaining work

A predeclared72-case arithmetic stress test spans roughness.001/.01/.1,
slope radii0/.001/.01/.1/1/10 and four angles, with a.002-square footprint.
Direct float32 edge summation produces **7 negative masses**, compared with0
in float64; no positivity clamp or case exclusion hides them. The maximum
absolute float32/float64 mass difference is4.23904e-8. This is a numerical
failure of the straightforward single-precision form, even though the main
double-precision analytic checks pass. Full raw masses are in the report.

The prototype is therefore **not production ready**. Precision/stability, oblique
views, degenerate or rapidly changing footprints, two-lobe/diffuse handling,
visibility boundaries, observation encoding and differentiable runtime cost
must be resolved before a controlled real-scene study. No shader, renderer,
sampler, optimizer, current budget or scene checkpoint was changed. The current
100k two-arm experiment remains the next real reconstruction-quality evidence.
