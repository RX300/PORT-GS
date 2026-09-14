# Cat: image and data diagnosis before round 2

2026-09-11. This diagnosis reads existing checkpoints, 32 fit frames and all 52
train-derived validation frames. Cameras and predictions remain unchanged.
The first round fixed opacity resets and simplified implementation, but Cat's
fixed-last validation PSNR changed from 21.699086 to 21.646310 dB. Gaussian count
fell from 318381 to 216456. Thus the verified scheduling repair reduced model
size without improving Cat image quality.

## What the images show

Even fit frames blur individual hairs and paper folds. The most difficult
validation frames, 233 and 158, also contain broad incorrect bright patches,
shadow-boundary displacement and silhouette offset. Best frame 496 preserves
large-scale shading more successfully while remaining visibly blurred. Frames
402/471, which deteriorated substantially, overpredict parts of the side and
paper brightness; the changes are not confined to background edges.

![Hard validation frame](../../runs/cat_refinement_r1_image_diagnosis/validation_0233.png)

![A fit frame](../../runs/cat_refinement_r1_image_diagnosis/fit_0122.png)

![Best validation frame](../../runs/cat_refinement_r1_image_diagnosis/validation_0496.png)

## Pixel measurements

Arithmetic mean across frames, using unaligned uint8-quantized predictions:

| Measurement | Fit32 | Validation52 |
|---|---:|---:|
| Error inside eroded object mask | 69.27% | 80.24% |
| Boundary error share | 24.04% | 16.27% |
| Background error share | 6.69% | 3.49% |
| Error retained after 16x16 block averaging | 37.46% | 51.25% |
| Prediction / GT two-pixel detail energy | 25.91% | 26.20% |
| Mean silhouette distance | 4.14 px | 3.78 px |
| Silhouette IoU at alpha .5 | .9536 | .9579 |
| Mean camera-weighted Gaussian support radius | 23.33 px | 23.61 px |

The detail measure uses the difference between pixel energy and 2x2-average
energy; it includes texture and edges. Block averaging describes coarse error,
not a uniquely identified lighting component. Support is weighted by actual
camera alpha compositing, excluding transparent background pixels.

Within overlapping object masks, validation MSE is .01980 for model visibility
.1–.5, versus .00961 for visibility above .5 and .01025 below .1. The partially
shadowed region has both overbright and underbright local errors: its aggregate
mean brightness bias is only +.01377. Consequently one global exposure/gamma
adjustment cannot explain the spatial error pattern. Model visibility is an
estimate, not independently measured shadow ground truth.

![Error summary](../../runs/cat_refinement_r1_image_diagnosis/summary_plot.png)

## Direction and sequence evidence from independent review

The old and new 52-frame PSNR vectors correlate .981. The four hardest frames
remain among the old five hardest. Of 52 frames, 24 improve and 28 deteriorate.

| Existing validation light group | Frames | Improved frames | Mean PSNR change |
|---|---:|---:|---:|
| azimuth -180 to -150 deg, elevation 30 to 60 deg | 15 | 12 | +.23137 dB |
| azimuth -90 to -60 deg, elevation 30 to 60 deg | 37 | 12 | -.16797 dB |

This pattern also appears within several capture sequences. Change versus
nearest fit camera/light angle is weak (correlations -.034 and +.038), as is
change versus alpha L1 (-.028). These facts point to direction-dependent RGB
behavior, with remaining sequence/view confounding; they do not prove an
individual material or camera error.

## Next hypothesis and limits

The evidence separates broad lighting/geometry errors from lost fine detail.
It does not support interpreting a point-count reduction as successful Cat
relighting. Existing camera-response and radiance-moment experiments also failed
to resolve the problem.

One candidate addresses the coupling between geometry and approximate shadows:
the current local radiance multiplies a geometry-derived visibility estimate,
but training detaches that entire estimate. Hence photometric error in a shadow
cannot directly update its occluder through this path. A candidate propagates
occluder/receiver derivatives while fixing adaptive sampling-grid choices and
numerical self-exclusion offsets within backward. This keeps all forward values
and checkpoint parameters unchanged. It is a hypothesis for the observed
partially-shadowed RGB errors, not an established explanation of all blur.

The initial 128-logit finite difference differed from autograd by 5.1% and did
not pass its criterion. Native gsplat thresholding and opacity-dependent tile
membership make that finite perturbation nonsmooth. The subsequent scale/receiver audit uses GPU float64 for the final scalar
reduction. At logit perturbation .001, weighted 128-logit error falls to .551%
and weighted single-logit error to .0163%. One real receiver gives errors below
.066% across .01/.003/.001. All tested geometry gradients are finite and forward
RGB is unchanged. These checks cover opacity finite differences and geometry
gradient finiteness, not exhaustive geometry finite differences.

After reviewing both image evidence and implementation, round 2 starts a fresh
30k Cat fit with the same 470/52 split and all round-1 settings. It changes only
the conditional shadow derivatives. The fixed-last full validation, fit images
and error maps will decide its quality; no control is retrained.

Artifacts: `runs/cat_refinement_r1_image_diagnosis/metrics.json`, `summary.json`,
the corresponding diagnostic images, and `runs/shadow_gradient_candidate_status.json`.


## Additional sampling-bound check

Independent GPU review of eight evenly selected fit frames finds that removing
all 39195 Gaussians below the native 1/255 opacity threshold leaves fitted
extent, near and far exactly unchanged. That subset therefore does not explain
frustum expansion in this checkpoint. Two points at opacity .01 control the
transverse extent: normalized centers roughly (.463,1.565,.129) and
(.359,1.532,.034), with maximum scales about .048 times object radius.

Using only opacity>.05 would narrow extent by factors 1.21–1.99, but those .01
points exceed the native threshold and their actual contributions have not been
measured here. This observation does not justify pruning them or changing the
sampling grid. The current full-set depth bins span about .0166–.0214 world units.
This read-only check leaves the running second round intact.


## Round-2 image outcome

The fixed-last 30k result improves mean validation RGB error by 12.13% and the
coarse 16x16 error by 19.93%. Detail energy rises from 26.20% to 27.10% of GT,
and camera-weighted support radius decreases from 23.61 to 21.81px. Images still
look blurred; the hardest frames 233/158 retain broad brightness and alignment
errors and slightly worsen. The outcome supports the selected coupling as one
useful optimization change, with clear remaining failure modes.

![Round-2 hard frame](../../runs/cat_shadow_gradient_r2_image_diagnosis/validation_0233.png)

![Round-2 fit frame](../../runs/cat_shadow_gradient_r2_image_diagnosis/fit_0122.png)

See [results](results.md) for PSNR/SSIM/LPIPS and exact protocol qualifications.
The final full fit uses the frozen 30k settings. A third structural round is
not needed for the measured improvement requested in this cycle.
