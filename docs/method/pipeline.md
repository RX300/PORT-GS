## Current30k default — 2026-09-16

The60k experiment was cancelled and its outputs deleted at user request.
The default budget is30000 steps, with shadows/ports starting5000 and
refinement stopping25000. Save only final last.pt, then evaluate full official
test when an experiment is explicitly launched. No new run is active.

## Current training schedule — 2026-09-15

The user-requested restart enables both shadows and directional ports at step5000,
stops Gaussian refinement at step25000, and trains through step30000.
validate_every=0 disables periodic validation and intermediate checkpoint saves;
only final last.pt is saved, followed by full official test evaluation.

## Current architecture — 2026-09-15

The active model is directional_port_v1: 512 spatial ports, four direction
channels and a shared material direction MLP. Direct and nonlocal radiance are
combined at pixel receivers. See [directional architecture](../architecture/modules/directional_transport.md).
The earlier architecture below is historical.

> Active representation restored on 2026-09-15: 512 learned spatial anchors,
> source irradiance pooling and the original material-response MLP (Git `9e9596a`).
> HashGrid and residual decoders are archived experiments. Current training
> also retains weighted loss logs/plots and the six-scene JSON launcher.

# Training and evaluation pipeline

1. Read official train metadata and create the deterministic training-light
   holdout. Cat uses 470 fit / 52 validation frames. Current-representation
   checkpoint initialization inherits its saved membership; `--fit-all` uses
   all 522 official train frames and an empty validation set.
2. Upload decoded samples to the selected GPU, estimate bounds from camera
   geometry on GPU, and initialize Gaussian geometry and the exchange model.
3. Sample a fit frame and compute source visibility. Rasterize base appearance,
   features, visibility and expected camera-Z using `RGB+ED`. Normalize attributes
   by alpha and back-project pixel centers through `K` and the view transform to
   form covered-pixel receivers. Integrate irradiance over all Gaussian sources,
   query exchange at receivers, and apply the spatial/angular material response
   per pixel. Alpha-composite the resulting foreground radiance.
4. Apply the shared observation model. PNG foreground encoding precedes alpha
   composition; HDR targets and predictions receive the same display transform.
   Optimize image, alpha and representation losses. Optional training-camera
   corrections belong only to the fit frames.
5. Refine geometry with the established opacity-reset and point-budget schedule.
   Defaults enable shadows at step 1,500, exchange at step 5,000, and stop
   refinement at step 15,000. During refinement, Gaussians whose screen radius
   exceeds 3% of the image's long edge enter split candidates, including broad
   supports with weak position gradients. Split and duplication are mutually
   exclusive. Screen-radius pruning is disabled because inherited parent radii
   are stale for newly split children; opacity/world-size pruning remains active.
   Save validation checkpoints during the 30k run.
6. Two structural rounds have completed their fixed-last validation and image
   diagnosis. The second round is accepted for perceptual/detail improvements,
   with its PSNR/outline tradeoffs recorded. Code and hyperparameters are frozen;
   first-round engineering startup corrections remain within round one.
7. Train a fresh model on all 522 Cat train frames and evaluate its final
   checkpoint on 66 official test frames using original camera/light calibration.
   Then train/evaluate Translucent and Bunny in sequence on the same GPU 0,
   applying only their predefined data/observation settings. This full-fit phase completed for all three scenes on September 13 JST;
   results retain the predefined data and metric boundaries.

`evaluate.py --split fit` restores saved training-camera corrections for fit
frames. Validation and test use original calibration. Standard LPIPS and the
quantized final metric protocol are described in
[setup](../experiments/setup.md). Actual completion and metrics belong in the
experiment record; this page describes the pipeline, not its execution status.
