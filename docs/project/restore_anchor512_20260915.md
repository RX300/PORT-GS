# Restore the learned-anchor512 implementation — 2026-09-15

At the user's request, restore the 512 learned spatial-node model from Git
`9e9596a`. The transport, renderer, evaluator and image-diagnosis implementation
match that revision. The original source pooling, learned centers/widths,
material exchange fractions, and 165-input material-response MLP are active
again. Default rank is512; shadows start at1500 and exchange at5000.

Training retains the later total/component loss history and automatic loss.png
generation. The JSON validation launcher is retained and now emits anchor512
arguments without HashGrid/residual fields or local tinycudann PYTHONPATH.
`configs/hashgrid.json` was removed from the active tree and remains available
in Git and prior experiment snapshots. Local third-party packages and all
experiment checkpoints/results are untouched.

`configs/validation.json` now describes a possible fresh
`anchor512_validation_20260915` run. No six-scene experiment or monitor is
launched by this rollback. Existing anchor512 results can be evaluated directly
with the restored evaluator. HashGrid/residual checkpoints use their source
revisions/snapshots, not this active anchor model.

Current test scripts again exercise the anchor representation; their new check
outputs use `runs/anchor512_restore_check/` so historical audit evidence is not
overwritten. Architecture/pipeline/principles docs describe the restored model.

Verification passed: the four core model/render/evaluation/diagnosis files match
`9e9596a` exactly; six-scene training argv match the saved anchor512 manifest
except output paths. The original operator/gradient audit passed. A two-step
train-interface smoke check produced weighted loss records and a valid loss.png.
The saved Cat anchor512 checkpoint loaded and its first-frame PSNR/SSIM/LPIPS
matched the historical evaluation within 1e-5 relative tolerance. Check artifacts
are in `runs/anchor512_restore_check/`; no full training run was started.
