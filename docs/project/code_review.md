# Independent review — 2026-09-11

The `review_port` subagent performed an initial read-only review and a separate
review of the implementation changes.

| Finding | Evidence | Resolution |
|---|---|---|
| P1: opacity reset never executed | Installed gsplat `strategy/default.py` line 195 has a chained-comparison precedence error | Correct schedule in `refinement.py` |
| P1: checkpoint initialization can fabricate validation membership | Train regenerated holdout before loading arbitrary weights | Inherit checkpoint split, explicit full-fit promotion |
| P2: hitting point cap stopped prune/reset permanently | Trainer rewrote `refine_stop_iter` | Bound growth, retain scheduled pruning/reset |
| P2: excess initial checkpoint points escaped new cap | Follow-up review of bounded growth | Trim low-opacity excess once during initialization |
| Redundant source/query shading and quadrature | Duplicate forward paths and area/volume code | One shared transport path and source measure |
| Redundant spatial bases | Source and target are identical in forward rendering | Reuse the same basis in that call |
| Whole-image GPU-to-CPU copies for bounds | Only camera matrices, K and dimensions are consumed | Batch bounds on GPU |

The final review confirmed the corrected callback, capacity arithmetic, source/query
transport semantics, unified observation transform, and initialization-budget
trimming with initially empty Adam state. Real-frame regression preserved RGB
and alpha exactly for the tested Cat/Pixiu checkpoints. The reviewer also checked
Pixiu metric protocols and retained the warning about missing historical defaults
and slightly increased alpha error. Cat second-round metrics and all 84 diagnostic-frame keys were independently
verified. The final test stage is complete.


## Second-round result review

The reviewer recomputed all 52 paired deltas and confirmed matching configurations
and frame membership. PSNR/LPIPS improve overall; SSIM, alpha L1 and silhouette
IoU are slightly lower. Both hardest frames remain poor. The 12.13% quantized
MSE and 19.93% coarse-error reductions are consistent with the per-frame data.
Visibility-bucket membership changes, so bucket means are descriptive. The
review supports freezing the r2 configuration for full fitting; official test
performance is recorded separately in the results.


## Final protocol review

The independent reviewer confirmed both complete full fits and tests: Cat
30k/all522/test66 and Pixiu 30k/all562/test71, validation empty, fixed `last.pt`,
`limit=0`, complete frame indices and training completion events. Results are
22.000071/.767526/.249252 and 20.972036/.847689/.163980 (PSNR/SSIM/standard LPIPS).
The historical Cat 100k result and missing Pixiu defaults remain appropriately
qualified. Source, experiment, image and final-result review are complete.
