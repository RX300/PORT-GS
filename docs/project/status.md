# In progress — 2026-09-14

A fresh rank-512 spatial-node validation is running on six fixed scenes: two
each from `Real_NRHints`, `Synthetic_GS3`, and `Synthetic_SSS-GS`. The selected
scenes, exact commands, source snapshot, and startup evidence are recorded in
[`rank512_validation_20260914.md`](../experiments/rank512_validation_20260914.md).
The scheduler state is
[`runs/rank512_validation_20260914/status.json`](../../runs/rank512_validation_20260914/status.json);
no historical rank-32 output is reused or overwritten.

# Completed — 2026-09-13

Two structural rounds and three fresh 30k full fits are complete. Round two was
selected on training-derived validation before official testing and frozen in
`cat_r2_source.tar`. The first-round screen-pruning and layout corrections were
engineering restarts within that round. All computation used GPU 0, now released. Final independent review verified
3/3 reports, source ownership and process exit; the final GPU snapshot shows
0% utilization, 14MiB and no compute process.

| Complete official test | Train / test | PSNR | SSIM | Standard LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Cat | 522 / 66 | 21.501174 | .766281 | .227281 |
| Translucent | 2000 / 400 | 28.304924 | .960368 | .051791 |
| Bunny small | 500 / 500 | 37.600658 | .986484 | .019684 |

Cat meets the original-calibration >20 dB target. Relative to historical full
Cat, PSNR decreases .498897 dB, SSIM decreases .001245, and LPIPS improves
.021971 (8.81%). LPIPS improves on 65/66 frames; alpha L1 worsens on all 66.
The result reflects perceptual/detail gains with pixel/outline costs. Broad
lighting and position errors remain. Bunny's weakest views score 12.143550 and
12.436795 dB; 7/500 views fall below 20 dB despite median 38.047665 dB.
Frame 332 has severe view/outline and shape errors, with no unique cause
established by that image. The high mean does not imply robust performance
across all views.

The accepted validation change improved detail energy .268290 → .339026 of GT
(+26.4%) and LPIPS .247135 → .228932, while PSNR decreased .224227 dB from round
one. Source-node conservation/reversibility has a precise discrete scope;
pixel receiver queries and expected-depth mixing remain approximations.

The full sequence used frozen settings, original test metadata and zero test
fitting. Existing shared datasets were reused; no duplicate downloads, other
baseline training or ablations were added. Published GS³ values use a different
budget and incompletely specified final metrics; SSS-GS's small-data score is
an aggregate rather than a Bunny-specific result. No strict SOTA claim follows.

Source archives `cat_r1_source.tar`, `cat_r2_source.tar` and `source_before.tar`
preserve their own checkpoint implementations and earlier outputs. Final code
counts are 2009 → 1531 lines for eight production modules (23.79% reduction),
and 4133 → 2242 for all top-level Python including tests (45.75%). Folder names
retain the September 12 launch date. The final aggregate records **3/3 complete**.

[Final results](../experiments/results.md) · [Decision](decisions.md) ·
[Setup](../experiments/setup.md) · [Cross-data protocol](../experiments/comparison_20260912.md) ·
[Independent review](review_20260912.md) ·
[Full results](../../runs/research_20260912/full_results.json) ·
[Code counts](../../runs/research_20260912/cleanup_files.json)
