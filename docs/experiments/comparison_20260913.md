# SSD-GS and PORT-GS full-queue comparison — 2026-09-13

This table records the seven completed SSD-GS evaluations on the official
Real_NRHints test split and the corresponding PORT-GS rows available at the
handoff. Values are copied from the result files; no metric was recomputed in
this documentation pass.

| Scene | Test frames | SSD PSNR | SSD SSIM | SSD LPIPS | PORT state | PORT PSNR | PORT SSIM | PORT LPIPS |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Cat | 66 | 18.000583 | 0.703937 | 0.248067 | complete | 21.501174 | 0.766281 | 0.227281 |
| CatSmall | 158 | 32.675672 | 0.967992 | 0.093708 | complete | 30.795341 | 0.943799 | 0.094551 |
| CupFabric | 145 | 34.742582 | 0.976249 | 0.074919 | complete | 34.055336 | 0.970301 | 0.063623 |
| Fish | 66 | 23.259054 | 0.821960 | 0.149528 | running | — | — | — |
| FurScene | 85 | 25.307714 | 0.878876 | 0.107913 | pending | — | — | — |
| Pikachu | 200 | 31.118807 | 0.961188 | 0.080664 | pending | — | — | — |
| Pixiu | 71 | 23.402195 | 0.862244 | 0.118944 | pending | — | — | — |

SSD values come from
`SSD-GS/runs/real_fixed_calibration_20260913/<scene>/metrics.json`; the files
store lowercase `psnr`, `ssim`, and `lpips` plus `unitRGBMSE`. The evaluator
used the official test transforms and original scene calibration. The fixed
SSD checkpoints are historical weights; their full training lineage is not
guaranteed free of prior data contamination, so these values are reported as
an evaluation reference rather than a clean causal baseline claim.

Completed PORT values come from the following result files:

- Cat: `runs/research_20260912/cat_full_s0/test/metrics.json` (reused by the
  full manifest).
- CatSmall:
  `runs/full_benchmark_20260913/Real_NRHints/CatSmall/test/metrics.json`.
- CupFabric:
  `runs/full_benchmark_20260913/Real_NRHints/CupFabric/test/metrics.json`.
- Synthetic_GS3/Translucent:
  `runs/research_20260912/translucent_full_s0/test/metrics.json`.
- Synthetic_SSS-GS/bunny_small:
  `runs/research_20260912/bunny_full_s0/test/metrics.json`.

The PORT result files store aggregate metrics under the nested `metrics` object
with uppercase `PSNR`, `SSIM`, and `LPIPS`. The seven SSD rows are therefore
the clean one-to-one scene mapping; Translucent and bunny_small are completed
PORT reference scenes without corresponding rows in the SSD table.

The machine-readable queue and result paths are declared in
`runs/full_benchmark_20260913/manifest.json`; live phase state is in
`runs/full_benchmark_20260913/status.json`. At this handoff PORT has five
completed result rows, while Fish is active and the remaining PORT rows are
pending. The current PORT run is not ranked against SSD across all scenes until
the matching rows finish.

GS3 and SSS-GS have no final full-queue results at this handoff and are not
included in a method ranking. The table also does not claim strict SOTA: the
methods can differ in training budget, checkpoint lineage, and metric
implementation even when the scene and test split match.
