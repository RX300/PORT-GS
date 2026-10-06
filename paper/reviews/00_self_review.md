# Self-review of the first complete draft (2026-10-06)

Draft reviewed: `paper/build/main.pdf` (8 pages + references) and `paper/build/supp.pdf`.
Method: end-to-end read of the LaTeX sources and the compiled PDF, plus a re-check of every
quantitative statement against the result files listed in `paper/README.md`.

## 1. Claim-by-claim evidence check

| Claim in the paper | Evidence | Status |
|---|---|---|
| 33.47 / .952 / .056 average, +1.89 dB over SSD-GS | common-GT JSONs, 18 scenes, 5,691 frames (`make_tables.py`) | verified |
| best PSNR on 9/18 scenes; wins 13/13/12 vs SSD-GS/GS3/OLAT-GS | `numbers.json` | verified |
| Hotdog +1.90, Translucent +1.91; Drums -1.09, AnisoMetal -0.85 | per-scene common-GT | verified |
| SSS: on par with OLAT-GS (40.63 vs 40.74), >= 2.8 dB over others | family means | verified |
| Real: +2.0 dB, 4/7 scenes, trails by 0.8-1.6 dB on 3 | family means / per scene | verified |
| Re-scoring changed baseline averages by <= 0.005 dB and no ranking | original vs common-GT | verified (per scene <= 0.04 dB) |
| **First draft said GT pipelines differ by <= 1 level / 699 channels** | `audit.json` of all 18 scenes | **wrong; fixed**: 6-10% of channels, up to 42 levels on CatSmall/CupFabric/Pikachu |
| Ablation: full 31.51 +- 0.11, no transfer -1.37, local residual +0.03, single level -0.44, moment only -0.24 | `summary.csv`, `repeatability.csv` | verified |
| Translucent +0.41 for the atlas on every seed; Soap/Drums favor the local residual on every seed | `results.csv` per seed | verified |
| Transfer share 6-9% opaque, 20-58% SSS | 8-view atlas previews of all 18 final models (new, read-only) | verified |
| Loss domain: Lego 23.02 -> 30.25, Drums 24.10 -> 30.10, all 200 views improve, median 7.2 / 6.0 | `radiometric_curriculum_results.json`; configs diffed: only the loss domain differs | verified |
| Refinement: display domain better on all metrics for both receivers; pixel best SSIM/LPIPS; Gaussian +0.19 dB PSNR | `summary.csv` (refinement group) | verified |
| Training 17-28 min (21 mean); 1.6-5.5x faster on profiled scenes | `training_times.csv`, supplementary `results.md` | verified |
| Test-light novelty median 1.1 deg | metadata only (no test image read) | verified |
| Views in Fig. 3 at the median per-view margin (75th pct. for Statue) | `frame_candidates.py` | verified |
| Supplement transfer-ablation caption said "dark faces lose light" | mean linear brightness nearly identical across variants | **unsupported; replaced** by error maps and a descriptive caption |

## 2. Narrative and positioning issues found and fixed

1. **Transfer term was initially framed as the main source of the margin.** The capacity-matched
   local residual matches it on average (31.54 vs 31.51 dB). The paper now says so explicitly,
   credits the atlas transfer only where it wins on every seed (Translucent), and states that the
   margin is more likely due to visibility and optimization, which E3/E4 must confirm.
2. **Gaussian receivers inherit the per-primitive shadow limitation** that motivates the paper. Added
   an explicit sentence (Sec. 3.4) explaining why they are used only while geometry moves.
3. **Target annealing is not new** (GS3 uses it for HDR data). Credited in intro, related work and
   Sec. 4.2; our claim is the controlled diagnosis plus keeping joint training linear for all data.
4. **Real-scene gains are confounded with registration.** Stated in Sec. 5.2 with E2 as a placeholder.
5. **Light extrapolation is untested** by the benchmarks (median novelty 1.1 deg). Stated in Sec. 5.3,
   limitations and conclusion; E1 table left as placeholder.
6. **Trade-off of the linear schedule** on SSS/real scenes vs the earlier display-domain variant is
   disclosed (limitations + Table S3), with the caveat that the versions differ in more than the loss.

## 3. Clarity and notation

- Symbol clash Phi_i (intercepted flux) / Phi (atlas features) / Phi_N (normal CDF): renamed to
  P_i and G.
- "exactly the fractions of light" now qualified by "within the splatting model" (depth-sorted,
  EWA-projected primitives).
- Single-seed ablation deltas are now read against the 0.11 dB seed deviation.
- Over-general phrasings removed ("mostly", "most relighting pipelines", "short-range scattering").

## 4. Formatting and policy

- Official CVPR author kit (main branch after the CVPR 2026 release; no 2027 kit published yet),
  review mode, anonymous, line numbers, `\confYear{2027}`.
- Main text ends on page 8; references start on page 9; supplement compiled separately with
  cross-references to the main paper (xr-hyper).
- All figures are generated from stored outputs by `paper/scripts/`; no image was edited by hand.

## 5. Residual risks before peer review

- Main results are single-seed (seed variation on the ablation panel is 0.11 dB).
- E1-E4 are not run; the visibility claim (E3) and the training-recipe ablation at the full
  protocol (E4) are the most important gaps for the current narrative.
- SSS-GS / BiGS are not compared (E5).
- The development used the benchmark scenes (no fully blind test set); the paper avoids claiming
  blind evaluation and states that one configuration is used for all scenes.
