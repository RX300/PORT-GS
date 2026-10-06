# LiSA material-head selection on held-out training lights

6/6 jobs completed, 0 failed (2026-10-01 23:25 – 2026-10-02 00:27 JST, GPUs 2/3). 30,000 steps, seed 0, trained
without `--fit-all`: whole 30-degree light-angle groups (~10% of the official training frames) are held out and
used only for this evaluation. No official test frame was rendered. Rule fixed before results: highest mean
validation PSNR over the two scenes, LPIPS as tie-break within 0.05 dB.

| Head | AnisoMetal (225 views) | bunny_small (51 views) | Mean PSNR |
|---|---:|---:|---:|
| compact (first run) | 27.5366 / 0.96158 / 0.03669 | 38.1230 / 0.98928 / 0.02130 | 32.8298 |
| reflect (+ reflection encoding, 4 bands, 4 layers, 2048 hint) | 27.6661 / 0.96217 / 0.03635 | 38.0492 / 0.98916 / 0.02115 | 32.8577 |
| **spatial** (reflect + 8-band positional encoding) | **28.0464 / 0.96383 / 0.03513** | **38.2360 / 0.98971 / 0.01916** | **33.1412** |

Cells are PSNR / SSIM / LPIPS on the validation split. Decision: `spatial` (+0.31 dB mean over compact, better on
both scenes and all three metrics). The positional encoding, not the reflection encoding, provides most of the gain:
it lets the local head represent spatially varying anisotropic streaks on brushed metal. It did not hurt the
light generalization of the SSS scene. Final six-scene run and ablations: `runs/light_atlas_final_six_scene_20261002`.
