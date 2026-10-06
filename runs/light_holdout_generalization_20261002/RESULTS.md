# Held-out light generalization and material-head comparison

15/15 jobs completed, 0 failed (2026-10-02 05:20–10:09 JST, GPUs 2/3). 30,000 steps, seed 0, trained WITHOUT
`--fit-all`: whole 30-degree light-angle groups (~10% of the official training frames) are held out and used only for
evaluation (`--split validation`). No official test frame is rendered. Held-out light novelty (angle to the nearest
training light): median 4.3° / max 13.5° (AnisoMetal), median 11.2° / max 15.9° (bunny), versus 0–2° for the official
test lights. LiSA AnisoMetal/bunny runs for the spatial and compact heads are reused from
`runs/light_atlas_head_selection_20261001` (same code and protocol). Cells: PSNR / SSIM / LPIPS.

| Method | AnisoMetal (225) | Translucent (225) | bunny_small (51) | dragon_small (50) | 4-scene mean PSNR | Pixiu (56) |
|---|---:|---:|---:|---:|---:|---:|
| `directional_port_v1` | **28.250** / 0.9599 / 0.0420 | 27.929 / 0.9581 / 0.0582 | 35.456 / 0.9851 / 0.0234 | 34.730 / 0.9700 / 0.0400 | 31.591 | — |
| LiSA, svbrdf head | 28.220 / **0.9639** / 0.0353 | 29.483 / 0.9708 / 0.0412 | 37.988 / 0.9891 / 0.0216 | 37.488 / 0.9817 / 0.0260 | 33.295 | 20.351 |
| LiSA, spatial head | 28.046 / 0.9638 / **0.0351** | 29.565 / 0.9712 / **0.0406** | **38.236 / 0.9897 / 0.0192** | **37.613 / 0.9819 / 0.0250** | **33.365** | 20.369 |
| LiSA, compact head | 27.537 / 0.9616 / 0.0367 | **29.642 / 0.9715** / 0.0407 | 38.123 / 0.9893 / 0.0213 | 37.519 / 0.9814 / 0.0261 | 33.205 | 20.365 |

LiSA vs `directional_port_v1` on held-out lights: +1.61 to +1.77 dB mean over the four synthetic scenes (svbrdf:
AnisoMetal −0.03, Translucent +1.55, bunny +2.53, dragon +2.76), with better SSIM and LPIPS everywhere. Compared with the
official test split (near-training lights: spatial head −0.96 / +1.84 / +2.69 / +1.88 dB), LiSA's advantage holds or grows
on novel lights and the AnisoMetal deficit disappears. Pixiu held-out scores are low for every head because held-out real
views keep their uncorrected test-style pose offsets; the heads tie there.

Mean learned visibility on eight held-out Pixiu views: compact 0.72, svbrdf 0.84, spatial 0.91 (none collapsed in this
training set). Decision registered before results (svbrdf within 0.1 dB of spatial on the 4-scene mean and Pixiu V < 0.98):
satisfied (−0.07 dB, 0.84) → the svbrdf head is the final configuration; its six-scene test run is
`runs/light_atlas_svbrdf_six_scene_20261002`.
