# LiSA 研发实验：材质头、消融与留出光照

本页整理既有研发记录，不是新分析。compact/spatial均为历史配置；当前主结果为svbrdf的[全18场景记录](results.md)。以下旧六场景对照和对齐诊断不与当前train-only全量表混用；其中机制解释保留为当时观察，不升级为已证实因果结论。所有实验均已结束，旧启动时间/PID仅供追溯。


## 2026-10-04 LiSA-v2 研发（根因诊断、train内选择、全量重跑）

全部证据单独记录于[LiSA-v2根因、修复与选型](lisa_v2_root_causes_20261004.md)：外壳/塌缩的单因素诊断批次
（`runs/lisa_shell_diagnosis_20261004`、`runs/lisa_collapse_diagnosis_20261004`、`runs/lisa_v2_early_light_pass_20261004`），
光照组留出选择（`runs/lisa_v2_selection_20261004`及`*_split_fg*`、`*_ramp_*`、`*_dense_init_*`扩展，部分为有记录的提前停止），
插值留出选择（`runs/lisa_v2_interp_selection_20261004`），正式全量（`runs/lisa_v2_full_dataset_20261004`）。

Method: [module](../architecture/modules/light_atlas.md) · [proposal and review](../research/lisa_light_space_atlas_20261001.md).

## Protocol

Identical to `refactored_six_scene_20261001` and `surface_attention_six_scene_20261001` except for the method:

- Scenes: Real_NRHints Cat/Pixiu, Synthetic_GS3 AnisoMetal/Translucent, Synthetic_SSS-GS bunny_small/dragon_small;
  full official train (`--fit-all`), full official test (66/71/400/400/500/500 frames).
- 30,000 steps, seed 0, native 3DGS from 20,000 random points, 400,000 cap, refinement until 25,000.
- 512 px Real/GS3, 256 px SSS; GS3 white background, others black; display gamma 2.2; SSS unit light intensity 1.
- Real scenes: training-camera rotation calibration with translation gauge (lr 1e-3 → 1e-5 from step 2000);
  synthetic scenes: no camera fitting. Test cameras and lights keep the original calibration; no test-time fitting.
- Method-specific: `--representation light_atlas` with defaults (512² atlas, 7 levels, 8 flux channels, neural
  visibility, atlas transport); light pass from step 2000 (`--shadow-start 2000 --port-start 2000`).
- Evaluation: `evaluate.py last.pt --split test --lpips --highlights --save-all` (uint8-quantized PSNR, SSIM
  11×11 σ1.5, VGG LPIPS on [-1,1]); last checkpoint only, no checkpoint selection on test.

## Entry points and provenance

- Canonical config `configs/validation.json` (name `light_atlas_six_scene_20261001`) →
  `bash launch_validation.sh` → `make_validation_manifest.py` → `run_benchmark.py` (tmux socket
  `port-validation-light_atlas_six_scene_20261001`).
- Output: `runs/light_atlas_six_scene_20261001/light_atlas/<family>/<scene>/` (`last.pt`, `history.jsonl`,
  `loss.png`, `test/metrics.json`, all `test/pair_*.png`); queue log and status beside the manifest.
- Source: `source.tar` in the run directory is the exact code. Git HEAD `bcb2f46` with uncommitted LiSA changes
  (`methods/light_atlas.py`, `methods/__init__.py`, `renderer.py`, `test_methods.py`, config, proposal);
  `git diff HEAD` SHA-256 `c00217b1…4087` recorded in `manifest.json`.
- Environment: `ssd-gs` (PyTorch 2.4.1, CUDA 12.1, gsplat 1.5.3), RTX 6000 Ada; workers on GPUs 2 and 3
  (the two GPUs assigned to this task), one serial job per GPU.

## Preflight (not part of the results)

Directory `runs/light_atlas_preflight_20261001/`.

1. `python -m unittest test_methods.LightAtlasTests` — 2 tests pass (cast-shadow visibility, gradients, linearity).
2. `test_method_integration.py --methods light_atlas --optimize-cameras` on a 4-frame Cat subset at 32 px —
   CLI train, final checkpoint, reload with identical render, corrected-fit CLI evaluation: PASS (`integration/`).
3. Geometry probe on the previous `directional_port_v1` AnisoMetal Gaussians (no training): the level-0 moment
   test places the dice shadows on the floor where GT has them; on lit pixels of five scenes it mislabels
   0.02–3.8% as shadowed (acne left for the learned residual).
4. AnisoMetal 3,000-step smoke run (light pass from 1000): crisp, correctly placed floor shadows after 2,000
   light-pass steps; 20-view test subset 23.40 dB at 3k steps (`aniso_3k/`). ~29 it/s at 110k Gaussians.

## Startup check (2026-10-01, JST)

Queue started; scheduler confirmed Cat (PID 218443, GPU 2) and Pixiu (PID 218444, GPU 3) at step 100.
Both passed step 2600 after the light pass and camera calibration switched on at 2000 (Cat l1 0.047 → 0.037,
Pixiu 0.035 → 0.022 across the switch); GPU utilization 48–69%, 4.4 GiB peak allocated. Remaining four jobs queued.

## Results (compact material head)

6/6 completed, 0 failed, 1,937 test frames; queue 2026-10-01 22:10–23:18 JST. Full table with all PORT methods and
baselines: `runs/light_atlas_six_scene_20261001/RESULTS.md` (CSV beside it).

| Scene | LiSA | directional_port_v1 | best other |
|---|---:|---:|---:|
| Cat | **25.300** / **0.8215** / 0.1659 | 25.258 / 0.8207 / 0.1673 | attention LPIPS 0.1647 |
| Pixiu | 26.247 / 0.8808 / **0.1024** | 26.320 / 0.8812 / 0.1056 | attention 26.388 / 0.8832 |
| AnisoMetal | 26.610 / 0.9566 / **0.0383** | 28.264 / 0.9581 / 0.0453 | attention 28.430 / 0.9602 |
| Translucent | 31.072 / **0.9767** / 0.0319 | 29.320 / 0.9651 / 0.0513 | GS³ 32.756 PSNR; SSD-GS 0.0302 LPIPS |
| bunny_small | **40.132 / 0.9913** / 0.0167 | 38.170 / 0.9879 / 0.0186 | SSD-GS LPIPS 0.0148 |
| dragon_small | **37.902 / 0.9830** / 0.0238 | 36.138 / 0.9766 / 0.0342 | SSD-GS LPIPS 0.0208 |
| **Six-scene mean** | **31.2105 / 0.93500 / 0.06316** | 30.5783 / 0.93162 / 0.07038 | SSD-GS 29.5412 / 0.91145 / 0.07871 |

Findings:

- Translucency/SSS: +1.75 dB (Translucent), +1.96 dB (bunny), +1.76 dB (dragon) over the default PORT method,
  improved on 80–97% of views and in every camera–light angle bin, including back-lighting. bunny/dragon also
  exceed SSD-GS's 100k-step results (+1.65/+0.52 dB).
- Real scenes: on par in PSNR (Cat +0.04, Pixiu −0.07), lower LPIPS; side/back-lit Cat views gain most.
  Fixed-calibration real-scene scores remain limited by the test-pose offsets documented in the calibration audit.
- AnisoMetal: −1.65 dB PSNR although LPIPS is the best of all methods and floor shadows are much sharper.
  Error concentrates on near-saturated metal reflections in front-lit views; the transport term carries only
  ~7% of metal radiance (vs ~50% on resin and fur), so the deficit lies in the compact local material head.
- Calibration-compensated diagnostic (secondary; `evaluate.py --shift-align 24`, translation fitted on GT, same
  procedure for both models; outputs in `runs/light_atlas_six_scene_20261001/diagnostics/`):

  | Scene | LiSA fixed → shift-aligned | directional_port_v1 fixed → shift-aligned | test offset (dy, dx) px |
  |---|---|---|---|
  | Cat | 25.300 → **29.324 / 0.9113 / 0.1491** | 25.258 → 29.262 / 0.9106 / 0.1504 | (+1.96, +0.42) vs (+2.00, +0.45) |
  | Pixiu | 26.247 → **32.857 / 0.9472 / 0.0864** | 26.320 → 32.221 / 0.9446 / 0.0908 | (−2.54, +0.68) vs (−2.55, +0.67) |

  Both models share the same test-pose offset (a calibration property), so the fixed-calibration metric hides
  LiSA's +0.64 dB appearance gain on Pixiu. These are the best aligned real-scene scores recorded in PORT-GS
  (previous best Cat 29.12, Pixiu 32.19; SSD-GS/GS³ Pixiu 30.90/30.75).
- Outliers: the two lowest bunny/dragon views per scene are data pose artifacts (identical scores for both methods).
  Four further bunny views where the default renders the front-lit bunny red (13.7–15.4 dB) are rendered correctly by
  LiSA (35.9–38.4 dB). Excluding every view below 25 dB for the default, LiSA still gains +1.74 dB (bunny) and
  +1.77 dB (dragon), so the improvement is not driven by outliers.
- Decomposition probes (scratch diagnostic) show physically sensible terms: visibility holds cast and attached
  shadows (fur self-shadowing on Cat), the transport term holds the resin/fur subsurface glow.

## Follow-up: material-head selection on held-out training lights

`methods/light_atlas.py` gained `--material-head {compact, reflect, spatial}` (compact = this run, default, so
these checkpoints still load and re-render bit-identically; reflect adds a reflection-vector encoding, 4 direction
bands, a 2048 highlight hint and a 4th hidden layer; spatial additionally adds an 8-band positional encoding) and
`--visibility-model gaussian` (ablation: per-Gaussian deep-shadow visibility splatted by the renderer).

Selection protocol, fixed before results: `runs/light_atlas_head_selection_20261001`, AnisoMetal and bunny_small,
30k steps, seed 0, trained WITHOUT `--fit-all` so whole 30-degree light-angle groups (~10% of train frames) are
held out; evaluation on that validation split only (no official test frame is rendered). Rule: choose the head
with the highest mean validation PSNR over the two scenes (LPIPS breaks ties within 0.05 dB); then rerun the full
six-scene test protocol with it, followed by ablations.

Selection result (`runs/light_atlas_head_selection_20261001/RESULTS.md`), validation PSNR / SSIM / LPIPS:

| Head | AnisoMetal | bunny_small | Mean PSNR |
|---|---:|---:|---:|
| compact | 27.537 / 0.9616 / 0.0367 | 38.123 / 0.9893 / 0.0213 | 32.830 |
| reflect | 27.666 / 0.9622 / 0.0364 | 38.049 / 0.9892 / 0.0212 | 32.858 |
| **spatial** | **28.046 / 0.9638 / 0.0351** | **38.236 / 0.9897 / 0.0192** | **33.141** |

`spatial` selected. Crops of AnisoMetal test views show why: the compact head renders brushed-metal faces too
smooth and dim, missing the spatially varying anisotropic streaks the positional encoding can represent.

## Final run and ablations

`runs/light_atlas_final_six_scene_20261002` (config name `light_atlas_final_six_scene_20261002`, started
2026-10-02 00:28 JST, GPUs 2/3, 24 jobs; source diff hash `cbc19504…` in the manifest). Variants, all with
`--material-head spatial` and the protocol above: `lisa` (full method), `no_transport` (`--light-transport none`),
`gaussian_visibility` (`--visibility-model gaussian`: per-Gaussian 64-bin deep-shadow visibility splatted as an
attribute, as in per-Gaussian shadow methods; the atlas still feeds the transfer term), `moment_visibility`
(`--visibility-model moment`: classical level-0 moment test, no learned residual). Startup confirmed:
Cat-lisa PID 230401 (GPU 2) and Pixiu-lisa PID 230400 (GPU 3) past step 100.

## Known limitation: brushed-metal anisotropic highlights (AnisoMetal)

With the spatial head AnisoMetal reaches 27.302 / 0.9598 / 0.0365 (compact 26.610 / 0.9566 / 0.0383), the best LPIPS
of all methods and above GS³ in SSIM/LPIPS, but 0.96 dB below `directional_port_v1` in PSNR (better on 7% of views;
largest deficit in front-lit views). Decomposition of test view 153: visibility is ≈1 on the dice faces (no acne
there) and the transfer term is ~6% of radiance; the missing energy is a bright, elongated specular band on the
brushed faces that the local head renders dimmer. Light-space transport cannot help here (the effect is local and
view-dependent). Candidate next step, not implemented: anisotropic spherical-Gaussian hints in a tangent frame
(Gaussian principal axis plus a learned per-Gaussian rotation), i.e. a classical anisotropic lobe bank feeding the
neural material, selected on held-out lights with the same protocol.

## Light-novelty analysis (why the six-scene test does not test light extrapolation)

Angle from each official test light to the nearest training light (directions from the scene origin): median
2.3° (Cat), 2.1° (Pixiu), 1.1° (AnisoMetal, Translucent) and 0.0° (bunny, dragon: test lights coincide with training
light positions; novelty is the camera–light combination). LiSA's per-view gain over `directional_port_v1` is
uncorrelated with this novelty (|r| ≤ 0.11). The six-scene gains therefore come from better shadow/translucency
modeling, not from extrapolation, and do not test the light-equivariance argument. The held-out light-group split
used for head selection is a real extrapolation test (validation-light novelty median 4.3° AnisoMetal, 11.2° bunny,
max 13.5°/15.9°); a matched comparison against `directional_port_v1` on that split follows the ablations.

## Final result (spatial head) — `runs/light_atlas_final_six_scene_20261002/RESULTS.md`

Six-scene mean **31.4588 / 0.93542 / 0.06215** (compact 31.2105 / 0.93500 / 0.06316; `directional_port_v1`
30.5783 / 0.93162 / 0.07038; SSD-GS 29.5412 / 0.91145 / 0.07871). Per scene: Cat 25.231 / 0.8197 / 0.1645,
Pixiu 26.188 / 0.8811 / 0.1019, AnisoMetal 27.302 / 0.9598 / 0.0365, Translucent 31.155 / 0.9767 / 0.0321,
bunny 40.861 / 0.9919 / 0.0144 (best of all methods on all three metrics), dragon 38.017 / 0.9833 / 0.0235.
Figures: `docs/figures/lisa_qualitative_20261002.png` (GT / default / LiSA on Translucent #45, AnisoMetal #308,
bunny #413, dragon #273) and `docs/figures/lisa_decomposition_20261002.png` (GT, render, V, local, transport).

### Decomposition trade-off of the spatial head

Mean visibility and transport share of linear radiance on probe views (compact → spatial): Cat #44 V 0.56 → 0.60,
transport 50% → 12%; Pixiu #18 V 0.95 → **1.00**, transport 50% → 11%; AnisoMetal #308 V 0.81 → 0.84, transport 9% → 8%;
dragon #273 V 0.74 → 0.74, transport 40% → 32%. The positional encoding lets the local term absorb part of the
light-dependent appearance: on Pixiu (real capture, calibration noise) the learned visibility switched off and the cast
shadow on the box is reproduced by the local term. Metrics are unaffected (real-scene PSNR within ±0.07 dB of compact),
and held-out-light validation still favoured spatial on AnisoMetal/bunny, but the decomposition is less physical.
Report both heads; a principled fix is to give positional capacity only to light-independent material parameters
(e.g. spatially varying weights of light-dependent lobes computed without position) so that shadows and transport
cannot be baked. The held-out light comparison below also runs the compact head on Translucent/dragon to test whether
this baking costs light generalization.

## Historical registration: held-out light comparison and `svbrdf` head (registered 2026-10-02 01:45 JST, before any result)

`light_holdout_generalization_20261002` launches automatically after the final queue (same held-out light-group
protocol as the head selection; validation split only, no test frames). Variants: `lisa_svbrdf` (new head, prototyped
in an isolated copy while the queues ran: position enters only through 8 light-independent coefficients that scale
light-dependent responses computed without position) on Pixiu, AnisoMetal, Translucent, bunny, dragon; `lisa_spatial`
and `lisa_compact` on Pixiu, Translucent, dragon (AnisoMetal/bunny reused from the head selection);
`directional_port_v1` on the four synthetic scenes.

Decision rule: `svbrdf` replaces `spatial` as the final head only if (a) its held-out PSNR averaged over the four
synthetic scenes is at least the spatial average minus 0.1 dB and (b) its learned visibility on held-out Pixiu views is
not collapsed (mean V < 0.98 on probe views); otherwise `spatial` stays final. If adopted, `svbrdf` is run once on the
six-scene test protocol. The generalization question is answered by LiSA vs `directional_port_v1` on the same held-out
lights.

## Ablations (final protocol, spatial head; `runs/light_atlas_final_six_scene_20261002/RESULTS.md`)

| Variant | Mean PSNR | Mean SSIM | Mean LPIPS | Largest per-scene effect |
|---|---:|---:|---:|---|
| full LiSA | **31.459** | **0.9354** | **0.0622** | — |
| no light-space transport | 30.797 | 0.9322 | 0.0674 | bunny −2.67 dB, AnisoMetal −0.72, dragon −0.73; Translucent +0.31 PSNR (SSIM/LPIPS worse) |
| per-Gaussian deep-shadow visibility | 31.351 | 0.9349 | 0.0630 | bunny −0.51 dB, Translucent −0.29 (but +0.6–0.8 at 30–90° lighting), ~1.35x slower |
| moment test without learned residual | 31.283 | 0.9342 | 0.0635 | bunny −0.87 dB |

All three components contribute on average; the light-space transport contributes most, concentrated on the
subsurface-scattering scenes. Real scenes are insensitive to every ablation (within ±0.15 dB), consistent with their
calibration-limited fixed-test metric. The Translucent transport result and the side-lighting visibility result are
documented weaknesses.

The held-out light comparison (`light_holdout_generalization_20261002`, 15 jobs) started 2026-10-02 05:20 JST after
the chained launcher installed the `svbrdf` head (unit tests 4/4) — startup confirmed: Pixiu-svbrdf PID 251879 (GPU 2),
AnisoMetal-svbrdf PID 251880 (GPU 3).

## Decision: `svbrdf` head adopted (2026-10-02 06:48 JST, before any svbrdf test result)

Held-out light PSNR / SSIM / LPIPS (validation split; AnisoMetal/bunny spatial and compact from the head selection):

| Head | AnisoMetal | Translucent | bunny_small | dragon_small | 4-scene mean | Pixiu (held-out) | Pixiu mean V |
|---|---:|---:|---:|---:|---:|---:|---:|
| svbrdf | **28.220** / 0.9639 / 0.0353 | 29.483 / 0.9708 / 0.0412 | 37.988 / 0.9891 / 0.0216 | 37.488 / 0.9817 / 0.0260 | 33.295 | 20.351 | 0.84 |
| spatial | 28.046 / 0.9638 / 0.0351 | **29.565** / 0.9712 / 0.0406 | **38.236** / 0.9897 / 0.0192 | **37.613** / 0.9819 / 0.0250 | **33.365** | 20.369 | 0.91 |
| compact | 27.537 / 0.9616 / 0.0367 | 29.642（最终值） | 38.123 / 0.9893 / 0.0213 | 37.519（最终值） | 33.205 | 当时未记录 | — |

Rule check: (a) 33.295 ≥ 33.365 − 0.1 ✓; (b) svbrdf Pixiu held-out visibility 0.84 < 0.98 ✓ → `svbrdf` replaces
`spatial` as the final head. Note the spatial head did not collapse V on this Pixiu run (0.91), so the earlier collapse
depends on the training set/optimization, not only on the head. `svbrdf` runs once on the six-scene test protocol
(`light_atlas_svbrdf_six_scene_20261002`, chained after the held-out queue). The spatial-head six-scene results and
ablations remain valid component evidence.

## Held-out light generalization (`runs/light_holdout_generalization_20261002/RESULTS.md`)

Held-out 30° light-angle groups (novelty median 4.3–11.2°, max ~16°; official test lights are within 0–2°):

| Method | AnisoMetal | Translucent | bunny_small | dragon_small | 4-scene mean PSNR |
|---|---:|---:|---:|---:|---:|
| `directional_port_v1` | 28.250 | 27.929 | 35.456 | 34.730 | 31.591 |
| LiSA svbrdf (final head) | 28.220 | 29.483 | 37.988 | 37.488 | 33.295 |
| LiSA spatial | 28.046 | 29.565 | 38.236 | 37.613 | 33.365 |
| LiSA compact | 27.537 | 29.642 | 38.123 | 37.519 | 33.205 |

LiSA leads by +1.6–1.8 dB on novel light directions (dragon +2.76, bunny +2.53, Translucent +1.55 for svbrdf), with
better SSIM/LPIPS on every scene; the AnisoMetal test deficit vanishes (−0.03 dB). This is the evidence for the
light-space operator's generalization that the official test split could not provide. The svbrdf six-scene test run
(`light_atlas_svbrdf_six_scene_20261002`) started 10:09 JST; startup confirmed (Cat PID 268218, GPU 2; Pixiu PID 268220,
GPU 3).

## 最终svbrdf配置

正式结果现统一于[全18场景记录](results.md)，不再单列旧六场景排名。

## Final checks (2026-10-02 11:40 JST)

`python -m unittest test_methods`: 92 tests pass (GPU 2). Registering `light_atlas` had made two generic CPU/float64
method tests (`test_method_selection_and_config_roundtrip`, `test_zero_light_linearity_and_optimization`) call its
CUDA-only light pass; they now evaluate light-space methods with CUDA float32 inputs (same round-trip bit-identity,
intensity linearity, zero-light and gradient checks) and skip them only when CUDA is unavailable. End-to-end coverage of
the final head is the six-scene svbrdf run (train, checkpoint, full official-test evaluation, 0 failures).
GPUs 2/3 released; no PORT-GS process is running (only the launchers' tmux servers with exited panes remain, as for
earlier runs).
