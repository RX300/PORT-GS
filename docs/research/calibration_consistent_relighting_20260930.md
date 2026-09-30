# Calibration-consistent relighting: diagnosis, method and research direction (2026-09-30)

## 1. Where the real-scene gap comes from

The [calibration audit](../experiments/calibration_audit_20260930.md) answers the question the previous
representation rounds could not: on Real_NRHints Cat/Pixiu the dominant error is geometric registration.

- Official training poses: 4-6 px per-frame error at 512 px (f ~ 3100 px, so 0.1-0.2 deg).
- Official test poses: small scatter but a nearly constant image-space offset relative to the training frame
  (Pixiu about (-2.7, -9.3) px, Cat about (+2.5, -1.3) px, fitted constant term; azimuth-dependent part <= 3 px).
- Pose-refining baselines (GS3, SSD-GS) drift their reconstruction frame while fixing training blur, which
  is why they score 18-19 dB on Cat under fixed calibration. Their published real-scene numbers come from
  test-view camera/light fitting (SSD-GS code), so they are not comparable with fixed-calibration numbers.
- PORT never corrected training poses. Every PORT model therefore averaged misaligned observations; after
  removing each test view's global offset it still trailed GS3/SSD-GS by 1.7 dB (Cat) / 3.4 dB (Pixiu).
  The trailing error is spread over silhouettes (3.3x GS3 MSE on Pixiu), highlights (2.4x) and interior (1.8x).
- Model selection since 2026-09-12 relied on fixed-calibration test differences of 0.1-1 dB, smaller than
  the alignment noise between variants (local transport: best fixed, worst aligned). Those decisions should
  be revisited under calibration-compensated metrics.

Secondary observations: per-view best linear gain between render and GT has 7-13% std on both scenes
(flash power and exposure variation or shading error), and older PORT renders are 6-8% too dark in linear
intensity on average. Input blur is not a dominant factor (no blurred subset in either scene).

## 2. Implemented improvement: calibration-consistent joint training

1. **Center-preserving rotational self-calibration.** For telephoto captures (9 deg field of view) image-plane
   registration errors are rotation-dominated. Each fit camera rotates about its calibrated optical center
   (`viewmat' = [dR|0] viewmat`); there is no anchor camera, and a shared rotation of all cameras is kept
   because it is observable when cameras surround the object. Sparse per-view Adam.
2. **Translation-gauge projection.** With fixed centers and free rotations, a global scene translation can be
   absorbed by re-aiming every camera. After every camera step the corrections are projected off this
   3-DoF gauge (`w <- w - C d*`) and the scene is translated by `-d*`, which leaves every fit view's object-
   center image unchanged. This pins the reconstruction to the calibrated frame instead of letting gradient
   noise choose it. This drift is what moved GS3/SSD-GS by 5-9 px on the test views.
3. **Per-frame light-position offsets** (optional), sparse, regularized, active after shadows and ports.
4. **Evaluation**: primary fixed calibration; shift-aligned metrics as the like-for-like appearance measure
   across methods; test-time calibrated metrics (scene frozen) for comparison with published numbers.

Novelty assessment (honest): joint pose refinement is standard (NeRF--/BARF, NRHints camera optimization,
GS3/SSD-GS cam/light optimization, 3R-GS global drift correction). The distinct parts are the
center-preserving parametrization with an explicit translation-gauge projection that keeps a relightable
model in the evaluation frame, and the benchmark finding itself (split-level test offset, frame drift of
pose-refining baselines, published protocol fitting test calibration). This is an enabling contribution and
an evaluation contribution. On its own it is not the paper's central novelty.

## 3. Proposal and research direction (after the round-1 evidence)

Round 1 (rotation camera fit, no gauge) already changes the ranking: Pixiu neural reaches fixed 24.07 / .8641 /
.1162 (above GS3/SSD-GS on all three metrics) and 32.05 dB under the literature protocol (SSD-GS's own
paper-protocol renders: 31.17 with the same metric code); Cat default reaches 29.12 / .9096 / .1506 aligned,
above SSD-GS's paper-protocol 27.25 / .9001 / .1661. The same run collapses the Cat fixed score (15.9-17.4 dB)
because the shared pitch correction drifted the frame by 8-12 px: the gauge problem is not hypothetical.

### Proposed paper-level contribution: gauge-consistent self-calibration + calibration-disentangled evaluation

Claim to test: *pose-refined relightable reconstructions can be evaluated at the original test calibration,
with no test-time fitting, if the refinement keeps the reconstruction in the calibrated frame.*
Center-preserving rotations remove the 7-DoF world gauge except a 3-DoF translation that per-view re-aiming
absorbs to first order; projecting that out after every step (closed form, image-preserving) fixes it.
What is new relative to published relighting systems: GS3/SSD-GS/NRHints refine cameras without a gauge and
then fit test cameras on test images; 3R-GS corrects global drift with a learned refiner in pose-free NVS but
does not preserve a calibrated evaluation frame. Evidence required: fixed-calibration scores restored on Cat
while keeping the aligned-quality gain (round 2), same trend for both representations (round 2b).

Novelty risk: moderate. Gauge freedom is textbook bundle adjustment; the contribution is the diagnosis in
relighting benchmarks (split-level test offsets, frame drift of pose-refining baselines, published protocols
that fit test calibration), a minimal fix that makes strict evaluation meaningful, and SOTA results under
three protocols. It should be framed as an analysis-plus-method paper, not as a new reflectance model.

### Representation direction (open)

- Side/back-lit transport is the remaining appearance gap (PORT's advantage over GS3 drops to -0.2 dB on Cat
  and +0.4 dB on Pixiu for light-camera angles > 90 deg; dark blotch in side-lit Cat views). A translucent-
  shadow-map style term using PORT's existing deep shadow volume (entry depth -> traversed thickness -> learned
  transmission kernel) is cheap, but overlaps conceptually with SSD-GS's dipole scattering module and SSS-GS; it
  would need a distinct, testable claim before being proposed as novel.
- PORT's own claim (nonlocal directional ports), tested for the first time without calibration noise: removing the
  ports under the final settings costs 0.72 dB (Pixiu, translucent) and 0.17 dB (Cat) in calibration-compensated
  PSNR. The ports are a real contribution and a natural place to address the side/back-lit gap.
- Base choice resolved: the default directional base beats the frozen neural BRDF on both scenes once calibrated
  (final R2b vs R2: Cat 29.18 vs 28.15, Pixiu 32.31 vs 31.94 aligned). Per-frame light offsets add nothing measurable.
- Evidence for the proposed claim is now complete: with the translation gauge, fixed-calibration scores are restored
  (Cat 15.9 -> 24.8-25.2 dB) while the calibration-compensated gain is kept, for both representations.
