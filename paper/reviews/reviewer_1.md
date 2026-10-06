# CVPR 2027 Review: Reviewer 1

**Paper:** LiSA: Light-Space Atlases for Relightable Gaussian Splatting

## 1. Summary

LiSA rasterizes the Gaussians once per point light, from the light, and reads the compositing weights as the flux each primitive intercepts. Depth moments and eight learned flux features form a 512² light-space buffer and a 7-level pyramid. Receivers (Gaussian centers in training, pixels later) read an MLP-refined variance-shadow-map test (Eqs. 4–5) and a transfer term linear in the gathered flux (Eq. 6). A three-stage schedule moves the loss from display to linear radiance. Against four retrained baselines on 18 scenes, LiSA reports 33.47 vs. 31.59 dB PSNR and 2.6–5× shorter training.

## 2. Strengths

- The formulation is clean and physically motivated. Its inputs are light-relative, and its filtered shadow test is differentiable at any receiver.
- The paper is unusually candid. It includes a local-MLP control for the transfer term (Tabs. 2–3), a registration caveat (l. 398–404), its development history (Tab. S3) and failure cases (Fig. S3). All 5,691 test images are scored against common 8-bit targets, and the light-novelty statistic (Tab. S2) is useful.
- The method is practical: about 21 minutes of training, interactive rendering, and sharper cast shadows on Hotdog and Translucent (Fig. 3).

## 3. Weaknesses

1. **Most of the headline margin comes from a confounded subset plus averaging** (l. 19–22; Tabs. 1, S1). Against each benchmark's strongest baseline, LiSA gains +2.00 dB on real captures, +0.43 dB on GS³ scenes and −0.11 dB on SSS scenes. The 1.89 dB margin over SSD-GS breaks down as:
   - 0.88 dB from Cat, Fish and Pixiu, where all baselines score 18–24 dB while the non-RNG baselines reach 31–36 dB on CatSmall and CupFabric;
   - 0.79 dB from the SSS scenes, where OLAT-GS beats LiSA;
   - 0.22 dB from the remaining ten scenes.

   On the four unconfounded real captures, LiSA trails GS³ and SSD-GS by about 0.3 dB. Yet Fig. 3 showcases Fish, whose SSD-GS and GS³ crops look equally sharp but score about 5 dB lower on a glittery mat, suggesting misalignment.
2. **Registration is handled asymmetrically** (l. 335–340, 359–364; S1 l. 057–067). LiSA's gauge-fixed refinement stays in the calibrated frame by construction. The baselines refine without an anchor and lose the published protocols' test-pose fitting. Mask parity is unclear too. LiSA adds hull initialization, a mask loss (l. 325–326) and gated background supervision (Tab. S3), which full-image metrics on black backgrounds reward.
3. **The central thesis is untested.** The transfer term, half of contribution 1, ties a capacity-matched local residual (31.54 vs. 31.51 dB). It loses on Soap despite a 58% share and wins only on Translucent (+0.41 dB). Visibility remains, but E3 is TBD and the K=1/g≡0 ablations are single-seed. Stage II shades Gaussian centers (l. 271–273), so the outlines in Fig. 3 may come from stage-III deferred shading, not the atlas. Yet Tab. 5 favors Gaussian receivers (34.24 vs. 34.05 dB).
4. **The novelty is incremental.** GS³ and SSD-GS already splat from the light. The "flux deposit" reading (l. 49–55, 499–501) is the emission–absorption semantics of deep and opacity shadow maps [24, 28]. RNG refines a shadow-map cue with a network and switches from forward to deferred shading, and NRHints feeds per-pixel shadow cues to a network. "Linear in the deposited flux" (l. 154, 238) is conditional, since W_k depends on the light through S and n·l. That is not PRT-style linearity.
5. **The loss-domain claim is overgeneralized** (l. 16–19, 107–110; Fig. 4 title). It rests on two HDR scenes, 16k iterations and one model. GS³ and SSD-GS reach 29.7–31.5 dB on Lego and Drums, so the collapse may be specific to the product of factors in Eq. (7). Against the revised version (Tab. S3; other changes also differ), staging gains 3.76 dB on GS³ scenes but loses 0.2 dB on real and 0.4 dB on SSS scenes. "Not model capacity" (l. 109) is never tested. RawNeRF argues the opposite, so intermediate weightings deserve study.
6. **Statistics, selection and baselines.** With one run per method and scene (l. 373) and seed spread up to 1.5 dB (l. 438–439), the win counts (l. 377–379) are within noise. Tab. S3 suggests design choices were scored on official test views, while the baselines ran untuned defaults. RNG (15–18 dB on Drums, Lego, FurBall and Pikachu) and OLAT-GS (about 20 dB on Drums and Lego) look like failed runs, unchecked against published numbers. SSS-GS and NRHints are absent from their own benchmarks.
7. **Generalization to new lights is unsupported** (l. 261–263, 442–445). Median test-light novelty is 0–2.3°, and on SSS-GS, test lights coincide with training lights (Tab. S2).

## 4. Questions for the authors

1. Do all methods receive identical masks? Please report foreground-masked metrics.
2. How do the baselines score without pose refinement, or with your gauge fix?
3. Did the display-domain run in Fig. 4 use the warm-up, and which loss domain do GS³ and SSD-GS use on HDR data?
4. How much of the Hotdog and Translucent gains remains if stage III uses Gaussian receivers?
5. Budgets differ (38k vs. 60–100k iterations). How many images per iteration does each method use, and how does quality compare at matched wall-clock time?

## 5. Missing or misrepresented related work

- **"RNG: Relightable Neural Gaussians" [14]** is cited only for its shadow-map cue (l. 39, 123–124). Its depth refinement network and hybrid forward-deferred fitting are the closest precedents for Eq. (5) and Sec. 3.4.
- "Learning Neural Transmittance for Efficient Rendering of Reflectance Fields" (Shafiei et al., BMVC 2021).
- "Neural Radiance Transfer Fields for Relightable Novel-view Synthesis with Global Illumination" (Lyu et al., ECCV 2022).
- "PRTGS: Precomputed Radiance Transfer of Gaussian Splats for Real-Time High-Quality Relighting" (Guo et al., ACM MM 2024).
- "Fourier Opacity Mapping" (Jansen & Bavoil, I3D 2010) and "Deep Opacity Maps" (Yuksel & Keyser, EG 2008).
- "Real-Time Realistic Skin Translucency" (Jimenez et al., IEEE CG&A 2010), whose shadow-map thickness is the classical counterpart of δ_k.

## 6. Presentation and writing

- Eq. (6) overflows into the right column, over the Sec. 3.4 heading (p. 4).
- Overloaded symbols: s (l. 211, 219, 251), h (l. 174, 237) and f (l. 169, 283).
- Lines 188 and 229 write "E dA", but E = I/r² (l. 176); it should be E cos θ dA.
- In Fig. 2, "ρ, s = MLP(f, n, l, v, h, r)" conflicts with l. 251–253 and S1, where the lobe weights are light-independent and come from a separate network.
- The caption of Fig. 1(e) attributes the transfer to the vase and dice, but the panel also lights the opaque floor, and Tab. S2 reports 6–9% transfer on opaque objects.
- The development narrative (l. 74–94) belongs in the supplement. Contribution 3 is an evaluation, not a contribution.
- Formulaic or padded passages:
  - the abstract's opening triad, repeated at l. 32–35;
  - "at what depth, and by what" (l. 55);
  - "What favors the light-relative design should show…" (l. 442–443);
  - the conclusion's restatement of the abstract;
  - "proved as important as the representation" (l. 505), which the results do not support.

## 7. Essential vs. optional planned experiments

- **E3 (essential)** is the only direct test of the thesis. Add Hotdog, Lego and a real capture. Composite per-Gaussian visibility in stage III, and add an RNG-style per-pixel hard shadow-map arm.
- **E2 (essential).** Make the rotation-only test-pose alignment mandatory and identical for all methods. Run the baselines without pose refinement, or with your gauge fix, and report the recovered offsets. A 2D shift of up to 24 px can fit textured backgrounds.
- **E1 (essential if the light-relative claims stay).** Held-out lights lie only 4–11° (median) from training lights (S5 l. 139–140). Add a contiguous held-out cap at least 20° away. Use three seeds for LiSA and the local residual, and pair Pixiu with E2.
- **E4 (essential for contribution 2).** Also apply the schedule to GS³ or SSD-GS.
- **E5 (optional).** Report the two SSS-GS scenes that converged now.

## 8. Preliminary rating

**3: Borderline Reject.** Confidence: 4/5.

The system is competent, fast and honestly analyzed, with better shadows on synthetic scenes. However, its representation claims are untested (E3, E1) or contradicted by its own transfer ablation. The headline margin mostly reflects registration-confounded captures and averaging across benchmarks. On unconfounded data the gain is +0.43 dB on GS³ and −0.11 dB on SSS.

## 9. What would most change my rating

1. E3 showing a multi-seed advantage for atlas visibility on shadow-dominated scenes.
2. E2 under symmetric registration that preserves a lead on real captures.
3. Per-benchmark margins in the abstract and introduction, with the transfer term demoted or backed by E1.
4. E4, plus the schedule applied to one baseline.
5. Validated baseline reproductions and confirmed mask parity.
