# Revision log after the mock CVPR 2027 reviews (2026-10-06)

Reviews: `reviewer_1.md` (OLAT relighting expert), `reviewer_2.md` (rendering / light transport),
`reviewer_3.md` (methodology and clarity). All three rated the first draft **3 (Borderline Reject)**
with confidence 4. Their concerns overlap strongly; they are grouped below by theme, with the
change made (or the experiment that remains to be run).

## A. Framing and claims

| Concern | Raised by | Change |
|---|---|---|
| Abstract/intro/Fig. 1 make the atlas transfer central although a capacity-matched local MLP matches it | R1, R2, R3 | Abstract and intro now state the limit of the transfer term explicitly; contribution 1 is "light-space visibility and transport"; Fig. 1(e) caption calls the transfer panel a learned split; Sec. 5.3 says the margin is *not* attributed to transfer and states the visibility/schedule attribution as a hypothesis for E3/E4 |
| Headline margin concentrated (real captures, averaging across benchmarks) | R1, R2, R3 | Intro and Sec. 5.2 report per-benchmark margins (+2.0 real, +0.4 GS3, -0.1 SSS vs OLAT-GS), the 46% share from Cat/Fish/Pixiu, the 3-3 split with GS3 on GS3 scenes, synthetic-only means (+1.77 dB vs SSD-GS, p=0.014; vs OLAT-GS p=0.12), Wilcoxon tests; the remaining positive claim is consistency (only method in the top two on all benchmarks, best LPIPS on all three) |
| Loss-domain claim overgeneralized; "not model capacity" untested; GS3/SSD-GS already anneal HDR targets | R1, R2, R3 | Scoped to "in our model"; "not model capacity" removed; GS3 and SSD-GS credited for HDR annealing; study limits stated (2 HDR scenes, 1 seed, 16k); control described exactly; E4 extended (offset in Gamma, RawNeRF weighting, schedule applied to a baseline) |
| Light-relativity asserted, not shown | R1, R2, R3 | Stated as a design property whose benefit is the subject of E1; E1 now runs on the ablation panel (official-test column filled with existing results) with a harder contiguous-cap split |
| Shadow gains may come from per-pixel stage III rather than the atlas | R1, R2, R3 | Sec. 3.4 states that Gaussian receivers inherit the per-primitive limitation; E3 now compares per-Gaussian visibility and an RNG-style hard shadow map at identical receivers (plus Chebyshev/beta sweep) |
| Formulaic sentences ("literally that of the light", "A good representation turned out not to be enough", "The culprit was...", "more than a shadow map", "as important as the representation", rhetorical question, Fig. 4 title) | R2, R3 | All rewritten or removed |
| Development story reads like a lab notebook | R1 | Reduced to one sentence in the intro; details in supplement S2 |
| Third contribution is only an evaluation | R1 | Replaced by the analysis contribution, including the benchmark light-novelty finding |

## B. Technical correctness (mainly R2)

| Issue | Change |
|---|---|
| "every pixel carries the same flux" is wrong (d omega ~ cos^3) | Abstract/intro now speak of fractions of each light pixel's flux; Sec. 3.2 defines the atlas as the deposit per unit incident flux and quantifies the solid-angle variation (<30% typical, up to 2.1x at the widest atlas corners; half-angles 8-30 deg measured on all 18 models) |
| "exactly the fractions" ignores sorting, dilation, alpha clamping | Qualified "within the rasterized splatting model" with citations (StopThePop, error analysis, volumetric consistency, Mip-Splatting); supplement names gsplat's classic mode |
| E(x) used with and without cosine | Normal-incidence irradiance renamed E-bar; identity written with cos(theta) |
| Moment test with beta = 3 under-shadows mixed footprints (half-transparent occluder: 0.98 instead of 0.5; Chebyshev exact) | Stated in Sec. 3.3 with the example, listed in limitations; E3 adds Chebyshev test and beta sweep |
| Center depths re-introduce a per-primitive approximation | Stated in Sec. 3.3 and limitations |
| Factoring the receiver's falloff out of the transfer term is off by (r_out/r_in)^2 | Stated in Sec. 3.3 with measured light-distance ranges (constant on synthetic data, within 1.8x on real captures) |
| "Linear in the deposited flux" imprecise; TSM "special case" wrong | Now "for a given geometry and light, linear in the deposited features"; TSM described as related, not a special case; possible double counting of direct light noted (R1 Q7) |
| Eq. 8 argument | Sharpened: l1 gradient magnitude independent of residual size; pure power law stated; untested remedies listed |
| Notation overload (s, h, f, mu, C, i) | Spread eta_k, descriptor q_k/Q, mean depth z-bar_k, display transform Gamma, networks G and K, composite c-bar, entry/exit x_in/x_out |
| Eq. 6 overflow; Fig. 2 "rho, s = MLP" | Eq. 6 split into an align block; Fig. 2 labels corrected |
| "Penumbrae" for a point light | Replaced by soft boundaries from filtering, sub-texel occluders and finite real lights |

## C. Experimental rigor (mainly R3, R1)

| Issue | Change |
|---|---|
| Single-seed rows and "every seed" (n = 3) | Single-seed rows labeled as suggestive; "positive on all three seeds but not significant"; E3 adds three seeds for them; E7 adds three seeds for all 18 scenes |
| Fig. 3 selection effects (largest-margin scenes, Fish confounded, Statue at 75th pct) | Rows now: two shadow scenes, a real capture where methods are close (Pikachu), and our worst synthetic scene (Drums); all at the median view; Fish, Statue, Pixiu moved to the supplement with the registration caveat |
| Mask parity | Stated: all methods use the masks (baselines to composite targets, OLAT-GS also for its surface stage, LiSA also for an opacity loss); foreground-masked PSNR over all 5,691 views (supp. Tab. S8) preserves the ranking within every benchmark (LiSA +1.7 dB real, +0.4 dB GS3; OLAT-GS +0.25 dB on SSS) |
| Suspicious baseline runs (RNG, OLAT-GS < 21 dB) | Disclosed in Sec. 5.1; E6 (reproduce one published configuration per baseline) added |
| Registration confound on real captures | Evaluation-only E2(b) computed for all methods and all 791 real test views (best 2D shift within 24 px, supp. Tab. S7): baselines need 4.0-4.6 px vs 1.7 px for LiSA; LiSA's real-capture lead shrinks from 2.0 to 0.7 dB (still ahead on Cat, Pixiu, FurScene; behind on CupFabric, CatSmall, Fish, Pikachu). Stated in the intro and Sec. 5.2. E2(a) and rotation-only alignment remain to be run |
| Use of the benchmarks during development | Disclosed in Sec. 5.1 and supplement S2 |
| Scene subsets without rationale | Ablation panel rationale stated (materials + weakest scenes); E1 moved to the same panel |
| Missing related work (RNG's forward/deferred switch and depth refinement, neural transmittance, NRTF, PRTGS, Gaussian shadow casting, deferred reflection, classical opacity/moment/translucency maps) | Added and credited |

## D. Items deliberately left as placeholders

E1 (held-out lights), E2(a) (retraining without pose refinement), E3 (visibility variants), E4
(optimization ablations), E5 (SSS-GS, NRHints, BiGS), E6 (baseline reproduction), E7 (seeds on all
18 scenes). Their protocols are in supplement S5 and `paper/README.md`.

## E. Layout consequences

To stay within 8 pages after adding the requested caveats and statistics, the cost table and the
refinement table moved to the supplement (Tabs. S5, S6) with their key numbers quoted in the text, Fig. 3
shrank slightly, and the conclusion became a two-sentence summary.
