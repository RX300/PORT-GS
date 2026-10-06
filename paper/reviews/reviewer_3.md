# Review of "LiSA: Light-Space Atlases for Relightable Gaussian Splatting" (Reviewer 3)

## 1. Summary

LiSA rasterizes the Gaussians once per point light and reads the compositing weights as intercepted flux. It stores depth moments and eight learned flux channels in a light-space atlas pyramid. Receivers read the pyramid to get a moment shadow test, refined by an MLP, and a "transfer" term that is linear in the gathered flux. A staged schedule moves the loss from display to linear radiance while the geometry forms. On 18 OLAT scenes, LiSA averages 33.47 dB against 31.59 dB for the strongest baseline, and trains in about 21 minutes. The paper itself finds that a capacity-matched local network matches the atlas transfer on average. Experiments E1–E4 have not been run.

## 2. Strengths

- **A simple, well-motivated idea.** Eq. 2 needs only one 512² pass per light.
- **A careful re-evaluation.** All 5,691 test images are scored against common targets, with no test-pose fitting. Per-scene results and timings are reported (Tabs. S1, S4), and the baselines get more iterations than LiSA.
- **Unusual candour.** The paper includes a capacity-matched control, paired seed differences, failure cases, its development history and light-novelty statistics. The E1–E5 protocols are fixed in advance (S5).
- **A clean result in Fig. 4, and fast training.** Training is 1.6–5.5× faster than the baselines (Tab. 6).

## 3. Weaknesses

1. **The framing contradicts the transfer results.** The abstract (l. 012–015), Fig. 1(e), the introduction (l. 043–046) and contribution 1 (l. 102–106) make the atlas transfer central. Yet the local residual performs on par (31.54 vs. 31.51 dB, Tab. 2), and it is exactly the kind of per-primitive network the introduction says "must memorize, light by light". It also wins on every seed on Soap (−0.29±0.11; my 95% CI is ≈ [−0.56, −0.02]). The only positive scene, Translucent (+0.41±0.26, n=3), has a CI of ≈ [−0.24, 1.06]. In Fig. S2 the local residual is even ahead on the displayed view (34.55 vs. 34.38 dB). Even so, Fig. 1 showcases Translucent and reads panel (e) physically, which l. 488–489 disclaims.

2. **The fallback attribution is untested.** L. 440–442 credits light-space visibility and the schedule, but E3 and E4 are TBD.
   - The single-seed K=1 and g≡0 rows ablate refinements of the shadow test, not atlas visibility against per-primitive visibility.
   - Geometry forms with per-Gaussian visibility (stage II, l. 269–278).
   - Pixel receivers lower PSNR in Tab. 5 (34.05 vs. 34.24 dB). That table uses Lego, Drums and Soap rather than the shadow scenes of Fig. 3.
   - RNG's shadow-map crops in Fig. 3 also look unfragmented.

3. **The margin is concentrated, and the seed reasoning is optimistic.** The following is my own analysis of Tab. S1.
   - LiSA beats each baseline across scenes (Wilcoxon p ≈ 0.003–0.016). However, 46% of its 1.89 dB margin over SSD-GS comes from Cat, Fish and Pixiu, where registration is suspected (l. 401–404).
   - On the six GS3 scenes, LiSA and GS3 split 3–3 in PSNR (median +0.09 dB), though LPIPS favours LiSA on 5 of 6.
   - Against the per-scene best baseline, LiSA wins 9 of 18, with a median margin of ≈ 0 dB.
   - Per-scene seed spread reaches 1.5 dB (l. 438–439), so single-run per-scene claims are uninterpretable.
   - Tab. S1 shows 12 wins, 1 tie and 5 losses against SSD-GS, not the 13 wins stated at l. 379.
   - An SD of 0.11 dB from three seeds has a 95% CI of ≈ 0.06–0.69 dB. Single-seed rows therefore cannot show that "both … matter" (l. 413–416).
   - "Every seed" with n=3 has a sign-test p of 0.25.

4. **The loss-domain study is narrow.** It uses two failure-selected HDR scenes, 16k of 38k iterations, no stage III, held-out training views and one seed (l. 451–458). It shows that the loss domain can decide collapse in this model. It does not show "not model capacity" (l. 109–110), since capacity was never varied, nor does it support the general claims at l. 159–161 and l. 505–506. In Tab. S3, the display-domain variant even leads on real and SSS scenes, although that comparison is confounded. Simple remedies for the singularity in Eq. (8), such as an ε offset, clipping or RawNeRF-style weighting, are untested.

5. **Selection effects.**
   - Fig. 3 shows the two largest GS3 margins and the largest real and SSS margins, with Statue at a 75th-percentile view.
   - It includes Fish, which the paper suspects of a registration effect, and failure cases appear only in Fig. S3.
   - Five different scene subsets are used without a stated rationale.
   - Development versions were scored on the test sets (Tab. S3).

6. **The benchmarks do not test what the design is for.** Test lights lie a median of 1.1° from the nearest training light (0.0° on SSS-GS, Tab. S2). The regime the design "was built for" (l. 512–513) awaits E1.

## 4. Assessment of main claims

| Claim | Evidence | Sufficient? |
|---|---|---|
| Atlas beats per-primitive visibility (l. 099–100) | Fig. 3; single-seed rows | No: E3 not run |
| Transfer carries light through objects (Fig. 1) | w/o transfer: −1.37 dB | No: a local residual matches it; the ablation only removes a radiance path |
| Loss domain, not capacity, caused failures | 2 scenes, 16k iterations | Partly: capacity and generality untested |
| Best average accuracy over 18 scenes | Tabs. 1, S1; one seed | Mostly: consistent, but concentrated and confounded by registration |
| About 20 minutes of training | Tabs. 6, S4 | Yes |
| Loss domain is "as important as the representation" | None | No |

## 5. Questions for the authors

1. Do the baselines get the same foreground masks? In Tab. S3, changes that include a stronger mask loss added 1.0 dB on real scenes, and the metrics use full images.
2. Which choices were made using official test views? What are the "development scenes" (S5, l. 138)?
3. Does the display-domain arm in Fig. 4 run the annealed warm-up?
4. h takes S, which contains M0,k. In what sense is L_tr then linear in the deposit, and what does that linearity buy?
5. Can you reproduce one published number per baseline? OLAT-GS at ≈ 20 dB on Drums and Lego, and RNG below 20 dB on six scenes, look like failed runs.
6. Were the "reused" SSD-GS runs (Tab. S4) made under this protocol?

## 6. Presentation and writing

The prose is dense and specific, with little padding. The problems are notation, overclaiming and leftover development narrative.

- **Notation.**
  - s denotes a spread (l. 212), a statistics vector (l. 220) and the lobes (l. 251). h, f (l. 283), μ, C and i (l. 228–231) are also overloaded.
  - E omits cos θ at l. 175 but includes it at l. 188.
  - Fig. 2 derives ρ and s from one MLP, while S1 (l. 041) uses two.
  - Eq. (6) overruns the column (l. 237), and Tab. 5 does not report seeds.
  - "Receiver", "deposit" and "flux features" each need a plain definition.
- **Overstatement.** The abstract says "every pixel of a view from the light carries the same flux" (l. 008–009). This conflicts with footnote 1 and with the 75° half-FOV cap (S1, l. 012; cos³75° ≈ 0.02).
- **Formulaic or overclaiming sentences:**
  - "We take a different vantage point, literally that of the light." (l. 047)
  - "A good representation turned out not to be enough." (l. 074)
  - "The culprit was the domain of the loss." (l. 079–080), which conflicts with "does not isolate the mechanism" (l. 295–296)
  - "is more than a shadow map" (l. 499–500)
  - "Where the loss is computed proved as important as the representation" (l. 505–506)
  - the Fig. 4 title, "The loss domain decides the geometry."
- **Suggested rewrites:**
  - l. 079–080 → "Changing only the loss domain removed the collapse (Sec. 5.4)."
  - l. 440–442 → "We hypothesize that the margin comes from light-space visibility and the schedule; E3 and E4 test this."
  - l. 016–019 → "In a 16k-iteration study on two HDR scenes, a linear-radiance loss raised PSNR on held-out training views by 6–7 dB."
  - Fig. 1(e) → "radiance assigned to the transfer term (a learned, not physical, split)."

## 7. Essential vs. optional planned experiments

**Essential:**
- E3, run with three seeds and paired CIs, adding Hotdog and a plain depth-shadow-map variant at pixel receivers.
- E1, with at least three seeds and covering the Tab. 2 panel.
- E2, since nearly half the headline margin depends on it.
- E4(a, b).
- Not currently planned but essential: three seeds for the single-seed rows of Tab. 2 and for LiSA on all 18 scenes, plus the checks in Q1 and Q5.

**Optional:** E4(c, d); E5; E1 on all scenes; a baseline retrained with the radiometric schedule.

## 8. Preliminary rating

**3: Borderline Reject.** The method is accurate on average, fast and candidly analysed. However, the mechanism the title names is not supported. A local network matches the transfer term, and the visibility comparison has not been run. The optimization claim rests on two scenes at 16k iterations, and nearly half the margin is confounded with registration.

**Confidence: 4.**

## 9. What would most change my rating

I would move to 5 if all of the following held, with the abstract, introduction and Fig. 1 rewritten to match Sec. 5.3:
- E3 shows a seed-robust gain of atlas visibility over per-Gaussian visibility on shadow-dominated scenes.
- E1 shows that the atlas degrades less than the local residual under held-out lights.
- E2 preserves a substantial margin on real captures.

Parity in both E3 and E1 would move me to 2.
