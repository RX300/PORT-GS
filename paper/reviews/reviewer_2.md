# CVPR 2027 Review: Reviewer 2

**Paper:** LiSA: Light-Space Atlases for Relightable Gaussian Splatting

## 1. Summary

LiSA rasterizes the Gaussians once from each point light and treats the alpha-compositing weights as the fractions of the light's flux that each primitive intercepts. The resulting atlas holds depth moments and eight learned flux features, and is filtered into a binomial pyramid. The pyramid feeds a Gaussian-CDF moment shadow test refined by a residual network (Eqs. 4–5) and a transfer term that is linear in the gathered features (Eq. 6), both added to a neural local BRDF (Eq. 7). A three-stage schedule fits the geometry with a linear-radiance loss on Gaussian receivers, then refines appearance on pixel receivers. Over 18 scenes LiSA reports 33.47 dB against 31.59 dB for SSD-GS, with about 21 minutes of training per scene.

## 2. Strengths

- The core observation is sound. Within the compositing model, w_i(u) are intercepted fractions. One extra 512² pass therefore yields a differentiable deposit, and because it is linear in w_i it can be filtered (Eqs. 2–3).
- The analysis is unusually candid. The paper includes a capacity-matched control for the transfer term (Tabs. 2–3), test-light novelty statistics (Tab. S2), the registration caveat (L398–404), failure cases, and the development history (Tab. S3).
- All four baselines are re-run under one protocol, scored against common targets, with no test-time fitting.
- Moving shadows on Hotdog and Translucent are crisp (Fig. 3), and training is 1.6–5.5× faster than GS3 and SSD-GS.

## 3. Weaknesses

1. **The headline margin is fragile.** Against the best baseline of each benchmark (Tab. 1), LiSA gains +2.00 dB on real captures, +0.42 dB on GS3 and −0.11 dB on SSS-GS. On the 11 synthetic scenes, where registration plays no role, the gain averages 0.18 dB. Main results are single runs, while the seed std is 0.11 dB and AnisoMetal alone spans 26.4–27.9 dB (L438–439). The real-capture gain comes from Cat, Fish and Pixiu, which the authors attribute partly to registration (L402–404). On the other four real scenes LiSA (31.11 dB) trails GS3 (31.42 dB).
2. **The atlas components lack support.** A local MLP that never reads the atlas matches the transfer term (31.54 vs. 31.51 dB, Tab. 2), and Fig. 1(e) shows "transfer" radiance on the opaque checkerboard floor. The authors concede this point (L439–440). The multi-scale and residual ablations are single-seed differences (0.44 and 0.24 dB) against a 0.11 dB seed std. Credit therefore shifts to atlas visibility, which E3 has not yet tested.
3. **The shadow gains belong to stage III, not to the atlas.** Fig. 3 and L384–387 credit the clean shadow outlines to the atlas. However, the geometry is formed on Gaussian receivers that "inherit the per-primitive shadow limitation" (L272–273), the limitation criticized in L41–46. Per-pixel evaluation arrives only on frozen geometry. Tab. 5 favors Gaussian receivers in PSNR and excludes shadow-dominated scenes, and no hard or PCF shadow map with pixel receivers is compared.
4. **Light-relativity is asserted, not shown** (L71, L261–263). Test lights lie a median of 1.1° from training lights (0° on SSS-GS). g, h and ρ all read the free per-Gaussian code f, so the claim that "position enters only through eight coefficients" (L248–251) understates how much the networks can memorize.
5. **The loss-domain claim is too broad.** It rests on two GS3 HDR scenes, 16k iterations, one seed, and held-out training views (Sec. 5.4). L490–494 and Tab. S3 show display-domain variants ahead on the real and SSS-GS data (confounded). The abstract's "alone raises … 6–7 dB" needs this scope stated.
6. **The relation to classical techniques is understated.** The visibility is a two-moment opacity/deep shadow map with a Gaussian reconstruction. The transfer is a TSM that drops the TSM's world-space distances. The RSM analogy (L56–60) is not realized: there are no sender positions, normals or form factors, and the gather is centered on the receiver's own projection. On current evidence LiSA is a differentiable learned TSM plus VSM.
7. **Scope.** The method handles one isotropic point light outside the object. The half-FOV is capped at 75°, and receivers outside the frustum are treated as lit (Supp. L011–022). Each additional light costs an atlas plus every network evaluation, which makes environment lighting expensive. Multi-bounce transport whose senders are distant in light space cannot be gathered.

## 4. Technical correctness issues

1. **"Every pixel … carries the same flux"** (L008–009, L047–050). *Error.* The pixel solid angle dω_u scales as cos³θ: a 2.2× variation at a 40° half-FOV and 58× at the 75° cap. Footnote 1 lets h absorb this, but h's inputs (S, f, n·l, n·v) contain no off-axis angle, FOV or light distance. Fix: weight the deposit by the known dω_u before filtering, which also makes "flux sums" (L204) true, and state the isotropic-emission assumption.
2. **"Exactly the fractions"** (L052–054). *Imprecision.* This holds only for point-sampled α with per-ray ordering. gsplat sorts per tile by center depth, adds a resolution-dependent 0.3 px dilation, clamps α and terminates early. The splat α is also not the transmittance of a 3D Gaussian, and the EWA projection error grows across wide frusta. "Weights are the flux" (L500–501) is dimensionally wrong, since the flux is I dω_u w_i. Fix: state these assumptions and quantify Σw_i against ray tracing.
3. **E(x) notation.** *Error.* E(x) is defined without cosθ (L175), but the identity E dA = I cosθ dA/r² (L188) requires the cosine. Fix: use separate symbols.
4. **Eq. 4.** *Missing justification.* Take a footprint where an occluder layer of opacity a lies in front of the receiver's own layer, with τ→0. Then t = √(a/(1−a)), independent of the depth gap. With β=3 this gives V₀ ≥ 0.5 whenever a < 0.9; for a = 0.5, V₀ = 0.98 where the true value is 0.5. With β=0 it over-shadows (0.16). The Chebyshev bound of VSM [13] is exact here (1−a). Fur and Gaussian edges are therefore under-shadowed, and V₀ alone shadows the highlights (L255–256). Because the bias is proportional to the spread, coarse levels read almost always lit. τ_k is a lateral texel width used as a depth spread, with no slope term (Supp. L015–017). Fix: use a Chebyshev or four-moment test with a slope-scaled absolute bias, sweep β, and show V₀ maps.
5. **Center depths** (L192–193, L267–268). *Missing justification.* A large, obliquely lit Gaussian deposits a single depth over its whole footprint. Off-center receivers differ by about offset·tanθ, which is far above 3τ, so the per-primitive approximation returns in the depth channel. Expected depth is also ill-defined at silhouettes and in fur (FurScene −1.62 dB). Fix: use ray–Gaussian intersection depth in both passes, and add each Gaussian's along-ray variance to M₂.
6. **Factoring out E(x)** (L257–260). *Imprecision.* Fixed angular kernels need the entry points' 1/r_i², but Eq. 7 applies the receiver's 1/r_o². For back-lit transmission through depth T the error factor is ((r_i+T)/r_i)². That is 4× for T = 2R with the light 3R from the center. The stated lateral condition omits this depth term. h sees δ_k in units of R but never the light distance, so its correction cannot transfer across distances. Fix: multiply by I and feed h the light distance or the world footprint size, or store positions as a TSM does [7].
7. **"Linear in the deposited flux"** (L014–015, L238). *Imprecision.* W_k depends on S, which is a nonlinear function of the same deposit. The only exact linearity is in I, which is trivial and not PRT-like [36]. The sum-of-Gaussians TSM is not a "special case" (L240–242): its exp(−d²/2v) acts per entry texel inside the integral, not on the footprint mean δ_k. At coarse levels δ_k does not measure "material crossed" (L239–240).
8. **Eq. 8.** *Imprecision.* The ratio is right (12.3× at 1% of white), but a linear sRGB toe or an ε makes it bounded. For ℓ1, f′(L̂) does not depend on the residual, so dark, near-converged pixels whose sign is set by noise dominate. Fix: test ℓ2 or Charbonnier losses and the (L+ε)^{1/γ} encoding. Also specify the domain of the SSIM term and how LDR data are linearized.

## 5. Questions for the authors

1. Is the light pass rendered with the classic or the antialiased gsplat mode?
2. What are the distributions of light distance (in units of R) and half-FOV, and what fraction of receivers falls outside the frustum?
3. How is logit V₀ handled when V₀ is exactly 0 or 1?
4. Why are the lobes shadowed by V₀ rather than by a stop-gradient V?
5. How are silhouette and fur pixels lifted? Does FurScene improve with Gaussian receivers in stage III?
6. How is linear radiance obtained for the real captures, and how are clipped highlights treated?
7. Which receivers will E3 use?

## 6. Missing related work

- Beyond Hard Shadows: Moment Shadow Maps for Single Scattering, Soft Shadows and Translucent Occluders
- Fourier Opacity Mapping
- Adaptive Volumetric Shadow Maps
- Deep Opacity Maps
- Convolution Shadow Maps
- Summed-Area Variance Shadow Maps
- Percentage-Closer Soft Shadows
- Real-Time Realistic Skin Translucency
- Splatting Indirect Illumination
- StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering
- Mip-Splatting: Alias-free 3D Gaussian Splatting
- On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy
- Volumetrically Consistent 3D Gaussian Rasterization
- RaDe-GS: Rasterizing Depth in Gaussian Splatting
- Gaussian Shadow Casting for Neural Characters
- PRTGS: Precomputed Radiance Transfer of Gaussian Splats for Real-Time High-Quality Relighting

## 7. Presentation and writing

- Eq. 6 overflows into the right column, over the Sec. 3.4 heading (p. 4).
- Notation is overloaded: s_k is the spread, bold s_k the descriptor, and s(x) the lobes; E appears with and without the cosine. In Fig. 2, r is undefined, and P_i(u) is never used.
- Fig. 2 writes "ρ, s = MLP(…)", which contradicts the light-independent lobe weights described in L251–253.
- "Penumbrae" (L226) is the wrong word: a point light casts none, and the softness comes from filtering. "Absorbs" (L183, L198) should be "intercepts".
- Several sentences read as formulaic rhetoric: "A good representation turned out not to be enough" (L74), "The culprit was the domain of the loss" (L79–80), "literally that of the light" (L47), "Is it the atlas that matters, or just an additive radiance path?" (L427), and "Where the loss is computed proved as important as the representation" (L505). Plain statements would read better.

## 8. Essential vs. optional planned experiments

**Essential:**
- E3, run with the same receivers in both arms and with an added PCF shadow-map variant.
- E1, extended with a light-distance split and a g≡0 variant.
- E2, because real captures supply 43% of the margin over SSD-GS.
- E4(a) and E4(c).

**Optional:**
- E4(b) and E4(d).
- E5.
- A β/Chebyshev ablation (cheap, and recommended).

## 9. Preliminary rating

**3: Borderline Reject.** The deposit observation is sound and the analysis is honest. However, several of the central physical claims are overstated or do not follow from the math (issues 1, 4, 6 and 7). The transfer term does not beat a local MLP, and atlas visibility, the component now credited with the gain, is untested. On clean synthetic data the margin is about 0.2 dB.

**Confidence: 4.**

## 10. What would most change my rating

- E3 showing a gain from atlas visibility beyond seed noise under identical receivers, especially on Hotdog and Translucent.
- E1 showing that the atlas variant degrades less than the local residual.
- E2 preserving the real-capture margin.
- Corrected formulations for issues 1, 4 and 6.
