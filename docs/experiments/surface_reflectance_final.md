> 2026-09-24清理更新：runs已按用户要求仅保留5套最终结果。中间实验、训练先验/种子和独立审计导出目录已删除；最终模型/指标/图片/源码仍在。历史路径仅作来源记录，重训缓存须重新生成。见[清理说明](runs_cleanup_final_only_20260924.md)。

# Surface-reflectance closing experiment: Cat and Pixiu

Both fresh30000-step runs, all137 official test frames, complete1084 training
frames, source/state audits, common-PNG comparison and manual review completed.
The new method does **not** reliably recover tiny highlights and is not promoted.
Relative to the default, Cat global quality worsens; Pixiu PSNR improves but
LPIPS and tiny recall worsen. The user's requested research cycle is closed;
cleanup and pause do not mean the broader highlight objective has been achieved.

## Fixed protocol and source lineage

Canonical snapshot: `runs/surface_reflectance_final/validation.json`; exact dirty
worktree snapshot: `source.tar`. Reused ssd-gs / CUDA12.1 / PyTorch2.4.1 /
gsplat1.5.3 without dependency upgrades. Seed0, 100000 surfels, native512,
four uniform64px core patches with5px SSIM halos, intersection GGX shading,
fragment cutoff0, lr horizon100000, original .8L1+.2(1-SSIM)+.05mask loss.
Fixed cameras, one shared positive scene light scale; no shadow casting,
densification, neural material decoder, RGB residual or auxiliary SDF branch.

Geometry starts from train-mask surface seeds only, using the previously fixed
95%-consensus construction. Fresh material/opacity/light, no checkpoint weights.
Cat uses all522 train frames, Pixiu all562; validation lists are empty. Exact
seed metadata, image paths, original capture paths and checkpoint lineage were
audited. No test images contribute to initialization or training; no test
exposure/pose optimization. Both camera and light metadata change, and previous
research observed these scenes: these are development-stage joint view/light
tests, not untouched blind tests or isolated pure-light generalization.

The preceding paired5000-step inverse-PDF proposal failed all three tiny-detail
improvement gates and manual review. Uniform sampling was selected before these
tests. No test-driven tuning, budget extension or checkpoint selection followed.
These are same30000-update practical comparisons, not equal pixel work, geometry,
priors, transport, learning-rate schedules or wall time. Baselines are existing
exports from their own archived source; they were not re-trained or re-rendered.

## Complete official-test results

Source evaluator scalar metrics are primary (LPIPS lower is better).
Legacy peak metrics come from the earlier verified canonical-PNG analysis,
checked against the current analysis because their original evaluator predates
peak/component metrics. The final model has original evaluator peak metrics.

| Scene / method | Frames | PSNR | SSIM | LPIPS | Tiny matched / GT pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cat / port_default | 66 | 21.536184 | 0.766465 | 0.230405 | 0/575 |
| Cat / port_neural | 66 | 22.479174 | 0.780930 | 0.255563 | 8/575 |
| Cat / surface_final | 66 | 21.119731 | 0.763317 | 0.287500 | 0/575 |
| Pixiu / port_default | 71 | 20.578249 | 0.845556 | 0.156690 | 46/5324 |
| Pixiu / port_neural | 71 | 21.478632 | 0.851217 | 0.161733 | 84/5324 |
| Pixiu / surface_final | 71 | 21.672545 | 0.848500 | 0.173014 | 6/5324 |

The final model's tiny contrast/GT is .03477794(Cat), .00352057(Pixiu), with tiny
RGB MAE .18915374/.36839494. Tiny recall0%/.112697% is not reliable recovery.
Cat passes2/8 numerical research gates; Pixiu3/8. Both manual gates fail.
Default method remains directional_port_v1.

The two-scene equal-weight mean is PSNR21.396138, SSIM.805909, LPIPS.230257.
Its PSNR gain over default is .338922dB but LPIPS worsens by .036709; this average
must not conceal Cat's degradation or be compared directly with six-scene means.

## Complete training fit, separately

| Scene | Frames | PSNR | SSIM | LPIPS | Tiny matched / GT pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cat | 522 | 20.317576 | 0.756272 | 0.292935 | 3/3999 |
| Pixiu | 562 | 22.129115 | 0.861285 | 0.167976 | 81/40429 |

Even training fit fails to restore most tiny peaks. Training/test score
differences reflect different image distributions; they do not prove that
held-out rendering is intrinsically easier or that geometry is accurate.
No 3D geometry ground truth was evaluated.

## Resources, verification and manual observations

Cat/Pixiu training5057.28/3409.14seconds, peak allocated12.335/12.982GiB.
GPUs0/1 were checked idle before launch; startup verified actual train children,
logs and GPU activity. Other GPUs/tasks were untouched. After the user's cadence
request, active progress checks were spaced by30minutes. All eight phases ended
2026-09-23T17:50:13UTC; no training or evaluation remains for this run.

The23 source/state/lineage/RNG/complete-frame audit conditions passed. Fit16
versus full-fit replay passed64 exact metric/PNG checks. Across all411 model/test
predictions, canonical GT matches exactly,417 component comparisons and integer
peak counts agree, fixed crops match the old GT-only selection, and independent
float64 NumPy/CV2 full-frame rings agree. Cat has6 fixed crops, Pixiu8; each has
4 fullframes. Root viewed all14 crops and8 fullframes.

Cat loses fur/facial texture and develops broad glossy spots and smooth/blobby
shapes. Pixiu loses small white forehead/body glints and narrow bright streaks,
with smeared shapes and base texture. No accurate tiny-peak position/shape
restoration was visible. These observations agree with the quantitative failure.

Strict source-versus-CPU scalar comparison is **not entirely passed**:
622 discrepancies at unchanged rtol1e-6/atol1e-8 (SSIM221, LPIPS401), maximum
absolute difference2.157688e-5. No PSNR violations. Source values remain primary;
all primary/secondary gate decisions agree. An initial read-only verifier failed
with KeyError because legacy reports lack peak fields; that failure is retained,
and the corrected provenance is explicit above. No training or metrics were
re-run/replaced to obtain a more favorable result.

Evidence under `runs/surface_reflectance_final/`: `research_completion.json`,
`audit/report.json`, `fit_replay_audit.json`, `analysis_verification.json`,
`analysis_verification_first_failure.json`, `test_gates.json`, `manual_review.json`,
`analysis/{Cat,Pixiu}/`, and each scene's `test/metrics.json` plus every test PNG.
Final models and train-only seed artifacts are retained. See
[method inventory](method_inventory.md) and the final cleanup record for retention.
