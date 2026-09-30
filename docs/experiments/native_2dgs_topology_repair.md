# Native 2DGS topology repair and retraining

2026-09-27. User authorized fixing the diagnosed training problems, retraining Cat/Pixiu, and proceeding with local neural light transport if geometry is good. No SDF or pretrained geometry supervision.

## Fix and validation

Project-local subclass preserves observed screen-radius history through append/prune, including exact clones; split children are unobserved until rendered. The original author files/CUDA, split sampling/shrinkage and optimizer state rules stay intact. Three regression scenarios cover no-growth screen pruning, clone-history propagation and displaced split/Adam-state preservation, alongside existing camera/projection/gradient/similarity checks. `logs/native_topology_checks.log` passes. Real800-step random1k/128px integration tests topology events and checkpoint reload; this is not quality evidence.

## Locked comparison

One canonical `configs/validation.json`, one launcher, three named variants, GPUs0/1. Each uses fresh30000/native512/seed0,40k fit-only occupancy positions, original cameras, SH3, alphaL1.2, normal.05 after7000, camera-extent normalization, author training schedules and original-world export. Cat470/52 and Pixiu506/56 fit/validation; no official test. Full fit and validation evaluation at terminal step only.

1. `prune_fixed`: same median depth1/distortion1000/camera split extent as the retained prior model; fixes radius bookkeeping only.
2. `expected_defaults`: fixes topology, uses generic author expected depth0 and distortion0 together; this is a two-setting reconstruction recipe comparison, not a single-factor attribution.
3. `object_split`: changes only split size reference relative to expected_defaults, to `.01*object_bound_radius` instead of `.01*camera_extent`.

Each step logs true clone counts, split parents/children and screen/opacity/world removal reasons (reason counts can overlap; total removed is their union). No hard point cap or hidden continuation. Outputs remain `runs/native_2dgs_topology_repair/<variant>/Real_NRHints/<scene>/`. The previous terminal model is retained as a reference during evaluation; no destructive edits to prior results while jobs run.

## Geometry gate before relighting

Review the same four evenly-spaced validation views, plus full-mask/fit/validation metrics and nearby-view depth consistency. Compare with retained failed native geometry using the same renderer; compare with previous PORT-DNA observationally (it used more train frames, including this validation subset). Look for removal of large planar/terraced patches and spikes, preserved support/concavities and coherent normals across views. Desired silhouette IoU is at least about.9 without material worsening from the previous profile. Point count or self-consistency alone cannot pass this gate. No claim of ground-truth geometric accuracy is available.

If both scenes have useful stable coarse geometry and visible improvement, initialize the local per-contribution neural transport using only the matching fit-only geometry payload, then evaluate Cat/Pixiu. If a scene still has major geometry failure, do not call that prerequisite met or freeze it as a teacher. Diagnose the observed failure and report it explicitly. User's conditional authorization requires actual quality review, not merely a successful training exit.


## Additional renderer and input checks

A read-only parity check on actual completed prune_fixed Cat/Pixiu models and fit camera0 compares the unmodified author `gaussian_renderer.render` against the adapter, with the same off-center projection supplied to both. RGB, alpha, rendered normal, surface depth and distortion are bit-identical for depth_ratio0 and1. The intentionally different depth-normal pixel-center convention has mean absolute difference around9e-6–1.9e-5. This rules out a forward-output mismatch in these checked cases, not all possible calibration error. `renderer_parity.json` retains values.

Read-only dataset inspection and SSD-GS `scene/dataset_readers.py` show the supplied points3d.ply files are consistent with the baseline's random cube initialization:100k points in[-.5,.5], uniform gray127, zero normals, with identical reported coordinate quantiles for both scenes. They are not evidence of an available colored SfM reconstruction and were not used as geometric supervision.

The first prune-only pair completed: Cat validation14.6285/.71642/.34365 PSNR/SSIM/LPIPS, IoU.94926; Pixiu18.2651/.84423/.17212, IoU.89657. Manual review still sees over-smoothing/terracing/spikes. The fix alone does not pass the geometry gate; other declared profiles continue.

The existing diagnostic entrypoint adds `--native-study RUN_ROOT --output NEW_DIRECTORY` for completed corrected-topology studies. It verifies event and cumulative population accounting, plots all population curves, and compares all four fixed validation clay views across variants using raw stored normals/alpha. It does not fit models or use GT masks to hide background geometry.


## Three-profile results and final timing correction

All three declared profiles completed both30k scenes and full fit/validation evaluation. `comparison/summary.json` verifies all event/cumulative population identities and retains metrics; fixed-view clay comparisons and population curves are also saved. Object-scale splitting restores substantial real splitting (Cat246936/Pixiu104143 cumulative split parents) and yields80798/37160 final points. Cat LPIPS improves to.30123; Pixiu.16425. Nevertheless, qualitative review still shows smoothing and planar/terraced/spiky structures, not accepted detailed geometry.

Review of the first correction revealed an additional timing mistake: it kept old radius maxima across refinement intervals, including early intervals when screen culling was disabled. That can cull points later using old large footprints even after they shrink. Final code uses the radius history for the current decision and resets after pruning. The new interval test explicitly fails against the archived first repair; all corrected tests pass. This is a correction to our first repair, not another upstream claim.

A final fresh pair `native_2dgs_repaired_geometry` is identical to object_split except this reset timing. Same30k/splits/seed/cameras/initialization. Its source snapshot is separate; the earlier study remains governed by its own immutable source archive. No relighting stage has been started; the geometry gate remains required after final review.


## Final completed result (2026-09-27)

The final interval-corrected Cat/Pixiu pair completed30000 steps each and all470/506 fit plus52/56 internal validation frames. This task ran four independent fresh profiles per scene. The retained model has30000 updates, not120000 continuous updates. No official test or relighting training was performed.

| Profile | Scene | Val PSNR | SSIM | LPIPS | Silhouette IoU | Stored points |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| prior_native | Cat | 14.5374 | 0.71802 | 0.31405 | 0.94959 | 98412 |
| prior_native | Pixiu | 18.2816 | 0.84538 | 0.16509 | 0.90494 | 54180 |
| prune_fixed | Cat | 14.6285 | 0.71642 | 0.34365 | 0.94926 | 18089 |
| prune_fixed | Pixiu | 18.2651 | 0.84423 | 0.17212 | 0.89657 | 17200 |
| expected_defaults | Cat | 14.6065 | 0.71059 | 0.36182 | 0.94800 | 5330 |
| expected_defaults | Pixiu | 18.2448 | 0.84397 | 0.17337 | 0.89222 | 12676 |
| object_split_first_repair | Cat | 14.5734 | 0.71939 | 0.30123 | 0.94831 | 80798 |
| object_split_first_repair | Pixiu | 18.2988 | 0.84583 | 0.16425 | 0.89911 | 37160 |
| final interval-corrected | Cat | 14.5795 | 0.71935 | 0.30067 | 0.94838 | 81018 |
| final interval-corrected | Pixiu | 18.2999 | 0.84586 | 0.16377 | 0.89938 | 37257 |

Final opacity-eligible counts are79861/35772 out of81018/37257 stored points (eligibility is an upper bound, not actual visible contribution). Cumulative split-parent counts245495/103105 and screen-removal counts153624/70109 confirm real subdivision and active screen culling. Each event and total population identity is checked. Reason counts can overlap and are not unique point IDs.

**Geometry decision: failed for both scenes. Relighting condition not met.** Cat regains some texture/paper-fold detail relative to over-pruned development controls, but remains broadly flat/over-smoothed with missing fine structure. Pixiu retains planar/terraced regions and edge spikes. The final interval fix is behaviorally correct, but its visual result is close to the preceding object_split profile. Relative to the previous failed native profile, appearance improves modestly and silhouettes regress slightly. No sufficient geometric superiority over the old PORT-DNA reference has been established.

The four fixed views of both scenes were inspected, along with a common-renderer comparison against PORT-DNA. That reference trained all train frames including these validation views, so it is an observational comparison. No GT geometry is available. Self-normal error is1-cos(theta); its drop under expected-depth smoothing is not proof of accurate surfaces and cannot be used alone to accept geometry. SH image scores here are not relighting scores.

The results demonstrate that a real pruning bug and the split-size mismatch were insufficient explanations of the remaining quality failure. They do not establish illumination, initialization or calibration as the unique cause. Further geometry research should isolate those factors rather than presume another parameter change or more iterations will work. The requested neural transport stage remains conditional and was not attached to this failed geometry.

## Final artifacts and verification

- Final checkpoints and author Gaussian PLY: `runs/native_2dgs_repaired_geometry/Real_NRHints/{Cat,Pixiu}/`. Both geometry payloads are finite, in original-world coordinates and have exact fit/validation lineage.
- [Cat gray comparison](../../runs/native_2dgs_repaired_geometry/review/Cat/summary.png) · [Pixiu gray comparison](../../runs/native_2dgs_repaired_geometry/review/Pixiu/summary.png). Columns: captured RGB, old PORT-DNA geometry-only clay, final repaired native geometry-only clay. Full four-view normal/clay and correspondence reports are in each review directory.
- [Development population curves](../../runs/native_2dgs_repaired_geometry/controls/repair_study/comparison/population.png); per-scene three-profile clay figures and exact accounting report are in the same folder.
- [Full machine-readable results](native_2dgs_repair_results.json), [final state/source/gate audit](../../runs/native_2dgs_repaired_geometry/final_model_audit.json), [retention mapping](../../runs/native_2dgs_repaired_geometry/retention.json).

Four native regression tests pass; the new interval test fails as expected against archived first-repair code. The earlier real800-step train/reload integration passes. Author/adapter forward parity on actual models passes for both depth settings. Final model finiteness, coordinate export and event/cumulative point accounting pass; six core training/evaluation source files match the final launch archive. Source checks do not convert a failed quality result into a successful reconstruction.

Following the user's final-only rule, only the final Cat/Pixiu native checkpoints remain. The previous native run and the three-profile development study are consolidated under final-run `controls/prior_native` and `controls/repair_study`. Their model weights and redundant PLY exports were removed; metrics, images, logs, source archives and terminal geometry-view buffers remain. Those buffers are diagnostic outputs, not trainable models. Immutable archives retain historical absolute paths; use retention.json to locate retained evidence. The five old relighting methods and their18 final models remain untouched.
