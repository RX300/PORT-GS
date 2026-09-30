# 当前清理结果 — 2026-09-22

按用户要求，旧实验/消融/诊断目录及未完成实验已清理，
surface_kernel和surface_diffusion的方法代码、相关说明与历史指标也已删除。
没有恢复或启动实验。

runs/现在只保留3个顶层目录：

- directional_port512_validation_20260915：默认3DGS方向端口的六场景基线。
- surface_attention_validation_20260922：已完成的六场景重光照结果与图像对比。
- directional_surfel512_validation_20260921：仅保留surface_priors中的六场景StableNormal/DA3监督。

最后一项约21 GiB，仍被当前配置及保留checkpoint的元数据引用，路径不变。
SSD-GS参考指标保存在docs/experiments/research_summary.json；保留方法的历史结果仍在docs中。
共享数据集、预训练模型、SSD-GS基线项目、原始研究文档与保留实验的源码快照不变。
旧文档中的实验路径是历史来源，可能已清理；当前可用文件以本清单为准。
调度器的默认manifest跟随configs/validation.json。

删除路径：

- runs/anchor512_restore_check
- runs/attention_integration
- runs/benchmark_comparison.csv
- runs/candidate_comparison.csv
- runs/direct_hashgrid_preflight
- runs/direct_hashgrid_validation_20260914
- runs/directional_port_validation_20260915
- runs/directional_surfel512_warmup5000_20260921
- runs/full_benchmark_20260913
- runs/hashgrid_preflight
- runs/hashgrid_validation_20260914
- runs/local_frame512_validation_20260917
- runs/method_refactor_20260917
- runs/paired_port512_validation_20260917
- runs/rank512_validation_20260914
- runs/research_20260912
- runs/residual_hashgrid_preflight
- runs/residual_hashgrid_validation_20260915
- runs/surface_attention_ablation_20260922
- runs/surface_attention_cosine_anisometal_20260922
- runs/surface_attention_cosine_bunny_20260922
- runs/surface_attention_cosine_check
- runs/surface_attention_cosine_dragon_20260922
- runs/surface_attention_followup_20260922
- runs/surface_diffusion_budget_check
- runs/surface_diffusion_check
- runs/surface_diffusion_pilot_20260922
- runs/surface_diffusion_validation_20260922
- runs/surface_kernel_check
- runs/surface_kernel_pilot_20260922
- runs/surfel_diagnosis
- runs/surfel_integration
- runs/surfel_padding_check
- runs/surfel_supervised
- runs/surfel_warmup_check
- runs/directional_surfel512_validation_20260921/Real_NRHints
- runs/directional_surfel512_validation_20260921/Synthetic_GS3
- runs/directional_surfel512_validation_20260921/Synthetic_SSS-GS
- runs/directional_surfel512_validation_20260921/manifest.json
- runs/directional_surfel512_validation_20260921/queue.log
- runs/directional_surfel512_validation_20260921/source.tar
- runs/directional_surfel512_validation_20260921/status.json
- runs/directional_surfel512_validation_20260921/validation.json

---

# Output cleanup — 2026-09-14

The user requested removal of obsolete experiment outputs. Removed 94 old trial
and temporary-check directories plus 63 obsolete root-level log/preview files
from `PORT-GS/runs/`, about 9.70 GiB of allocated file storage. The output tree
now occupies approximately 2.4 GiB. No active PORT-GS Python training process
was found before deletion. The retained comparison manifests had no references
into the deletion targets.

The exact 157 paths and their pre-deletion allocated sizes are recorded in
[output_cleanup_20260914.json](output_cleanup_20260914.json). All listed paths
were deleted permanently; ignored experiment outputs are not stored in Git.

Retained directories:

- `hashgrid_validation_20260914`: current six-scene results and checkpoints.
- `rank512_validation_20260914`: the six-scene learned-anchor512 comparison.
- `full_benchmark_20260913`: historical benchmark results, comparison inputs,
  and queue evidence, including its pre-existing failed status.
- `research_20260912`: baseline32 results, source archives, and the Cat r1
  checkpoint required by current transport/receiver regression tests.
- `hashgrid_preflight`: operator and receiver audit reports; the two-step
  training checkpoint and one-frame reload evaluation were deleted.

Compact root-level JSON/CSV summaries, patches and research scripts are retained.
Source code, Git history, documents, uploaded papers, datasets, dependencies and
other method projects were not cleanup targets. Older documentation may link to
removed trial outputs; those links are historical evidence references, not
instructions to recreate deleted directories.

## JSON follow-up — 2026-09-21

Removed 19 obsolete root-level launch/command JSON records (25455 bytes)
whose output directories no longer exist and whose filenames have no references
in project Python, shell, Markdown, JSON or JSONL files. Retained experiment
configuration, splits, metrics, queue manifests/status, diagnostic reports and
launch records cited by the experiment documentation.

Removed files under `runs/`:

- `aniso_10k_launch_commands.json`
- `bunny_basis_control_s0_launch.json`
- `bunny_basis_normalized_s0_launch.json`
- `bunny_light_only_s0_launch.json`
- `bunny_localized_frame_encoded_full_s0_launch.json`
- `bunny_localized_frame_encoded_s0_launch.json`
- `bunny_localized_frame_full_s0_launch.json`
- `bunny_localized_frame_s0_launch.json`
- `bunny_localized_full_s0_launch.json`
- `bunny_localized_s0_launch.json`
- `cat_basis_control_s0_launch.json`
- `cat_localized_object_step_s0_command.json`
- `cat_localized_pose_s0_launch.json`
- `cat_localized_s0_launch.json`
- `cat_material_angular_s0_launch.json`
- `cat_moment_control_s0_launch.json`
- `cat_moments_s0_launch.json`
- `cat_response_control_s0_launch.json`
- `pixiu_refinement_full_s0_launch.json`

## Remaining root JSON cleanup — 2026-09-21

At the user's request, removed all 33 remaining JSON files directly under
`runs/` (916160 bytes). These were historical summaries, diagnostic
reports and launch records, not current training/evaluation inputs. Existing
documentation references to these reports are now historical references to
deleted artifacts. JSON files inside experiment subdirectories, checkpoints,
CSV summaries and scripts were preserved.

Removed files under `runs/`:

- `benchmark_comparison.json`
- `bunny_transport_collapse.json`
- `camera_gradient_diagnostic.json`
- `candidate_benchmark_launch.json`
- `candidate_comparison.json`
- `candidate_protocol_audit.json`
- `cat_camera_audit.json`
- `cat_camera_wide_audit.json`
- `cat_camera_wide_audit_prefix_selection.json`
- `cat_final_test_summary.json`
- `cat_geometry_conditioning_audit.json`
- `cat_metadata_contract_audit.json`
- `cat_refinement_full_s0_launch.json`
- `cat_refinement_r1_s0_launch.json`
- `cat_round2_paired_validation.json`
- `cat_shadow_gradient_r2_s0_launch.json`
- `cat_shared_intrinsics_diagnosis.json`
- `cat_shared_rotation_diagnosis.json`
- `cat_support_audit.json`
- `code_review_summary.json`
- `deep_shadow_stats.json`
- `deep_shadow_unit.json`
- `full_sphere_shadow_check.json`
- `panel_splat_footprints.json`
- `pixiu_refinement_r1_s0_launch.json`
- `pixiu_refinement_r1_summary.json`
- `reference_pixel_audit_20260911.json`
- `research_iteration_summary.json`
- `shadow_derivative_audit.json`
- `shadow_gradient_candidate_status.json`
- `sss_benchmark_s0_summary.json`
- `translucent_fit_angle_audit.json`
- `translucent_footprint_audit.json`

## Script cleanup — 2026-09-21

Removed 11 historical one-off scripts: the fixed-date benchmark launcher,
two checkpoint-bound anchor restoration audits, and eight run-local analysis,
collection and launch scripts. Historical documentation mentioning these files
describes past checks; the scripts are no longer maintained entrypoints.
Retained current method/data/alpha/integration tests, parameterized diagnosis,
monitoring, and the shared validation launcher/manifest/scheduler. Future runs
reuse these entrypoints and edit the canonical configuration in place.
Models, metrics and archived source snapshots were preserved.

- `test_transport.py`
- `test_receiver_rendering.py`
- `launch_full_benchmark.sh`
- `runs/cat_shared_rotation_diagnosis.py`
- `runs/directional_port512_validation_20260915/collect_results.py`
- `runs/directional_port_validation_20260915/collect_results.py`
- `runs/reference_pixel_audit_20260911.py`
- `runs/research_20260912/cat_r1_s0/analyze_checkpoint.py`
- `runs/research_20260912/cat_r2_s0/analyze_checkpoint.py`
- `runs/research_20260912/operator_factorization_profile.py`
- `runs/research_20260912/run_full_sequence.py`

## Additional project cleanup — 2026-09-21

Removed the unused legacy Codex/Luna hourly-review monitor, three historical
patch files, 46 old trial/build/audit logs from `logs/`, two dated configuration
copies identical to retained run snapshots, and project bytecode caches.
The shared launcher does not call the removed monitor; use `run_benchmark.py
--manifest <run>/manifest.json --status` for explicit progress checks. Historical
monitor commands in old reports describe retired tooling. Current diagnosis,
regression tests, model/results files, research documents, vendor dependencies
and source archives remain available. Configuration links now target retained
run snapshots.

Removed files:

- `hourly_monitor.py`
- `runs/refinement_changes.patch`
- `runs/repair_changes.patch`
- `runs/shadow_gradient_candidate.patch`
- `logs/anchor512_restore_eval.log`
- `logs/anchor512_restore_operator.log`
- `logs/anchor512_restore_train.log`
- `logs/direct_hashgrid_receiver_audit.log`
- `logs/direct_hashgrid_reload_smoke.log`
- `logs/direct_hashgrid_train_smoke.log`
- `logs/hashgrid_receiver_audit.log`
- `logs/hashgrid_reload_smoke.log`
- `logs/hashgrid_train_smoke.log`
- `logs/pixiu_asymmetric_black.log`
- `logs/pixiu_bounded_area.log`
- `logs/pixiu_bounded_volume.log`
- `logs/pixiu_camera_lr.log`
- `logs/pixiu_camera_scale.log`
- `logs/pixiu_capacity.log`
- `logs/pixiu_corrected_ggx.log`
- `logs/pixiu_corrected_neural.log`
- `logs/pixiu_deferred_fixed.log`
- `logs/pixiu_deferred_smoke.log`
- `logs/pixiu_dense_init.log`
- `logs/pixiu_local_fixed.log`
- `logs/pixiu_local_ports_fixed.log`
- `logs/pixiu_local_ports_joint.log`
- `logs/pixiu_mask_control.log`
- `logs/pixiu_mask_strong.log`
- `logs/pixiu_no_shadow_fixed.log`
- `logs/pixiu_port_fixed.log`
- `logs/pixiu_port_fixed_retry.log`
- `logs/pixiu_port_joint.log`
- `logs/pixiu_radiometric.log`
- `logs/pixiu_shadow_fixed.log`
- `logs/pixiu_surface_port.log`
- `logs/pixiu_symmetric_black.log`
- `logs/pixiu_warmup.log`
- `logs/residual_hashgrid_query_audit.log`
- `logs/residual_hashgrid_receiver_audit.log`
- `logs/residual_hashgrid_reload_smoke.log`
- `logs/residual_hashgrid_train_smoke.log`
- `logs/tcnn_build.log`
- `logs/translucent_dense_init.log`
- `logs/translucent_local_fixed.log`
- `logs/translucent_local_ports_fixed.log`
- `logs/translucent_port_fixed.log`
- `logs/translucent_radiometric.log`
- `logs/translucent_surface_port.log`
- `logs/translucent_warmup.log`
- `configs/local_frame512_validation_20260917.json`
- `configs/paired_port512_validation_20260917.json`
