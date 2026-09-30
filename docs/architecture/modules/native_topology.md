> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

# Corrected native 2DGS population bookkeeping

`native_topology.corrected_model_class(AuthorModel)` returns a local subclass, leaving the author's source and rasterizer intact. Native reconstruction selects it through `--native-topology corrected`. Legacy inference remains compatible because checkpoints store tensor capture and canonical world geometry, not Python model objects.

## Radius lineage

The original append routine zeros all screen-radius history before its pruning test. The subclass preserves old rows, appends inherited history for exact geometric clones, and appends zero/unobserved history for displaced and shrunk split children. Author prune methods retain/slice this array together with Gaussian attributes and Adam moments. A clone therefore cannot evade an oversized-screen decision just by becoming a new row. A split child must be rendered before its own new footprint is judged.

The clone/split selection rules, random tangent displacement, two-child scale shrinkage1/1.6, SH/opacity inheritance and opacity/world pruning retain author semantics. Optional object-bound split size is a separate training configuration; it rescales percent_dense only and does not alter learning-rate scale or the .1*camera_extent world-prune threshold.

## Accounting and checks

Every call returns before/after counts, exact clones, removed split parents, added split children, final pruned union and each overlapping pruning reason. Counts obey `after = before + cloned + split_children - split_parents - pruned`. Cumulative event counts are not unique point counts.

`NativeReconstructionTests.test_corrected_screen_pruning_clone_lineage_and_split_state` fixes scales/opacities/gradient values independently of kNN initialization and verifies (1) no-growth oversized deletion, (2) parent+clone deletion using inherited radius, (3) split child scale and new observation state, and (4) Adam moment shapes plus a successful optimizer update after topology changes. Existing analytic projection/normal and CUDA gradient tests also pass. Real short training validates wiring/checkpoint reload.

## Observation-window timing correction

The radius maximum describes the current refinement interval, like accumulated gradients. The decision uses that history, then clears all retained radii after pruning, including intervals in which screen culling is disabled. The first local repair incorrectly kept surviving radii across intervals; before screen culling activates this can retain stale large footprints even after a Gaussian shrinks. The final correction moves the reset to the end of the decision, preserving both actual screen culling and interval semantics. A two-interval test fails against the archived first repair and passes against the final code; existing clone/split/Adam/projection checks remain passing.
