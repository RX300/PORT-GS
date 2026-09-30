# 训练代码重构的行为一致性验证（2026-09-30）

目的：确认[训练包重构](../architecture/modules/training.md)不改变默认训练、评价和 manifest 行为。
基线为已推送快照 `54fd659`（`origin/feature/selectable-transport-methods`）；重构在 `refactor/port-gs-structure`。
环境 `ssd-gs`（PyTorch 2.4.1、gsplat 1.5.3），RTX 6000 Ada，GPU0–2；未改共享环境，未启动正式实验。

## 方法

gsplat 反向使用原子加法，同一代码同种子重复训练本身不逐位一致（首步后 history 相对差约 1e-7，
含增密的 3100 步运行完全发散）。因此采用三项判据：

1. 首步前向（无原子加法）损失及各损失项逐位一致，验证初始化、RNG 顺序、抽帧、校正与损失组装。
2. 无结构差异：history 行字段、loss_terms 键与顺序、checkpoint 键/张量形状、评价 metrics 键全部相同。
3. 数值差异与“旧代码对旧代码”的噪声同量级；链式场景统一从同一组旧代码源 checkpoint 启动。

10 个场景覆盖三种方法及全部辅助分支（Cat 12 帧子集 32px，另有完整 Cat 512px GGGS 导入）：
directional 3100 步（增密、opacity 重置、验证/best、中间保存、anchor 相机、灯位、光强尺度、高光损失）；
R2b 配方 30 步（GGGS 导入、rotation 相机、平移规范、学习率衰减）+ test（LPIPS、高光、shift-align、test 期标定）与 fit 评价；
surface_attention（2DGS、几何预热、cosine、intersection 深度）；neural_material 300 步；
其上的 SDF（primitive、法线场、detail）、SDF shading、SDF 体积（峰值/上下文采样、hint 编码）、体积独享阶段、
辐射残差（配对上下文、multiply、角度库）及其续训。

## 结果

- 10/10 场景首步损失与损失项逐位一致，项顺序一致。
- 冻结阶段（residual、residual_resume、volume_only）本身确定，新旧 history、checkpoint 与评价指标**逐位一致**。
- 其余场景无结构差异；新旧差异与旧旧差异同量级（如 attention history 3.4e-7 vs 1.8e-7，R2b 3.9e-6 vs 9.7e-7，
  checkpoint 均约 3e-2 相对最大差，来自原子加法累积）。directional 3100 步在 100–500 步的相对损失差落在 6 对旧旧差异范围内；
  终点验证（2 帧，32px）：旧代码 5 次 15.64–15.84 dB（均值 15.70），新代码 3 次 15.28/15.30/15.84 dB，SSIM 均值 0.5997 vs 0.5988；
  秩检验无显著差异，长程增密运行的终点差异属混沌发散。
- 锁步检查（同一 directional 配置 60 步，覆盖 anchor 相机、灯位、光强尺度、高光损失、验证，保存第 1/2/5/30/60 步）：
  新旧各跑两次，任意两两 checkpoint 最大相对差在第 1 步均为 4.4e-8、第 2 步 ≤9.7e-8，此后新旧差异与旧旧差异同量级，
  说明 optimizer 顺序、学习率、校正与正则更新逻辑一致。
- manifest：当前 `configs/validation.json`（变体）、全部评价开关+表面预处理、SDF 评价分支、按场景选择的变体四种配置，
  新旧 JSON 除新增 `source_provenance`/`source_dirty` 外完全相同，错误信息相同。
- 单元测试：`test_methods.py`、`test_data_contract.py`、`test_encoded_alpha.py` 共 93 项通过（新增 6 项）。

新增行为单独验证：rotation 冻结阶段不再报错并逐位复用保存校正；`opacity_reset_every=0` 不重置且不改剪枝调度；
分块划分前向逐位一致；manifest 记录 HEAD、脏树状态、diff 哈希及 third_party 修订。

验证脚本与运行输出位于会话临时目录，未加入仓库（一次性检查代码不进入项目）。
