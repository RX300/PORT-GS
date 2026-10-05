# 训练入口与 `training/` 包

2026-09-30 重构：`train.py` 从 1087 行的单函数拆为编排循环（约 370 行）和 `training/` 下的职责模块。
默认训练行为不变；验证见[重构记录](../../experiments/training_refactor_20260930.md)。

## 模块

| 模块 | 职责 |
| --- | --- |
| `train.py` | 固定顺序的训练循环：抽帧、校正、渲染、损失、optimizer 步、增密、投影、日志、保存与验证 |
| `training/options.py` | `build_parser` / `validate_arguments` / `parse_arguments`；只依赖命令行的组合检查与冻结阶段覆盖 |
| `training/source.py` | 读取 `--init-checkpoint`；依赖已解析几何（3DGS/2DGS）或源 checkpoint 的检查；残差/体积阶段继承源的观察契约与调度 |
| `training/initialization.py` | 训练内帧划分、表面先验、GPU 样本与峰值掩码、物体中心/半径与光强尺度、初始 Gaussian/transport（含 GGGS 导入） |
| `training/schedule.py` | 几何预热阶段切换、means/网络学习率、相机/灯位指数衰减 |
| `training/pose.py` | `CameraFit`（相机校正、optimizer、平移规范）、`LightFit`（逐帧灯位）、`LightScaleFit`（全局光强尺度） |
| `training/fields.py` | `SurfaceFields`（2DGS 监督 SDF 与可选体积辐射头）、`NormalFieldFit`（法线残差场） |
| `training/residual.py` | `ResidualStage`（冻结主 GS 的辐射残差、配对采样与审计）、帧池与优化统计 |

每个可选分支对外只暴露同一组时机：`begin`（清梯度、设学习率、采样或校正）、`loss_terms`/`regularization`、
`step`、`log_fields`、`checkpoint_fields`。`train.py` 在原位置调用它们，因此：

- **RNG 契约**：先构造完整 Gaussians 与 transport（即使随后被 checkpoint/导入几何替换），分支按
  残差 → 光强尺度 → SDF/体积 → 法线场 → 相机 → 灯位 的原顺序构造；辅助网络仍在 `fork_rng` 中初始化。
- **损失项顺序**：光度项 → mask → normal_field_reg → highlight → camera_reg → light_reg → 表面 → SDF/体积；
  `sum(loss_terms.values())` 的浮点结果与重构前一致。
- **optimizer 顺序**：SDF/体积 → 残差或几何 → 网络 → 法线场 → 相机（含规范平移）→ 灯位 → 光强尺度 → 增密 → 投影。

checkpoint 键、history 行字段、config.json 字段与重构前相同（新增 `opacity_reset_every` 配置键）。

## 冻结阶段

`--radiance-residual` 与 `--sdf-volume-only` 固定 Gaussian 教师、transport、法线场和已保存相机校正。
`CameraFit(trainable=False)` 只应用已保存校正，不创建 optimizer、不计算正则、不做规范投影。
此前该阶段用 0 学习率创建 optimizer；rotation 模式的 SparseAdam 拒绝 lr=0，导致从 R2b/neural 旋转校正
checkpoint 启动残差阶段时直接报错。

## 本次修复与新增选项

- `--opacity-reset-every N`（默认 3000，与原行为相同）：只控制 `--refine-stop` 前的 opacity 重置，0 关闭。
  `reset_every=3000` 仍决定细化暂停与 gsplat 大尺度剪枝的起始步，因此关闭重置不改变剪枝调度。
  用于检验保留导入 GGGS 不透明度的方案（R2b 默认每 3000 步把所有 opacity 压到 ≤0.01）。
- `--val-limit` 必须为正；`--validate-every 0` 的帮助文字明确“关闭包括最终一次在内的全部验证，只写 last.pt”。
- `--init-checkpoint` 与 port 格式 `--init-geometry` 继承源的 `scene_gauge_shift`（累计规范平移记录）；此前续训从 0 重新计数。
- 端口空间划分 `spatial_partition` 按 65536 行分块并在反向时重算，不保存 [N, ports, 3] 偏移：
  400k 源 ×200k 像素 ×512 端口的传输前后向峰值显存 10.99 → 4.32 GiB。前向逐位不变；端口参数梯度只改变求和顺序（相对差 ≤1.3e-6）。

## 2026-10-04 新增选项（LiSA v2；默认值保持此前行为）

根因与证据见[LiSA v2根因记录](../../experiments/lisa_v2_root_causes_20261004.md)。

- `--split-scale2d-stop N`：此前代码固定把 gsplat `refine_scale2d_stop_iter` 设为 `--refine-stop`，即在 25k 步前
  无条件分裂所有屏幕半径超过图像 3% 的高斯（与梯度无关）。省略时保持该行为；`0` 关闭此规则，只保留基于梯度的
  克隆/分裂（原始 3DGS、GS³、SSD-GS 的规则）。该规则是 Lego/Drums 外壳的触发因素，但也是 LiSA 实际的增密机制：
  LiSA 延迟着色的 absgrad 约 1e-4，远低于阈值，关闭后点数停在约 2–17 万并欠拟合。LiSA-v2 保留该规则（不传此选项）。
- `--foreground-appearance`：渲染器把前景辐射的梯度按 GT alpha 加权（`foreground*w + foreground.detach()*(1-w)`），
  GT 背景像素只监督覆盖。防止初始随机高斯覆盖黑背景时把共享外观网络推入 softplus 饱和。评价与推理不受影响。
- `--foreground-appearance-until N`（默认 0）：只在第 N 步及以前施加上述加权（LiSA-v2 用 2000）。早期防止黑背景塌缩，
  之后恢复完整梯度，让边缘像素可以着色为背景；全程加权会使白背景场景的残余覆盖变暗（变体F的Drums下降约3 dB）。
- `--holdout-every N`（默认 0）：train 内每 N 帧留出 1 帧（offset N//2）作为验证，近似官方 test 的插值分布；
  省略时仍按 30° 光照角度组留出（外推）。与 `--fit-all`、`--init-checkpoint` 互斥。
- `--refine-start`（默认 500）、`--grow-grad2d`（省略时 absgrad 0.0008 / 否则 0.0002）、`--budget-ramp N`
  （点数上限从初始数在 `--refine-start` 线性升到 `--max-points` 于第 N 步；0 为固定上限）：增密调度实验开关。
- 日志：transport 若提供 `diagnostics`（LiSA：可见度均值、学习残差、局部/镜面/传输线性辐射均值、rho），
  每 100 步写入 history 行的 `transport_stats`。
- `run_benchmark.py`：`worker_gpus` 中重复列出的 GPU 得到多个并发 worker（名称 `<gpu>.<i>`）；只列一次时名称不变。
  当前用户限制最多两张空闲GPU，每张仅一个worker；任务配置不得重复GPU。

## LiSA几何预热开发

`--geometry-warmup-steps N`同样允许`light_atlas`的3DGS几何。该阶段只用逐高斯正基色乘已知I/r²与light_scale，
冻结神经材质/传输，保留alpha监督；learned模式开放材质法线的梯度。未开启辐射域选项时保留原观察变换。
结束时保留已经拟合的反射响应及base优化器状态，
再开放完整神经着色与联合几何优化。旧2DGS RGB预热的base重置行为保持不变。
当前以统一8k预热+22k联合、对照30k全程联合验证；尚未替代v2正式配方。

注意：不透明度重置在 `--refine-stop` 前每 3000 步执行，保存步若恰好是重置步，权重的不透明度≈0.01；
30k/25k 默认配置不受影响，诊断性前缀运行应避免以 3000 的倍数结束。

## 已知限制（未改变行为）

逐高斯LiSA着色的开发路径中，早期前景外观保护以两次相同几何的RGB合成实现：一次正常着色，一次颜色detach，
按GT alpha混合两者的梯度。前向数值不变，背景仍能更新位置、尺度及不透明度。两次camera raster的屏幕梯度在
`Refinement.step_post_backward`相加后累计，每帧的可见计数只加一次；避免前景保护丢失增密信号。
[开发协议](../../experiments/shading_refinement.md)。

- GGGS 导入在初始化时跳过 `project_geometry`，但第一步后照常执行 opacity≤0.99 与尺度范围投影；只有 `--freeze-geometry` 时完全保留。
- 端口在 `--port-start` 处突变接入（初始 gate≈0.12），未加渐入；属于研究改动，未实现。
- 平移规范只移动 Gaussian，不移动灯光和 `gaussians.center`，见[相机模块](cameras.md)。

## LiSA开发入口（2026-10-04）

`--initialization hull` 使用fit轮廓筛选体积种子并按三近邻RMS初始化尺度，不读取外部几何；metadata写入输出的
`initialization.json`。它与`--init-checkpoint/--init-geometry`互斥，只支持3DGS，默认camera种子未改变。
`--normal-model learned` 时几何预热开放feature前三维材质法线的梯度；其它latent在预热中不参与着色。
LiSA预热保留配置的训练相机起始步和学习率日程，测试相机协议不变。

多阶段checkpoint记录source_checkpoint_step与training_steps_before，最终报告累计所有阶段步数。
残差/方差增密和贡献归一化原型已移除；当前仅保留原图像梯度增密及逐高斯前景保护所需的梯度合并。

规范配置可用`appearance_train`覆盖第二阶段选项。`make_validation_manifest.py`顺序编排原train到`geometry/`、
以其last.pt继续外观训练到`appearance/`、最终评价；仍由原队列每张GPU串行执行。相同配置用于所有场景，
只继承家族观察与相机协议。冻结LiSA外观时保存相机从step1启用、无optimizer或规范平移；跳过几何投影并禁止point cap裁剪。
`--lr-decay-steps`同时控制means、network和训练相机日程；16k阶段配30k日程复现原30k训练的前16k，
不会将真实相机学习率提前压缩到16k。默认省略时仍以`--steps`结束；冻结阶段不更新相机。

## 显式辐射目标过渡

`--radiometric-curriculum`只用于新LiSA3D训练并要求非零geometry_warmup_steps；不会修改数据或评价gamma。
渲染输出在线性域计算光度损失，前2/3几何预热目标等于普通观测，最后1/3目标幂指数连续到1，然后一直采用线性目标。
PNG先还原前景gamma再按alpha合成；HDR沿用全图编码契约。所有场景共享该规则。
保存/评价的质量节点必须已经完成目标过渡；当前16k受控结果已显示困难场景大幅改善，跨类别验证仍在进行。


该选项的动机是：在`f(L)=L^(1/gamma)`后算L1，其辐射梯度还乘`f'(L)`，使近暗处的权重远大于亮处。
线性域目标消去该输出导数；早期较亮目标再过渡到真实线性辐射，以改变几何/基色共同成形的过程。
Lego/Drums的匹配16k对照提升7.23/6.00dB，支持整套训练域修正，但尚未单独证明两个组成部分各贡献多少。
默认关闭时仍执行原target_image/render_observation流程，正式评价的gamma、alpha合成及量化指标始终不变。
