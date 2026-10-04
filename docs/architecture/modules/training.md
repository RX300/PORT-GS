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
  克隆/分裂（原始 3DGS、GS³、SSD-GS 的规则）。该规则是 Lego/Drums 外壳的触发因素。
- `--foreground-appearance`：渲染器把前景辐射的梯度按 GT alpha 加权（`foreground*w + foreground.detach()*(1-w)`），
  GT 背景像素只监督覆盖。防止初始随机高斯覆盖黑背景时把共享外观网络推入 softplus 饱和。评价与推理不受影响。
- `--refine-start`（默认 500）、`--grow-grad2d`（省略时 absgrad 0.0008 / 否则 0.0002）、`--budget-ramp N`
  （点数上限从初始数在 `--refine-start` 线性升到 `--max-points` 于第 N 步；0 为固定上限）：增密调度实验开关。
- 日志：transport 若提供 `diagnostics`（LiSA：可见度均值、学习残差、局部/镜面/传输线性辐射均值、rho），
  每 100 步写入 history 行的 `transport_stats`。
- `run_benchmark.py`：`worker_gpus` 中重复列出的 GPU 得到多个并发 worker（名称 `<gpu>.<i>`）；只列一次时名称不变。
  LiSA 每 GPU 两个 worker 吞吐约为单个的 1.47 倍。

注意：不透明度重置在 `--refine-stop` 前每 3000 步执行，保存步若恰好是重置步，权重的不透明度≈0.01；
30k/25k 默认配置不受影响，诊断性前缀运行应避免以 3000 的倍数结束。

## 已知限制（未改变行为）

- GGGS 导入在初始化时跳过 `project_geometry`，但第一步后照常执行 opacity≤0.99 与尺度范围投影；只有 `--freeze-geometry` 时完全保留。
- 端口在 `--port-start` 处突变接入（初始 gate≈0.12），未加渐入；属于研究改动，未实现。
- 平移规范只移动 Gaussian，不移动灯光和 `gaussians.center`，见[相机模块](cameras.md)。
