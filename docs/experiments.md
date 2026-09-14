# PORT-GS 实验记录

记录日期：2026-09-06。所有主表实验都使用 256 分辨率（Pixiu smoke 除外），
并由训练 JSON 内的光源方向组划分 fit/validation：Pixiu 为 506/56，
Translucent 为 1775/225。训练器只访问训练元数据和训练图像；本轮没有读取
或评价任何 test 图像。

表中 validation 指训练内 light-held-out 验证，`raw_MSE` 是日志中的原始均方
误差。它们是开发信号，不是正式 test 或 SOTA 成绩。

## 结果

| 运行 | 场景 | 配置摘要 | 状态 | 最后一次 validation（PSNR / SSIM / raw_MSE） | 用时 |
| --- | --- | --- | --- | --- | ---: |
| `pixiu_smoke` | Real_NRHints / Pixiu | 200 步，128 分辨率，5000 点 | 已达到配置步数 | 17.7472 / 0.7652 / 0.017192 | 5.38 s |
| `pixiu_warmup` | Real_NRHints / Pixiu | 5000 步，几何优化，最终 74263 点 | 已完成 | 18.7642 / 0.8127 / 0.014351 | 58.79 s |
| `pixiu_port_fixed` | Real_NRHints / Pixiu | 5000 步，冻结 geometry，warmup checkpoint | 失败 | — | — |
| `pixiu_port_fixed_retry` | Real_NRHints / Pixiu | 5000 步，冻结 geometry，PORT，最终 74263 点 | 已完成 | 20.0184 / 0.8338 / 0.010510 | 62.76 s |
| `pixiu_local_fixed` | Real_NRHints / Pixiu | 5000 步，冻结 geometry，local，最终 74263 点 | 已完成 | 18.9635 / 0.8180 / 0.013621 | 41.99 s |
| `translucent_warmup` | Synthetic_GS3 / Translucent | 5000 步，几何优化，最终 200773 点 | 已完成 | 21.5961 / 0.9026 / 0.030466 | 68.19 s |
| `translucent_port_fixed` | Synthetic_GS3 / Translucent | 5000 步，冻结 geometry，PORT，最终 200773 点 | 已完成 | 22.2885 / 0.9165 / 0.029331 | 92.32 s |
| `translucent_local_fixed` | Synthetic_GS3 / Translucent | 5000 步，冻结 geometry，local，最终 200773 点 | 已完成 | 21.9418 / 0.9066 / 0.029957 | 64.06 s |
| `translucent_local_ports_fixed` | Synthetic_GS3 / Translucent | 5000 步，冻结 geometry，local ports，最终 200773 点 | 已完成 | PSNR 21.9605869651（SSIM/raw_MSE 未整理） | — |

指标来自各运行的 [`history.jsonl`](/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/runs)。完成状态由对应日志末尾的 `event=complete` 确认；`pixiu_smoke` 的 history 已到达配置的第 200 步，但没有单独的完成事件。

## PORT sanity 修复

`pixiu_port_fixed` 在训练开始前失败，没有产生 history。冻结 geometry 后，
gsplat strategy 的 trainable 参数只有 `base` 和 `features`，但 optimizer 仍
包含 `means`、`opacities`、`quats`、`scales`，因此 `check_sanity` 报错：

```text
trainable parameters and optimizers must have the same keys,
but got {'features', 'base'} and dict_keys(['base', 'features',
'means', 'opacities', 'quats', 'scales'])
```

随后修正冻结 geometry 时的 optimizer 参数集合；`pixiu_port_fixed_retry` 在
相同 5000 步配置下完成，sanity 不再报错。该修复只说明训练流程可运行，不
说明模型已经优于基线。

## 复现实验命令

命令从 `PORT-GS` 目录执行，使用 `ssd-gs` 环境。以下命令与 warmup 配置
一致：

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
CUDA_VISIBLE_DEVICES=0 /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python train.py \
  --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Pixiu \
  --output runs/pixiu_warmup \
  --steps 5000 --resolution 256 --points 20000 --max-points 200000 \
  --feature-dim 16 --width 64 --rank 16 \
  --port-start 6000 --shadow-start 1500 --refine-stop 4500 \
  --validate-every 1000 --val-limit 32 --background 1.0 --seed 0
```

评价入口的 validation 用法如下；`test` 必须在配置冻结后显式选择，本轮没有
执行该 split：

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python evaluate.py \
  runs/pixiu_port_fixed_retry/last.pt --split validation \
  --output runs/pixiu_port_fixed_retry/eval --lpips
```

当前入口只覆盖 `Real_NRHints` 和 `Synthetic_GS3`；`Synthetic_SSS-GS` 尚未
实现。训练内 holdout 使用 256 分辨率和约 10% 的光源方向组，不能与官方
test 或正式 SOTA 表格直接比较。测试图像不参与训练、验证、早停、调参或本
轮文档整理。

## 30K joint runs

在 Pixiu 上继续运行了两项 30K joint 实验。两者均使用 256 分辨率、
`max_points=250000`、`rank=16`、训练内 506/56 fit/validation 划分，并在
30K 步完成：

| 运行 | 初始化与分支 | 最后一次 validation（PSNR / SSIM / raw_MSE） | 用时 |
| --- | --- | --- | ---: |
| `pixiu_port_joint` | `pixiu_port_fixed_retry/last.pt`，PORT，最终 181183 点 | 21.7624 / 0.8534 / 0.007371 | 545.09 s |
| `pixiu_local_ports_joint` | `pixiu_local_ports_fixed/last.pt`，local ports，最终 156270 点 | 20.0873 / 0.8316 / 0.011392 | 516.22 s |

两项日志末尾都有 `event=complete`，没有发现异常。它们仍然是训练内
light-held-out 结果，不是正式 test 或 SOTA 成绩。

此前观察到的 `pixiu_local_ports_fixed` 已完成 5000 步，validation 为
18.9882 / 0.8182 / 0.013534；它不属于上面的主表比较组。随后补录的
`translucent_local_ports_fixed` 已完成 5000 步，最终 PSNR 为 21.9605869651；
其余指标尚未在可用记录中整理。
## GGX surface prior 对照

`pixiu_surface_port20K` 与 `translucent_surface_port20K` 增加了 GGX
surface prior，对应结果如下：

| 运行 | 场景 | 最后一次 PSNR | 最佳 PSNR | 结论 |
| --- | --- | ---: | ---: | --- |
| `pixiu_surface_port20K` | Real_NRHints / Pixiu | 20.5703 | 20.8423 | 未稳定改善 |
| `translucent_surface_port20K` | Synthetic_GS3 / Translucent | 20.8357 | 22.6613 | 未稳定改善 |

两项对照都没有形成稳定收益，因此 GGX surface prior 未纳入主方案。GGX
属于标准的局部物理先验/技术组件，不能作为 PORT-GS 的创新点；PORT-GS 的
研究判断仍以成对空间端点传输和相同容量对照为核心。

## Mask weight 对照

两项 Pixiu 运行都从相同的 joint 初始 checkpoint 开始，训练 10K 步，只改
mask loss weight：

| 运行 | mask weight | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 状态 |
| --- | ---: | --- | --- |
| `pixiu_mask_strong` | 0.5 | 21.5523 / 0.8501 / 0.0076790 / 0.0199418 | 已完成 |
| `pixiu_mask_control` | 0.05 | 21.2079 / 0.8485 / 0.0081990 / 0.0230490 | 已完成 |

这组结果只用于记录 mask 权重敏感性，仍是训练内 light-held-out 验证，不是
正式 test 结果。

## Shadow hint 对照

两项 Pixiu 运行都从 `pixiu_port_joint/last.pt` 开始，固定 geometry 训练
5K 步，只改变近似 shadow hint：

| 运行 | shadow hint | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 状态 |
| --- | --- | --- | --- |
| `pixiu_no_shadow_fixed` | 关闭 | 21.3550 / 0.8495 / 0.0083768 / 0.0219874 | 已完成 |
| `pixiu_shadow_fixed` | 保留 | 21.9847 / 0.8554 / 0.0071332 / 0.0219874 | 已完成 |

结果显示该配置下保留 shadow hint 的训练内验证值更高，但这仍是单场景、单
种子、训练内划分的诊断对照，不能替代正式 test 评价。

## Dense initialization 诊断

`pixiu_dense_init` 与 `translucent_dense_init` 使用同一初始化合同：独立生成
100000 个均匀随机点，范围为 `[-0.5, 0.5]^3`（`--init-radius 0.5`），点上限
为 600000；不读取 `points3d.ply`，也没有 GT 点。data audit 已确认两场景的
原始 `points3d.ply` 完全相同，因此这里仍明确使用独立随机初始化。两项均训练
20K 步、256 分辨率，validation 仍来自训练灯光方向组。

| 运行 | 场景 | 最终点数 | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 最佳 PSNR | 用时 |
| --- | --- | ---: | --- | ---: | ---: |
| `pixiu_dense_init` | Real_NRHints / Pixiu | 202690 | 19.8030 / 0.8300 / 0.0114894 / 0.0359044 | 19.8030 | 343.04 s |
| `translucent_dense_init` | Synthetic_GS3 / Translucent | 586326 | 21.2665 / 0.8984 / 0.0299627 / 0.0062479 | 23.2193 | 811.71 s |

这是初始化合同诊断：它用于比较随机点范围、点数和 densification 行为，结果
不能归因于 PORT 传输创新，也不能与正式 test 或 SOTA 成绩混用。

## Black-background 对照与 capacity 诊断

以下运行使用 512 分辨率和黑色背景。`pixiu_symmetric_black` 与已完成的
`pixiu_asymmetric_black` 是相同条件下的对照，均为 10K 步；`pixiu_capacity`
从头开始，使用 asymmetric local 分支、`feature_dim=32`、`width=128`、
`rank=32`，训练 30K 步。所有指标仍来自训练灯光方向组 validation。

| 运行 | 配置 | 最后一次 PSNR | 最佳 PSNR | 其他记录 |
| --- | --- | ---: | ---: | --- |
| `pixiu_asymmetric_black` | asymmetric，10K 步 | 22.249612093 | 22.2583 | 已完成 |
| `pixiu_symmetric_black` | symmetric，10K 步 | 22.300240874 | 22.3303 | SSIM 0.8550，raw_MSE 0.0063766，已完成 |
| `pixiu_capacity` | asymmetric local，feature32/width128/rank32，30K 步 | 22.630539238 | 22.7088 | SSIM 0.8584，raw_MSE 0.0064120，已完成 |

局部颜色响应中交换两个方向参数的结构对称，只是表示层约束，不等于 BRDF
互易性，也不能单独证明整个 Gaussian 合成满足互易。当前还在用控制实验核查
离散测度、alpha 权重和背景处理的影响；因此表中的单视图诊断（另有约 56.5 dB
的高值）只能作为数值 sanity check，不能报告为跨视角或未见光照泛化成绩。
这些 black-background 与 capacity 结果也没有正式 test 评价。

## Camera principal-point geometry diagnostics

`diagnose_geometry.py` 只优化 alpha mask 的 binary cross-entropy，使用训练相机
的固定 32 视图子集报告 fit/validation alpha L1 与 IoU，不使用 RGB、光照或
relighting PSNR。`pixiu_mask_metadata` 保留元数据主点，`pixiu_mask_center`
只将训练及留出相机的主点改为输出图像中心，不 warp 图像；两项都已生成
`geometry.pt`：

| 运行 | 主点模式 | 最终点数 | fit alpha L1 / IoU | validation alpha L1 / IoU | 状态 |
| --- | --- | ---: | --- | --- | --- |
| `pixiu_mask_metadata` | metadata | 49064 | 0.0203143 / 0.932493 | 0.0213897 / 0.937918 | 已完成 |
| `pixiu_mask_center` | center | 40213 | 0.0344801 / 0.889038 | 0.0323622 / 0.917398 | 已完成 |

这组结果只诊断相机/几何轮廓一致性，不能称为重光照成绩或 SOTA 结果。

## Camera scale / position LR 对照

`pixiu_camera_lr` 与 `pixiu_camera_scale` 都是 Pixiu、512 分辨率、30K 步的
neural 训练。两者都使用 camera extent 作为 means position LR；前者使用
object scale 做 densification scale，后者使用 camera scale：

| 运行 | position scale | densification scale | 最终点数 | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 最佳 PSNR |
| --- | --- | --- | ---: | --- | ---: |
| `pixiu_camera_lr` | camera | object | 90024 | 23.6654 / 0.8634 / 0.0045133 / 0.0242329 | 23.7460 |
| `pixiu_camera_scale` | camera | camera | 59468 | 23.9492 / 0.8643 / 0.0042338 / 0.0304533 | 23.9788 |

`camera_scale` 的 split threshold 约为 0.14738，大于当时的最大 sigma 约
0.14087，因此它改变了 duplication/split regime。该对照不能作为孤立的
position-LR 收益证明；两项也都只是训练内 validation，没有正式 test 评价。

## Half-vector / multiscale 与 SSS 小场景（历史修正）

Translucent 结果仍是训练内 light-held-out validation，不是正式 test 或 SOTA，且未
启用 camera optimization。SSS 的 `unit_light_intensity=1.0` 只是显式等功率假设，
不是辐射定标。

| 运行 | 场景与配置 | 最终点数 | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 最佳 PSNR | 用时 |
| --- | --- | ---: | --- | ---: | ---: |
| `translucent_half_multiscale` | GS3 / Translucent，30K，512，half-vector + multiscale ports，depth shadow | 400000 | 25.11744851 / 0.94758978 / 0.005056056 / 0.002446097 | 25.11744851 | 1224.59 s |
| `bunny_port` | **作废**：旧 SSS split/灯光坐标读取错误，实际使用官方 val | — | 撤回，不保留质量结论 | — | — |

`translucent_half_multiscale` 与 `translucent_half` 使用相同训练预算和模型规模，
multiscale 结果仅记录为实验性参数化对照。

旧 `bunny_port` 运行已撤回：当时 SSS loader 将 split 错映射为 `val`，并错误翻转
灯光坐标，因此该运行实际使用官方 val 元数据，不能称 train-only 结果，也不能保留
其 PSNR/SSIM 结论。官方 test 图像未被加载；作废依据见
[`runs/bunny_port/INVALID.md`](../runs/bunny_port/INVALID.md)。新的
`bunny_world_deep_fine` 正在用修正后的 train split 和 raw world light convention
重新运行，尚未形成可报告的完整结果。

## Radiometric observation contract

`pixiu_radiometric` 与 `translucent_radiometric` 使用 512 分辨率、30K 步和
`display_gamma=2.2`。Pixiu 的训练背景为黑色（`background=0`），Translucent
的训练背景为白色（`background=1`）。模型输出先视为线性辐射并做 gamma 2.2
显示映射后再和观测比较。

PNG GT 保留原编码并做 alpha 合成；EXR GT 先合成再做 gamma 2.2，以对齐
baseline 的 train GT 存档。抽查的三张量化图像中，新的 GT 与 baseline GT
超过 99.98% 像素一致。该 gamma 是观测/指标合同修正，不是根据 test 图像调参。

| 运行 | 场景 | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 最佳 PSNR | 用时 |
| --- | --- | --- | ---: | ---: |
| `pixiu_radiometric` | Real_NRHints / Pixiu，黑背景 | 23.4655 / 0.8614 / 0.0047431 / 0.0258117 | 23.4655 | 424.23 s |
| `translucent_radiometric` | Synthetic_GS3 / Translucent，白背景 | 21.8532 / 0.9192 / 0.0091674 / 0.0035410 | 23.2919 | 1201.54 s |

这里的 `raw_MSE` 指未裁剪的 observation-domain MSE；在 gamma 合同启用后，
它不应再被解读为线性 HDR 辐射域误差。两项结果仍是训练内 light-held-out
验证，没有正式 test 评价。

## Corrected neural 与 deferred smoke

`pixiu_corrected_neural` 使用 512 分辨率、`display_gamma=2.2` 和 area
transport measure，训练 100K 步。它是当前 neural 分支的长程 sanity run，
validation 仍只来自训练灯光方向组：

| 运行 | 最终点数 | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 最佳 PSNR | 用时 |
| --- | ---: | --- | ---: | ---: |
| `pixiu_corrected_neural` | 69237 | 22.8600 / 0.8558 / 0.0056745 / 0.0244535 | 23.6417 | 1441.84 s |

`pixiu_deferred_smoke` 是 200 步、128 分辨率的短 smoke，采用标准的 feature
splat 后逐像素着色路径：

| 运行 | 最后一次 validation（PSNR / SSIM / raw_MSE / alpha_L1） | 状态 |
| --- | --- | --- |
| `pixiu_deferred_smoke` | 21.1626 / 0.8323 / 0.0087698 / 0.0337885 | 已完成 |

deferred smoke 只用于检查逐 Gaussian 恒定颜色拟合瓶颈；它是标准渲染路径诊断，
不能作为 PORT 创新或泛化收益的证据。

## GGX corrected run failure

`pixiu_corrected_ggx` 在第 20600 步因
`AssertionError: Exterior-light shadow map contract violated` 停止，没有完成
记录。诊断显示 GGX thin-ratio regularizer 会通过增大切向 scale 继续降低 loss；
出现低 opacity（约 0.0029）和大 Gaussian 尺度（约 1.20）后触发视锥合同。
后续实现已改为 absolute thin scale/radius regularizer，并加入
`max_scale=0.1*radius` 投影，但上述运行仍记录为失败，不能把中断前指标当作
GGX 结果。GGX 仍属于标准局部物理先验，不是 PORT-GS 创新点。

## Camera-scale 与 mask geometry 后续结果

以下结果均来自已有日志或显式 evaluation 输出，仍是训练内 validation/fit
诊断，不是正式 test 或 SOTA。Translucent 两项运行均未启用 camera offsets；
Pixiu 的 `mask_pose` 与 `pose_fit` 使用旧相机学习率 `.01`，`mask_pose_small`
使用当前记录的 `.0003`。`mask_decay`、`mask_fine` 不优化相机。

| 运行 | 场景与配置 | points | 最后一次指标 | 用时 |
| --- | --- | ---: | --- | ---: |
| `translucent_camera_scale` | Translucent，30K，512，camera-scale，no camera opt | 400000 | validation PSNR/SSIM/raw_MSE/alpha_L1 = 25.00245625 / 0.94738403 / 0.005369313 / 0.002241302 | 1178.31 s |
| `translucent_mask_init` | Translucent，30K，512，初始化自 `translucent_mask_metadata/geometry.pt`，no camera opt | 400000 | validation = 24.92925876 / 0.94656670 / 0.005380701 / 0.002231532 | 1221.37 s |
| `pixiu_mask_decay` | Pixiu alpha geometry，10K，metadata K，no camera opt | 43773 | fit L1/IoU = 0.01907056 / 0.93784810；validation = 0.02019627 / 0.94365779 | 28.16 s |
| `pixiu_mask_pose` | Pixiu alpha geometry，10K，camera LR `.01` | 39989 | fit = 0.03424120 / 0.87622576；validation = 0.03550873 / 0.88888922；corrected-fit = 0.03411355 / 0.87908938 | 76.32 s |
| `pixiu_mask_pose_small` | Pixiu alpha geometry，10K，camera LR `.0003` | 40644 | fit = 0.02047536 / 0.90806978；validation = 0.01933546 / 0.92371687；corrected-fit = 0.00871518 / 0.97126190 | 74.18 s |
| `pixiu_mask_fine` | Pixiu alpha geometry，10K，`grow_scale3d=.001`，no camera opt | 69747 | fit = 0.01903881 / 0.93885545；validation = 0.01991478 / 0.94576843 | 23.69 s |
| `pixiu_pose_fit` | Pixiu relighting，30K，camera LR `.01` | 49266 | validation PSNR/SSIM/raw_MSE/alpha_L1 = 20.69944918 / 0.84269640 / 0.008783363 / 0.03188432 | 537.37 s |

`translucent_camera_scale/best.pt` 的显式 CPU/GPU evaluation（未启用 LPIPS）为：

| 输出 | split 与帧数 | PSNR / SSIM / raw_MSE / alpha_L1 |
| --- | --- | --- |
| [`fit32`](runs/translucent_camera_scale/fit32/metrics.json) | fit，32 帧 | 25.55634123 / 0.94575125 / 0.003902950 / 0.002240814 |
| [`validation_full`](runs/translucent_camera_scale/validation_full/metrics.json) | validation，225 帧 | 24.79263056 / 0.94311072 / 0.004514062 / 0.002299091 |

相机优化的 corrected-fit 数值只用于诊断训练相机修正是否改善 fit，不能与未修正
validation 混写；validation 始终使用原始相机。`.01` 是旧相机学习率试验，和
当前 `.0003` 默认记录不同。上述结果没有正式 test 评价，不能宣称 SOTA。

## Translucent deep-fine 联合优化

`translucent_deep_fine` 是 30K、512 分辨率的联合几何/传输运行，使用
half-vector、deep shadow、`absgrad` 和 object densification scale；位置学习率仍
使用 camera scale，未启用 camera optimization。最终点数为 `174585`，最后一次
validation（step 30000）为 PSNR/SSIM/raw_MSE/alpha_L1
`28.21223307 / 0.95896222 / 0.003187131 / 0.002073688`，最佳 validation PSNR
为 step 27000 的 `28.29036617`，总用时 `667.19 s`。

相较于旧 `translucent_half` 的 30K 运行（最后 PSNR `25.04845190`、最佳
`25.06062478`、最终点数 `400000`），该运行同时改变了 shadow 深度策略、absgrad
梯度聚合和 densification scale，且点数/几何轨迹不同。因此这只是联合配置结果，
不能把全部收益归因于 half-vector、deep shadow 或 absgrad 中的单一组件，也不是
正式 test/SOTA 结果。

## Translucent appearance 70K（阶段性记录）

`translucent_appearance_70k` 从 `translucent_deep_fine_layers4/best.pt` 的最佳
checkpoint 继续做冻结几何的外观拟合；优化器重新初始化，因此这是阶段性续训，不能
称为 exact resume。日志中的 20K、30K、40K 是本阶段的局部步数，不应与源运行步数
直接相加为统一的累计步数。

| 阶段局部步数 | 点数 | validation PSNR / SSIM / raw_MSE / alpha_L1 | 用时 |
| ---: | ---: | --- | ---: |
| 20000 | 172476 | 29.46182531 / 0.96359545 / 0.0019742062 / 0.0020784988 | 492.46 s |
| 30000 | 172476 | 29.66749775 / 0.96448878 / 0.0020070821 / 0.0020784988 | 740.17 s |
| 40000 | 172476 | 29.81090337 / 0.96492779 / 0.0018233173 / 0.0020784988 | 987.46 s |
| 70000 | 172476 | 29.93312341 / 0.96524530 / 0.0016398676 / 0.0020784988 | 1727.87 s |

该 Translucent 记录仍是训练内 validation，不能作为正式测试或 SOTA 结论。

## 首次正式测试：Bunny seed 0

`bunny_benchmark_s0` 按实验协议第8节从正确的全部500个官方 train 帧从零训练，
没有 validation checkpoint 选择，固定评价60K last.pt；未继承作废 bunny_port。
完整官方 test 的500条记录已评价，没有测试期相机、曝光或颜色优化。

| 指标 | PORT-GS | 本地 SSD-GS 参考 |
| --- | ---: | ---: |
| PSNR | 36.28149348 | 38.48003387 |
| SSIM | 0.98308009 | 0.98909527 |
| LPIPS（基线 [0,1] 输入约定） | 0.02485137 | 0.01475585 |

PORT-GS 的标准 [-1,1] LPIPS 为0.03115610。该场景 PSNR 落后参考约2.20 dB，
尚未达到SOTA；不能把之前33.88 dB内部验证或作废实验当成正式测试成绩。
结果在 `runs/bunny_benchmark_s0/test/metrics.json`。

子任务随后触发额度上限，停止重试代理。为继续完成已冻结配置的其他四个SSS场景，
当时的 `benchmark_sss.py`（现已扩展并替换为 `benchmark.py`）使用物理GPU0/1的两个串行队列执行60K训练与完整测试，
不搜索配置、不中途改模型、不覆盖已有输出。Candle/Soap分配GPU0，Dragon/Statue分配GPU1。
`runs/sss_benchmark_s0_summary.json` 记录逐场景结果；只有五场景全部成功才写完整macro。
单种子宏平均仍不是最终统计稳健性证明，后续重复种子要求未被取消。
