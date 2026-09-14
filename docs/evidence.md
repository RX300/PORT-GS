# 证据、参考指标与当前状态

2026-09-06。此文件记录初始审计；当前内部训练结果见
[experiments.md](experiments.md)，仍无正式 test/SOTA 结果。

## 参考分数

本地结果来自 18 个标准输出目录的 `results_test.json`，Real/GS3 取 `ours_100000`，SSS 取 `ours_60000`。这里按场景简单宏平均；未读取 SSD-GS 方法代码。论文参考来自 [SSD-GS Table 1 / Table 3](https://arxiv.org/pdf/2604.13333) 的测试列，表格已视觉核对。

| 数据集 | 论文 PSNR / SSIM / LPIPS | 本地 PSNR / SSIM / LPIPS |
|---|---|---|
| Real_NRHints | 32.1075 / 0.9499 / 0.0795 | 31.9972 / 0.9502 / 0.0804 |
| Synthetic_GS3 | 32.2919 / 0.9736 / 0.0307 | 31.5780 / 0.9657 / 0.0381 |
| Synthetic_SSS-GS | 38.3542 / 0.9864 / 0.0158 | 38.3476 / 0.9871 / 0.0165 |

SSS 论文 Table 1 的汇总 w/ Opt 为 38.3542 / 0.9863 / 0.0158；w/o Opt 为 37.4409 / 0.9843 / 0.0186。由 Table 3 已四舍五入的逐场景数据重新平均，与原汇总可能有末位差异。Opt 的评价条件未核实，不推断其含义，也不据此指控数据泄漏。

本地 SSS 的 PSNR 宏平均略低于论文，SSIM 较高、LPIPS 较差；Soap 的本地 PSNR 明显更低；不能把不同来源的逐场景最大值拼成一种实际存在的方法。GS3 的 SSD 论文结果也并非每场景都优于表中所有基线。

完整数值和每条来源路径保存在 [baseline_metrics.json](baseline_metrics.json)。这些数字是基线既有结果，不是新方法成绩。

## 基线评价协议核查

18 个标准 SSD-GS output 的 `cfg_args` 与 `cameras.json` 数据字段已汇总到
[baseline_protocol.json](baseline_protocol.json)。历史运行均记录
`resolution=-1`、未记录 `max_reso`/`json`、`images="images"`、`eval=true`、
`data_device="cpu"`、`view_num=2000`；实际相机尺寸由 `cameras.json` 记录。
`source_path` 全部指向当前不存在的旧工作区路径，因此尚未证明这些历史结果与
当前 `/workspace/datasets/SSD-GS/data` 逐像素相同。

已确认的背景与原生尺寸合同：Pixiu 为黑背景、512×512；GS3 Translucent 为白
背景、512×512；SSS-GS bunny 为黑背景、256×256。不能用单一白背景 256×256
配置声称复现三者的基准评价。

Pixiu 训练颜色合同只读取训练图像：baseline `train/ours_100000/gt` 与当前
`data/Real_NRHints/Pixiu/train` 都有 562 张。对两边均缩为 32×32、只比较固定
规则的 `RGB*alpha`、`(RGB*alpha)^(1/2.2)`、`(RGB*alpha)^2.2` 后，原始
`RGB*alpha` 得到 562 个互异最近邻；`gt/00000..2.png` 的候选分别为
`r_536.png`、`r_355.png`、`r_292.png`，32×32 MSE 分别为
`4.47e-8`、`6.95e-8`、`9.93e-8`。512×512 全像素核对中，按原始合成值取整后
三帧 exact fraction 分别为 `0.9999682`、`0.9999619`、`0.9999644`，最大
通道差均为 1/255；两种 gamma 规则的 RMSE 约为 `0.107/0.122/0.116` 和
`0.075/0.128/0.112`，明显不匹配。因此已核实为训练帧顺序置换加 PNG 量化，
没有 gamma 变换证据。该核对没有读取 Pixiu test 或 baseline 预测。

## 训练指标实现核对

只对 Pixiu 与 GS3 Translucent 的 `train/ours_100000/{renders,gt}` 做了 CPU
PNG 配对计算。全部配对的逐图 PSNR 宏平均与 `results_train.json` 一致：Pixiu
562 张为 `33.44349209006873`，记录值 `33.4434928894043`，差
`-7.99e-7`；Translucent 2000 张为 `33.3390170147654`，记录值
`33.339019775390625`，差 `-2.76e-6`。逐图 PSNR 与对应
`per_view_train.json` 的最大绝对差分别为 `4.86e-6` 和 `6.46e-6`。

对文件 `00000.png`、`00001.png`、`00002.png`，PORT 的 `evaluate.ssim` 与
`per_view_train` 一致到约 `1e-7`。LPIPS 只各取 Pixiu/Translucent 的
`00000.png` 一对做 VGG CPU 核对：`lpips.LPIPS(net="vgg", version="0.1")`
直接接收 `[0,1]` PNG 张量时分别得到 `0.09466355293989182` 和
`0.029003553092479706`，与记录值 `0.09466402232646942` 和
`0.029003042727708817` 相差约 `5e-7`。因此保存结果的 LPIPS 合同是 VGG
v0.1 + `[0,1]` 输入；PORT 当前将输入映射到 `[-1,1]` 的路径不能宣称与该
基准 LPIPS 等价。该核对没有读取任何 test 或把 baseline 预测输入训练。

## NRHints 训练相机优化来源核查

官方 NRHints README 对真实采集场景推荐开启 camera optimization，并明确说明这
是论文投稿后加入的改进，细节应看 author's version。该 README 只说明真实场景的
训练选项，不能据此推断 SSD-GS 历史 baseline output 是否启用了相机优化。

author's version 的 §3.5 将初始视点 `R0,t0` 修正为
`R=ΔR·R0`、`t=Δt+ΔR·t0`，其中 `ΔR∈SO(3)`、`Δt∈R³` 为可学习修正，并说明
重建损失在训练期间反传到这些修正。论文将其动机归因于真实相机标定误差造成的
细节模糊；没有说明用 heldout/test GT 像素优化评价相机，也没有给出 evaluation-only
pose fitting 流程。论文的相机采集段还说明第二相机图像用于标定光源位置，棋盘格
用于辅助相机标定。

当前 PORT 的 `TrainCameraOffsets` 只为 fit/train frame 建立偏移，旋转界为
`0.03*tanh(raw[:3])`，CV 相机平移界为
`0.03*object_radius*tanh(raw[3:])`；fit index 0 固定为零锚点，validation 始终使用
原始相机。调用方默认相机学习率记录为 `0.0003`；早期 `0.01` 初试不作为协议或
推荐配置。单 fit 的梯度诊断只证明偏移参数可反传；小 mask 运行中 corrected-fit
指标有所改善，不能解释为 validation 改善或正式基准成绩。

来源：[NRHints 官方 README](https://github.com/iamNCJ/NRHints)、[论文 PDF（数据采集与校准段，pp.5--6）](https://nrhints.github.io/pdfs/nrhints-sig23.pdf)、[author's version §3.5](https://arxiv.org/html/2308.13404v1)。

## 条件编码与 CPU 核查

当前实现新增标准 half-vector 条件：对 `normalize(light_dir + view_dir)` 做 4-band
位置编码（27 维），追加 `dot(light_dir, view_dir)`（1 维），再接到原有 54 维方向
特征之后；因此局部第一层输入增加 28 维，每个隐藏宽度增加 `28*width` 参数。
该条件同时用于普通前向和 deferred 查询前向，当前 GPU 集成尚未重新测试。它是
标准视线/光线 half-vector 参数化，不作新颖性声明；参考
[Rusinkiewicz, 1998](https://www.cs.princeton.edu/~smr/papers/brdf_change_of_variables/)。

CPU 已通过梯度有限性、光强线性叠加与零光强 blackout 检查；原有（无 half-vector）
路径保持 bit-exact。上述检查不构成 GPU 或最终指标验证。

## 本地数据审计

Luna Max 只读检查了数据元数据与结果文件，没有读取测试图像，没有运行 GPU。

| 数据集 | 训练条数 | 验证状态 |
|---|---|---|
| Real_NRHints | Cat 522; CatSmall 1258; CupFabric 1153; Fish 526; FurScene 676; Pikachu 1500; Pixiu 562 | transforms_valid 各 400 条，但所指 EXR 在数据树中不存在 |
| Synthetic_GS3 | 六场景各 2000 EXR | 无 val |
| Synthetic_SSS-GS | 五场景各 500 条元数据 | train/val/test 均 500 条；所有被引用 PNG 存在 |

SSS 引用的唯一 PNG 数（总条数均 500）：

| 场景 | train | val | test |
|---|---:|---:|---:|
| bunny_small | 498 | 496 | 498 |
| candle_small | 497 | 493 | 498 |
| dragon_small | 497 | 493 | 498 |
| soap_small | 498 | 498 | 496 |
| statue_small | 497 | 495 | 496 |

SSS 每 split 的 `(transform_matrix, light_positions)` 唯一组合为 500；train/val/test 的组合交集为 0，精确比较以及保留 6/8 位小数比较一致。**这不等于已经排除所有图像内容重复或分别重复的相机/光源方向**：未读取图片，也没有做内容指纹。相同 PNG 被不同元数据引用本身需要在评价说明中披露。

Real/GS3 有 pl_pos 与 pl_intensity。SSS 只有 light_positions/light_pos，没有 intensity/power/radiance/strength 字段；仅能确认位置已知。当前 SSS 运行必须显式给出 `unit_light_intensity=1.0`，它只是等功率 unit-radiance 假设。

所有场景有 points3d.ply，带位置、法线和 RGB；其来源没有核实，不能默认可作训练监督。未看到单独 normal/albedo/roughness/material 图片，不声称存在完整 GT 材质。

示例来源：

- [Real 元数据](/workspace/datasets/SSD-GS/data/Real_NRHints/Cat/transforms_train.json)
- [SSS 训练元数据](/workspace/datasets/SSD-GS/data/Synthetic_SSS-GS/bunny_small/transforms_train.json)
- [SSS 训练标注](/workspace/datasets/SSD-GS/data/Synthetic_SSS-GS/bunny_small/train/anno.json)
- [本地 Cat 指标](/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/SSD-GS/output/real_nrhints_Cat_20260430/results_test.json)

## SSS-GS 端到端 split 与坐标核对

2026-09-06 对 `Real_NRHints/Pixiu`、`Synthetic_GS3/Translucent` 和
`Synthetic_SSS-GS/bunny_small` 做了只读 metadata 核对。`SceneDataset` 的
`train`、`test` 分别读取同名 `transforms_train.json`、`transforms_test.json`；
Real 的 `val` 读取 `transforms_valid.json`，SSS 的 `val` 读取
`transforms_val.json`，GS3 没有 `transforms_val.json`，请求会明确失败。测试逐条
比较了每个 JSON 的完整 frame 记录和全部引用路径，构造数据集时未解码 val/test
图像；只解码了一个正确 SSS train 样本。可复现实验位于
[test_data_contract.py](../test_data_contract.py)。

官方 `cgtuebingen/SSS-GS` 的相机 reader 将输入 `M=transform_matrix` 的相机局部
Y/Z 轴乘以 `F=diag(1,-1,-1,1)`，再计算 `W=inverse(M@F)`，并以转置后的
`R`、`T` 装配 world-view；这与 PORT 的 `c2w=M`、
`viewmat=inverse(c2w@F)` 一致。官方代码见
[dataset_readers.py](https://github.com/cgtuebingen/SSS-GS/blob/main/scene/dataset_readers.py)、
[dataset.py](https://github.com/cgtuebingen/SSS-GS/blob/main/scene/dataset.py)、
[scene/__init__.py](https://github.com/cgtuebingen/SSS-GS/blob/main/scene/__init__.py)
和 [graphics_utils.py](https://github.com/cgtuebingen/SSS-GS/blob/main/utils/graphics_utils.py)。

灯坐标的端到端结果与早期只看 reader 的结论不同：官方 `dataset_readers.py` 在
构造 `CameraInfo` 时先翻转 Y/Z，而 `CameraDataset.__getitem__` 又翻转一次，
所以最终 `light_pos` 是原始 `frame["light_positions"][0]`（`D²p=p`）。本地
`train/anno.json` 也存储这个 raw world 坐标。此前记录的“SSS 灯坐标单次翻转”
已被这次端到端核对否定；角度划分保留的一次 Y/Z 变换只属于固定的 180° X
分组基底，不是 renderer 坐标变换。

对 train frame 0 `r_192_l_61`，`c2w[:3,3]=[0.86273503,1.5693238,3.6114972]`，
当前 `viewmat[:3,:4]` 与 `anno.RT` 最大绝对差为 `1.2931e-7`；500 个 train
frame 的最大差为 `8.4066e-6`，`K` 和 raw light 字段逐项一致。raw light 为
`[-0.7209595,-3.6245086,1.5307338]`；一次翻转会错误地得到
`[-0.7209595,3.6245086,-1.5307338]`，官方端点和 anno 均为 raw 值。

此前的 `runs/bunny_port` 还使用了已修正的 split 映射，`split="train"` 实际
读取了 `transforms_val.json`。因此它的 23 dB 结果是作废诊断，权重不可复用；
此前称其为“train-only”的声明也被本次端到端核对否定。该核对没有读取官方
test 图像。

## 环境与工作边界

已确认 Python 3.10.19、Torch 2.4.1+CUDA12.1，并发现 gsplat/tinycudann/scipy/Pillow/lpips；diff_gaussian_rasterization 包未发现。没有修改该环境。PDF 阅读依赖仅临时装在 /tmp 独立目录，不进入训练环境。

用户要求跳过 WORKFLOW.md，因此没有读取或沿用该工作流；没有查看其他自研项目。SSD-GS 仅作为结果与论文表格参照，未读取方法代码；PDF 表格提取曾附带输出相邻正文，随后改为表格区域裁剪，没有据此开展方法分析。后续检索 GS3 数据采集条件时，宽泛搜索结果也意外附带了 SSD-GS 摘要片段；未点击这些结果或用它们设计组件。此后相关检索应限定已知的非 SSD-GS 官方来源域名，避免搜索摘要再次越过用户要求的阅读范围。PORT 核、半角输入和 deep shadow 设计均有此前独立推导、代码与经典来源记录，不能把本记录理解为“从未在工具输出中看到任何 SSD-GS 正文”。

当前 GPU 使用为 0 张。后续最多两张当时空闲 GPU；是否能达到 SOTA 由按协议运行后的结果决定。

## 检查记录

CPU 小型随机矩阵检查通过：成对核对称、低秩乘法与显式矩阵一致、光强叠加与零光响应一致，最大绝对误差低于 1e-12；非负输入输出非负。该检查只验证离散公式的代数性质，不验证渲染质量、物理真实性或数据集泛化。另核对了 18 条本地指标文件路径以及文档内本地链接。未训练模型，也未执行正式评价。
