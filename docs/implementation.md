# 实现与运行记录

## 当前实现

2026-09-06：独立 PyTorch + gsplat 实现。`gaussians.py` 管理几何，`transport.py` 实现局部与成对空间传输，`renderer.py` 使用原生光栅化；`train.py` 和 `evaluate.py` 分离训练与评价。数据读取由 `data.py` 负责。

已用 Real_NRHints/Pixiu 的真实训练相机/光源做 128×128 CUDA 前向反向检查，输出形状正确，几何梯度全部有限。没有使用 SSD-GS 权重、方法代码或其他用户项目。

## 与初始设计的具体化

- 点光源均位于物体外部时，灯侧采用一个动态拟合场景的透视相机。视锥覆盖 Gaussian 的 3-sigma 支持；若支持进入灯相机后方则报错，不能把部分视锥当作完整球面可见性。该已核实的外部灯条件无需六面图。预期深度产生平滑遮挡提示，当前不对该提示反传，因此只是近似可见性。
- u/v 的空间权重由 Gaussian 特征和标准化位置解码，方向项采用可学习球面 Gaussian。成对角度轴独立，RGB 强度独立；R=16 为初值。
- 空间求积权重由 opacity × scale 体积导出并归一化；求积权重暂时 detach。该权重是有效离散化约定，不代表已恢复真实表面积或严格能量守恒。
- 灯强归一化为仅从拟合训练图像元数据计算的 `median(I / distance_to_training_camera_center²)`，该单一常数保存在模型内，对验证/测试不重新估计。它只改变数值单位，不是图像校准。
- 训练光照 holdout、纯训练相机初始化；不使用提供的点云。开发阶段所有图像仅来自 transforms_train。
- 内部评价显式使用存储域 RGB、clamp [0,1]、unit-range PSNR、11×11 Gaussian SSIM，另保存 raw MSE。LPIPS 只在显式评价时启用 VGG。该协议尚未证明与 SSD-GS 的最终报告完全一致，所以内部开发分数不作 SOTA 声明。

## 核查来源

仅读取以下原始数据集项目的数据 I/O 约定，没有基于其方法实现设计模型：

- [NRHints data parser](https://github.com/iamNCJ/NRHints/blob/main/data/data_parser.py)：Real intrinsics 顺序为 cx,cy,fx,fy；PNG /255，EXR 保留数值；alpha 背景合成。
- [GS3 dataset reader](https://github.com/gsrelight/gs-relight/blob/main/scene/dataset_readers.py)：相同数据数值读取习惯，训练/测试带 pl_pos/pl_intensity。

读取代码不做 gamma 不能证明 PNG 的物理辐射响应线性。当前存储域结果先用于遵循数据约定的重建开发；物理光强缩放检验需分别说明真实图像与线性 EXR 的适用范围。

## 开发阶段决策（未读取正式测试图像）

- 同几何、同训练预算、同参数量的空间消融：Pixiu 完整 PORT 为 20.0184 dB，禁止跨位置传输为 18.9882；Translucent 对应 22.2885 与 21.9606。它们是训练内留出光照的 256px 开发结果，尚不能外推到完整基准。
- Pixiu 后续联合训练后完整 PORT 为 21.7624、局部端点控制为 20.0873。两者各包含 5K 预训练、5K 固定几何和 30K 联合训练。
- 标准 GGX + 与 Gaussian 朝向绑定的表面法线/薄片正则没有稳定改善两个开发场景，保留为 `--local-model ggx` 的研究控制，当前主方案仍是 neural 局部核。GGX 采用标准微表面公式，不是新贡献；参考 https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory 。GGX 分支的单一全场景辐射尺度仅从训练优化，不允许测试 GT 校准。
- 增大 mask 权重只略微改善轮廓误差，未解决整体不足。固定几何关闭 shadow hint 后 Pixiu 21.3550，保留为 21.9847，因此当前保留该近似提示。
- 初始 frozen 对照结束后，修正了灯图采样的半像素偏移：gsplat 以 j+0.5 为像素中心，grid_sample 的标准化坐标应为 2*projection/resolution-1。旧实验记录原样保留；联合阶段及其后实验均使用修正版本，旧 checkpoint 重新评价时会遵循当前修正渲染器，不能冒充旧日志的精确复现。
- DefaultStrategy 调用顺序核对安装版本 gsplat v1.5.3 官方 example，optimizer.step 在 step_post_backward 之前是其明确示例。没有采用不成立的顺序修改建议。
- 原 max-points 为停止下一次增长的软阈值，导致初轮 Translucent 达到 200773。后续实现超过阈值时按低透明度裁剪并同步 optimizer/state，成为实际点数上限。
- 原数据 Pixiu/Translucent 的 points3d.ply 完全相同：100000 个 [-.5,.5]^3 均匀点、零法线、常数127 RGB，符合原 GS3 数据读取器的随机初始化分支。新诊断仍自行随机生成点，不将任何数据点云、法线、颜色当作监督。`--init-radius .5 --points 100000` 对齐其初始范围/数量；这同时影响模型归一化与优化器场景尺度，不能把增益仅归于点数。
- SSS loader 已实现，但其光功率未知。必须显式传 `--unit-light-intensity`，记录等功率假设；端到端官方 reader/CameraDataset 对灯坐标各翻转一次，因此 loader 输出 `light_positions[0]` 的 raw world 坐标。角度划分保留的一次 Y/Z 变换只属于分组基底，不是 renderer 坐标变换。当前训练器的 validation 仍是正确的 train 内留出，不会自动使用官方 val。

## 评价合同核实与局部响应约束

从三个标准输出的 cfg_args 仅提取数据/评价字段，未读取方法代码：Pixiu resolution=-1, white_background=False，cameras.json尺寸512×512；Translucent resolution=-1, white_background=True，512×512；SSS bunny resolution=-1, white_background=False，256×256。记录的旧source_path目前不存在，因此仅能核实记录的协议，尚未证实旧输入图像与当前数据逐字相同。此前白背景256开发实验只支持内部消融，不能直接用于SOTA比较。

同一训练视图的诊断（不是基准成绩）：从 Pixiu PORT 联合模型出发，只拟合一个允许的train图像1000步，固定geometry可达到34.48 dB；允许geometry优化可达到56.52 dB。诊断模型不用于后续训练或评价，没有把此分数当作泛化结果。这排除了渲染器完全无法表达目标图像的情况，仍需解决多视角/光照一致拟合。

发现局部响应存在测度问题：若 c=I/r² * f_r(l,v) * max(n·l,0)，即便 f_r 满足互易，整体响应 rho=f_r*max(n·l,0) 通常也不满足 rho(l,v)=rho(v,l)。初版直接对拟合颜色响应施加交换对称，未显式分离入射余弦和投影测度，因此可能过度限制表示。增加 `--asymmetric-local` 作为同参数量控制，允许局部响应表达方向非对称；成对非局部结构保持不变。不得把离散矩阵对称的代数测试当作完整成像过程物理互易的证明。

## 固定颜色观察模型

核查 baseline TRAIN GT（不是 test、也不是用于蒸馏的预测）：Translucent 前三张保存图分别有 >99.98% 量化通道等于 `round(clamp(EXR*alpha + white*(1-alpha),0,1)^(1/2.2)*255)`。Pixiu 的 train GT 是原训练 PNG 的次序置换与黑背景合成，前三张对应r_536/r_355/r_292，直接量化一致率 >99.996%；未发现对Real输入再做gamma的证据。

`--display-gamma 2.2` 模式使输出传输保持线性辐射，损失和指标应用固定gamma观察映射。PNG GT保持其原编码合成值；EXR GT先线性合成再固定gamma。该转换对整个数据集固定，不按图像拟合、不依据test优化。Real预测采用该近似观察模型属于对未知真实相机响应的建模假设，不将PNG文件本身视作辐射标定证据。旧 `display_gamma=1` 实验保留为旧观察域开发记录，禁止与新域直接拼表。

历史字段 raw_MSE 在此模式下表示未clamp的观察域MSE，而不是线性HDR误差；JSON协议已明确。正式比较前仍需完成与基准TRAIN存档的指标实现核查。

## 数值稳定性审查后的修正

独立CPU审查发现：旧 Pixiu joint 模型27.34%的opacity已在float32中精确等于1，且对logit导数精确为0；asymmetric_black达37.66%。旧joint尺度长短比中位约1.21e5，GGX surface薄轴/半径中位约1.61e-13。因此，单纯加大mask损失无法修改已死亡的opacity梯度。

新训练使用投影约束：opacity<=.99，所有scale>=scene_radius*1e-4。它们在参数更新及densify/prune之后执行，初始化/载入继续训练时也执行。该尺度下限远小于典型像素足迹，目的是避免数值退化，不是用GT约束几何。`--opacity-cap 1 --min-scale 0` 可明确复现无界控制。旧模型纯evaluation不会被偷偷修改。

旧PORT入射权重opacity*sx*sy*sz随薄轴趋零失效：joint仅84/181183个点占50%积分质量；surface仅3/139511。新 `--transport-measure area` 使用 opacity*(sx*sy+sy*sz+sz*sx)，在薄片极限仍有非零面积权重。保留volume用于同条件消融。该权重是近似求积，不保证split-invariance或能量守恒；已删除原先不成立的'归一化可保证增密不改变通量'表述。几何固定阶段的核比较不受动态split影响。

## 指标实现核查

仅使用baseline TRAIN存档核对：Pixiu562张、Translucent2000张逐PNG PSNR平均与results_train相差不到3e-6 dB；PORT的11x11 sigma1.5零填充SSIM匹配到约1e-7。基准LPIPS数值与VGG v0.1直接输入[0,1]匹配，和通常要求的[-1,1]不同。

正式evaluate.py将预测和GT按uint8量化后计算PSNR/SSIM；同时保留标准输入[-1,1]的LPIPS，以及明确单列的 `LPIPS_baseline_01`。后者只用于对齐历史基准约定，不能与标准LPIPS混称。训练内快速验证仍采用未量化观察值。没有使用baseline预测监督PORT，没有读取正式test图像。

## Deep shadow 与 fit-all integration checks

`visibility_hint(..., mode="deep")` 沿现有外部灯光视锥使用 256 x 256、64 个
深度 bin；每个 Gaussian 的深度在其 3-sigma near/far 范围内向相邻 bin 线性
分配，再对每像素各 bin 的 alpha 做前缀累积。接收深度查询使用条件射线 sigma
和至少 1.5 个 bin 的 z 偏移；越过 near 的查询明确返回 visibility=1。这是
有限深度分辨率的自遮挡排除近似，不是精确可见性，也没有新增缓存、CUDA 内核
或学习式阴影网络。经典参考为
[Deep Shadow Maps](https://lightfield.stanford.edu/papers/deepshadows/)，该
标准技术本身不构成 PORT-GS 创新。

GPU 两 Gaussian 控制记录在
[`runs/deep_shadow_unit.json`](/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/runs/deep_shadow_unit.json)：前方遮挡 Gaussian alpha 为
0、0.2、0.8 时，后方 Gaussian 的 visibility 分别为 1、0.800035、0.200142；
默认 `mode="depth"` 与显式 `mode="depth"` bit-exact。由于检查只覆盖构造的
前后遮挡样例，不能据此宣称完整场景的精确阴影或物理互易。

`fit_all_smoke` 已完成 2 步（100 点、32 px），同时启用 `fit-all`、half-vector、
multiscale ports 与 deep shadow；该运行不使用 validation，
`best_validation_psnr=null`。它只验证入口组合能启动和收尾，不是重光照指标或
泛化结果。

## Training-camera adjustment

`cameras.py` 提供 train-only 的 `TrainCameraOffsets`。每个 fit frame 有一个 6 维
零初始化 raw 参数；第一个 fit frame 是零锚点。旋转使用
`0.03*tanh(raw[:3])`，CV 相机平移使用
`0.03*object_radius*tanh(raw[3:])`，并以 `matrix_exp(skew(rotation))` 构成左乘
SE(3) 修正。当前调用方默认相机学习率记录为 `0.0003`；早期 `0.01` 初试保留为
失败诊断，不作为默认协议。

`correct(sample, local_fit_index)` 只更新训练 sample 的 `viewmat/c2w`，validation
始终使用原始 metadata 相机，不把 train offset 转移到 heldout。单 fit 梯度诊断只
验证偏移参数有有限、非零梯度；小 mask 运行中 corrected-fit 指标改善不等于
validation 改善，也不构成基准成绩。

NRHints author v1 §3.5 记录了训练期间联合优化 viewpoint 与表示的公式及校准误差
动机，但没有明确 GT/test pose fitting。来源：[NRHints README](https://github.com/iamNCJ/NRHints)、
[author v1 §3.5](https://arxiv.org/html/2308.13404v1)。不要据此推断 SSD-GS
历史 baseline 使用了相机优化。

## 后续数值边界与着色粒度诊断

GGX corrected运行在约20.6K因完整阴影视锥断言失败而停止。20K检查点中最大scale=1.2009、opacity=.00293；旧thin ratio sz/min(sx,sy)在sz碰到下限后仍可通过增大切向scale降低损失，产生低透明度巨型primitive。新约束补充scale<=.1*scene_radius（与原DefaultStrategy的尺度裁剪量级一致），thin正则改为sz/scene_radius，避免奖励切向发散。没有用缺失阴影的替代输出隐藏失败。

独立核查bounded-volume检查点：opacity零梯度点数从数万降至0，max opacity=.99000007；min scale/radius>=1e-4。该几何下volume求积有效点数约333，area约5166；这仅证明数值支持更合理，不证明最终指标改进。

`--deferred`新增标准神经延迟着色控制，限定neural局部模型。先splat世界位置/base/features/visibility，再alpha归一化，于像素查询点着色；nonlocal入射积分始终来自完整源Gaussian集合。复杂度为O((N+HW)R)加光栅化，不引入图像ID、屏幕空间自由参数、GT深度或额外神经图像后处理。

CPU用真实训练checkpoint检查17个Gaussian查询与原完整前向对应输出，误差0；query backward梯度有限。128px、真实train数据CUDA smoke完成200步，未出现NaN。其21.16 dB为小样本低分辨率功能检查，不是完整基准成绩。延迟着色本身是已有技术，研究贡献仍需由跨位置传输的等容量消融证明。

当前局部条件还追加标准 half-vector 特征：`normalize(light_dir+view_dir)` 的 4-band
位置编码 27 维与 `dot(light_dir,view_dir)` 1 维，接在原 54 维方向特征之后；普通
前向与 deferred 前向均使用，局部第一层增加 `28*width` 参数。GPU 集成仍待测试，
CPU 梯度、光强线性叠加和 blackout 已通过，旧路径 bit-exact。该参数化遵循标准
half-vector 处理，不作新贡献声明；参考
[Rusinkiewicz 1998](https://www.cs.princeton.edu/~smr/papers/brdf_change_of_variables/)。

## Multiscale PORT 消融

`multiscale_ports=False` 默认保留原有的 softplus concentration 参数化。打开
`multiscale_ports=True` 时，使用可学习的 log-sharpness 初始化
`log([0.7, 2, 8, 32])` 并按 rank 重复，普通前向和 deferred 前向都用 `exp` 还原
sharpness；替换参数不增加总参数量。动机是 CPU checkpoint 审计中，18K 步的
Translucent/Pixiu 旧 spherical-Gaussian sharpness 约停留在 `0.6--1.35`，全局
PORT 因而偏向宽核；这只是实验性参数化理由，不是新颖性或有效性证明。

CPU 已在 `multiscale_ports=True`、`half_vector=True` 下核对普通/deferred 前向：
17 个 Gaussian query 与完整前向对应输出的最大绝对误差为 `0`，普通/查询分支
梯度均有限；光强线性叠加误差约 `1.4e-9`，deferred 约 `9.3e-10`，zero-light
blackout 均为 `0`。默认与 multiscale 参数量均为 `1520`（该 CPU 配置），初始
sharpness 为 `[0.7,2,8,32]` 循环。GPU 集成和质量收益仍待测试，不把该消融写成
benchmark 结果。

## Full-train final protocol

默认按灯光角度分组的约 10% train holdout 保持不变，用于开发阶段选择设置与步数。
设置锁定后，最终重训通过 `train.py --fit-all` 使用全部官方 train 帧，保持锁定的
配置和步数。该模式将 `val_indices` 设为空，不计算 validation 指标、不做 validation
checkpoint 选择，只周期性保存 `last.pt`；完成事件的
`best_validation_psnr` 为 `null`。最终模型完成后才显式运行 `evaluate.py --split test`，
不使用 test GT 做相机、颜色或模型校准。

对没有 validation 的 checkpoint 请求 validation 评价会明确报错，评价调用必须选择
实际存在的 split。`runs/fit_all_smoke` 已完成两步 GPU 集成检查：checkpoint step=2，
fit_indices 包含全部 562 个训练帧，val_indices 为空；这不是质量实验。

## Fit diagnostics and local capacity

`evaluate.py --split fit --limit 32` 在保存的 fit_indices 上固定间隔抽样，所有相机
保持原始 metadata；它不会应用训练相机的 nuisance offsets。`--split validation`
默认评价全部 heldout 帧，正式 test 应保持 `--limit 0`。报告保存实际帧数和 limit。

局部 MLP 默认保留两层隐藏层，可用 `--local-layers 4` 检验训练欠拟合是否来自
局部容量。旧两层 checkpoint 的 CPU 加载以及四层网络前向/反向有限性已通过；
Translucent 的四层 GPU 运行已验证，最佳训练内 validation PSNR 为 `28.65394`。
它是容量控制，不是独立创新声明。

`--absgrad` 使用 gsplat 原生 raster 输出和 `DefaultStrategy` 的 abs-gradient
路径，梯度阈值为 `0.0008`；普通梯度对照阈值为 `0.0002`。这只改变增密触发信号，
不引入自定义 rasterizer 或额外监督，质量收益需按同一训练协议比较。

旧 `runs/bunny_port` 使用过已修正的 split 映射，`split="train"` 实际读取了
SSS `transforms_val.json`；其 23 dB 结果与权重均作废，不得复用。该结论与
[`docs/data_contract.md`](data_contract.md) 和 [`docs/evidence.md`](evidence.md)
中的端到端坐标与 split 核对一致。

## 失败分析用传输控制（2026-09-06）

- `--normalized-ports`：面积加权 L2 空间基，deferred 复用全部源点归一化。Bunny 30k 未改善，默认关闭。
- `--view-independent-local`：在构造编码前将局部 view direction 置零，保持网络维度；Bunny 30k 比对照低 2.07 dB，默认关闭。
- `--localized-ports`：学习空间中心和宽度，partition-of-unity 汇聚/分配源照度。默认关闭；与前述 normalized 或 local-ports 冲突时直接报错。局部神经响应保持原样。

三个选项均保存在 config/checkpoint，并由 evaluate 恢复；旧 checkpoint 默认关闭。数学验证脚本为 verify_normalized_ports.py、verify_view_independent_local.py、verify_localized_ports.py。研究结果和解释边界见 failure_analysis.md。
