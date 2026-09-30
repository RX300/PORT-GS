# SDF体渲染亮点采样对照

日期：2026-09-23。状态：两组各3000步、同16训练视图评价、图像比较与最终审计全部完成。
亮点采样降低了GT峰处RGB误差，但产生过宽亮斑并明显损害整体拟合；未恢复准确细小高光，不设为默认。
下一轮独立机制对照见 [局部邻域监督与提示编码](sdf_peak_shape.md)。

## 问题与已有证据

前轮 `sdf_volume_detail_pilot` 已完成。几何细节网格相对仅外观网格使同16训练视图的辅助分支
PSNR从23.122提高至23.502，alpha误差下降约10.6%，但仍未恢复准确的细小高光。
主GS、主传输和相机未改变；该结果没有证明真实几何或未见灯光泛化改善。
本轮从已完成的几何网格模型继续，检验训练输入与稀疏射线预算是否限制亮点拟合，不重复网格对照。

`data.py` 对PNG采用预乘alpha后的 `INTER_AREA` 缩放，再恢复前景颜色。
这是现有数据契约；本轮不改变颜色空间、gamma或缩放实现。
在与前轮相同的16训练视图上，先用原生512px目标和 `evaluate.neutral_peak_mask`
确定5390个中性局部亮点代理像素，再检查128/256px输入在这些相同像素处保留的信号。

| 输入分辨率 | 原生亮点位置neutral均值 | 仍高于neutral .55的比例 | 原生局部对比保留比例 |
|---|---:|---:|---:|
| 128px | .549026 | 45.44% | 40.28% |
| 256px | .667793 | 72.28% | 74.39% |
| 512px | .754874 | 100% | 100% |

此处neutral为RGB最小通道。缩放后图像以最近邻还原到512px，使用固定原生GT亮点集合和相同11px
原生窗口计算局部对比，先算每帧对比比值再按该帧亮点数加权。
它测量输入细节衰减，不是模型渲染质量，也不是物理高光真值。
各分辨率上直接重算的亮点掩码使用固定像素窗口，计数不具尺度不变性，不能直接当成亮点保留率。

原采样器每步有放回抽取256条前景射线和256条SDF定义域内射线。
以源checkpoint的SDF立方体、保存的训练相机校正和同16帧，精确统计采样池大小：

| 分辨率 | 每步期望命中GT亮点射线数 | 单像素在新增3k步内期望被采次数 | 单像素在新增3k步内从未采到的概率 |
|---|---:|---:|---:|
| 128px | 4.109664 | .556114 | 59.31% |
| 256px | 3.138915 | .137958 | 87.30% |
| 512px | 2.104730 | .034395 | 96.63% |

上述为16帧算术平均，假设每步在全部562训练帧中均匀独立选帧，只计算新增3k步。
它不表示模型此前没有见过这些像素，也不排除共享网络从其它像素获得间接监督。
5390个原生亮点均在SDF定义域内。另检查全部562帧512px输入，每帧都有亮点代理，
数量93至1039，均值378.972，中位357，10%/90%分位为209/588.9。

诊断已归档为 `runs/sdf_volume_peak_sampling_pilot/input_diagnosis.json` 和
`diagnostic_source.tar`。JSON另含全部562帧的亮点计数。

## 假设与单变量设计

128px输入会消减原生小亮点；直接提高到512px又会进一步降低固定512射线预算对单个像素的覆盖。
本轮两组统一使用512px原生输入，单独比较亮点定向采样是否改善辅助辐射分支的训练拟合。
这不直接估计提高分辨率的因果效应；要判断分辨率本身的作用，需另做相同评价分辨率的对照。

对照组 `uniform` 设置 `sdf-volume-peak-fraction=0`，保持原256前景+256定义域采样。
候选组 `peaks` 取 `.25`，每步128条训练GT亮点、192条前景、192条定义域射线，总预算仍为512。
所有采样均有放回，亮点候选与有效定义域取交集；无亮点时整批回到原前景/定义域分配。
这组Pixiu数据在输入检查中没有空亮点帧。

候选组不做逆概率补偿，是显式提高亮点权重的优化目标。
在上述16帧统计下，其期望亮点射线数约129.579/步，为原采样约61.6倍。
不能将其称为无偏估计或只降低原目标的采样方差。
预测是亮点RGB误差和局部对比可能改善，但过宽亮斑、误报增加或整体图像质量下降也可能出现，
因此必须同时检查亮点位置、precision/recall、整体指标及可视化，记录负结果。

## 固定训练协议

- Run名称：`sdf_volume_peak_sampling_pilot`，仅 `Real_NRHints/Pixiu`。
- 共同初始化：
  `runs/sdf_volume_detail_pilot/geometry_grid/Real_NRHints/Pixiu/last.pt`。
  保留已有几何与外观细节网格、辐射头和训练步数，不重复初始化或头预热。
- 两组各追加3000步，seed0，512px，全部562训练帧，`fit-all=true`。
- 每步512射线、64粗区间+64重要性细采样，两组总射线数与优化步数相同。
- `sdf-volume-only=true`、`freeze-sdf=true`、`sdf-detail=true`、`sdf-volume-detail=true`。
  固定SDF距离场参数、全部GS属性、主传输模型及已保存相机校正；不增密、不重置材质或曝光。
- 辐射MLP、外观细节网格与 `log_sharpness` 仍学习。
  距离场固定不等于体渲染权重或渲染alpha/depth完全固定，不能将本轮称为固定全部体渲染几何。
- 复用现有RGB L1、alpha L1及GS→SDF方向的渲染depth/normal项；仅更改采样分布。
  更改分布会一起改变RGB、mask及有效depth/normal损失的加权分布。
  固定距离场时不更新该场的Eikonal项，depth/normal仍可能通过合成权重影响sharpness。
- 使用ssd-gs环境和CUDA12.1，不新增或升级依赖。最多使用两张启动前检查空闲的GPU；
  实际设备、命令、配置与源码由manifest、日志和快照记录。

RGB监督仍处于既有观察空间，黑底，display-gamma=2.2；数据集PNG目标不额外做gamma变换。
已有 `--highlight-weight` 是主GS的图像loss，在volume-only中会被过滤，本轮不借该参数给aux加权。
新的 `--sdf-volume-peak-fraction` 直接控制aux射线分配。

## 评价与有效结论边界

两组均在原生512px上评价相同16个已参与训练的视图：
`[0, 35, 70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525]`。
使用保存的相同训练相机校正，渲染SDF volume分支；共同源模型已在相同分辨率上评价，
不能把前轮128px的PSNR直接作为本轮512px的初始分数。另记录冻结主GS在同16帧512px上的参考结果。

报告PSNR/SSIM/LPIPS、alpha L1，以及GT中性局部亮点RGB MAE、局部对比/GT、
2px邻域内precision/recall和亮点像素计数，检查原图与局部放大。
亮点mask来自训练GT，属于图像代理而非真实specular分解；不得只凭亮点变亮宣称恢复了正确高光。
固定距离场的审计应逐位比较SDF参数、GS、相机及主传输，并分别报告辐射头、外观网格、sharpness的变化与有限性。

本轮只做训练拟合诊断，不使用official test调参，也不构造伪validation。
共同初始化及其上游模型均已拟合全部562训练帧；现在抽出的任何这些帧都不能称为未见验证集。
后续未见灯光验证必须在GS、相机、SDF、辐射头等任何学习之前固定划分，核查完整初始化来源。
无几何GT且主GS冻结，本轮不能证明真实几何精度、主2DGS重光照或未见灯光泛化改善。

## 完成结果

以下均为同16个已参与训练的视图、原生512px、相同5390个GT亮点代理像素。
亮点指标使用量化至uint8的观察空间RGB，11×11局部对比窗口，2px Chebyshev邻域匹配；
按像素数汇总亮点指标，precision/recall从总计数计算，不先平均逐帧比值。

| 分支 | PSNR | SSIM | LPIPS | alpha L1 | 亮点RGB MAE | 亮点对比/GT | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 源SDF，512px重评 | 22.863321 | .858262 | .164227 | .026464 | .333829 | .014393 | 无预测峰 | .000000 |
| uniform | 23.284552 | .864632 | .157497 | .026278 | .343188 | .016977 | 1.000000 | .009091 |
| peaks | 21.152530 | .845624 | .165776 | .026846 | .114115 | .136336 | .203553 | .154174 |
| 冻结主GS参考 | 26.078235 | .898818 | .118444 | .020691 | .158159 | .451671 | .673069 | .603525 |

`uniform`相对源SDF的PSNR提高.421231dB，16帧全部提高，但亮点MAE只有1帧下降，
且仅预测15个峰像素；因此precision=1不能当作有效高光恢复。
`peaks`相对`uniform`的GT峰处RGB MAE下降66.75%，16帧全部下降，
但PSNR下降2.132022dB，16帧全部退步。预测峰像素6136个，其中仅1249个在GT峰2px邻域内；
GT峰中831/5390被预测峰邻域覆盖。局部对比仍仅为GT的13.63%。
这些结果支持“原监督对峰处亮度覆盖不足”，不支持“增加峰权重已恢复准确窄峰”。
不把亮点MAE优于主GS单项解释为辅助分支整体或重光照质量优于主GS。

两组各绘制1536000条训练射线，实际命中GT峰分别6533条（.4253%）与388950条（25.3223%）。
日志记录训练耗时234.61/236.77秒，峰值allocated均9.46382GiB；这些是本轮日志统计，非跨方法性能结论。
距离场固定，但sharpness从源183.883957分别学至206.844711/157.693710，
说明采样目标也改变了体渲染合成权重；不能将全部变宽归因于辐射MLP。

## 首4帧局部邻域审计

另在已保存的训练图像对比PNG上审计帧0、35、70、105，GT半幅逐位核对为相同量化目标。
定义11×11方形膨胀的邻环：`max_pool(GT_peak,11) & ~GT_peak & (alpha>.9)`，
即Chebyshev半径5；不是半径11。邻环总计24920像素，峰总计1210像素。
以下正向误差为区域内 `max(predicted_neutral-target_neutral,0)` 的平均，包含误差为0的像素。

| 首4训练帧，11×11邻环 | uniform | peaks |
|---|---:|---:|
| 邻环neutral正向误差 | .032172 | .177974 |
| 邻环neutral MAE | .079963 | .191492 |
| 邻环neutral超过GT .05的像素比例 | 20.66% | 68.64% |
| 邻环RGB MAE | .074143 | .154654 |
| 邻环外前景neutral正向误差 | .027206 | .057112 |
| 邻环外前景RGB MAE | .056760 | .081312 |

23×23膨胀的复核也呈同方向：邻环neutral正向误差.031551→.139940。
因此峰附近的额外亮度扩散有直接证据，不能只凭峰处误差下降宣布成功。
审计只覆盖首4训练帧，不代替完整16/562帧统计；尚未隔离法线、方向表达和体积分厚度的影响。
证据为 `runs/sdf_volume_peak_sampling_pilot/neighborhood_audit.json`，分析程序为同目录的
`port_peak_neighborhood_audit.py`，归档状态以run根实际文件为准。

## 参数、协议及来源审计

`final_audit.json`确认两组GS、主传输、保存相机及全部SDF参数均逐位等于共同源，
点数仍330930；所有张量有限。场步数仍8000，辐射头步数由6000变为9000，
外观网格、辐射MLP和sharpness均更新。两组配置仅输出目录和peak fraction不同，split相同。
31个已记录训练步的frame_index一致，不将其扩大表述为完整3000步轨迹已经逐项审计。
同16评价帧、校正相机、逐帧GT峰计数、量化目标逐位一致，汇总亮点指标已独立重算并一致。

完整初始化链已核查，以下路径均接 `Real_NRHints/Pixiu/last.pt`：

| 来源顺序 | run/variant | 本阶段训练 | fit/validation |
|---|---|---:|---:|
| 1 | `neural_material_sdf_fresh_validation/gaussian` | 从随机初始化30k | 562/0 |
| 2 | `neural_material_camera_refine_pilot/camera` | 相机联合精修5k | 562/0 |
| 3 | `neural_material_calibrated_sdf_pilot/sdf` | 固定校正相机，SDF精修5k | 562/0 |
| 4 | `sdf_volume_feasibility/fixed_field` | 固定场，辅助辐射头3k | 562/0 |
| 5 | `sdf_volume_detail_pilot/geometry_grid` | 几何/外观细节网格3k | 562/0 |

这五个源checkpoint的config、split及checkpoint内索引均一致指向全部562训练帧。
GS、材质/主传输、SDF、辅助头、相机和场景归一化均继承相应训练或metadata依赖。
共享冻结BRDF decoder来自程序化预训练，未拟合这些场景像素；它的独立性不能消除其它组件的来源依赖。

还核查了已有先验：单图法线每帧独立生成；DA3深度使用全部562帧上下文。
对未来拟议506/56 train内部划分，所有506个fit目标的上下文都包含被留出的公共锚帧202。
所以仅切片目标帧不足以使现有DA3先验干净，必须重新按fit目标和fit上下文生成。
移除`fit-all`、冻结参数、只重置材质或仅迁移几何也不能将当前模型变为未见留出初始化。

保存的训练源码、路径/帧metadata和先验上下文审计未发现official-test进入模型拟合；
但祖先的official-test结果曾用于研究分析，不能称为新的独立盲测。
本轮未追加official-test评价；上述结论是已保存来源记录内的审计结论，不是对未记录外部行为的绝对证明。
完整来源证据在 `runs/sdf_volume_peak_sampling_pilot/initialization_lineage.json`
（工作副本 `/tmp/port_initialization_lineage.json`）。

## 入口与产物

本轮通过canonical `configs/validation.json`和现有入口启动；canonical以后可用于新run，
重现本轮时以run根保存的 `validation.json`、manifest和源码快照为准：

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
bash launch_validation.sh configs/validation.json
```

`make_validation_manifest.py`、`run_benchmark.py`、`train.py`、`evaluate.py` 继续承担生成任务、
训练与评价，不创建新的启动器或带日期配置副本。
固定 `eval_branch=sdf_volume`、`eval_split=fit`、`eval_limit=16`。
实现位置为 `sdf_volume.select_rays`、`train.py` 的aux采样与训练GT掩码缓存；
评价复用 `neutral_peak_mask` 和已有亮点指标定义。

输出根为 `runs/sdf_volume_peak_sampling_pilot/`：
各 `{uniform,peaks}/Real_NRHints/Pixiu/` 保存config/split/history/checkpoint/fit_sdf，
run根保留canonical配置快照、manifest、日志和源码快照；汇总比较图、指标、输入诊断及参数审计也归档于此。
完整命令以保存的manifest为准。最终汇总为 `comparison.json`、`comparison.png`、
`comparison_crops.png`、`final_audit.json`，分析源码在 `analysis_source.tar`。
两组训练和评价均已完成；负结果保留，不替换默认方法。
