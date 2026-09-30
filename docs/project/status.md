## 2026-09-30：训练代码重构完成（分支 refactor/port-gs-structure）

重构前代码快照 `54fd659` 已推送至 `origin/feature/selectable-transport-methods`。
重构后 `train.py` 约 370 行，职责移入 `training/`；默认训练/评价/manifest 行为经验证不变。
已修复残差/体积独享阶段在 rotation 相机源上的启动崩溃；新增 `--opacity-reset-every`；manifest 记录源码来源。
未启动正式实验；R2b 等已有结果无需重跑。[验证记录](../experiments/training_refactor_20260930.md)。

## 2026-09-30：按用户选择收敛为三个方法

用户明确保留 directional_port_v1、surface_attention、neural_material；其余六个方法/几何入口删除。
移除专用训练、评价、初始化、诊断、manifest 分派及测试；保留共享数学、神经材质、相机校正与已有 GGGS 初始化读取。
历史模型、数据、结果、规范实验配置不变，退役源码与 Wrapping 依赖先归档。
本次不重新排名三个保留方法，也不启动正式实验；Attention 的最新相机协议效果仍未验证。
[范围与验证](../experiments/method_retirement_20260930.md)。

## 2026-09-30（最终）：全部标定实验完成；推荐default+旋转相机校正+平移规范（R2b）

固定标定official test（主指标）：Cat **25.203/.8201/.1674**、Pixiu **26.264/.8811/.1041**（原PORT 22.623/.7797/.2393与21.571/.8505/.1580；
GS3 19.124/.7169/.2319与23.724/.8641/.1173；SSD-GS 18.004/.7037/.2482与23.404/.8621/.1193），两场景三项指标均领先。
文献口径（冻结场景拟合test相机/灯光）Cat 29.40/.9169/.1464、Pixiu 32.43/.9469/.0879，高于SSD-GS自身论文口径渲染（同一评价代码）27.249/.9001/.1661与31.174/.9450/.0911。
逐帧灯位校正（R2c）无可测收益，未采用。去掉ports消融（R3）：Pixiu对齐/文献口径-0.72dB，Cat -0.17dB，ports在标定正确后是有效贡献。
固定标定分数受test相对train的常量偏移限制（Cat +2.0px、Pixiu -2.6px），训练数据无法确定该偏移。
所有队列已结束（calib_gauge_20260930中两项neural gauge_light为主动取消），GPU已释放。[完整结果](../experiments/calib_camrot.md)。

## 2026-09-30（更新2）：平移规范修正生效，Cat/Pixiu固定标定三项指标均领先GS3与SSD-GS

`calib_gauge_20260930`（neural+旋转相机校正+平移规范）：Pixiu固定标定**26.271/.8803/.1061**，Cat **24.849/.8142/.1767**
（原PORT 21.571/.8505/.1580与22.623/.7797/.2393；GS3 23.724/.8641/.1173与19.124/.7169/.2319；SSD-GS 23.404/.8621/.1193与18.004/.7037/.2482）。
共享竖直校正被投影为场景平移（Pixiu +6.7→+0.2px、Cat +12.2→+0.04px），test偏移回到原train/test差值（-2.6/+2.0px），对齐质量不变。
运行中：default基底+规范（`calib_gauge_default_20260930`，GPU2）、default+规范+逐帧灯位（`calib_gauge_default_light_20260930`，GPU1）、
去掉ports消融（`calib_noports_20260930`，GPU0）。原neural gauge_light两项在启动前取消。[报告](../experiments/calib_camrot.md)。

## 2026-09-30（更新）：第一轮相机自标定结果；第二轮平移规范实验运行中

Pixiu（neural，旋转相机校正）：固定标定test **24.066/.8641/.1162**（原21.571/.8505/.1580），三项均超过GS3 23.724/.8641/.1173
与SSD-GS 23.404/.8621/.1193（SSIM与GS3仅差4e-5，视为持平）；对齐诊断31.86/.9426/.0918，文献口径（冻结场景拟合test相机/灯光）
32.05/.9445/.0902，高于SSD-GS自身论文口径渲染用同一评价代码的31.17/.9450/.0911。拟合PSNR 24.6→32.5。
Cat（neural）：对齐质量27.90/.8932/.1630（原24.69），已超过GS3/SSD-GS对齐值；但共享俯仰校正约+12px使坐标系竖直漂移，
固定标定test跌至15.92（test偏移+14.7px）。这正是平移规范问题，第二轮`calib_gauge_20260930`加入`--camera-gauge translation`
（及可选逐帧灯位），GPU0/1运行中，Cat优先。[实验报告](../experiments/calib_camrot.md)。

## 2026-09-30：真实场景标定审计完成；训练相机自标定实验运行中

**主要瓶颈不是重光照材质，而是相机标定。** Cat/Pixiu官方train位姿逐帧误差约4–6px（512px），
official test位姿相对train存在近似常量偏移（Pixiu约(-3,-11)px，Cat约(+2,-1)px）。
固定标定test分数主要反映各方法坐标系恰好漂移到哪里：GS3/SSD-GS因训练相机优化在Cat上下漂7–9px，
PORT因无相机优化继承Pixiu约10px水平偏移。逐视角最佳平移对齐后（仅作诊断），
PORT落后GS3/SSD-GS约1.7dB（Cat）/3.4dB（Pixiu），主因是未校正训练位姿造成的整体模糊。
此前各PORT变体的固定test排名与对齐后排名不一致（如local_transport固定Pixiu最高、对齐后最低）。
[审计报告](../experiments/calibration_audit_20260930.md)。

新增：`--camera-mode rotation`（固定相机中心、无锚点相机、逐视角稀疏更新）、`--camera-gauge translation`、
`--optimize-lights`，以及次要诊断指标`evaluate.py --shift-align`和外部渲染评分`diagnose_image_errors.py --external-renders`。
主指标仍为原始标定的official test。实验`calib_camrot_20260930`（neural/default×Cat/Pixiu，GPU0–2，30k）运行中，
启动已核对（子进程5474–5476、step500、GPU占用）。[相机模块](../architecture/modules/cameras.md)。

## 2026-09-29：残留清理完成

已删除缓存及编译/打包残留，重复权重硬链接去重，释放约1.01GiB。所有最终模型和实验结果保留，运行库导入及权重读取检查通过。[记录](../experiments/cleanup_executed_20260929.md)。

## 2026-09-28：100万高斯上限对照已完成

Cat/Pixiu均30k及完整test/fit完成，已核对对照图、目标图像与几何。
Cat最终747578个高斯，PSNR22.52035（40万上限22.62266）；Pixiu248502个，PSNR21.53561（此前21.57110）。
Cat训练耗时增加59%，显存13.27→25.06GiB，质量无明确提升，几何仍粗糙；不推广100万设置。
Pixiu两轮均未触及40万，微小差异不应解释为上限收益。原结果保留，无本轮活动训练。
[完整结果与图像](../experiments/gggs_neural_material_1m.md)。以下是历史记录。

## 2026-09-27：GGGS初始化＋默认/DNA联合实验全部完成

本轮 **GGGS初始几何＋Neural Material** 已完成：Cat/Pixiu各30,000步，几何与材质持续联合优化，共享神经BRDF解码器冻结。
完整测试PSNR为22.6227/21.5711dB，SSIM为.779711/.850532，LPIPS为.239291/.157998。
较GGGS＋DNA的PSNR提高.078/.386dB，但细节、高光和表面粗糙化仍未解决，**不替换默认参考**。
[完整结果、对照图与几何/换光预览](../experiments/gggs_neural_material_joint.md)。本轮训练和评价均已结束，无本轮运行任务。

按用户指定顺序，先默认directional_port_v1，再DNA distribution_material三维适配；均从同一原始GGGS+法线/深度几何独立初始化。
四次训练各30000步，位置/尺度/旋转/透明度、材质和高斯数量均可更新；完整test66/71及fit522/562完成。
默认：Cat21.81926/.770223/.228127，Pixiu20.54358/.845065/.155993。
DNA：Cat22.54456/.774314/.240020，Pixiu21.18482/.846024/.158111（PSNR/SSIM/LPIPS）。
DNA相对新默认PSNR+.7253/+.6412dB，但两场景LPIPS变差；几何轮廓与粗糙度未保持初始GGGS表现。
同一GGGS连续深度回放也进行了核对，图像代理仍显示退化；无几何GT，不宣称测得真实3D误差。
全部模型/损失有限、划分/目标图像一致、保存加载与光强线性通过；四个最终模型保留，临时短测模型已删。
当前无本轮后台训练或评价。默认方法不变，当前代码仍为6个重光照＋3个几何入口；新增的是GGGS初始化适配选项。
[两轮结果与图片](../experiments/gggs_dna_joint.md) · [默认数值](../experiments/gggs_default_joint_results.json) · [DNA数值](../experiments/gggs_dna_joint_results.json)。

以下为历史过程记录，运行中/未启动等状态由本节更新。

## 2026-09-27：默认联合实验已完成，DNA联合实验正式运行

默认两场景各30k及完整test/fit完成：Cat21.81926/.770223/.228127，Pixiu20.54358/.845065/.155993。
相对旧默认，Cat小幅提升，Pixiu基本持平；固定灰模和轮廓指标显示几何退化，未推广。
DNA两场景600步3D联合短测/加载通过，按用户要求在默认全量评价结束后正式启动30k。
来源是相同原始GGGS权重，几何未冻结，使用DNA材质与三维法线接口。实际GPU/step100已核对。
[默认结果](../experiments/gggs_default_joint.md) · [DNA协议/结果](../experiments/gggs_dna_joint.md)。

## 2026-09-27：追加DNA联合优化，排在默认实验之后

用户要求默认实验结束后，再用同一原始GGGS几何独立初始化DNA并继续优化几何。
3D适配和五个关键测试已完成；不将3D高斯压成圆盘，旧DNA2DGS仍可读取。
当前只运行默认实验，DNA正式训练尚未开始。[DNA协议](../experiments/gggs_dna_joint.md)。

## 2026-09-27：GGGS初始化接默认重光照，允许联合几何优化

用户明确允许relighting继续优化几何。当前已启动directional_port_v1的Cat/Pixiu各30k联合训练：
GGGS+先验几何初始化，默认gsplat延迟着色，完整训练图监督及默认增密/裁剪；无SDF或新教师损失。
几何转换测试、两场景600步短测及保存加载评价通过；正式子进程/GPU和step100日志均已核对。
这不是DNA或上一轮local_transport。[协议/结果](../experiments/gggs_default_joint.md)。

## 2026-09-27：GGGS固定几何＋局部神经光传输完成

复用GGGS+StableNormal/DA3的30k几何，Cat/Pixiu各完成30k新材质训练及全部test66/71、fit522/562评价。
逐贡献求值、三点条件高斯积分；几何全部buffer与输入逐位一致，无SDF，不启用联合几何更新。
测试PSNR22.51677/21.90448，较DNA+.37509/+.40152dB；LPIPS .263330/.174197，Cat略好、Pixiu更差。
固定图像仍见毛发模糊、缺失小高光和边界拖影，没有解决质量问题，不替换Default。
权重/损失有限、划分完整、光强线性及保存加载检查通过；三方法全部目标图像一致。
所有训练、评价与预览已完成，无本轮后台任务。只保留两个最终材质权重，临时短测权重已清理。
[完整报告与图片](../experiments/local_transport.md) · [数值/审计](../experiments/local_transport_results.json)。

以下为历史阶段记录，其未进入relighting等描述已由上述最新用户要求和结果更新。

## 2026-09-27：GGGS法线/深度先验实验完成，整体未可靠提升

Cat/Pixiu各fresh30000/512px与完整fit470/506、validation52/56全部完成；最终128596/57741高斯。
StableNormal法线.05、DA3相对深度.1，1000步起渐进启用；只用fit教师，无SDF/official test。
PSNR分别只+.00318/+.00149dB，LPIPS变差；Pixiu边界起伏减轻，但IoU .91058→.90100、F1 .38011→.33346。
Cat仍缺局部形状并有折痕，两场景几何不通过，未进入relighting。
另外修正GGGS评价中的尺度敏感深度离群值：保持训练坐标求解、输出世界深度，新旧模型均完整重评。
三个关键测试、实际教师梯度、输入划分、有限权重/损失、人口计数及修正后保存加载检查通过。
本轮训练与所有重评价均已结束。[完整结果/对照图](../experiments/gggs_normal_depth.md) ·
[数值与审计](../experiments/gggs_normal_depth_results.json)。当前5个relighting方法+3个几何入口；先验是GGGS选项。

## 2026-09-27：Gaussian Wrapping 两阶段完成，几何仍未通过

无先验与StableNormal(.05、7000步起)两条件，Cat/Pixiu各fresh30000、512px、完整fit/validation全部完成。
先验仅训练划分470/506帧，无SDF/深度教师，无official test。四次训练均完成五次补壳和独立加载评价。
加先验最终Cat222330/Pixiu136980高斯，验证IoU .94711/.87920、边界F1 .30575/.29414。
Pixiu粗糙度下降但底座锯齿、头部尖突及形变仍在；Cat局部形状未恢复。两场景几何均不通过，未进入relighting。
权重有限性、训练/验证划分、先验梯度、补壳计数、最终坐标渲染一致性核验通过。
本轮训练与评价进程已完成；当前5个relighting方法+3个几何入口，StableNormal是Wrapping选项。
[完整实验与对照图](../experiments/gaussian_wrapping.md) · [数值](../experiments/wrapping_results.json)。

## 2026-09-27：继续删除 paired_port 和 local_frame

两种方法的实现、注册和专用测试已删除，无保留方法依赖它们。
当前5个重光照方法＋2个几何重建入口，共7个。现有最终权重和DNA方法未修改。
历史设计与结果仅作追溯；前文9个方法的记录已由本条更新。

## 2026-09-27：删除两条旧方法代码

按用户列出的surface_reflectance、directional_surfel删除其注册、实现、专用渲染/训练/采样与测试入口。
当前7个重光照方法＋2个几何重建方法，共9个；DNA实现和已有模型不变。
历史实验指标和源码归档仍作追溯，不属于当前方法入口。
[删除记录与验证](../experiments/method_retirement_20260927.md)。

## 2026-09-27：通用方法调研与首轮实验完成，几何仍未通过

按用户新方向筛选通用表面重建方法：Geometry-Grounded GS、Gaussian Wrapping、CoMVS-GS。
Cat/Pixiu的train-only经典匹配probe通过覆盖仅1.649%/.433%，没有用于稠密几何监督。
GGGS核心后端已接入，离心K、连续深度/梯度/坐标、Mip滤波保存加载检查通过。
关闭Mip的首轮分别在8000/3600步失败；恢复作者默认滤波后两场景均完成fresh30000及完整fit/validation。
验证IoU .95086/.91058，LPIPS .29148/.16110；Cat局部几何不足、Pixiu大片层状结构仍在，均未通过几何门槛。
本轮所有训练/评价已结束，未进入relighting，未评价official test，没有SDF/预训练几何监督。

[通用调研](../research/general_surface_reconstruction_20260927.md) ·
[最终实验与灰模](../experiments/gggs_core_geometry.md) · [完整数值](../experiments/gggs_core_results.json)。
下一优先级为移动光源下的几何观测/外观解耦；不继续只轮换SH-only表面内核，也不把本轮核心消融当作完整论文复现。

项目清理：删除约11.43GiB旧缓存/冗余实验文件，保留必要源码配置指标归档；再去掉约163MiB嵌套归档，清除58个已退出旧tmux。
当前保留默认6、PORT-DNA2、原生2DGS2、GGGS2，共4套最终目录/12个模型；短训删除，匹配诊断并入GGGS目录。
[清理依据与清单](../experiments/project_cleanup_20260927.md)。

以下保留历史记录，旧路径/旧状态以本节及当前runs索引更新。

## 2026-09-27：修复、重训与验收已完成，重光照条件未满足

已修复屏幕尺寸剪枝记录、精确克隆的尺寸继承，以及本周期先剪枝后清零的统计时序；四项回归检查通过。Cat/Pixiu本轮各完成四组独立fresh30000及完整fit/validation，最终模型为最后一组30000步。
最终Cat/Pixiu保留81018/37257点，验证IoU .94838/.89938、LPIPS .30067/.16377。与前轮失败原生模型相比，RGB小幅改善，但Cat仍过平滑、Pixiu仍有片层/尖刺；两场景均未通过几何门槛。本轮未接入或训练relighting，未评价official test，无SDF或预训练几何监督。
[完整修复与结果](../experiments/native_2dgs_topology_repair.md) · [数值与验收](../experiments/native_2dgs_repair_results.json)。最终模型在`runs/native_2dgs_repaired_geometry/Real_NRHints/{Cat,Pixiu}/last.pt`；旧原生和开发对照证据归入该run的`controls/`，中间模型已清理。当前共6个结果目录、20个最终模型；此前5套重光照最终模型未改。本轮训练与评估均已结束。

以下保留历史阶段记录；其中旧run路径/进行中表述由本条及retention.json更新。

## 2026-09-27：最终统计窗口修复正在验证

三组重训已结束；点数/分裂/剪枝计数核验通过，但几何质量尚不满足接入重光照。复核首版修复发现历史半径跨周期残留的问题，现已改成每个增密周期先按历史半径剪枝、再清零。新增时序测试在归档旧版稳定失败，最终四项检查通过。
最终`native_2dgs_repaired_geometry`正按相同object_split配置fresh30000重训Cat/Pixiu，只改变统计清零时机；等待终端几何评估。

## 2026-09-27：修复并重训已授权

已实现项目内屏幕半径记录修复，覆盖旧点、克隆点、分裂点及Adam状态的检查通过。按“剪枝修复→通用深度/正则→物体尺度分裂”三个可复核配置重训Cat/Pixiu；几何通过后再接局部神经光传输。
[协议与结果](../experiments/native_2dgs_topology_repair.md)。继续不使用SDF或预训练几何监督。

## 2026-09-27：优先纠正训练链路诊断

用户质疑差几何是否来自训练参数。复核确认：所用作者版本的屏幕尺寸剪枝在比较之前被clone/split清零；GPU隔离复现通过。三组都用median depth1，未对照作者通用默认expected depth0。四帧终端梯度诊断还否定了“distortion标量小所以几何影响弱”的推断；Pixiu位置梯度可达photo的1.3–5.6倍。
[完整审计与证据](../experiments/native_2dgs_parameter_audit.md)。优先验证人口管理、默认深度配置与正则设置；数据/标定不是已证明的主因。本轮未改训练代码/参数、未重训，尚无修复后的质量结果。

## 2026-09-26：原生2DGS几何阶段完成，质量门槛未通过

作者原生模型/CUDA已接入，依赖项目内编译，复用ssd-gs环境。Cat/Pixiu分别完成三组fresh30000：原生默认、alpha约束+深度畸变、相同设置+坐标归一化。完整内部fit/validation评价完成。
最终轮廓IoU .94959/.90494，背景曲面显著减少，但Cat过度平滑、Pixiu片层/尖刺仍在；不能称为“几何重建好了”。坐标归一化未实质改善质量。无SDF或预训练几何监督，未进行本阶段官方test评价。

[完整协议、结果和下一步依据](../experiments/native_2dgs_geometry.md)。最终模型在`runs/native_2dgs_normalized_geometry/Real_NRHints/{Cat,Pixiu}/last.pt`；灰模对比在`review/`。最终模型/PLY已还原世界坐标，接口和最终状态检查通过。只保留这两个最终几何模型，两组开发对照的图/指标/日志/源码归档在同一run的`controls/`；此前5套重光照结果未改。

几何是下一阶段的前提：优先固定光照多视图，或先证明当前图像的经典特征/标定与稳健多视图约束可用。现有移动光源数据对原生SH外观的假设不匹配，稀疏对应也有1–3px级排查线索。局部神经传输尚未开始；不将当前粗几何作为已验收的教师。本轮训练、评价和对比均已结束。

## 2026-09-26：质量诊断完成，下一主线设计待验证

只读复核保留结果、补评当前模型完整训练集、核查权重/cutoff/角度带宽和论文来源。
确认训练高频拟合不足、超过半数存储点低于原生alpha阈值；这些是证据，不是唯一病因证明。
建议重构有效几何分配与局部光传输求值；Cat/Pixiu分别检验几何假设，继续不用SDF或预训练几何。
[完整分析](../experiments/quality_diagnosis_20260926.md) · [方案/顺序/停止条件](../research/local_transport_redesign_20260926.md)。
本轮未修改生产模型/配置/数据，未开始新训练；新增fit_full在原最终run内，仍只有5套最终目录。

## 2026-09-24 用户授权：runs仅留最终结果

清理完成：5套最终run、18个模型保留；97个非最终run、中间权重及训练缓存已删，释放约30.8GiB。
[保留索引](../../runs/README.md) · [范围/限制/删除记录](../experiments/runs_cleanup_final_only_20260924.md)。
以下历史文字中“全部保留”的说法由本次明确清理要求更新；历史路径不保证仍存在。

# PORT-DNA-2DGS 本轮已完成（2026-09-24）

新方法实现、五项针对性检查、真实短训/加载渲染、Cat/Pixiu各fresh30000、
全部137张官方测试、共同GT核验、8固定整图/14裁剪人工检查及神经分支诊断均完成。
没有SDF、预训练几何、旧检查点。PSNR比默认+.605494/+.924710dB，LPIPS均差；
高频细节/高光目标仍未解决，默认保持directional_port_v1。

[方法与结果](../experiments/distribution_material.md)；模型和完整证据：
`runs/distribution_material_final/`。本轮无正在运行的训练或评价，未追加测试后调参。
用户本次要求的提出方法、实现、训练和评价已完成；方法质量不标记为整体成功。
下面保留上一轮研究记录。

---

# PORT-GS 当前状态

本轮方法研究、最后一轮受控改进、Cat/Pixiu完整官方测试、终端复核与冗余文件
清理均已完成。按用户要求停止追加研究并暂停。当前没有本轮训练、评价或审计
任务在运行。更广泛的“可靠恢复GT细小高光”目标尚未实现，不能标记为研究成功。

## 最终结果

| 场景 | 全train / test | PSNR | SSIM | LPIPS | tiny召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| Cat | 522 / 66 | 21.119731 | .763317 | .287500 | 0/575 |
| Pixiu | 562 / 71 | 21.672545 | .848500 | .173014 | 6/5324 |

新基础surface_reflectance使用fresh30000、fit-only表面种子、uniform采样和
局部GGX交点着色；无旧模型权重、test曝光/位姿适配或测试后调参。
Cat较默认PSNR低.416453dB，Pixiu高1.094297dB，但LPIPS均变差，小高光未恢复。
Cat2/8、Pixiu3/8数值门槛通过，人工检查失败。默认directional_port_v1保持不变。

最后的tiny逆PDFproposal受控实验只增加监督覆盖，未通过质量门槛，已记录失败；
没有继续增加训练预算。详细结果、来源和界限见：

- [8种方法与结果统计](../experiments/method_inventory.md)
- [Cat/Pixiu最终测试报告](../experiments/surface_reflectance_final.md)
- [表面初始化对照](../experiments/mask_surface_initialization.md)
- [最后一轮采样改进](../experiments/surface_patch_proposal.md)
- [实际清理清单](../experiments/output_cleanup_20260924.md)

## 完整性与保留

所有137张test和1084张训练图均已评价。来源/状态/RNG/完整帧数审计、GT一致性、
417组件核对、64次fit图重放与独立环域检查通过。严格跨后端SSIM/LPIPS检查
622项差异保留，未放宽容差或替换主指标；主/次门槛判断相同。旧baseline原报告
没有高光字段，因此其高光组件来自之前已完成的统一PNG审计，来源已明确记录。
初次读取该旧字段失败的记录也保留，不冒称全程无失败。

已删除117个冗余文件，约2.072GiB：62个无训练依赖短测模型、54个项目字节码、
1个已被实际清单替代的草稿。59个受保护训练输入及最终模型/完整测试输出均保留。
共享数据、环境、预训练/几何监督、SSD-GS/GS3和原始研究资料未清理。

精简前README及完整docs已归档至
`runs/surface_reflectance_final/pre_cleanup_docs.tar.gz`，历史研究细节仍可追溯。
当前canonical配置为已完成的surface_reflectance_final，不能同名重跑。
没有下一项自动实验；只有用户明确恢复后才继续。
