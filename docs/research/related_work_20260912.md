# 外部重光照方法与可执行数据集

检索日期：2026-09-12。范围为外部方法的官方论文、项目页、README 和数据卡；
SSD-GS 方法未纳入研究。本页提供数据与协议依据，未启动外部方法训练。

## 已确认发表的方法

| 方法与会议 | 输入与评价目标 | 官方代码及数据入口 | PORT 当前接入结论 |
| --- | --- | --- | --- |
| [GS³，SIGGRAPH Asia 2024](https://gsrelight.github.io/) | 多视角、已知点光源 OLAT；合成场景每个 2,000 train / 400 test，512²；测试新视角与新灯光的 RGB GT | [代码与数据说明](https://github.com/gsrelight/gs-relight)、[官方数据](https://huggingface.co/datasets/gsrelight/gsrelight-data) | 已有 6 个合成场景，`pl_pos`、`pl_intensity`、EXR 均由现有 loader 支持 |
| [SSS-GS，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/dc72529d604962a86b7730806b6113fa-Paper-Conference.pdf) | 半透明物体多视角 OLAT；合成渲染或真实光场采集图像作为 GT | [代码](https://github.com/cgtuebingen/SSS-GS)、[官方数据卡与下载](https://huggingface.co/datasets/CGTuebingen/SSS-GS) | 已有全部 5 个 `*_small` 合成场景；灯位置齐备，光强需显式等功率假设；真实子集需要扩展 loader |
| [RNG，CVPR 2025](https://whois-jiahui.fun/project_pages/RNG/) | 面向硬表面和毛发等软边界的视角、灯光条件化重光照；公开代码接收 NRHints 数据格式 | [官方代码及运行协议](https://github.com/sssssy/RNG_release)、[NRHints 原始入口](https://github.com/iamNCJ/NRHints) | 现有 Real_NRHints 可作为相同数据家族；训练图像数、原始标定与评价域仍须统一后比较 |
| [ReCap，CVPR 2025](https://jingzhi.github.io/ReCap/) | 同一物体在多个未知环境光下采集，测试新环境图重光照；RelightObj 提供 Blender GT | [官方代码及数据说明](https://github.com/jingzhi/ReCap)、[官方数据下载](https://drive.google.com/drive/folders/1TH9RXfjrpR7SCcODjzcH47sI5NHieqLR?usp=sharing) | 当前 PORT 为单点光源输入；环境光积分与训练入口需另行实现，本轮优先已有 OLAT 数据 |

GS³ 的会议是 SIGGRAPH Asia 2024；RNG 的会议是 CVPR 2025。两者在搜索摘要中
可能因相邻参考文献而混淆，以上以各自官方项目页为准。

## 方法定位

GS³ 使用空间和角度 Gaussian 表达外观；SSS-GS 面向半透明材质；RNG 直接学习
光照与视角条件化外观；ReCap 用跨环境观测约束材质与光照的分离。
因此，单纯换成光照条件 MLP 或增加一个残差分支，已有明确的相关工作覆盖。
PORT 的新方案应明确独立的光传输结构与可检验性质，而把数据标定和渲染正确性
作为实验成立的前提。[GS³ 项目](https://gsrelight.github.io/)、
[SSS-GS 项目](https://sss.jdihlmann.com/)、[RNG 项目](https://whois-jiahui.fun/project_pages/RNG/)、
[ReCap 项目](https://jingzhi.github.io/ReCap/)。

## 推荐本轮使用 GS³ Translucent

本地目录：`/workspace/datasets/SSD-GS/data/Synthetic_GS3/Translucent`。
2,000 个训练和 400 个测试图像全部存在。场景包含明显的次表面散射，适合观察
新光传输表示是否改善非局部外观。当前 PORT 数据合同提供 512px、白背景和已知
点光源位置/强度入口。以 train 内灯光留出集选择设置，冻结配置后使用全部 train
训练，再一次性评价官方 test。该建议只指定一个外部数据集场景，实验执行由主任务安排。

GS³ 官方论文补充 Table 4 报告该场景 **PSNR 32.34 dB、SSIM 0.9740、LPIPS 0.0318**。
原论文指定 512²、2,000 train / 400 test、逐测试图像指标取平均。
这些是论文报告值；本地文件齐全仅证明现有 split 的引用完整。比较前仍须核对
EXR 显示变换、HDR 裁剪、alpha 合成、背景、SSIM 与 LPIPS 实现，匹配后才能报告
严格的数值差距。[GS³ 论文 §5.1 与补充 Table 4](https://arxiv.org/html/2410.11419v1)。

## 其他数据的协议边界

- SSS-GS 数据卡区分 800px `*_full` 和 256px `*_small`；本地为后者，
  每场景 500 train / 500 val / 500 test。数据卡记录 full 使用 Filmic tonemapping。
  `unit_light_intensity=1` 只是 PORT 的等功率假设。分辨率、实际 split 与显示域统一后
  才能和对应论文数字比较。官方页面的旧介绍与新数据卡对物体总数描述不同；下载盘点
  使用当前数据卡和文件列表。[SSS-GS 数据卡](https://huggingface.co/datasets/CGTuebingen/SSS-GS)。
- RNG README 示例以最多 1,000 张输入、512px 训练，并有两阶段训练安排。复用
  NRHints 文件格式并不能证明与 PORT 的原始相机/灯光固定协议一致。
  [RNG 官方运行说明](https://github.com/sssssy/RNG_release)。
- ReCap RelightObj 包含 13 个对象、8 个环境，每环境提供 200 train 与 200 test
  视角。发布的跨环境混合 JSON 有专门视角选择；teapot/coffee 某些环境使用缩放后的
  HDR 强度。PORT 需要保留这些记录并支持环境光，才可进行该协议。
  [ReCap 数据说明](https://raw.githubusercontent.com/jingzhi/ReCap/main/README.md)。

## 下载决策

已有 GS³ 与 SSS-GS 两套可用的外部合成数据，本轮直接复用，避免重复下载。
RelightObj 与 SSS-GS 真实子集尚未在共享根目录中发现；当前输入接口不完整支持它们，
本轮未下载。后续选择这些数据时分别放在 `/workspace/datasets/ReCap/` 与
`/workspace/datasets/SSS-GS/`，使用上表官方入口，保留原始 split。
现有文件盘点见 [数据清单](datasets_inventory_20260912.md)。


## 官方评价代码审计补充

审计时间：2026-09-12。读取官方公开代码的图像 I/O 与评价部分，并核对其依赖
版本。以下区分论文报告数字、公开训练日志指标和导出图像评价；本次没有运行实验。

### GS³：显示域匹配，论文最终 metric 实现仍有缺口

官方 `dataset_readers.py` 对 EXR 原值先做 `RGB * alpha + white * (1-alpha)`，
保留 float32；`syn_train.sh` 的 Translucent 使用 `--hdr --white_background`、
2,000 输入和 100k 步。`syn_render.sh` 明确传 `--gamma`，因此 `render.py`
对合成后的 GT 和 prediction 均做 `pow(1/2.2)`，再通过 torchvision 保存 PNG。
PORT 当前的 HDR 白背景合成、gamma 2.2 和 uint8 显示域与该导出路径一致。
来源：[EXR reader](https://github.com/gsrelight/gs-relight/blob/main/scene/dataset_readers.py)、
[训练入口](https://github.com/gsrelight/gs-relight/blob/main/syn_train.sh)、
[合成渲染入口](https://github.com/gsrelight/gs-relight/blob/main/syn_render.sh)、
[PNG 导出](https://github.com/gsrelight/gs-relight/blob/main/render.py)。

公开 `train.py` 的 `training_report` 是另一种口径：prediction 和 GT 在线性域
裁剪到 [0,1] 后计算 PSNR，gamma 只用于 TensorBoard 图像。其 helper 接收
`C x H x W`，逐通道计算 PSNR 再平均。PORT 的逐帧 RGB 总 MSE、显示域量化 PSNR
与这个训练日志指标不同。官方 SSIM helper 为 11x11、sigma 1.5、零 padding，
与 PORT 的 SSIM 核心公式一致；仓库的 LPIPS helper 默认 AlexNet，公开入口没有
给出生成论文 Table 4 的完整离线 metric 脚本，故无法认证论文 LPIPS 网络与输入范围。
来源：[training_report](https://github.com/gsrelight/gs-relight/blob/main/train.py)、
[PSNR helper](https://github.com/gsrelight/gs-relight/blob/main/utils/image_utils.py)、
[SSIM helper](https://github.com/gsrelight/gs-relight/blob/main/utils/loss_utils.py)、
[LPIPS helper](https://github.com/gsrelight/gs-relight/blob/main/lpipsPyTorch/__init__.py)。

因此 Translucent 的 32.34 / .9740 / .0318 应标记为**论文参考值**，目前可确认
输入数量、分辨率和导出显示变换相符；完整同协议差值还缺论文最终 metric 的证据。
若需额外对齐公开训练日志，可从固定 checkpoint 的线性合成图直接计算逐通道
PSNR 后取平均，另行标记 `official-training-report-linear`。这是纯评价定义，
不涉及相机、光照或曝光拟合，也不替代主要的标准显示域指标。
严格复刻官方 PNG 导出时可使用 `floor(clamp(x,0,1)*255+0.5)`；PORT 当前
round 的恰好半整数取舍可能造成末位差异。[导出函数依赖](https://docs.pytorch.org/vision/stable/_modules/torchvision/utils.html)。

### SSS-GS：small 对应 256²/500 输入，Bunny 专属 PSNR 未确认

论文 Table 1 的 small 对应设置为 256²、500 输入，报告合成集合
**PSNR 35.01±1.01、SSIM .972±.01、LPIPS .040±.01**。
800²/11,200 输入的集合结果为 37.35±2.13 dB。两者均为聚合结果。
本次查阅的官方论文与数据卡没有提供可确认的 `bunny_small` 单场景 test PSNR；
封面约 37 PSNR 展示和 Bunny intrinsic RMSE .057 属于其他说明，不能代替该数字。
[SSS-GS 论文 Table 1/4](https://arxiv.org/html/2408.12282v2)、
[small/full 数据定义](https://huggingface.co/datasets/CGTuebingen/SSS-GS)。

官方 `CameraDataset` 读取 PNG 为 [0,1] RGB 并乘 alpha；GT 没有额外 gamma。
`render.py` 裁剪 prediction 并将 prediction/GT 保存为 PNG。PORT 的 SSS GT
原数值乘 alpha、256px 黑背景与这一目标域一致；PORT 在自己的预测分支对
foreground 使用 gamma 是模型输出映射，不需要再对官方 PNG GT 编码一次。
[数据读取](https://github.com/cgtuebingen/SSS-GS/blob/main/scene/dataset.py)、
[导出](https://github.com/cgtuebingen/SSS-GS/blob/main/render.py)。

但官方离线 `metrics.py` 与 PORT 的度量定义有实质差异：

| 项目 | SSS-GS 公开离线代码 | PORT 当前主要指标 |
| --- | --- | --- |
| 输入 | 保存后 PNG 转 [0,1]，逐帧计算再平均 | 显示图裁剪、uint8 量化，逐帧平均 |
| PSNR | TorchMetrics 默认动态范围，由 GT 值域推定 | 固定峰值 1、全部 RGB 的 MSE |
| SSIM | TorchMetrics 默认动态范围；reflect padding 后取内部区域均值 | 固定范围 1、零 padding 后全图均值 |
| LPIPS | VGG，`normalize=True`，内部把 [0,1] 转为 [-1,1] | VGG，显式 [-1,1] |

官方 environment 固定 TorchMetrics 0.11.4。该版本 SSIM 默认 sigma 1.5、
11x11 核，动态范围取 prediction/GT 两者较大的 max-min；其均值排除 5px 边界。
来源：[metrics.py](https://github.com/cgtuebingen/SSS-GS/blob/main/metrics.py)、
[environment.yml](https://github.com/cgtuebingen/SSS-GS/blob/main/environment.yml)、
[TorchMetrics 0.11.4 PSNR](https://github.com/Lightning-AI/torchmetrics/blob/v0.11.4/src/torchmetrics/image/psnr.py)、
[TorchMetrics 0.11.4 SSIM](https://github.com/Lightning-AI/torchmetrics/blob/v0.11.4/src/torchmetrics/functional/image/ssim.py)。

合法的评价适配是在已有固定 prediction/GT PNG 上调用相同版本、相同配置的
官方 metric 函数，另存 `sss-official-metrics`。这只改变指标计算规则，不改变模型、
预测、GT、标定或光强。动态范围属于官方 metric 定义，而不是从 GT 拟合图像变换。
即使度量代码对齐，Bunny 单场景也仍应与其自身公开结果比较；集合均值仅列论文参考。
PORT 的显式 `unit-light-intensity=1` 假设、训练步数和实际 split 也应随结果记录。

## 补充：已发表的 2026 工作与适配边界

| 方法 | 官方发表依据与公开入口 | 数据/光照形式 | 本轮适配结论 |
| --- | --- | --- | --- |
| GHPT，CVPR 2026 | [官方项目](https://nwu-vislab.github.io/GHPT/)、[CVPR 节目](https://media.eventhosts.cc/Conferences/CVPR2026/CVPR_main_conf_2026_15.pdf) | 静态场景的材质/环境图逆渲染；Synthetic4Relight、TensoIR Synthetic 环境光重光照 | 静态但输入为环境光，不是已知单点光源 OLAT；当前 PORT loader 与该协议不匹配 |
| LumiMotion，CVPR 2026 Highlight | [官方项目](https://joaxkal.github.io/LumiMotion/)、[CVF 收录](https://openaccess.thecvf.com/CVPR2026?day=2026-06-07)、[官方代码](https://github.com/joaxkal/LumiMotion) | 五个合成场景、四种照明，各含静态/动态版本；材质和 environment map 训练 | 包含静态版本但采用环境光；动态版本还需时间/变形表示，本轮不下载 |

GHPT 的 [NWU 官方仓库](https://github.com/NWU-VISLAB/GHPT) README 明确该仓库
仅托管项目网站；其链接的 `1BenXiaoHai1/GHPT` 在此次核查返回 404，故当前没有
从该入口确认可执行训练代码或新的单独数据下载。项目展示使用既有公开环境光
基准，不据此宣称发布了新的点光源数据集。

LumiMotion 官方 README 给出 [Zenodo 合成数据](https://zenodo.org/records/18894615)
下载入口，真实 ENeRF/DNA 数据需按各自访问约定获取。
现有 `/workspace/datasets/LumiMotion/` 目录在前次盘点中已出现，故后续若改变任务
范围，也应先盘点其内容。此次仅核对公开协议，保留已有数据。
[官方数据说明](https://github.com/joaxkal/LumiMotion/blob/main/readme.md)。


## 2026-09-22：第二阶段研究更新

本次查阅官方论文、项目和代码。以下内容用于设计依据，不将不同任务的论文数字并表为同协议SOTA。

| 工作 | 已核实的相关点 | 本项目如何使用或区分 |
| --- | --- | --- |
| [SSD-GS, ICLR2026](https://arxiv.org/abs/2604.13333)，[官方代码](https://github.com/irisfreesiri/SSD-GS) | 表面方向/材质条件、神经阴影提示修正和次表面项；NRHints/GS³ 100k、SSS 60k训练 | 表面与阴影条件可补足当前神经着色；不是复现其解析ASG/specular或dipole模型。论文的test校准优化需单独区分 |
| [RadioGS, ICLR2026](https://arxiv.org/abs/2603.01491)，[项目](https://qbhan.github.io/radiogs-page/) | 2D surfel辐射与PBR在额外方向上的一致性，结合可微光追 | 说明照片拟合不等于物理解耦；其重光照含辐射适配，和当前直接输入OLAT光参数不同 |
| [RadiosityGS, SIGGRAPH Asia2025](https://arxiv.org/abs/2509.18497)，[项目](https://raymondjiangkw.github.io/radiositygs.github.io/) | 2D surfel间在SH空间的可微光传输、可见性和求解 | “表面间传输”已有直接先例；当前attention不等于其SH方程求解 |
| [RNG, CVPR2025](https://whois-jiahui.fun/project_pages/RNG/)，[代码](https://github.com/sssssy/RNG_release) | Gaussian特征条件化神经着色、阴影提示，先forward再deferred训练 | 可参考几何/着色优化策略，不能将小型神经着色网络视为首创 |
| [Spec-Gloss Surfels and Normal-Diffuse Priors, WACV2026](https://openaccess.thecvf.com/content/WACV2026/html/Kouros_Spec-Gloss_Surfels_and_Normal-Diffuse_Priors_for_Relightable_Glossy_Objects_WACV_2026_paper.html)，[代码](https://github.com/gkouros/SpecGloss-GS) | 2DGS、spec-gloss着色与StableNormal/StableDelight先验 | 2DGS+预训练法线已不是新颖点；环境光和材质逆渲染协议不同 |
| [MaterialClusterGS, 2026预印本](https://arxiv.org/abs/2606.09018) | 共享材质原型减少逐primitive材质歧义 | 不为追求新模块同时叠加材质字典，本轮不实现 |
| [DIAMOND-SSS, CVPR2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Araneda_DIAMOND-SSS_Diffusion-Augmented_Multi-View_Optimization_for_Data-efficient_SubSurface_Scattering_CVPRF_2026_paper.html) | 稀疏数据的扩散增强与跨视角一致性 | 是Findings而非主会论文；数据稀缺任务与当前完整OLAT拟合不同，不引入额外生成栈 |

当前本地SSD-GS保存了100k Real_NRHints模型的原test校准评价：Cat PSNR18.000583，
Pixiu23.402195，见SSD-GS/runs/real_fixed_calibration_20260913。论文校准优化数字不能替代此协议结果。
训练时相机/灯光是否优化、测试是否使用原始标定、LPIPS输入域和预算都要随比较列明。
未经统一指标重算和相同训练协议复现，这些数字只能辅助定位差异，不能给出完整SOTA排名。

相关经典依据：[Burley 2015，Extending the Disney BRDF to a BSDF](https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf)
用两指数曲线描述归一化扩散，并按RGB提供传播距离控制。
[Jensen等2001](https://graphics.stanford.edu/~srm/publications/SG01-subsurf-abstract.html)
是不同表面点入射/出射与扩散近似的经典基础。

本地SSD-GS基线已实际复算全部1937帧，统一PORT-GS指标，见experiments/results.md。
合成四场景平均PSNR33.962519，预算100k/60k；它是可核实的强基线参考，并非所有最新论文的完整SOTA排名。
Real旧训练使用test相机/灯光优化，其固定测试标定输出也仍有训练来源限制，不能纳入严格干净的SOTA认证。

## 2026-09-23：基础表示与亮度约束的定向复核

[DP-GES，CVPR2026，论文§3–7](https://arxiv.org/html/2605.25345v1)
使用不透明内区、半透明边缘的surfel以及周围3D Gaussian；surfel用逐像素depth peeling排序，
Gaussian细节分支按层透射率加权。其外观为视角相关Spherical Beta，论文评价是NVS，
并将物理交互/阴影列为后续方向。因而本项目只把它视作改变opacity/visibility基础的候选依据，
不将其细节或FPS数字当成OLAT重光照、真实几何或细小高光恢复证据。
当前交点渲染器已经按真实hit depth完整排序；仅加入depth peeling并不能自动解释/修复缺峰。
替换成不透明内区还会改变初始化、几何梯度和遮挡，需单独受控实验，不能在运行中替换。

对已列出的[SpecGloss-GS，§3.3–4.2](https://arxiv.org/html/2510.02069v2)补核：
其negative-only clipping针对HDR环境图，保留正向HDR峰；StableDelight diffuse先验仅作早期软约束，
并承认先验失败情形。不能把“去掉HDR环境图上限”曲解为去掉最终图像评价裁剪。
本地surface_reflectance已经使用独立diffuse/F0、解析点灯、可优化正值全场景光强尺度；
shader及observation_image没有将正向辐射硬限制到1，评价的[0,1]裁剪另属指标域。
这是本地代码核对，不能据论文推定当前存在同一种HDR上限错误。运行中的fresh两组无这些先验，协议保持不变。
