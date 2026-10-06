# LiSA 最终版：光源图集、多尺度查询与阴影的相关工作审计

核查日期：2026-10-06（Asia/Tokyo）。对象为当前工作区的 LiSA-staged：
`methods/light_atlas.py`、论文方法/优化章节及最新补充实验；不是最初六场景版本。
本次仅检索、阅读与记录，没有修改模型或启动实验。

## 1. 判断与检索边界

本次查到了实质性组件重合，但尚未找到一篇已核实论文完整实现以下组合：

> 从逐场景 OLAT 图像联合学习 Gaussian 几何与特征；每个新光源重新 splat 出
> alpha 加权的深度矩和学习特征；线性滤波为共享多尺度光源图集；接收点查询该图集，
> 用于矩阴影修正及非负特征加权外观；最终用像素接收点精修。

“未找到完整同构方法”不是首次提出的证明，也不是接收概率判断。
核心组件不能分别声称首次提出；特定组合仍可能形成研究贡献，需用适合本任务的对照说明其价值。
本报告的“强重合”指某项贡献表述会被已有工作覆盖，不等于整篇 LiSA 与其相同。

检索同时覆盖经典实时渲染和 2024–2026 Gaussian 重光照，按机制检索而非仅按标题：
`Gaussian relighting mipmap`、`Gaussian light-space features`、`neural reflective shadow maps`、
`learned translucent shadow maps`、`Gaussian shadow map moments`、`light-space feature pyramid`、
`shadow map Gaussian CDF` 等，并追踪原论文引用和作者代码。
不采用搜索结果的相对“发布时间”作为论文日期；以论文、会议和作者元数据为准。

## 2. 本地方法的准确边界

- 光源 pass 存储 `sum(w_i phi_i)`、`sum(w_i z_i)`、`sum(w_i z_i^2)` 和 `sum(w_i)`；
  `w_i = T_i alpha_i`。其中 `z_i` 是高斯中心的归一化光源轴向深度。
- 8 个学习特征通道加 3 个统计通道，512² 起始、7 层固定模糊/下采样。
  每个接收点在所有层的同一投影位置采样；没有经典纹理 mipmapping 的像素 footprint→LOD 选择。
- 可见度先用 `1-M0*Phi((z-mu)/sqrt(var+tau^2)-3)`，再在最细层结果的 logit 上加学习修正。
- 传输分支用接收点属性与深度统计预测非负权重，再线性组合采样到的特征。
  这不是显式源点—接收点路径积分，也不保证分支对应真实间接光。
- 前30k使用Gaussian接收点，冻结几何后8k使用像素接收点。

来源：[实现](../../methods/light_atlas.py)、[最终方法](../../paper/latex/sec/3_method.tex)、
[优化](../../paper/latex/sec/4_optimization.tex)。

## 3. 最接近的工作

### A. Translucent Shadow Maps，EGSR 2003

**最接近“光源空间 mipmap 聚合散射”的基本结构。** 原文§3–4已经存储入射辐照度、深度和法线，
在接收点的光源投影邻域查询；21点分层采样使用不同mip层，并使用alpha缓冲避免背景污染。
它还区分局部和较远范围的散射响应。

与LiSA的区别：已知几何和解析扩散响应；采样具有横向偏移并利用源—接收几何；
LiSA由图像学习Gaussian及特征，在各层同一投影位置读取混合统计，以学习权重近似响应。
不能把“光源图的多尺度聚合用于SSS”或“alpha归一化图集”单独列为新意。

核查：原PDF§3–4（含mipmap采样、alpha归一化）；
[出版页](https://diglib.eg.org/items/5af01fd6-37dc-43df-b741-396d34105fc4)、
[原PDF](https://diglib7.eg.org/server/api/core/bitstreams/61d24d66-9f4c-4ed0-9bc0-cf8df86b0cd1/content)。

### B. Shadow Map Filtering with Gaussian Shadow Maps，VRCAI 2011；后续C&G 2013

**直接覆盖“从深度均值/方差构造正态CDF阴影检验”的核心。** 2011原文Eq.(4)使用
正态分布CDF拟合深度分布；2013扩展讨论指数变换。这里的Gaussian指统计分布，不是3DGS。

LiSA额外包含alpha合成的深度分布、覆盖率、尺度项、3σ偏置与学习修正。
这些改变需要独立论证，但CDF本身不能作为新的阴影原理。

核查限制：2011大学库PDF直连返回403；已核对大学收录年份、原PDF被索引的公式段落，
并用2013出版社技术摘要交叉确认。未宣称成功通读2011全文。
[2011大学收录](https://recerca.uoc.edu/documentos/5ecfaae7299952687c553a02)、
[原PDF入口](https://repositori.uji.es/server/api/core/bitstreams/c174cfc0-4c6b-4a73-9330-49dd0def8c13/content)、
[2013出版社页](https://www.sciencedirect.com/science/article/abs/pii/S009784931200180X)。

### C. NeuMIP: Multi-Resolution Neural Materials，SIGGRAPH 2021

**直接覆盖“神经特征金字塔＋MLP解码多尺度外观”。** §3.2–3.3以UV位置、尺度及入射/出射方向
查询神经纹理，并以小MLP解码；各层纹理独立优化，不是从最细层固定滤波生成。

与LiSA的区别：NeuMIP是材质空间的多尺度BTF表示，层级由查询尺度选择；
LiSA图集随光源重新渲染，表示当前场景受光和遮挡，所有层通过固定滤波相连并共同参与查询。
不能宣称首次提出“可学习mipmap”，也不能把NeuMIP等同于LiSA的动态光源图集。

核查：[原文§3.2–3.3](https://arxiv.org/html/2104.02789v1)、
[作者项目](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/)。

### D. RNG: Relightable Neural Gaussians，CVPR 2025

**同任务中很近的结构先例。** §3.3从相机深度恢复像素接收点，再投影到光源深度图；
以接收点与光源深度对应点的距离作为阴影提示。§3.4明确先forward、后deferred，
理由是前者几何较好，后者阴影更清晰。

LiSA增加深度二阶矩、多尺度信息、学习特征聚合及具体的可见度分支。
但“在Gaussian重光照中逐像素查询shadow map”和“用forward/deferred阶段兼顾几何与阴影”已有先例。

核查：[原文§3.3–3.4](https://arxiv.org/html/2409.19702v5)、
[CVPR正式PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Fan_RNG_Relightable_Neural_Gaussians_CVPR_2025_paper.pdf)。

### E. Real-Time Object Composition in 3D Gaussian Splatting via Physics-Driven Light Transport Factorization，PG 2026

**新查到的Gaussian矩阴影近邻。** 出版摘要明确涉及不透明度加权的光源深度矩和deferred VSSM。
已进一步核对正文Shadow Modeling、Eq.(22)与作者`shadow6.py`，实际顺序是：

1. 光源视角alpha合成得到平均深度代理图`Z_L`；
2. 对`Z_L`及其平方建立Summed-Area Tables；
3. 以blocker search估计半影范围，在相应邻域求空间均值/方差，使用Chebyshev界计算可见度。

**不能误写为与LiSA相同的矩：** 它对已合成的深度图平方再滤波；LiSA在光栅化前就把
每高斯的`z_i^2`作为属性合成。通常`(sum(w_i z_i)/sum(w_i))^2`不等于
`sum(w_i z_i^2)/sum(w_i)`，后者保留同一光线内的深度方差信息。

该工作处理预训练场景和已知PBR资产的免训练物体合成，使用SAT/VSSM；
没有在核查的阴影路径发现LiSA式学习特征金字塔与OLAT联合重建。
它实质覆盖“Gaussian光源深度＋矩阴影＋逐像素查询”的宽泛主张，但不等于完整LiSA。

日期：Eurographics条目元数据`dc.date.available`为2026-09-17T13:23:38Z；
这是库元数据，不推断它是最早公开时间。不要采用搜索摘要“9个月前”或作者主页不一致的会议月份。

核查：[出版社条目](https://diglib7.eg.org/items/0a144ab0-f5dc-42a9-8f4a-d292f9736fd6)、
[作者PDF](https://raw.githubusercontent.com/xx-luozi-xx/3DGS-Object-Composition/main/assets/paper.pdf)、
[已核查代码版本](https://github.com/xx-luozi-xx/3DGS-Object-Composition/blob/dc7fdd09302b6f255060f1a15b74e9d8b1141593/shadow6.py)。

### F. LightFormer，SIGGRAPH 2024

**覆盖“光源观察写入学习表示、接收点学习聚合非局部光照”的宽泛思路。** 原文§4.2的neural RSM
记录深度、位置、法线、反射通量，将虚拟点光源编码为学习表示，§5用pixel-light attention聚合。
因此LiSA不能只凭“将手工RSM变成可学习的写入/读取”主张整个方向原创。

区别在于给定几何/材质的神经前向渲染、VPL/光源embedding与attention；LiSA逐场景反演几何，
保持二维可滤波图集并在局部多尺度投影邻域聚合，而不是该attention算子。

核查：[作者项目](https://wylighting.github.io/lightformer/)、
[原文§4.2、§5](https://wylighting.github.io/lightformer/static/pdf/paper.pdf)。

### G. Neural Shadow Mapping（SIGGRAPH 2022）与 Kernel Predicting Neural Shadow Maps（SIGGRAPH 2025）

前者由G-buffer和阴影图提示学习阴影；后者学习屏幕空间滤波核，并结合四阶Moment Shadow Maps。
因此“矩阴影＋神经修正/学习核”不是空白。它们主要面向给定几何的阴影生成，
不等同于LiSA的光源空间特征传输及Gaussian逆渲染。

核查：[NSM原文](https://sayan1an.github.io/pdfs/neuralShadowMapping.pdf)、
[KPNSM原文§3–5](https://doi.org/10.1145/3721238.3730645)、
[KPNSM作者项目](https://hoosus.github.io/kpnsm/)。

## 4. 其他相关与易混淆工作

| 工作 | 已核查内容及与LiSA的关系 | 一手来源 |
|---|---|---|
| Variance Shadow Maps / Summed-Area VSM | 深度矩的可滤波性、mipmap/SAT已有先例；是阴影统计的基础 | [作者技术章节](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps) |
| GS³，SIGGRAPH Asia 2024 | 光源splat后将光线结果汇总为逐高斯可见度，再学习修正；另有局部RGB残差。官方实现已有HDR目标gamma过渡。无LiSA式共享特征金字塔 | [原文§4](https://arxiv.org/html/2410.11419v1)、[官方训练实现](https://github.com/gsrelight/gs-relight/blob/main/train.py) |
| Deep Gaussian Shadow Maps，2026预印本 | 将解析光学厚度/透射率按方向和径向深度存入八面体图集，接收点采样；不是二维多尺度学习特征金字塔，也不做LiSA式OLAT联合拟合 | [原文§3](https://arxiv.org/html/2601.01660v1) |
| RGS-DR，2025预印本 | 环境光cubemap mip链对应粗糙度积分；另有球面特征mipmap残差，原文说明换光时不启用该残差。区别于每灯重新splat的场景光源图集 | [原文§4.2–4.3](https://arxiv.org/html/2504.18468v2) |
| Lightweight Attention-based Indirect Illumination，SIGGRAPH 2026 / AMD技术介绍 | RSM反射通量、G-buffer、1spp提示经attention生成间接光；与学习光源聚合相关，没有在官方介绍中看到LiSA的矩/特征金字塔逆渲染 | [作者技术介绍](https://gpuopen.com/learn/lightweight-attention-based-indirect-illumination/) |
| Moment-Based 3D Gaussian Splatting，2025预印本 | 从沿射线Gaussian密度的解析矩重建连续透射率，实现顺序无关体渲染；不能将其统计量与LiSA的alpha沉积深度矩混为一谈 | [原文§3](https://arxiv.org/html/2512.11800v1) |
| GTS，PG 2026 | 已读正文：逐高斯Spherical Voronoi方向可见度、顺序无关混合与环境光mip链；不是动态光源投影图集 | [作者PDF](https://cseweb.ucsd.edu/~ravir/jiamupg.pdf)、[出版社条目](https://diglib7.eg.org/items/4b27e167-0f87-4422-b6b4-19df6a2c39ed) |
| Mip-Splatting，CVPR 2024 | 3D频率约束与屏幕空间2D Mip滤波针对采样率/抗混叠；不是点光源图集，也不能用其结果支持LiSA的抗混叠保证 | [正式论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html) |

补查线索包括GOGS、F-RNG、Gaussian Codec Avatars、RadiosityGS和NeLiF。
前几项分别涉及方向mipmap、前馈重光照、环境光预滤波或显式surfel传输，
不以关键词相同就判为同构。NeLiF仅确认了官方摘要：从灯具多视图编码3D lighting field，
再结合G-buffer/阴影提示解码，并使用生成的HDR Gaussian灯具表示。
其完整方法页受访问限制，本轮未获得一手全文，不能声称已经排除全部结构重合。
[NeLiF官方摘要](https://sa2025.conference-schedule.org/presentation/?id=papers_1990&sess=sess106)。

## 5. 对论文表述的具体影响

| 原先可能使用的主张 | 审计后的边界 |
|---|---|
| 首次从光源视角splat Gaussians求阴影 | GS³等已有；不成立 |
| 首次使用深度矩/正态CDF估计阴影 | VSM、Gaussian Shadow Maps已有；不成立 |
| 首次用光源mipmap处理非局部散射 | TSM已有；不成立 |
| 首次用神经mipmap解码材质/外观 | NeuMIP已有；不成立 |
| 首次把Gaussian阴影改为逐像素shadow-map查询 | RNG等已有；不成立 |
| 首次把Gaussian深度与矩阴影结合 | PG2026合成工作等使该宽泛主张不成立；具体矩定义仍有区别 |
| 所有层来自同一可微alpha沉积，深度统计和学习特征共同服务接收点查询，并与OLAT重建联合学习 | 本次未找到完整同构组合；可作为具体设计描述，尚不能宣称全球首次 |

可以采用的谨慎定位是“面向Gaussian重光照逆渲染的可学习多尺度光源图集”，
正面承认TSM/矩阴影/神经材质的来源，再解释Gaussian重建中的表示与优化选择。
不要把“已知几何改成可学习几何”自动当作充分贡献，也不要因组件已知就自动否定整个组合。

## 6. 现有实验与最直接的区分证据

补充实验的seed0五场景：full 31.5295、single_scale 31.0912（差0.4383 dB）；
三seed：full 31.51213、local_residual 31.53948。
后者仍保留图集深度矩阴影，因此只限制“特征传输必然优于局部残差”的主张，
不能解释为多尺度图集整体无效。见[补充结果](../experiments/supplementary/results.md)。

最直接的后续验证建议（本次未执行，也不要求全部外部方法都复现）：

1. 同一接收点/训练流程下比较单尺度、多尺度深度统计及多尺度特征，隔离是哪类信息有用。
2. 同一几何下比较可实现的普通深度/PCF或VSM查询与LiSA阴影，避免把逐像素着色收益归给mipmap。
3. 在本任务的留出灯位上比较学习图集与局部条件残差；不额外要求材质编辑或场景编辑。

来源阅读的完整度已在各项标明。结论是目前可公开检索并核实的证据范围内的研究判断；
未公开、未索引、无法取得全文的工作仍可能影响新颖性定位。
