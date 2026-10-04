# 2025年3月起：逐场景OLAT重光照论文与本地数据出处

初次核查：2026-10-03（JST）；系统补查：2026-10-04。用户已明确起点包含3月，当前正式发表窗口为2025-03-01至2026-10-03。此前直接从4月1日起算过窄，已纠正并补回BiGS。排除以光照泛化为主要目标的逐场景研究。本轮仅调研、读取数据元信息和更新文档，没有修改模型、转换数据或启动训练。

10月4日进一步明确：只找以重光照为核心任务的论文。Stoch3DGS和Splat the Net仍是技术参考，不计主候选。当前新增主候选为BiGS、RadiosityGS；MLI-NeRF有直接NRHints实验与官方入口，但主旨偏内在分解，单列；Practical Inverse Rendering有官方点光OLAT代码，但要求已知网格且需数据管线适配，单列条件候选。不能把这四项都宣称为满足相同条件的基线。

## 1. 筛选口径

- 时间：2025-03-01至2026-10-03，按正式会议发表时间计算，而非仓库更新、检索日期或论文集后补入库日期。BiGS、RNG虽在2025年会议发表，arXiv初稿是2024年；这里不会把预印本日期和正式发表日期混用。
- 会议：检查CVPR、ICCV、ECCV、ICLR、NeurIPS、ICML、SIGGRAPH/SIGGRAPH Asia及3DV等领域会议的正式论文，包含相应TOG论文；逐项标明实际venue，不将Findings、workshop或仅预印本混入。
- 范式：一个对象/场景独立优化几何、外观、材质或光传输；排除跨场景前训练后的前馈重光照，也按用户进一步指示排除以OOD光照泛化为主要研究目标的MetaGS。
- 主要任务：重光照必须是论文的核心研究目标，并有相应方法设计和主要实验；以通用表示、普通NVS或训练/渲染算法为主、将重光照作为应用验证的论文不进入主候选。不能以“做过一个relighting实验”替代这个条件。
- 数据合同：多个已知相机视角，图像间点光源位置变化，光源与相机可以独立移动；输入RGB和相机/灯光标定，评价同一场景的重光照。仅在推理时“能加一个点光源”，或理论上改写渲染器后可能支持，不足以入选。
- 优先证据：正式论文实验节/表格、作者项目页与官方代码。明确区分同一数据版本、相同数据家族、相同物体资产与仅格式兼容。
- 支持程度与复现成本分别记录：论文不必已跑过我们的全部18场景，也不要求无需格式转换即可运行；额外几何输入、完整相机×灯光采样、固定相机灯光关系等实质性要求必须说明。

## 2. 已跑对手与新增候选

SSD-GS和RNG已跑过，不计入新增方法。新增主候选为BiGS和RadiosityGS。BiGS专门从OLAT数据学习可重光照物体；RadiosityGS把高效重光照作为核心目标之一，在主文中进行GS³来源数据的重光照定量比较，同时研究几何重建。以下为已核实清单，不声称检索已经穷尽所有研究。

| 方法 | 正式发表 | 逐场景与数据证据 | 与本地18场景关系 |
| --- | --- | --- | --- |
| SSD-GS: Scattering and Shadow Decomposition for Relightable 3D Gaussian Splatting | ICLR2026 | 论文§5.1和表1明确采用三类OLAT数据；官方train.py逐场景优化 | 直接覆盖同名7个真实、6个GS³合成、5个SSS合成场景；我们的评测集合以此为参照 |
| RNG: Relightable Neural Gaussians | CVPR2025，6月 | 正式论文§4.2、表1与官方逐场景训练入口明确支持NRHints点光源数据 | 真实7场景全部在论文表1中；另含FurBall/Hotdog/Lego等NRHints合成场景，但不等于本地GS³重渲染版本 |
| RadiosityGS: Differentiable Light Transport with Gaussian Surfels via Adapted Radiosity for Efficient Relighting and Geometry Reconstruction | SIGGRAPH Asia2025 / TOG，12月 | 论文§9明确使用GS³ Synthetic/RenderCapture已知点光源数据；官方README明确支持GS³灯位/光强格式 | 实际使用过GS³数据家族，正文展示Translucent、FurBall、Hotdog等；论文改为25/50张训练、其余约2000张测试 |
| BiGS: Bidirectional Primitives for Relightable 3D Gaussian Splatting | 3DV2025，3月 | 核心任务为OLAT到可重光照高斯；官方训练脚本逐帧读取灯位/光强，直接调用点光渲染器并计算距离平方衰减 | 支持相同类型的多视图点光OLAT，作者使用自建合成/捕获数据；没有确认原论文跑过本地NRHints、GS³或SSS-GS版本；需普通GS几何初始化及格式转换 |

### 2.1 SSD-GS

这是直接比较三类数据覆盖最完整的对手，官方给出相同的三层目录组织。需要继续保留当前train-only主协议：原始测试标定、没有测试GT拟合。其论文中三个真实场景使用1024px，与本地512px不同，数字不能直接拼表。

来源：[正式论文](https://fanglue.github.io/papers/SSDGS.pdf) · [官方仓库](https://github.com/irisfreesiri/SSD-GS) · [官方数据说明](https://raw.githubusercontent.com/irisfreesiri/SSD-GS/main/README.md)。

### 2.2 RNG

核对的是CVPR2025正式PDF，不使用早期arXiv版本的旧表。表1列出Cat、CatSmall、Cluttered、CupFabric、Fish、Pikachu、Pixiu；Cluttered对应本地FurScene。其论文将训练输入限制到最多1000张，512px，合成背景为黑色；本地GS³为2000张/白背景，SSS也不是RNG原论文的已确认实验集。因此，“我们已经把RNG适配到18场景”不等于“RNG作者原本评测了同样18场景”。

来源：[正式论文§4.2与表1](https://openaccess.thecvf.com/content/CVPR2025/papers/Fan_RNG_Relightable_Neural_Gaussians_CVPR_2025_paper.pdf) · [作者项目](https://whois-jiahui.fun/project_pages/RNG/) · [官方代码](https://github.com/sssssy/RNG_release)。

### 2.3 RadiosityGS

重点是显式可微全局光传输与几何/材质优化，使用Gaussian surfels和radiosity求解器；属于逐场景方法。本轮对其“支持我们这种数据”的证据最直接：官方README明确接受GS³提供的点光源位置和强度。也给出逐场景train.py及完整评价脚本。

适配限制：官方要求HDR监督、场景缩放到单位立方体，不含额外环境底光；论文暂不处理相机参数求导，排除了含相机误差的数据。因而优先对应我们的Synthetic_GS3，不能直接宣称Real_NRHints/PNG格式SSS也能无改动运行。官方重实现与论文数值略有变化，须注明版本；25/50视图协议与本地完整训练集应分别报告。

来源：[论文§9](https://arxiv.org/html/2509.18497v2) · [作者项目与会议](https://raymondjiangkw.github.io/radiositygs.github.io/) · [官方代码及输入要求](https://github.com/RaymondJiangkw/RadiosityGS)。

### 2.4 BiGS：应补回的专门重光照方法

作者项目页明确标注3DV2025、BibTeX为2025年3月，DOI为10.1109/3DV66043.2025.00099。其核心是用双向球谐函数表示高斯的视线/光照相关散射，并分解直接/间接光传输；逐场景优化，包含近场点光、环境光、半透明与毛绒等外观，不以跨场景前馈或OOD光照泛化为主要目标。

代码证据比“推理时能加点灯”更直接：`scripts/training.py`第249–265行从训练样本读取`light.position`与`light.intensity`并调用`render_bigs_with_point_light`；`relight/bigs_olat_dataset.py`第135–138行从每帧元数据读取灯位/光强；点光渲染器第28–37行计算逐高斯光线方向和距离平方衰减。核查版本为main分支`8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613`。

原论文合成采集为40灯×48相机，另有全灯开启图像用于几何初始化。官方流程先训练普通GS，再训练重光照分量；README建议使用中性光图像初始化，提供`olat_all_on`示例。我们没有该额外采集分支，后续必须明确几何初始化来源，不能宣称只是改一个路径就能复现，也不能使用测试GT生成初始化。原版训练脚本默认选择40个灯组，数据组织需要适配；此次只读取代码，没有做这些改动。

来源：[作者项目与正式发表信息](https://desmondlzy.github.io/publications/bigs/) · [论文§5](https://arxiv.org/html/2408.13370) · [官方训练入口](https://github.com/desmondlzy/bigs/blob/main/scripts/training.py) · [官方输入读取](https://github.com/desmondlzy/bigs/blob/main/relight/bigs_olat_dataset.py) · [官方点光渲染器](https://github.com/desmondlzy/bigs/blob/main/relight/render_bigs_with_point_light.py)。

## 3. 技术参考与排除项

### 3.1 Stoch3DGS：移出主候选

其核心贡献是用于普通及可重光照3DGS的随机光线追踪和无偏梯度估计，重光照是通用算法的应用之一。虽有主文§5.2、表3的NRHints重光照实验和GS³/RNG对照，按用户要求仍不作为专门重光照方法计数；保留为阴影与训练算法的技术参考。表3有Cat、Pixiu、Lego、Hotdog、FurBall、Basket，不能仅凭合成场景同名认定与本地GS³版本相同。

代码边界：当前公开仓库master分支README只给出MipNeRF360普通新视角合成复现流程；本轮没有核实到完整OLAT训练入口。不能把“仓库公开”表述为“重光照分支已可直接复现”。

来源：[作者项目与CVPR2026信息](https://iliyan.com/publications/RayTracingReconstruction/) · [论文§5.2/表3](https://arxiv.org/html/2603.23637v2) · [公开README](https://raw.githubusercontent.com/XuPaya/Stoch3DGS/master/README.md)。

### 3.2 Splat the Net：移出主候选

它用椭球范围内的浅层神经密度场替代固定解析高斯核，逐场景优化，研究重点是表示能力、存储与渲染效率。重光照仅为附录应用，故不符合“专门做重光照”的要求；不再列为新增重光照对手。

核对ICLR2026正式论文PDF共21页：第19页附录D.2给出光照条件颜色网络，第20页明确使用Bi等（2024，即GS³）及Kang等（2019）提供的OLAT数据，包含合成场景渲染图和真实捕获资产的渲染图；图12展示重光照。其“real captures的rendered images”不能替换为我们的Real_NRHints真实照片，也不能仅凭Hotdog/Lego同名就断定全部帧与本地划分一致。

证据强度与代码限制：重光照在该论文中是附录的应用展示，没有查到重光照PSNR/SSIM/LPIPS比较表；主文的NVS指标不能当作重光照指标。官方仓库main分支在本轮读取时为`3b069fda26a004b9f31307ff7b1f25f0ad139ac1`（提交日期2025-11-14），只有main公开分支。README提供NeRF Synthetic、MipNeRF360等普通NVS流程；递归目录、参数定义和数据读取文件未显示OLAT训练入口或灯位读取。因此应列为“论文确实做过相近数据，重光照复现流程待确认”，不能承诺拉取仓库即可在本地18场景运行。此次只读取公开文件，没有安装或执行作者代码。

来源：[ICLR2026正式条目](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad7922fd4650f8aba5d8b067e622ca84-Abstract-Conference.html) · [正式论文D.2，第19–20页](https://proceedings.iclr.cc/paper_files/paper/2026/file/ad7922fd4650f8aba5d8b067e622ca84-Paper-Conference.pdf) · [作者HTML全文D.2](https://arxiv.org/html/2510.08491v2) · [官方代码](https://github.com/huynguyenbao/splat-the-net)。

### 3.3 MLI-NeRF：数据支持最直接的非高斯相关方法，主旨偏内在分解

MLI-NeRF（Multi-Light Intrinsic-Aware Neural Radiance Fields，3DV2025）确实逐场景训练，并且实际使用DNL/NRHints真实物体数据。官方README的“Run on the real object dataset”直接要求从NRHints下载数据，提供`NRHints_Pikachu_a/b.yaml`与`run_real.sh`；脚本包含逐场景训练、伪标签构建和`transforms_test.json`评测。主文表2比较Random Lights下的NVS+重光照，补充材料表4还比较真实物体重光照。因此它不是只有理论适配可能，也不应再因3月日期被排除。

但论文主要贡献是多光照帮助反射率/阴影的内在分解，重光照是正文明确评测任务之一。按用户“专门做重光照”的严格口径，不把它与BiGS/RadiosityGS无差别计数；它是具备直接数据证据、值得单独考虑的非高斯对手。合成数据为作者为获得内在分解GT重新渲染的版本，也不等于我们的GS³合成文件。

来源：[作者论文§4.1、§4.4–4.5及补充材料](https://arxiv.org/html/2411.17235) · [官方NRHints说明与入口](https://github.com/liulisixin/MLI-NeRF) · [逐场景运行脚本](https://github.com/liulisixin/MLI-NeRF/blob/main/run_real.sh)。

### 3.4 Practical Inverse Rendering：官方点光OLAT已确认，额外输入仍是限制

此前只根据论文中的人脸面光采集给它作否定分类不够充分。本次检查官方代码，`scene_configuration.py`明确定义`Setup.OLAT`，`scene_preparation.py`第433–456行专门处理`type=point`的光源并包装成可切换的OLAT灯；`optimization.py`也提供逐灯优化及逐传感器更新点光位置的流程。所以它确实有官方点光OLAT实现，而不只是Mitsuba理论上支持点光。

仍不能当成当前18场景的直接替换基线：它优化已知网格上的纹理材质/SSS，要求初始网格、相机与灯光；公开通用优化器按相机×灯光二维参考图数组组织数据，还需核对和适配我们不完整的相机/灯光配对。官方没有给出NRHints/GS³/SSS-GS数据入口或相同基准实验。该工作主要是材质/散射外观重建，重光照为目标应用，故按当前要求单列条件候选，不增加严格主清单数量。

来源：[SIGGRAPH2025作者论文](https://weiphil.s3.eu-central-1.amazonaws.com/practical_reconstruction_low_res.pdf) · [官方点光OLAT处理](https://github.com/google/practical-inverse-rendering-of-textured-and-translucent-appearance/blob/main/python/practical_reconstruction/scene_preparation.py) · [官方优化流程](https://github.com/google/practical-inverse-rendering-of-textured-and-translucent-appearance/blob/main/python/practical_reconstruction/optimization.py)。

### 3.5 其他专门方法及未满足条件的工作

没有因表示类型预先排除mesh、物理逆渲染或图像方法；但在当前日期和数据要求下，以下工作不能直接加入核心清单。

| 工作 | 核查结果/排除理由 |
| --- | --- |
| Practical Inverse Rendering of Textured and Translucent Appearance，SIGGRAPH2025 | 已确认官方点光OLAT代码，见§3.4；要求额外网格与输入适配，主旨为外观重建，单列条件候选 |
| Efficient Object Reconstruction with Differentiable Area Light Shading，SIGGRAPH Asia2025 | mesh+材质逐对象优化，主方法是主动面光LTC；虽然论文有点光对照和Lego/Hotdog资产，但使用重新渲染的TensoIR/Shiny Blender版本，不能据此认定支持我们的点光OLAT训练管线 |
| PBR-NeRF，CVPR2025 | 正式实验是NeILF++/DTU及固定照明模式；未确认逐图变化的独立点光源OLAT输入合同，故不凭“逆渲染”标题纳入 |
| IRGS，CVPR2025；GHPT，CVPR2026 | 确实以可重光照逆渲染为主要目标，但官方训练/评测是环境光下的Synthetic4Relight、TensoIR Synthetic等；没有确认我们三类近场点光OLAT训练输入的官方支持，不能因名称含relight就直接作为兼容基线 |
| SU-RGS，ICCV2025 | 专门研究稀疏视角、变化环境光下的逐场景重光照；正式实验为NeRF-OSR、TensoIR Synthetic、NeRF On-the-go，没有我们三类点光OLAT实验或明确输入支持 |
| MetaGS，NeurIPS2025 | 实际是逐场景、同一场景不同灯位间的meta-learning，并非仅因名字就认定跨场景模型；但其核心目标是OOD光照泛化。用户明确要求连此类逐场景研究也排除，所以本轮不列入候选 |
| F-RNG、RelitLRM、GenLit等 | 前馈/跨场景先验或生成式重光照，不符合用户指定范式 |
| GS³、SSS-GS、OLAT Gaussians | 正式会议均为2024年，不符合新日期限制；不能因SSS-GS某些引用写2025或代码较晚发布就将其当作2025新论文 |
| PIR（Photometric Inverse Rendering），3DV2025 | 日期已满足；但论文§3.3明确假设相机与点光源相对位姿在所有图像间固定，联合优化一个刚性偏移；不同于我们逐图独立灯位。不能只看“不必完全共位”的摘要就认为支持任意独立灯位 |
| Differentiable Inverse Rendering with Interpretable Basis BRDFs，CVPR2025 | 官方论文与代码为多视图闪光摄影，论文§4实验采用共位/相机携带的闪光灯；尚未确认独立灯位OLAT训练入口。它与MVSCPS不是同一方法，也不是因为没有点光模型而排除 |
| MVSCPS: Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues，ICCV2025 | 官方README确实允许相机/灯光逐张独立移动，且逐场景训练；但论文§3.1明确假设单个方向光，§3.2按灯光方向与RGB强度建模，实验用DiLiGenT-MV和自采数据。没有近场点灯位置/距离衰减模型或我们三类数据实验的明确证据，不能仅凭OLAT关键词纳入 |
| GLOW: Global Illumination-Aware Inverse Rendering of Indoor Scenes Captured with Dynamic Co-Located Light & Camera | 逐场景近场逆渲染，但正文采集及训练依赖相机与灯光共位，使用自制室内场景。没有核实对相机/灯光独立移动的本地三类数据的官方支持；推理时能移动光源不足以证明训练输入兼容 |
| SpotlessGS，IROS2026已接收 | 主要针对机器人机载聚光灯造成的不均匀照明，逐场景优化灯与相机的相对位姿，再恢复均匀照明。未确认本地三类独立点光OLAT数据支持，也未核实其在指定截止前的正式会议发表日期，不能当作已满足全部条件的候选 |
| NRHints、DNL | 分别SIGGRAPH2023、SIGGRAPH Asia2020，超出时间范围；仍是解释数据来源的重要文献 |
| OpenSubstance、OLATverse、HumanOLAT | 数据集/基准工作，不据此补足方法baseline数量；也不属于本次已跑数据 |

非高斯两篇全文已读取，而非仅据标题分类：
[Practical Inverse Rendering官方论文](https://weiphil.s3.eu-central-1.amazonaws.com/practical_reconstruction_low_res.pdf)、[官方代码](https://github.com/google/practical-inverse-rendering-of-textured-and-translucent-appearance)；
[Area Light Shading作者论文](https://wanghmin.github.io/publication/gao-2025-eor/Gao-2025-EOR.pdf)。
其余依据：[PBR-NeRF正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_PBR-NeRF_Inverse_Rendering_with_Physics-Based_Neural_Fields_CVPR_2025_paper.pdf)、[MetaGS正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fec4c7df77668fb15a3ff6a90908c995-Abstract-Conference.html)、[SSS-GS会议条目](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc72529d604962a86b7730806b6113fa-Abstract-Conference.html)。

10月4日补查依据：[MVSCPS正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_Neural_Multi-View_Self-Calibrated_Photometric_Stereo_without_Photometric_Stereo_Cues_ICCV_2025_paper.pdf)、[MVSCPS官方仓库](https://github.com/CyberAgentAILab/MVSCPS)、[GLOW作者全文](https://arxiv.org/html/2511.22857v2)。对MVSCPS的排除基于方向光与近场点光的输入差异，而不是将文中的“generalization to view-unaligned OLAT”误判为用户排除的光照OOD泛化任务。

针对“专门做重光照”的补查来源：[IRGS官方代码与数据](https://github.com/fudan-zvg/IRGS)、[GHPT作者项目与实验](https://nwu-vislab.github.io/GHPT/)、[SU-RGS正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_SU-RGS_Relightable_3D_Gaussian_Splatting_from_Sparse_Views_under_Unconstrained_ICCV_2025_paper.pdf)、[SpotlessGS作者全文](https://arxiv.org/html/2608.14713)、[作者会议说明](https://arxiv.org/abs/2608.14713)。

3月边界重查与闪光采集证据：[PIR作者论文§3.3](https://arxiv.org/html/2408.06828)、[Interpretable Basis BRDFs正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Chung_Differentiable_Inverse_Rendering_with_Interpretable_Basis_BRDFs_CVPR_2025_paper.pdf)。

## 4. 三类实验数据从哪里来

**不是一次采集、同一篇工作首次发布的数据集。当前18场景是沿用SSD-GS的跨来源评测组合。** SSD-GS的目录名称和统一下载入口是组织方式，不是全部数据的原始作者归属。

| 本地类别 | 数量与场景 | 论文/采集来源 | 本地保留的直接下载包 |
| --- | --- | --- | --- |
| Real_NRHints | 7：Cat、CatSmall、CupFabric、Fish、FurScene、Pikachu、Pixiu | NRHints（SIGGRAPH2023）使用并发布的真实场景集合，其中4个继承自DNL（2020），3个是NRHints新采集 | _sources/gsrelight-data/NRHints/*.zip；来自GS³官方数据仓库的NRHints子目录 |
| Synthetic_GS3 | 6：AnisoMetal、Drums、FurBall、Hotdog、Lego、Translucent | GS³（SIGGRAPH Asia2024）发布的合成点光渲染版本；物体资产和场景设计有更早来源 | _sources/gsrelight-data/Synthetic/*.zip |
| Synthetic_SSS-GS | 5：bunny_small、candle_small、dragon_small、soap_small、statue_small | SSS-GS（NeurIPS2024）发布的合成散射数据small版本 | _sources/SSS-GS/synthetic/*_small.tar |

以上相对下载包路径都位于/workspace/datasets/SSD-GS。展开后数据位于同一根目录的data/。

### 4.1 Real_NRHints的更早血缘与改名

NRHints论文§4明确指出7个真实场景中4个来自Gao等人的DNL工作，新采集的是Cat on Decor、Cup and Fabric、Pikachu。对应关系为：

- DNL旧场景：Cat、Fish、Pixiu、Cluttered/FurScene。
- NRHints新增：Cat on Decor/CatSmall、Cup and Fabric/CupFabric、Pikachu。
- 本地别名：Cat_on_Decor → CatSmall；Cup-Fabric → CupFabric；Cluttered → FurScene。

**CatSmall是另一个场景Cat on Decor的别名，不是把Cat下采样得到的版本。**

GS³论文§5.1说明真实场景使用NRHints相同的训练/测试数据，数据卡也要求同时引用NRHints。因此“从GS³仓库下载”与“来源是NRHints/DNL”并不矛盾。

来源：[NRHints论文§4](https://arxiv.org/html/2308.13404v1)、[NRHints官方数据列表](https://github.com/iamNCJ/NRHints)、[GS³论文§5.1](https://arxiv.org/html/2410.11419v1)、[GS³官方数据卡](https://huggingface.co/datasets/gsrelight/gsrelight-data)。

### 4.2 Synthetic_GS3：同名资产不等于相同数据

本地版本每场景2000 train / 400 test、EXR、512px。NRHints也发布过Lego、Hotdog、FurBall、Translucent等同名/同类场景，但原论文合成协议是500 train / 100 test，灯位/视角及渲染版本不能假定相同。RNG和Stoch3DGS表中的Lego不应直接标记成“已跑过我们这份2000/400的GS³ Lego”。

这里的Synthetic_GS3表示采用GS³公开的那一版多视图点光图像，不表示Lego/Hotdog三维资产首次由GS³创造，也不是普通固定光照NeRF Synthetic可以不加处理地替代。

来源：[GS³论文§5.1](https://arxiv.org/html/2410.11419v1)、[NRHints原始合成数据说明](https://github.com/iamNCJ/NRHints)。

### 4.3 Synthetic_SSS-GS：small是官方版本

官方提供full与small版本，当前用的是官方256px small包；每场景本地有500 train、500 val、500 test。原始数据集还有真实捕获部分及800px full合成版本，本次18场景没有使用这些部分。不能将我们跑的5个合成small场景描述成整个SSS-GS数据集。

来源：[SSS-GS官方数据卡](https://huggingface.co/datasets/CGTuebingen/SSS-GS)。精确帧数以本地JSON及源包核对为准；数据卡中的概括说明不能覆盖具体文件。

## 5. 本轮本地来源核验

读取/workspace/datasets/SSD-GS/_sources/的13个zip和5个tar，分别与data/中对应场景的全部transforms JSON作解析后内容比较：18/18场景、48/48份JSON一致，包括帧顺序、相机/灯位、图像引用与其他字段。没有解包写回、改名、重划分或修改数据。

检查包括Real的train/test/valid、GS³合成的train/test、SSS合成的train/test/val。该结果证明现有元数据与保留源包一致；没有重新从远端下载整套图像做逐字节认证，也没有凭此声称所有图像与远端当前版本完全相同。

机器记录：[dataset_provenance_audit_20261003.json](dataset_provenance_audit_20261003.json)；既有逐场景数量：[数据清单](datasets_inventory_20260912.md)。

三类数据共同满足多视图点光OLAT任务，但图像格式、光强约定、场景尺度、背景与分辨率不同。评测时使用统一指标，并保留分类平均，不能只用一个跨来源总平均代替各类表现。[本地评价协议](../experiments/setup.md)

## 6. 对后续比较的建议

已完成的SSD-GS/RNG继续保留，但不计新增。新增专门重光照对手为BiGS与RadiosityGS：前者输入类型支持明确，需解决几何初始化和格式；后者实际跑过GS³来源数据，需对齐HDR、尺度和视图预算。MLI-NeRF是直接支持NRHints的非高斯相关对手，但任务主旨偏内在分解；Practical Inverse Rendering是有额外几何输入的条件候选。Stoch3DGS与Splat the Net保留技术参考。上述建议没有触发安装、改数据或实验。

当前确认两项新增主候选，不将其误写为整个时间窗口只有两篇重光照论文，也不声称除此之外不存在合格研究。没有一项新增工作已核实覆盖本地全部18场景，不能据一个数据家族的实验推定全部兼容。

## 7. 2026-10-04系统补查覆盖与证据边界

本次直接读取CVPR2025/2026、ICCV2025、ECCV2026、ICLR2025/2026、NeurIPS2025官方目录，用重光照、逆渲染、反射率、光传输、外观采集等关键词筛查标题，再对相关候选读取摘要、论文实验节与官方代码。SIGGRAPH2025/Asia2025/2026结合作者项目、论文和引用追踪补查；目录镜像访问失败，不能声称其全部目录已完整遍历。3DV2025补查BiGS、MLI-NeRF、PIR，3DV2026检查SpotLight。

前向引用检索使用Semantic Scholar发现线索：GS³返回63条、NRHints返回75条、SSD-GS返回2条引用关系，三个接口均无下一页。这是检索当时数据库返回的140条关系，不是140篇去重论文，更不是140篇全文都已精读；索引还会遗漏或拆分会议/预印本版本。最终技术分类仍依据原论文与作者代码，不用元数据或自动解读代替。

本轮新增核查的部分专门方法如下。没有官方点光OLAT输入证据时，标记为“未确认”，不宣称理论上绝对不能适配。

| 候选及正式来源 | 已核实的数据/输入条件 | 本轮处理 |
| --- | --- | --- |
| [SVG-IR，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SVG-IR_Spatially-Varying_Gaussian_Splatting_for_Inverse_Rendering_CVPR_2025_paper.pdf) | TensoIR Synthetic、Aria Digital Twin，环境光逆渲染 | 未确认独立点光OLAT支持 |
| [GS-ID，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Du_GS-ID_Illumination_Decomposition_on_Gaussian_Splatting_via_Adaptive_Light_Aggregation_ICCV_2025_paper.pdf) | 环境光和局部SG光源估计；TensoIR、ADT等 | 局部光源表示不等于逐图已知灯位输入；未确认 |
| [TensoFlow，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_TensoFlow_Tensorial_Flow-based_Sampler_for_Inverse_Rendering_CVPR_2025_paper.pdf) | TensoSDF合成、Stanford-ORB；逆渲染重要性采样 | 未确认独立点光OLAT支持 |
| [SGS-Intrinsic，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Niu_SGS-Intrinsic_Semantic-Invariant_Gaussian_Splatting_for_Sparse-View_Indoor_Inverse_Rendering_CVPR_2026_paper.pdf) | Interiorverse、TensoIR、MipNeRF、FIPT、DL3DV；环境光+局部SG灯光 | 点灯重渲染展示不足以证明逐图点灯训练兼容 |
| [SunFaded，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chang_SunFaded_Illumination-Aware_Gaussian_Splatting_for_Dark_Scenes_with_Camera-Mounted_Active_CVPR_2026_paper.html) | 主动灯光刚性固定在相机上 | 采集条件不同 |
| [GAINS，ECCV2026](https://patrickbail.github.io/gains/) | 逐场景稀疏视图，TensoIR、Shiny Blender、Ref-Real等 | 未确认独立点光OLAT支持；不因使用先验就误判为前馈 |
| [Dynamic Inverse Rendering，ECCV2026](https://arxiv.org/abs/2607.09329) | 刚体运动提供不同光照观测，估计环境光 | 未确认当前OLAT输入支持 |
| [DR-GS，ECCV2026](https://arxiv.org/html/2606.29379) | GlossySynthetic，PBR重光照与变形 | 未确认独立点光OLAT支持 |
| [Diffusion-Based Material Regularization，ECCV2026](https://arxiv.org/html/2606.31065) | Synthetic4Relight、Stanford-ORB、DTC-Synthetic，环境图重光照 | 逐场景物理优化但数据条件不同；DTC不是我们的三类数据 |
| [ROGR，NeurIPS2025](https://neurips.cc/virtual/2025/poster/117687) | 生成环境光训练图，再逐对象拟合光照条件NeRF；TensoIR、Stanford-ORB | 不误判成纯跨场景前馈；仍未确认点光OLAT支持 |
| [LumiTokens，ECCV2026](https://eccv.ecva.net/virtual/2026/poster/5551)；[NeAR，CVPR2026](https://arxiv.org/abs/2511.18600) | 前者学习场景token编辑器，后者明确无需逐对象优化 | 不符合用户指定范式 |
| [PRTGaussian，APSIPA2024](https://arxiv.org/abs/2408.05631)；[PRTGS，ACM MM2024](https://arxiv.org/abs/2408.03538) | 名称相近但不是同一工作，前者是OLAT拟合，后者主要环境光PRT | 不在当前发表窗口 |
| [TIRPL，ICASSP2025](https://ieeexplore.ieee.org/document/10888949) | 已定位论文元信息，未取得足以核查输入/数据集的原文和代码 | 证据未完成，不按标题认定兼容或凑入主清单 |

检索记录：[relighting_search_audit_20261004.json](relighting_search_audit_20261004.json)。
