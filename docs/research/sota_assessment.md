# LiSA的SOTA判断与文献可比性

核查日期：2026-10-03（JST）。对象为当前light_atlas + svbrdf最终配置，18场景、30k步、seed0。结论针对本次能核查的公开来源及本地证据，不声称检索穷尽所有论文。

## 结论

**目前不能宣称全面或文献级SOTA。** 可以准确表述为：在本地固定原始测试标定的18场景、四方法对照中，LiSA总体SSIM与LPIPS最佳，Real_NRHints和Synthetic_SSS-GS三项平均指标最佳；总体PSNR低于SSD-GS和GS³，Synthetic_GS3三项均落后于这两种方法。

单独报告“真实数据/SSS平均领先”应附带本地对照、评价协议与单种子限制；不能省略限定后称为这两个领域的SOTA。[完整成绩与诊断](../experiments/full_dataset_analysis.md)

## 1. 已覆盖的对手与结果

本地已完成LiSA、GS³、SSD-GS、RNG各18场景，全部72项通过结果审计。总体PSNR依次为31.3760、31.4885、31.5819、25.8328；LiSA的SSIM/LPIPS为0.9431/0.0646，优于这三项复现。统一到同一组GT再计PSNR后排名不变。

当前成绩不能支持全面SOTA有两个独立原因：已有对手在重要数据集和总体PSNR上领先；对照集合与验证设计尚不足以覆盖最新相关工作及稳定性。增加论文引用不能弥补本地缺少可比实验，增加训练步数也不能自动改变当前结论。

## 2. 第一手公开来源核查

| 工作及时间 | 任务/技术和本项目关系 | 当前比较状态 |
| --- | --- | --- |
| [GS³，SIGGRAPH Asia 2024](https://gsrelight.github.io/) | 多视图点光源输入，空间/角度高斯、三次splat；直接相关 | 已有18场景本地100k结果 |
| [RNG，CVPR 2025](https://whois-jiahui.fun/project_pages/RNG/) | 点光源神经高斯，前向→延迟着色、深度修正与阴影提示；直接相关 | 已有18场景本地30k+70k结果；其混合着色/深度修正可作为几何问题的设计参考 |
| [SSD-GS，ICLR 2026](https://fanglue.github.io/papers/SSDGS.pdf) | 散射/阴影/反射分解，包含本项目三类数据；核心baseline | 已有18场景本地结果，正式表使用train-only真实权重 |
| [Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting，CVPR 2026](https://iliyan.com/publications/RayTracingReconstruction/) | 通用随机光追训练算法，含点光源重光照实验 | 仅技术参考；按用户“专门做重光照”的要求移出主候选；[官方代码](https://github.com/XuPaya/Stoch3DGS)的完整OLAT入口亦未确认 |
| [RadiosityGS，SIGGRAPH Asia 2025](https://raymondjiangkw.github.io/radiositygs.github.io/) | 逐场景可微全局光传输；论文使用GS³ Synthetic/RenderCapture，官方输入支持点灯位置/强度 | 未纳入；原论文25/50视图、HDR与相机误差限制需先对齐 |
| [Splat the Net，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad7922fd4650f8aba5d8b067e622ca84-Abstract-Conference.html) | 通用神经图元表示；附录D.2含GS³来源OLAT定性展示 | 仅技术参考；按用户“专门做重光照”的要求移出主候选，不能挪用普通NVS指标 |
| [BiGS，3DV 2025](https://desmondlzy.github.io/publications/bigs/) | 专门从OLAT训练可重光照物体；官方训练逐帧读取灯位/光强，含距离平方衰减 | 用户确认包含2025年3月后补回主候选；需明确几何初始化与格式转换，未做同协议本地对照 |
| [MLI-NeRF，3DV 2025](https://github.com/liulisixin/MLI-NeRF) | 主旨偏多光照内在分解；实际使用NRHints，官方有Pikachu配置与run_real.sh | 数据直接相关的非高斯对手，按“专门重光照”的限制单列；未纳入本地比较 |
| [SSS-GS，NeurIPS 2024](https://sss.jdihlmann.com/) | 半透明物体重光照；SSS类别必须讨论的专门方法 | 不在当前四方法18场景表内，不能用其他目录的异协议结果或论文均值混入 |
| [F-RNG，2026-05预印本](https://arxiv.org/html/2605.25975v1) | 6视图、预训练LRM/内在分解先验、前馈重光照 | 输入/先验/数据集不同，应单列稀疏输入前馈路线，不能直接同表排名 |

### Stoch3DGS的技术参考价值与任务边界

[论文第5.2节与表3](https://arxiv.org/html/2603.23637v2)直接比较RNG和GS³的点光源重光照。其训练先做15k几何初始化，再85k神经外观优化，采用光线追踪的可见度。它使用的场景集合、测试处理和训练流程未与本地18场景逐项对齐，因此此处不将其论文数字直接插入本地表。其主要贡献是通用光追训练算法；用户明确只找专门重光照工作后，不再将其列为本轮新增主候选。

### SSD-GS论文数字为什么不能直接拼接

SSD-GS论文对CatSmall/CupFabric/Pikachu使用1024px，本地统一为512px；[官方训练代码](https://raw.githubusercontent.com/irisfreesiri/SSD-GS/main/train.py)在启用相机/灯位优化时包含测试相机标定优化分支，高斯优化器则受not opt_test保护。不能据此声称该分支直接更新高斯，也不能把这种协议的成绩当成完全未接触测试GT的成绩。

本地正式表明确使用train_only真实权重、原始测试标定。[SSD-GS本地协议](../../../SSD-GS/docs/experiments/setup.md)记录了替换与历史权重区别。故论文值、本地历史标定拟合值与当前严格固定标定值必须分表；不能选取其中最高数字组合排名。

## 3. 当前证据尚不能支持的扩展结论

1. **不能称普遍的未知光照外推SOTA。** 旧六场景的官方test光方向通常很接近train，SSS部分灯位甚至重合。已有真正留出灯组的结果支持LiSA优于PORT directional_port_v1约1.70 dB，但只覆盖四个合成场景，尚未同协议比较GS³/SSD-GS/Stoch3DGS。[研发记录](../experiments/development.md)
2. **不能将真实场景增益全归于光传输。** Cat/Pixiu既有诊断发现标定与坐标规范对固定测试分数有明显影响。LiSA和baseline训练相机/灯位策略也有差异。当前固定标定成绩合法，但不是光传输模块的独立消融结果。
3. **不能由30k步直接推导速度SOTA。** 各方法每步计算量、最终高斯数、GPU与训练日程不同，尚无完整统一的时间/显存/FPS比较。
4. **不能报告统计显著胜出。** 只有seed0；逐帧样本有视角/灯光相关性，不能代替多种子方差。0.206 dB的总体差距与SSIM/LPIPS优势都没有跨训练置信区间。
5. **不能把当前分解当成真实物理分量。** 冻结探针中五个场景出现传输分支承担近全部辐射的情况；具有多分支结构不等于成功解耦。

## 4. 可使用的研究表述

> LiSA在统一固定测试标定的三类数据、18场景本地实验中，以30k步训练取得最优总体SSIM/LPIPS，并在真实捕获与SSS数据上取得最优类别平均指标；总体PSNR尚未超过SSD-GS和GS³，复杂硬表面几何及局部高光仍存在显著短板。

“统一”指本地共同分辨率、显示域、划分及指标定义；GT逐像素预处理还存在已量化的小差异，见分析报告。投稿前应统一该细节并补足多种子、预算与近期方法对照。

## 5. 检索与读取范围

实际检查了作者项目页、官方GitHub与论文正文/表格：GS³、RNG、SSD-GS、Stoch3DGS、BiGS、SSS-GS、F-RNG。检索也覆盖2026年近期点光源/OLAT/relightable Gaussian关键词；不同任务的单图资产生成、室外重光照等未强行纳入同表。

应用Hugging Face论文阅读技能尝试公开论文API；请求遇到未索引404后，按技能回退至arXiv正文和作者PDF。结论采用原论文/作者代码，未采用自动生成的论文解读页作为技术依据。

本轮不安装新方法，不下载模型，不启动补充训练。需要实施的验证单独列在[改进提案](improvement_proposal.md)。

## 2026-10-03：进一步限定的文献核查

用户随后将候选范围限制为2025年3月起正式发表、逐场景优化，并排除以光照泛化为主要目标的研究。[专项报告](per_scene_olat_literature_202503_202610.md)记录数据血缘及排除项；10月4日又按“专门做重光照”要求收紧主候选，最终范围以下节为准。

补充Stoch3DGS的代码边界：公开仓库存在，但当前master README只列MipNeRF360普通NVS复现；本轮未核实到完整OLAT训练入口。上文“代码已可访问”不等同“重光照部分已可直接运行”。RadiosityGS则明确提供GS³点灯格式和逐场景训练入口，但其论文采用稀疏视图协议。

## 2026-10-04：排除已跑方法后的补查

用户明确起点为2025-03-01，包含3月，终点仍为2026-10-03；此前从4月1日起算过窄。新增主候选修正为BiGS与RadiosityGS，SSD-GS/RNG已跑过不计新增。MLI-NeRF有直接NRHints实验和官方入口，但主要贡献偏内在分解，单列相关对手；Practical Inverse Rendering已确认官方点光OLAT实现，但要求初始网格及数据适配，单列条件候选。Splat the Net和Stoch3DGS仍为技术参考。

本次按CVPR、ICCV、ECCV、ICLR、NeurIPS官方目录与GS³/NRHints/SSD-GS前向引用关系交叉检索，并继续核实非高斯方法。IRGS、GHPT、SU-RGS等虽专门研究可重光照逆渲染，但尚未确认当前近场点光OLAT训练支持。具体证据、分类及检索边界见[专项报告](per_scene_olat_literature_202503_202610.md)。本次没有启动任何新实验，原SOTA判断不变。
