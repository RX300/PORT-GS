# 通用表面几何重建：筛选与验证

日期：2026-09-27。用户明确要求面向不同物体的通用方案，不采用毛发、玉石等类别专用模型。
继续不使用 SDF 或预训练模型做几何监督；先验收几何，再进入 relighting。

## 判断

优先研究“让高斯具有明确、跨视角一致的表面定义”，而不是继续扩大原有神经材质容量。
当前 shortlist 为 Geometry-Grounded GS（GGGS）、Gaussian Wrapping、CoMVS-GS。
这些工作都不能证明任意材质、任意移动光源条件下的几何可以无歧义恢复。
“通用”指没有类别模板/材质专用结构，不意味着对输入条件没有假设。

## 一手文献与选择

| 方法 | 时间/出处 | 改善几何的机制 | 当前适配判断 |
|---|---|---|---|
| [Geometry-Grounded Gaussian Splatting](https://arxiv.org/abs/2601.17835) | 2026-01；[作者代码](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting) | 将高斯解释为随机实体，连续透射率求 0.5 等值深度，并对所有射线贡献者回传梯度，减少离散深度台阶 | 首个核心模块实验。不需要独立学习的 SDF 或预训练几何教师；完整论文还使用多视图约束 |
| [From Blobs to Spokes / Gaussian Wrapping](https://arxiv.org/abs/2604.07337) | 2026-04；ECCV 2026；[作者代码](https://github.com/diego1401/GaussianWrapping) | 有向高斯、法线/占据场、法线误差驱动的补充增密，形成更完整的表面壳层；再提取网格 | 后续候选。改动比 GGGS 大；并非二维 surfel 原样替换。官方实现的可选 `depth_order` 和 `milo` 分支须关闭 |
| [CoMVS-GS](https://arxiv.org/abs/2608.18413) | 2026-08 预印本 | 传统 MVS 稠密初始化，PatchMatch 与 GS 迭代互相提供深度，最后 Delaunay graph-cut 成面 | 有价值的独立几何证据路线；现有移动光源下先验证对应关系可靠性，不直接把全部匹配深度当教师 |
| [Multi-view Normal and Distance Guidance GS](https://arxiv.org/abs/2508.07701) | 2025-08 预印本 | PGSR 上增加跨视角距离重投影和法线关联 | 可借鉴约束，单纯预测之间的一致性并不构成真实几何证据 |
| [MILo](https://anttwo.github.io/milo/) | SIGGRAPH Asia 2025 / TOG | 训练期间可微地提取网格，GS/网格双向约束 | 使用由高斯计算、优化的有符号距离值，与本轮无 SDF 几何监督方向不合，暂不作为主线 |
| [2D-SuGaR](https://github.com/Divyam10/2D-SuGaR)、[GSRecon](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_GSRecon_Efficient_Generalizable_Gaussian_Splatting_for_Surface_Reconstruction_from_Sparse_ICCV_2025_paper.html) | Eurographics 2026；ICCV 2025 | 分别使用单目深度/法线先验和训练得到的跨视图几何网络 | 有研究价值，但不能在当前“无预训练几何监督”条件下原样套用 |

不能把论文宣传中的“最好”理解为所有指标和场景都最好。例如 Wrapping 的论文 Table 1
分别报告前景裁剪与全场景两组，以及均匀采样与虚拟扫描两种评估；GGGS/PGSR/Wrapping
的名次随协议变化。附录 DTU 上 GGGS 30k 平均 CD 为 .48、Wrapping 30k 为 .49，
不存在“更新的一定全面更好”的证据。上述数字仅是该论文表格，不能外推为本项目测量。

GGGS 的限制也应保留：其连续体积建模主要用于深度，颜色和法线仍采用 splatting
近似；复杂外观造成的几何—颜色歧义不会被新深度定义自动消除。

## 为什么此前训练步数足够，几何仍然不佳

1. 原生 2DGS 规定局部表示是平面，但没有强制所有面片组成单一、真实表面。
   少量贴错位置的 surfel 可用 alpha/颜色解释训练图像；法线与自身深度一致仍可能是错误表面。
2. 数据的相机和点光源共同变化。SH 只按观察方向变化，无法独立描述所有光照状态。
   几何参数因此可能吸收阴影、高光和散射变化。此处是机制假设，不是已完成的因果归因。
3. 轮廓种子只能约束外壳，不能充分定位内部凹陷；原始 `points3d.ply` 也不是可靠 SfM。
4. 已修复的增密/剪枝时序问题是真实实现缺陷，但修复后 30k 仍未通过人工几何验收。
   继续只调此类参数，证据不足。

## 已完成：经典匹配可行性测试

入口：`diagnose_image_errors.py --multiview-probe SCENE --output OUTPUT --resolution 256`。
数据只读 official train 中的 fit 子集；每场景固定等间隔四个 anchor。
从 fit 中选择相机中心夹角 3–25° 的四个邻居，优先相似已知光源方向。
使用 384 个相机边界深度假设、9×9 ZNCC、前三个邻居的聚合代价；完整 K 与原始位姿。
置信度要求 NCC>.7、不同局部极小值的 margin>.02；三个源视图分别通过
1px 往返重投影与 .01 场景半径深度一致性。没有读取旧模型深度或测试图。

| 场景 | anchor | 通过像素数 | 相对前景覆盖率 |
|---|---|---:|---:|
| Cat | 0 / 168 / 337 / 521 | 1979 | 1.649% |
| Pixiu | 0 / 184 / 369 / 561 | 242 | 0.433% |

[Cat 图](../../runs/gggs_core_geometry/diagnostics/matching/Cat/matching_panel.png) ·
[Pixiu 图](../../runs/gggs_core_geometry/diagnostics/matching/Pixiu/matching_panel.png)。
过滤后分布稀疏，不能作为高质量稠密表面监督。
这是简化的 frontoparallel plane sweep，不是完整 PatchMatch 或 CoMVS-GS 复现，
负结果不能证明所有 MVS 都不可用；遮挡、局部平面假设、标定误差也会造成拒绝。
一致性本身也不是精确度，Cat/Pixiu 目前没有可用于本轮评价的参考网格。

初版置信度错误地把同一极小值附近的相邻深度样本视作竞争匹配，合成平面检查失败；
该次真实 probe 在检查结束前启动，已判无效、删除输出并保留 rejected 记录。
修正后合成平面曝光变化、空白区域拒绝、旋转相机/离心 K 往返测试通过，再重跑得到本表。

## 已验证的首个候选：GGGS core

作者提交 `23b418c2f0f19b30b251acad6fe13445dae968c7`。保留作者 Gaussian 模型与
连续深度 CUDA 算法；使用原有 fit-only 轮廓种子，每场景 fresh 40k 点、512px、30k 步。
几何表示改为 3D Gaussian，而非把它称为原生 2DGS。通过后才能讨论向 2D surfel/网格转移。

这是相同数据/预算下的候选后端实验，**不是严格只改一个变量**，也不是完整论文复现：
增密采用作者 GOF 策略及默认 `percent_dense=0`；优化器采用作者学习率。
外观仍用 SH，没有 exposure 网络、多视图 NCC、SDF 或几何教师；保留作者默认 Mip filter；增加同样的
轮廓约束。深度法线约束仅在可靠前景内计算。原生 2DGS 作为历史对照保留。

相机适配必须覆盖 CUDA 深度射线：作者 Python 和 CUDA 都假设主点居中，只改投影矩阵不够。
本适配在更大的居中虚拟传感器上渲染，再依照原始 K 对输出做一次双线性采样。
已验证射线一致性、已知平面深度、非零有限梯度及坐标缩放一致性。

### 后续顺序与门槛

1. 已完成 GGGS core Cat/Pixiu 各30000步，比较了相同固定验证视角的灰模、法线、轮廓。
   同时报完整 fit/validation 指标及自一致性拒绝比例，RGB PSNR 不作为几何通过依据。
2. 若仍有表面孔洞/双层而全局形状更稳定，再验证 Wrapping 的有向法线及补壳增密。
   若形状仍明显随照明偏移，则先处理成像模型与几何梯度解耦，不能把补壳当作形状监督。
3. 后续通用方案采用共享几何 + 光照条件外观解释，但限制外观容量、交替更新及可靠几何像素
   筛选；这仅服务于变化光照下的几何恢复，完整 relighting 质量评价放在几何通过之后。
4. 没有数据支持的地方不补成“真几何”。若可靠对应/遮挡证据仍不足，需固定光照多视图
   或更多独立几何观察，才能进一步区分数据不可辨识与模型表达不足。

严格沿用先前限制，未切换为毛发模板、透明折射专用网络、SDF、单目深度教师或类别生成先验。

### 数值稳定性预检发现

首轮关闭 Mip filter 时，Pixiu 在3600步、Cat在8000步增密的随机采样中因尺度异常退出。
两次都没有最终模型；源码/配置/日志归档到 `docs/experiments/retired/gggs_unfiltered_failure.tar.gz`。
恢复作者默认3D Mip filter，每次增密后及后期每100步按训练相机更新其采样尺度。
检查点另存世界坐标下的滤波尺度，重新加载渲染一致性检查通过；每次 backward 后检查几何/SH梯度，
异常时保存首个失败状态以定位，不用 NaN 替换或继续输出貌似正常的模型。
过滤是否为退出的根因，须看重跑和诊断证据，当前不预先断言。

## 本轮实验结论

[最终报告](../experiments/gggs_core_geometry.md)：两场景完整30000步重跑成功，几何验收失败。
Cat验证IoU .94838→.95086，Pixiu .89938→.91058；灰模仍有明显缺失/片层。
这提供了“只改通用表面定义不足以解决当前数据”的证据，但未隔离照明、初始化、标定各自的因果贡献。
后续不应再把SH-only候选内核轮换当作主要方案，应先使几何阶段的成像模型适配移动光源，
再引入可靠的跨视角约束。GGGS/Wrapping仍是通用几何表示候选，经典MVS只在高置信区域约束。
