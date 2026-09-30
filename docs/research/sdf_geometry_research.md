# SDF与高光：相关研究及当前选择

检索日期2026-09-23；依据作者论文、项目页及官方代码说明。
以下方法的数据协议不同，论文数字不与本项目Cat/Pixiu直接比较。

| 研究 | 可借鉴机制 | 本项目适用边界 |
|---|---|---|
| [GSDF，NeurIPS 2024](https://city-super.github.io/GSDF/) | GS与SDF双分支、几何一致性与采样/密度引导 | 本轮先验证连续场约束，不重建NeuS渲染器 |
| [GS-ROR²，TOG 2025](https://arxiv.org/html/2406.18544v3) | 深度/法线双向stop-gradient，SDF预热，SDF剔除漂浮Gaussian | 最接近当前目标；本轮用窄带距离拟合替代其TensoSDF深度积分，没有声称复现全方法 |
| [DiscretizedSDF，ICCV 2025](https://arxiv.org/html/2507.15629v1) | 每个Gaussian存SDF值、SDF转opacity、投影到零面并约束深度一致性 | 改写opacity与点属性，属于另一个结构实验；本轮先保持点数/光栅化相同 |
| [GS-2DGS，CVPR 2025](https://arxiv.org/abs/2506.13110) | 2DGS与单目基础模型几何先验 | 已有StableNormal/DA3属于相近方向，先验不是几何真值 |
| [SpecGloss-GS，WACV 2026](https://gkouros.github.io/projects/SpecGloss-GS/) | 显式F0/roughness，deferred shading，法线与去高光漫反射先验，保留HDR亮峰 | 使用环境光协议；本项目已是逐像素着色、已知近场点光，不能直接照搬环境图 |
| [PS-GS，2025预印本](https://arxiv.org/html/2507.18231v1) | 同视角多光照法线监督与2DGS可见性 | 本地Cat522/Pixiu562个训练相机矩阵（6位小数）均各不相同，不具备直接UPS输入 |

## 由证据形成的判断

1. 连续SDF能提供跨视图统一表面，但不能凭空提供真实几何；如果只拟合错误GS，其零面也可能错误。
   因此使用窄带有符号约束、Eikonal、视线前方空域，并保留现有图像/几何先验。
2. 目前模型逐像素计算BRDF，直接specular没有被非局部gate衰减，材质decoder也已改善窄峰拟合。
   仍有GT小高光缺失，几何是合理待验证假设，而非已证实的唯一原因。
3. 先用相同38k初始化和额外3k预算比较SDF与control；不能只拿继续训练的SDF模型与训练更短模型比较。
4. 同时报告高光区域、整体图像和几何一致性；法线/深度夹角减小不代表真值误差减小。
5. 若几何一致性改善但高光仍缺失，下一候选是独立显式GGX/F0/roughness头，检验冻结低维材质码的
   表达/优化瓶颈。保持直接点光与PORT间接传输，单独改变局部材质；先train内固定子集筛选，
   再固定预算进行完整test，避免在test上搜参。此项尚未实现或验证。
6. 若SDF反向几何约束导致退化，先检查场的法线误差、零面位置及预热收敛，再决定是否需要
   更长场预热、可信区域选择或光线上的几何积分，不堆叠未验证的损失。

实现与固定协议见[模块](../architecture/modules/sdf.md)和[实验](../experiments/sdf_geometry.md)。

## 续训学习率审计

当前--init-checkpoint只加载权重，optimizer及几何位置学习率重建。
Cat position_scale=6.054558，均值初始lr=9.6873e-4，原30k末端=9.6873e-6；
Pixiu position_scale=14.791554，初始=2.36665e-3，原末端=2.36665e-5。
本轮3k追加会从约100倍原末端位置lr开始。这是明确的配置行为，不是SDF梯度泄漏。
控制组已出现test PSNR退化；学习率重启可能是原因之一，但尚无独立消融证明。
完成当前SDF对照后，应优先检验小步几何续训，再决定是否更换材质或增加场容量。

## 2026年9月最新补充

- [RGS，arXiv 2026-09-16](https://arxiv.org/html/2609.19421v1)：利用VGGT多视图深度先验、
  反射区域法线平滑及针对反射区域的增密，重点评价反光物体的新视角合成。
  值得借鉴的是跨视图几何先验；其环境反射/NVS协议不能直接证明近场换光效果。
- [GS-PI，arXiv 2026-09-17](https://arxiv.org/html/2609.19907v1)：3D条件扩散预测材质再蒸馏，
  与逐场景联合优化不同。主表重光照使用统一GT法线隔离材质质量，不能作为端到端几何更好的证据。
  本项目没有这种3D材质扩散训练数据，当前不把它作为直接替换方案。
- 本地prepare_surface_priors.py的DA3调用是`model.inference([image], ...)`，即逐张推理，
  虽然模型具备多视图能力，现有深度先验并未利用它。若小步续训仍无法修复几何，
  后续可在已知训练相机下采用重叠多视图DA3输入，建立共享尺度的几何先验，再与SDF耦合。
  这将引入来自图像的新几何证据，而非单靠SDF平滑既有错误表面；需要先检查相机约定、
  多视图尺度对齐及强换光条件下的可信区域。目前只完成接口调查，没有生成或采用新先验。

当前SDF损失约束alpha合成后的深度/法线，未直接约束每个Gaussian中心。
GS-ROR²也指出合成表面一致仍可能留下单独漂浮点。若场已拟合可靠而合成表面仍有杂层，
可单独检验SDF距离引导的点约束/剔除；必须保持可见细节并与相同点预算对照，
不能把删除点数或场残差下降当作质量改善。

## 若辅助约束仍不足：连续法线参与着色的候选（未实现）

当前SDF只拟合2DGS伪表面，再反向施加弱几何损失，RGB没有直接通过场法线更新SDF。
另一可检验方案是用normalize(grad f(x))作为逐像素几何着色法线，使BRDF的RGB梯度直接更新
连续表面，保留GS/SDF双向几何项及Eikonal约束。它使法线受可积连续场约束，而非仅让
独立Gaussian法线近似某个场。需要对SDF梯度求导并明确推理也加载SDF的checkpoint契约。
此方案可能引入几何/材质补偿或高频噪声，必须同预算训练对照，不把训练高光增强当作泛化提升。
这只是研究候选，当前运行仍为训练辅助SDF，未改变渲染公式。

## 下一结构候选：连续细节着色法线（尚未实现）

SDF正则和单点约束已减少若干几何自一致性误差，GT亮点半向量探针则显示仍存在方向调整空间。
下一候选保留宏观GS/SDF几何，从世界接收点查询一个跨视图共享的高频normal-residual MLP，
补充当前通过Gaussian混合得到的法线特征。输入只有归一化位置，不输入相机、灯光或图像。
输出作为features[6:9]的logit残差，在现有0.5*tanh和切向投影之前相加，使角度范围保持相同。
末层零初始化，初始渲染与原模型一致，检验连续查询是否改善属性混合下的细节表达。
建议3×128 ReLU、8频率Fourier位置编码、3维输出，约4万参数；独立Adam、沿用RGB/亮点监督。
保留frozen SDF几何精修与原冻结BRDF，做同初始化/同预算对照，不能同时更换材质头。
这是待检验假设；此前失败不证明Gaussian属性混合是唯一原因。

另外核查了辐射幅度约定：point_light以场景中位irradiance归一化；Cat/Pixiu各自pl_intensity恒定。
SSD-GS原渲染器没有单独使用metadata intensity，不能据此断言PORT的处理错误。
冻结物理BRDF与相机曝光/光强尺度仍有辨识性问题，当前尚未作为已证实根因，也没有修改它。

## 交点深度与官方2DGS代码核对（2026-09-23）

原实现项目：https://surfsplatting.github.io/
官方forward核：https://github.com/hbb1/diff-surfel-rasterization/blob/main/cuda_rasterizer/forward.cu
当前main确实由局部(u,v)与Tw计算交点Z、丢弃近裁剪面后的无效贡献；低通时改用中心Z的行是注释，
并非当前生效逻辑。原distortion使用归一化深度的成对平方距离，也不同于本机gsplat的中心Z代理。
PORT-GS本轮仅隔离接收点深度这一变量：保持本机gsplat RGB/alpha/排序，用交点Z，
但屏幕低通支撑仍用中心Z，避免引入不匹配的近裁剪贡献或在低通支撑外取平面延长线深度。
因此本轮是明确的混合深度对照，不是完整复现原2DGS渲染器。后续判断必须依实际训练效果。
稀疏深度梯度参与几何优化；原CUDA的absgrad增密统计没有新增深度分支的逐像素绝对梯度。
本轮refine-stop=0，不受该增密统计差异影响；若从头增密训练，须单独处理或明确协议边界。

## 新视角残差与真正重光照的边界（2026-09-23）

RGS-DR（3DV2026）的论文v6：https://arxiv.org/html/2504.18468v6
项目：https://gkouros.github.io/projects/RGS-DR/
源码：https://github.com/gkouros/RGS-DR/blob/main/render.py#L114-L116
它在延迟PBR之后加方向编码的残差，补偿原环境下的新视角细节；论文明确在换光照时关闭残差，
官方render.py也设置use_residual=False。不能把这类固定光照NVS高光增益直接当作移动点光源重光照增益。
PORT-GS已经在像素接收点上做延迟着色，不能仅移植“deferred shading”名称解决现有缺失。

## 冻结材质码的标准高光拟合诊断

pretrain_material.py新增--fit-lobes，仅CPU优化6维材质logits，冻结decoder与法线，不使用场景图像。
训练入射角15/45/70度、正负镜面偏角对数采样；独立检查30/60度，用标准介质/导体GGX的4档粗糙度。
1000步Adam、.02到.0001余弦下降，encoder输出初始化。source/config/拟合码/曲线保存在
runs/material_lobe_fit_diagnosis；这是单次局部拟合诊断，不是冻结decoder的全局表达能力上界。

30度未见角结果：介质roughness=.005的log RMSE .2825→.0516，峰值/teacher1.085、半高宽/teacher1.011；
导体.005误差 .9415→.1284、峰值.990、宽度1.138。说明场景自由材质码能纠正部分encoder误差。
最窄.001仍有误差：介质峰值.595、宽度1.632；导体峰值.764、宽度1.737。
60度未见角的最窄宽度比例均约1.882；.02导体半高宽约teacher一半，说明偏差并非单向变宽。
因此，不能用encoder曲线误差直接断言场景decoder完全不能表达高光；也不能认为改良预训练已消除近似误差。

后续候选：在修正几何的统一初始化上，增加直接优化解析GGX材质参数的对照，隔离“冻结神经BRDF近似”因素。
保持真实入射方向、出射方向、点光源距离与强度、表面法线和非局部传输的相同定义，不添加只看相机的RGB残差。
先完成当前30k几何对照，再决定是否执行这个材质对照；没有提前认定它胜出，也不改变当前运行中的模型。

## PNG上限与几何独立约束的核查

CPU统计全部训练帧，采用target_image相同alpha合成、5x5前景内域与原neutral_peak_mask。
Cat/Pixiu高光代理像素8927/212982；其中任一通道>=254.5/255比例7.718%/6.596%，
>=254/255比例29.248%/34.619%。后者只是接近上限，不能全部当作确切传感器饱和。
数据、阈值、逐通道计数在runs/specular_cue_coherence/sensor_saturation_audit.json。
中心/交点2k模型全fit raw_MSE与由逐帧PSNR还原的截断量化MSE，差额占raw分别0.6875%/0.7149%。
量化也贡献微小差额；不能将这整个差额全部归为饱和。现证据不足以认定这是高光缺失主因，不改当前loss。
此外Cat的颜色去饱和亮点代理覆盖很少，不能单独据它判定全部高光质量，仍需完整图像比较。

当前辅助SDF主要拟合GS的伪表面；Eikonal与窄带约束提供形状正则，但不直接引入另一套RGB表面观测。
因此“互相一致”并不保证真实几何正确。计划用从头10k检查点做CPU多视图轮廓一致性审计：
64个训练视图、轮廓膨胀2px，检查高opacity Gaussian中心以及SDF零面投影，排除画面外/相机后点。
若两者仍共同违反GT轮廓，下一几何候选是训练掩码导出的SDF自由空间约束，提供独立于GS的几何信息。
这只是待审计的候选；当前30k对照不增加任何新项，也不把视觉外壳当真实凹面几何。


## 2026-09-23：高光提示编码与重光照边界复核

NRHints原文§3.1/3.2将高光视作稀疏高频输运：[原文](https://arxiv.org/html/2308.13404)。
官方commit `0da5b82b851622a4d61aeb35e79fda889efe0441` 的
[提示代码](https://github.com/iamNCJ/NRHints/blob/0da5b82b851622a4d61aeb35e79fda889efe0441/models/neus_hint_model.py#L581-L616)
使用体权重平均法线/hit point、粗糙度.02/.05/.13/.34的4个GGX响应，沿ray复制。
[辐射网络](https://github.com/iamNCJ/NRHints/blob/0da5b82b851622a4d61aeb35e79fda889efe0441/fields/reflectance_network.py#L41-L53)
对这些原始提示做四频率PE；[编码实现](https://github.com/iamNCJ/NRHints/blob/0da5b82b851622a4d61aeb35e79fda889efe0441/fields/encodings.py#L155-L175)
使用[1,2,4,8]，没有π/2π因子。它不是cos幂提示，也没有本项目的log变换。
当前SDF辅助分支则使用逐样本法线计算三个log(1+b(1-n·h))/log(1+b)，未对提示本身做PE；
已对入射/出射方向做四频编码。两者差异是明确的建模选择，不能把当前分支称为NRHints复现。
提示增加PE可作小型表达对照，但论文并不保证收益：其去highlight-hint消融整体PSNR32.02→31.96，
局限部分明确提示依赖法线准确性。即使PE改善，也仅支持当前参数化/优化更易拟合，非旧MLP表达上限证明。

[GOGS，2025 §4.2](https://arxiv.org/html/2508.14563v1#S4.SS2)的球面Mip高光补偿在新灯光重光照时关闭。
与已记录RGS-DR一样，不能将其固定光照NVS残差作为移动光源重光照改善依据。
