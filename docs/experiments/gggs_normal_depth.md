# GGGS core + 法线/深度先验（2026-09-27）

用户明确要求给GGGS加入法线和深度先验并重新训练，覆盖此前禁止预训练深度的限制；仍不使用SDF。
复用当前GGGS core适配及作者连续深度CUDA，不声称完整论文复现。

## 协议

Cat/Pixiu各fresh30000、512px、seed0、40000轮廓占据种子，默认Mip、15000步结束增密。
与`runs/gggs_core_geometry`相同fit/validation划分470/52、506/56，无official test。
保留原图像损失(.8 L1+.2 DSSIM)、alpha L1(.2)、7000步起自深度法线一致性(.05)。
不开NCC或曝光网络，不从旧模型继续训练。规范入口configs/validation.json / launch_validation.sh，GPU0/1。

唯一新增监督为StableNormal法线(.05)和DA3-Large单目相对camera-Z深度(.1)，
1000步开始、2000个训练步线性升至全权重（2999步），让先验在增密早期影响形状。
法线约束实际高斯几何法线和深度导数法线各一半，不添加可绕过几何的独立法线特征。
深度残差为log(rendered Z)-log(teacher Z)，前景内去均值后SmoothL1(beta=.1)，
消除未知单目尺度，保留相对深度形状；不强制教师绝对位置，不拟合任意深度平移。
监督区域为GT alpha>.9且预测alpha>.05、正深度；法线导数额外要求整个3x3邻域有效。
保留原GGGS自一致性掩码(.9)及其它设置。

## 先验与环境

StableNormal直接复用上一轮精确fit-only张量，硬链接到本轮priors；不额外占用重复张量空间。
DA3只独立预测fit图，depth-views=1，不含验证上下文；沿用项目既有prepare_surface_priors.py入口。
官方源码 https://github.com/ByteDance-Seed/Depth-Anything-3 ，固定
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4；DA3-LARGE权重revision
c54c26b16ec04d218e8d584ecf4bce082a9fcc20，Hugging Face CLI下载config.json/model.safetensors。
历史相机条件DA3的绝对跨视图位置不可靠，因此本轮采用相对深度，不将其当作几何GT。

训练复用ssd-gs/PyTorch2.4.1/CUDA12.1。DA3单独使用
`third_party/da3_env`，由ssd-gs Python `-m venv --system-site-packages`创建，
依赖`pip install --no-deps`安装，作者代码`pip install --no-deps -e third_party/Depth-Anything-3`。
未升级共享Torch或安装xformers。实际隔离依赖见gggs_prior_environment.txt，安装日志logs/gggs_da3_env*.log。
StableNormal沿用上一轮隔离环境及权重。

## 检查和验收

合成CUDA检查：法线/深度先验分别产生有限且非零的位置梯度；相对深度尺度不变、
形状变化有梯度且背景不产生监督；相机射线、连续深度和保存加载保持正确。
命令`python -m unittest test_methods.GroundedGeometryTests -v`，3项通过。
真实两张Pixiu fit图已完成DA3预测；现有GGGS终端模型用于实际教师梯度预检（不是新训练初始化）。
预检记录gggs_prior_preflight.json、图../figures/gggs_prior_preflight.png。

完整训练后比较原GGGS与新模型的全量validation/fit，固定Cat36/236/406/517、Pixiu32/207/381/548。
关注轮廓IoU、2px边界F1、向外轮廓p95距离、边界深度法线粗糙度，并查看真实深度灰模。
RGB PSNR/SSIM/LPIPS只作附加指标。单seed、无几何GT，不能把教师符合度或平滑程度当几何精度。
真实教师梯度预检与全部训练/评价已完成。完整结果与评价坐标修正如下。


## 完整结果：局部尖刺减轻，没有可靠的整体提质

Cat/Pixiu均完成fresh30000及全部fit470/506、validation52/56。最终128596/57741高斯，
相对无先验GGGS的153718/65760点更少。没有中间检查点、官方test评价或新relighting训练。
以下基线和候选均使用修正后的同一GGGS评价路径，完整数值见
[gggs_normal_depth_results.json](gggs_normal_depth_results.json)。

| 场景 | 设置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | 边界F1 ↑ |
|---|---|---:|---:|---:|---:|---:|
| Cat | GGGS core | 14.58354 | 0.71924 | 0.29148 | 0.95086 | 0.34308 |
| Cat | + 法线/深度先验 | 14.58672 | 0.71974 | 0.29407 | 0.95017 | 0.33594 |
| Pixiu | GGGS core | 18.33461 | 0.84623 | 0.16110 | 0.91058 | 0.38011 |
| Pixiu | + 法线/深度先验 | 18.33610 | 0.84669 | 0.16285 | 0.90100 | 0.33346 |

| 场景 | 设置 | 轮廓向外偏移p95 px ↓ | 边界深度法线粗糙度p95 ° ↓ |
|---|---|---:|---:|
| Cat | GGGS core | 8.36 | 14.63 |
| Cat | + 法线/深度先验 | 8.57 | 13.06 |
| Pixiu | GGGS core | 16.48 | 29.09 |
| Pixiu | + 法线/深度先验 | 13.91 | 13.76 |

PSNR仅增加Cat .00318dB、Pixiu .00149dB，SSIM轻微提高，LPIPS反而变差；
单seed且存在CUDA非确定性，不能当作可靠提点。
Pixiu的底座锯齿、大片阶梯起伏明显减轻，边界外扩p95下降约15.6%，粗糙度下降约52.7%。
但轮廓IoU下降.00959、边界F1下降.04665，形状仍不准确；Cat局部形状未恢复且出现新的折痕。
两场景几何门槛均未通过，不进入relighting。局部平滑改善与整体准确性分别报告。

[Cat 固定深度灰模](../../runs/gggs_normal_depth_geometry/review/Cat/depth_clay_closeup.png) ·
[Pixiu 固定深度灰模](../../runs/gggs_normal_depth_geometry/review/Pixiu/depth_clay_closeup.png)。
每行观测RGB / 原GGGS / 加先验；数字是内部验证帧索引，不是训练步数。
同目录保留完整视野 depth_geometry_comparison.png 与高斯几何法线图，预测轮廓未用GT裁掉。
这些是渲染深度导出的灰模，未提取水密网格，没有真实几何GT。

## 终端评价坐标修正

终端核验发现GGGS连续深度求解器对全局坐标尺度并非完全数值不变。
Pixiu验证frame32中，直接把高斯搬到世界坐标后，两个前景像素深度约.366/.367，
而相同模型在训练归一化坐标求解再乘尺度为14.56010/14.47575世界单位；RGB MAE只有1.4e-7。
99%有效像素的深度差/场景半径小于8.2e-6，但极少数错误像素使整体mean检查超过阈值。
没有放宽阈值宣称原路径通过，也没有截断深度或删除这些像素。

修正`evaluate.py --gggs`：从capture恢复训练坐标模型、保持相机原始K和姿态关系，
在训练尺度求解连续深度，最后仅把深度乘回世界单位；法线方向和alpha不变。
canonical世界坐标高斯参数仍原样保存；参数映射、全部有限性和人口增删计数核验通过。
此修正不改训练权重或训练目标，也不改Wrapping/原生2DGS评价路径。

两场景旧GGGS和候选均重新完整评价fit/validation，共8次评价，使用空闲GPU0/1。
精确命令及完成状态在run的reevaluation.json，修正源码在evaluation_source.tar，
覆盖原训练source.tar内同名评价文件。最终metrics/对照图均为修正后结果。
原世界坐标评价的指标/图在controls/world_coordinate_evaluation，重复tensor/点云已清理；
逐像素差异证据evaluation_depth_fix.json，两个示例像素已和独立直接capture渲染核对。
相机、先验梯度、深度损失与训练坐标保存加载3项测试在修正后再次通过。

## 先验符合度与限制

四个固定fit视角、各自有效前景内的未加权教师损失均值如下；这是训练诊断，不是几何GT，
也不代表独立验证泛化。两组有效掩码随预测alpha略有变化。

| 场景 | 法线损失 旧→新 | 相对深度损失 旧→新 |
|---|---:|---:|
| Cat | 0.19322 → 0.13375 | 0.01419 → 0.01392 |
| Pixiu | 0.31222 → 0.11526 | 0.02418 → 0.02025 |

法线监督确实改变几何；深度符合度的改变量更小。这是联合添加两种先验的实验，
没有分别消融，不能把局部改善单独归因于法线或深度，也不能证明所有权重组合均无效。
独立逐帧教师并不保证跨视角一致；当前效果更像压制起伏，尚未解决表面位置、厚度与细节。
后续应先区分法线与深度各自贡献、检查跨视角先验冲突，再决定权重或约束形式，尚未启动额外训练。

对照审计：第1步loss一致、1000步前采样序列一致；900步Cat点数46083/46112，
Pixiu28899/28780，记录于control_audit.json。全部先验恰好覆盖fit，validation重叠0，
DA3每个上下文只有目标fit自身。模型、输入、坐标修正及保留清单均在当前run中。
