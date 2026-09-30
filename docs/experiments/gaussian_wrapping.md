# Gaussian Wrapping：先无先验，再按需 StableNormal 对照

用户2026-09-27要求尝试Gaussian Wrapping，关注Pixiu边界尖刺；效果仍不足时，
允许StableNormal法线先验。该授权明确覆盖先前“不用预训练几何监督”的限制，但未授权使用SDF。

## 协议

Canonical `configs/validation.json` / `launch_validation.sh`，现有ssd-gs和CUDA12.1。
无先验输出 `runs/gaussian_wrapping_geometry`。Cat/Pixiu各fresh30000，512px、seed0、
40000轮廓种子，fit/validation分别470/52、506/56，与旧原生和GGGS相同。
官方test不参与训练/调参。此前GGGS完整validation重渲染以补充相同边界指标，放在本轮references中。

完整实现及与作者的差异见[模块](../architecture/modules/wrapping_reconstruction.md)。
使用有向法线学习、五次补壳增密、作者多视图NCC/几何约束、曝光L1和默认Mip filter。
alpha权重.2、深度法线.05，法线场对齐.03，NCC.6、几何重投影.02。

第一阶段不使用任何法线/深度教师。若Pixiu边界仍有明显尖刺或整体表面仍错误，
第二阶段采用相同fresh初始化/预算/损失，唯一增加StableNormal法线损失（初始权重.05，
与深度法线项同起点7000步），对渲染几何法线和深度法线共同约束。
不能仅凭法线贴图变平滑就判定几何修复；结合边界F1/外扩距离/粗糙度、轮廓和灰模检查。

## 环境

Wrapping的ours、RaDeGS、Mini-Splatting2及warp-patch-ncc扩展均项目内build_ext --inplace，
未替换共享PyTorch/CUDA。simple-knn、trimesh使用已有项目依赖。
StableNormal使用项目内`third_party/stablenormal_env`，venv继承ssd-gs的PyTorch2.4.1，
隔离diffusers0.28.0、transformers4.36.1、HF Hub0.23.0等旧推理依赖。
另一个轻量hf_cli_env用于官方权重下载，避免CLI需要的新Hub版本影响旧diffusers。
只下载fp16权重，不下载重复fp32 `.bin`。

先验权重来源：Stable-X/yoso-normal-v0-3 @2202fcb69960d94b437e06c19c556a12ceeb57c0，
Stable-X/stable-normal-v0-1 @4362d11c636ef42ed117dd76984c7f2d83e6abd9。
使用Stable-X/StableNormal官方代码；只采用法线输出，不启用额外分割/深度网络。

## 预检

合成检查覆盖相机、连续深度和法线、每项MVS梯度、补壳clone和checkpoint。
真实Pixiu800步、128px、2000点，提前开启全部几何项并执行一次补壳；结果2100点，
保存后重新加载两张验证图通过。首次因相机缺少image_name元数据退出，已修正重跑。
[预检记录](wrapping_preflight.json)，日志`logs/wrapping_geometry_checks.log`和`logs/wrapping_preflight*.log`。

## 正式结果：两阶段均完成，几何未通过

无先验与 StableNormal 对照均为 Cat/Pixiu 各 fresh 30000 步；两阶段共四次完整训练。
全部 fit470/506、validation52/56 评价完成，未使用 official test、SDF 或深度教师。
无先验在 GPU0/1，对照在 GPU1/3；均 RTX6000 Ada。Pixiu 对照后段出现外部共享 GPU 负载，
因此不比较运行时间。每个终端模型均完成五次补壳和独立加载评价。
无先验最终 Cat229374/Pixiu138689 点；加先验 Cat222330/Pixiu136980 点。

下表是完整内部 validation 的逐图均值。边界 F1 容差2px；外扩距离是每图预测轮廓向 GT 外部偏离的 p95，
再跨图平均。粗糙度是 GT 边界8px带内深度法线相邻夹角的 p95；降低可能来自过平滑，不能单独验收。

| 场景 | 设置 | IoU ↑ | 边界 F1 ↑ | 外扩 p95 px ↓ | 边界粗糙度 ° ↓ |
|---|---|---:|---:|---:|---:|
| Cat | GGGS core | 0.95086 | 0.34308 | 8.36 | 14.63 |
| Cat | Wrapping | 0.94804 | 0.31183 | 9.14 | 10.55 |
| Cat | Wrapping + StableNormal | 0.94711 | 0.30575 | 9.46 | 10.64 |
| Pixiu | GGGS core | 0.91058 | 0.38011 | 16.48 | 29.09 |
| Pixiu | Wrapping | 0.88150 | 0.31331 | 18.94 | 23.54 |
| Pixiu | Wrapping + StableNormal | 0.87920 | 0.29414 | 17.96 | 19.67 |

| 场景 | 设置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Cat | GGGS core | 14.58354 | 0.71924 | 0.29148 |
| Cat | Wrapping | 14.62367 | 0.71907 | 0.31754 |
| Cat | Wrapping + StableNormal | 14.63208 | 0.71897 | 0.31460 |
| Pixiu | GGGS core | 18.33461 | 0.84623 | 0.16110 |
| Pixiu | Wrapping | 18.49184 | 0.84677 | 0.16597 |
| Pixiu | Wrapping + StableNormal | 18.51582 | 0.84700 | 0.16484 |

RGB 表格评价的是移动光源图像上的 SH 重建，并非 relighting 性能，也不是几何精度。
完整 fit/validation 数值、模型审计和比较来源见 [wrapping_results.json](wrapping_results.json)。

### 固定视角结论

- Cat：无先验已较平滑，加入先验没有恢复面部和身体局部形状；边界指标略退化，几何不通过。
- Pixiu：无先验比 GGGS 减轻大片层状起伏，但轮廓明显退化；加入先验又降低局部粗糙度，
  外扩 p95 从18.94降到17.96px，但底座锯齿、头部尖突和整体形变仍明显。
  边界 F1 从.31331降到.29414、IoU从.88150降到.87920，不能称为解决了尖刺。
- 两场景均不满足进入 relighting 的条件，本轮没有训练新重光照分支。

[Cat 深度灰模放大对照](../../runs/gaussian_wrapping_stablenormal/review/Cat/depth_clay_closeup.png) ·
[Pixiu 深度灰模放大对照](../../runs/gaussian_wrapping_stablenormal/review/Pixiu/depth_clay_closeup.png)。
两图都是观测 RGB / 无先验 / 加先验。行号是原训练元数据中的帧索引，不是训练步数。
Cat固定36/236/406/517，Pixiu固定32/207/381/548，均属于未参与拟合的内部验证集。
放大范围统一覆盖 GT 与双方预测 alpha>.05 的并集，并留8px边距，未裁掉外扩突起。
同目录保留完整视野 depth_geometry_comparison.png 和高斯几何法线 geometry_comparison.png。
这些是连续渲染深度的灰模，不是已提取的水密 mesh；本轮不声称完成作者的纹理网格管线。

### 法线先验确实生效，但不等于形状正确

StableNormal权重.05，从7000步开始，同时监督实际高斯几何法线与深度法线，
不能只通过独立有向法线特征满足监督。两组各四个固定 fit 视角的未加权先验误差：
- Cat：0.17581 → 0.14493。
- Pixiu：0.22063 → 0.12100。

这只是训练视角与教师的符合程度，掩码采用各自训练有效区域，不能充当地面真值或严格的几何误差。
解析球体预检中 negate_xyz 转换平均角误差13.2°（次优符号45.9°），用于确认坐标约定，
不代表真实场景先验精度；见 [坐标预检](wrapping_normal_convention.json)。
全部先验恰好覆盖470/506张fit，有限且分辨率正确，与validation重叠为0。

### 实现与对照核验

真实800步预检、逐项CUDA梯度、实际位置上的先验梯度、补壳clone、保存加载与Mip极小尺度检查通过。
终端全部权重和训练loss有限；5次补壳计数、30000步数、fit/val一致性、世界坐标导出核验通过。
最终模型在归一化训练坐标和世界坐标各渲染同一验证视角：RGB MAE分别1.18e-6/2.76e-7，
深度差/场景半径分别8.13e-7/1.20e-6，因此不是导出尺度变换导致当前几何退化。
证据在对照run的 model_audit.json、prior_audit.json 与各场景 coordinate_render_audit.json。

两组相同seed的第1步loss一致、7000步前采样帧序列相同，但CUDA训练不是逐位确定：
6900步Cat点数107018/107446，Pixiu63555/62893；前期loss最大差.00275/.00160。
这是单seed同协议对照，不能把微小差异视为统计显著；见 control_audit.json。

### 当前判断及后续候选（尚未实验）

本轮证据只能支持“这套适配与当前正则强度没有解决几何”，不能推论原论文对所有物体无效。
Wrapping的自监督补壳会围绕已有错误表面继续细化；局部法线约束也不直接确定表面位置、
厚度和轮廓，而且本轮先验有效区排除了低alpha及最外缘像素，对外部突起缺少直接约束。
另外，NRHints相机和点光源一起改变，SH外观和多视图NCC的对应假设可能继续把光照变化写入几何。
这些是实现和数据所支持的原因假设，尚未通过独立因果消融证实。

下一项更有判别力的实验是保持当前初始化/法线/轮廓，单独比较启用与关闭NCC对几何的影响，
并统计训练轮廓外突起的约束覆盖；先验证错误对应和边界监督是否是主因，再考虑材质联合优化。
不建议仅增加训练步数或凭更平滑的法线图继续接入relighting。
