> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

# PORT-GS系统架构

更新：2026-09-23。方法选择与通用训练/渲染解耦。
2026-09-30：训练入口拆为编排循环 `train.py` 与 `training/` 包，默认行为不变，见[训练模块](modules/training.md)。

```text
train.py / evaluate.py
        ↓ checkpoint.config.representation
methods注册表 → 所选TransportBase子类
        ↑ 统一forward接口
renderer.py：Gaussian属性、阴影、期望深度 → 像素接收点
        ↓ 线性前景RGB
可选冻结主GS的光照条件残差 → 非负截断
alpha合成 → 统一观察变换 → 损失 / 指标
```

| 模块 | 职责 |
| --- | --- |
| data.py | 元数据、图像数值域、相机/灯光约定及训练内留出划分 |
| gaussians.py | Gaussian参数、初始化与几何optimizer |
| methods/base.py | 公共照明与直接光着色，定义可扩展的光传输接口 |
| methods/directional.py | 方向化端口基线（默认方法） |
| methods/attention.py | 2DGS源表面到像素接收点的单次cross-attention |
| methods/neural_material.py | 冻结共享BRDF decoder替代局部材质MLP，沿用2DGS方向端口 |
| materials/ / pretrain_material.py | 程序化多层材质、6维编码器和通用decoder预训练 |
| methods/__init__.py | 方法名→实现类，方法参数/CLI/构造的唯一注册入口 |
| renderer.py | 属性和深度光栅化、像素接收点重建、近似阴影、alpha合成 |
| surface.py / prepare_surface_priors.py | 表面监督损失与离线StableNormal/DA3预测 |
| refinement.py | 增密、分裂与opacity重置 |
| train.py | 固定顺序的训练循环编排：抽帧、校正、渲染、损失、optimizer、增密、日志、保存与验证 |
| training/ | 命令行与组合检查、源 checkpoint 契约、初始化、学习率调度、相机/灯位/光强拟合、SDF/体积/法线场与辐射残差分支 |
| evaluate.py | 根据checkpoint构造方法，统一观察变换与评估 |
| make_validation_manifest.py | 多场景命令、方法标识、源码快照、源码与 third_party 来源记录和输出隔离 |

[方法接口与扩展](modules/methods.md)和[两个新方法](modules/research_methods.md)定义当前接口。
公共forward返回线性RGB，方法内部不做背景和gamma；renderer通过geometry选择3DGS或2DGS光栅化。
所有方法使用像素接收点和完整Gaussian源集合；attention在方法内部将源聚合到非空网格。
2DGS使用切平面分裂和圆盘阴影自遮挡偏置，表面监督仅在训练时使用。
neural_material的decoder只初始化一次，随后保持冻结，权重随场景checkpoint保存；
法线、6维材质latent和漫反射颜色仍接受图像梯度。[神经材质接口](modules/neural_material.md)。

像素接收点来自期望深度，可能处于多层贡献者之间；材质属性来自alpha归一化插值。
这仍是表示近似。方向化方法及新方法不声明完整渲染器具有能量守恒/互易性。
历史空间交换在离散源测度上的性质有更小的适用范围。

## SDF几何辅助分支

train.py可启用[sdf.py](modules/sdf.md)：多视图2DGS深度/法线拟合连续场，预热后场反向约束几何。
场存入checkpoint；默认辅助模式推理使用2DGS/transport，--sdf-shading模式额外恢复field并查询梯度法线。

可选[连续法线细节场](modules/normal_field.md)在像素接收点查询位置残差，补充Gaussian法线特征；
与SDF分开存储和优化，训练/评价共用显式normal_field接口。


## 材质与尺度的可选诊断分支

`materials/ggx.py`实现无预训练、无decoder权重的两叶片GGX，复用neural_material的6维码/法线接口。
material_model由方法配置与checkpoint明确保存，默认神经材质不变；显式重置才允许改变码含义。

训练器可用--optimize-light-scale创建一个临时log参数及独立Adam；每步恢复的正值写入既有
transport.light_scale buffer。无需改变模型权重键或评价入口，重载模型使用保存尺度。
相对增益记录于history；优化器仍按已有续训契约重建。详见[材质与尺度模块](modules/neural_material.md)。

## 固定主GS的可选光照残差

`materials/radiance_residual.py`通过原renderer接收点，查询带实际灯光/视角的有符号修正，
在原前景之后、alpha之前加入。末层零初始化保持原图；原GS/传输/相机/光强全部冻结。
只对抽样观察RGB L1训练新头及其空间网格；evaluate按独立保存配置完整查询残差。
支持几何或原材质法线作为冻结输入，既有7方法与默认保持。
[接口、冻结边界及验证](modules/radiance_residual.md)。


## 2026-09-24: PORT-DNA-2DGS

Implemented 8DNA-inspired photo-supervised distribution material on 2DGS.
No SDF or pretrained geometry supervision. See [module](../architecture/modules/distribution_material.md) and the distribution-material experiment report.


## 2026-09-26 Geometry-first stage

The user now prioritizes author-native 2DGS reconstruction before relighting.
A separate native_reconstruction module uses SH and author CUDA/model with calibrated
NRHints adaptation; new material training waits for geometric review. See
[protocol](../experiments/native_2dgs_geometry.md).
