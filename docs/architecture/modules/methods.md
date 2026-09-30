# 当前光传输方法与接口

2026-09-30：唯一活动注册表 `methods.METHODS` 只包含三个方法，默认 `directional_port_v1`。
所有训练、评价和检查点加载都使用显式 `representation`；不再默认回退到旧 Anchor 方法。

| 注册名 | 实现 | 表示 |
|---|---|---|
| directional_port_v1 | methods/directional.py | 空间端口内的方向池化与读取；默认 3DGS |
| surface_attention | methods/attention.py | 固定空间聚合、单次 cross-attention；2DGS |
| neural_material | methods/neural_material.py | 冻结共享 BRDF 解码器、材质编码与方向端口；默认 2DGS，可导入 3D GGGS |

`methods/base.py` 中的 TransportBase、NeuralTransport、PortTransport 是公共实现，不是额外方法。
`materials/`、`surface.py`、`refinement.py`、`cameras.py` 继续为保留方法服务。

## 公共接口

```python
forward(gaussians, receivers, eye, light_pos, light_intensity, source_visibility, port_active=True)
```

输出线性前景 RGB。渲染器负责几何光栅化、阴影和合成；观察变换负责 gamma/背景。
`--representation` 选择方法；方法类声明 defaults/cli_fields，注册表补全配置并创建模块。
三种方法的既有参数名和权重布局不变。旧已删除方法的权重只能用历史源码复现。

## 初始化和相机

`--init-checkpoint` 恢复同方法权重；`--init-geometry` 只导入几何。
`--init-geometry-format gggs` 当前仅适用于 directional_port_v1 / neural_material，保留三维协方差与滤波透明度。
`gggs_reconstruction.py` 只保留读取已有几何及连续深度诊断；`native_reconstruction.py` 只保留公共相机/深度数学和边界指标。
没有独立的 native_2dgs / gggs_core / gaussian_wrapping 训练或评价 CLI。

三种方法共享可选相机优化。最新 rotation + translation gauge 已在默认与 Neural 上完成 Cat/Pixiu 实验；Attention 尚未验证同条件最终质量。
[相机说明](cameras.md) · [清理记录](../../experiments/method_retirement_20260930.md)。

## 已退役

learned_anchor_exchange、distribution_material、local_transport 及三个几何重建入口已从活动代码删除。
更早删除的 surface_reflectance、directional_surfel、paired_port、local_frame 未恢复。
历史实验及论文分析不是活动方法清单。
