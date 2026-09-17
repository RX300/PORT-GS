# PORT-GS系统架构

更新：2026-09-17。方法选择与通用训练/渲染解耦。

```text
train.py / evaluate.py
        ↓ checkpoint.config.representation
methods注册表 → 所选TransportBase子类
        ↑ 统一forward接口
renderer.py：Gaussian属性、阴影、期望深度 → 像素接收点
        ↓ 线性前景RGB
alpha合成 → 统一观察变换 → 损失 / 指标
```

| 模块 | 职责 |
| --- | --- |
| data.py | 元数据、图像数值域、相机/灯光约定及训练内留出划分 |
| gaussians.py | Gaussian参数、初始化与几何optimizer |
| methods/base.py | 公共照明与直接光着色，定义可扩展的光传输接口 |
| methods/anchor.py | 原空间irradiance交换 |
| methods/directional.py | 方向化端口基线 |
| methods/paired_port.py | A：独立入光/出光空间支持 |
| methods/local_frame.py | B：直接光局部坐标系 |
| methods/__init__.py | 方法名→实现类，方法参数/CLI/构造的唯一注册入口 |
| renderer.py | 属性和深度光栅化、像素接收点重建、近似阴影、alpha合成 |
| refinement.py | 增密、分裂与opacity重置 |
| train.py | 通用训练循环、配置、权重保存与日志 |
| evaluate.py | 根据checkpoint构造方法，统一观察变换与评估 |
| make_validation_manifest.py | 多场景命令、方法标识、源码快照和输出隔离 |

[方法接口与扩展](modules/methods.md)和[两个新方法](modules/research_methods.md)定义当前接口。
公共forward返回线性RGB，方法内部不做背景和gamma；renderer不识别具体子类。
所有方法继续使用像素接收点和完整Gaussian源集合，保留原阴影及几何细化实现。

像素接收点来自期望深度，可能处于多层贡献者之间；材质属性来自alpha归一化插值。
这仍是表示近似。方向化方法及新方法不声明完整渲染器具有能量守恒/互易性。
历史空间交换在离散源测度上的性质有更小的适用范围。
