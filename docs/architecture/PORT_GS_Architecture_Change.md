# PORT-GS：方向化空间端口架构修改说明

## 1. 总体架构

保留原直接光网络和像素接收点渲染，只替换非局部光照交换。

空间端口是可学习的虚拟节点。所有 Gaussian 向端口贡献光照；光栅化后重建的像素三维接收点读取端口，不改成逐 Gaussian 着色。

```text
源 Gaussian 的位置、材质特征、当前光源
    ├─ 源端入射光 E
    ├─ 入射方向基 b_in
    └─ 空间分配权重 W_source
                ↓ 加权汇总
        端口入射状态 Z [R,B,3]
                ↓ 每个端口内部的方向矩阵 C
        端口输出状态 Y [R,B,3]
                ↓ 接收点按位置加权读取
        接收点方向状态 H [M,B,3]
                ↓ 与出射方向基 b_out 做内积，再乘交换比例
        非局部 RGB [M,3]
                + 原网络计算的直接光 RGB
                ↓
        原 alpha 合成与观察变换 → 图像
```

默认 R=64、B=4。空间分配仍使用原来的稠密 softmax；本版不加入 top-k、端口间图网络或额外 RGB decoder。

## 2. 模块定义

| 模块 | 修改后的定义 |
|---|---|
| Gaussian 材质特征 | 保留 32 维，不增加每 Gaussian 的独立方向参数 |
| 直接光网络 `local` | 保留 `165→128→128→128→128→3`，只负责直接光 |
| 交换比例 `exchange` | 保留 `Linear(32,3)+sigmoid`，源端和接收端均为 RGB 三通道比例 |
| 空间端口 | 64 个可学习中心和宽度，沿用原 `partition()` 定义 |
| 方向参数网络 `direction_net` | 新增共享 MLP：`32→32→32→12`，隐藏层使用 SiLU |
| 端口方向矩阵 `C` | 新增 `[64,3,4,4]` 参数，使用 `softplus(raw_C)` 保持非负 |

方向网络输出三个方向轴及三个宽度：每个方向轴 3 维、宽度 1 维，共 12 维。方向轴归一化为单位向量 μ，宽度 κ 使用 softplus 保持正值。方向轴采用不同的非零初始化。

方向基定义为：

\[
b(f,\omega)=\left[1,\exp\{\kappa_1(\mu_1^T\omega-1)\},\exp\{\kappa_2(\mu_2^T\omega-1)\},\exp\{\kappa_3(\mu_3^T\omega-1)\}\right].
\]

源端与接收端共享 `direction_net`：

```text
源端：Gaussian 特征 → 方向参数 → 用 Gaussian 指向光源的单位方向求值 → b_in
接收端：插值材质特征 → 方向参数 → 用接收点指向相机的单位方向求值 → b_out
```

方向网络不输入光强，也不直接输出 RGB。端口矩阵变换的是同一端口内部的方向通道，不是端口之间的连接。

## 3. 替换后的计算

令 j 为源 Gaussian，r 为端口，b 为方向通道，c 为 RGB 通道，x 为像素接收点。m_j 沿用 `quadrature_mass()`；E_j 和 E(x) 沿用原光强、逆平方衰减及可见性计算。

### 源端方向汇总

\[
a_{jc}=\operatorname{sigmoid}(\operatorname{exchange}(f_j))_c,
\qquad D_{rc}=\sum_jm_ja_{jc}w_{jr}.
\]

\[
Z_{rbc}=\frac{\sum_jm_ja_{jc}w_{jr}b^{in}_{jb}E_{jc}}{D_{rc}}.
\]

Z 的尺寸为 `[R,B,3]`。分母保留 RGB 三通道，且不包含当前光强。实现沿用原代码的稳定源权重归一化方式，并对数值分母作保护。

### 端口方向变换

\[
Y_{r,:,c}=C_{r,c}Z_{r,:,c}.
\]

C 的尺寸为 `[R,3,B_out,B_in]`，Y 的尺寸为 `[R,B,3]`。每个端口、每个颜色通道各自执行一个 4×4 矩阵乘法。

### 接收点空间插值与方向读取

\[
H_{xbc}=\sum_rw_r(x)Y_{rbc},
\qquad L^{nonlocal}_{xc}=a_{xc}\sum_bb^{out}_{xb}H_{xbc}.
\]

接收端不再用原始灯光方向调制非局部输出。非局部项也不再乘接收点的直接光可见性、光强、距离衰减或原 `local` 响应。

### 与直接光合并

\[
R_{direct}=\operatorname{softplus}(base+local(inputs)).
\]

端口启用时：

\[
L=(1-a)\odot E\odot R_{direct}+L_{nonlocal}.
\]

端口关闭时：

\[
L=E\odot R_{direct}.
\]

`⊙` 表示逐 RGB 通道相乘。关闭端口时不保留 `(1-a)` 衰减。合并后的线性前景 RGB 交回原 renderer，保留原 alpha、背景和观察变换。

汇总和读取使用矩阵乘法或 einsum。允许 `[N,R]`、`[M,R]`、`[N,B,3]` 和 `[M,B,3]`，不要构造 `[N,R,B,3]` 或 `[M,R,B,3]`。

## 4. 代码修改范围

| 文件 | 修改内容 |
|---|---|
| `transport.py` | 保留原 `Transport`；原有方向编码、离散权重等工具可复用 |
| `directional_transport.py`，新增 | 实现 `DirectionalTransport`、共享方向网络、方向汇总、矩阵变换及读取；保持原 `forward` 调用签名 |
| `train.py` | 增加 `representation=directional_port_v1`、`rank=64`、`dir_dim=4`、`dir_width=32`；将全部新参数交给网络 optimizer |
| `evaluate.py` | 按 checkpoint 保存的 representation 和尺寸构造模型，保留原模型加载路径 |
| 新模型配置 | 保存方向网络、方向基及矩阵参数化设置，避免加载时依赖未记录的默认值 |
| `renderer.py` | 保持现有 Gaussian 属性光栅化、像素接收点重建及图像合成流程 |

不修改阴影算法、Gaussian 几何表示、细化机制、原图像损失或数据读取。
