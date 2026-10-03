> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

## LiSA（`light_atlas`）原理 — 2026-10-01

点光源强度I对任意立体角元送出相同通量 I·dω：受光表面的辐照度面积元 E·dA = I·cosθ/r²·dA = I·dω。
因此从光源对Gaussian做普通alpha合成时，第i个Gaussian得到的权重 T_i·α_i 正是它接收到的通量份额，不需要法线。
投影阴影、次表面/半透明传输（光在光源可见表面进入后到达接收点）和近邻一次反弹都可写成“在光源空间对沉积通量做随接收点变化的核积分”。
经典方法固定核与存储量（PCF/VSM、translucent/reflective shadow maps、和式高斯扩散剖面）；LiSA让每个Gaussian沉积学习的通量特征，
用多尺度金字塔近似世界尺度核，由接收点条件的网络学习核权重，并保持对通量线性（类似光源空间PRT）。
可见度以沉积深度的前两阶矩做阴影检验（t=(z-μ)/s，3σ偏置），再加零初始化的学习残差。
网络只读取光源相对量（深度偏移、厚度、沉积通量），光源移动时图集重新渲染，属于结构上的等变，而不是对灯位输入插值。
局部神经材质的位置容量只进入与光照无关的系数（`svbrdf` 头：8个位置相关系数缩放不含位置的光照相关基响应），使随光源移动的阴影和传输必须经由光源空间算子，而不能被位置相关的局部项“烘焙”。
不声称物理精确的散射参数，也不覆盖光源完全不可见区域经多次反弹才到达的光。[提案](../research/lisa_light_space_atlas_20261001.md)。

## 当前可选择方法架构 — 2026-09-22

当前支持directional_port_v1（默认）、paired_port（方案A）、local_frame（方案B）
及learned_anchor_exchange，另有2DGS方法directional_surfel、surface_attention和neural_material。
2DGS与StableNormal/DA3改变的是几何表示和训练约束，不保证重光照指标更高。
surface_attention将受光源表面聚合到固定非空网格，像素接收点以单次attention读取其RGB。
Q/K只使用位置、表面特征与方向，V携带光强；固定光源位置时保持光强线性。
cosine评分对Q/K归一化，用于限制角度分数的饱和，未增加网络参数。
该表示不保证能量守恒、互易性或真实直接/间接光分离。
neural_material先用程序化多层BRDF预训练统一6维latent decoder，再冻结网络，
通过图像拟合每Gaussian材质code与独立漫反射色。直接光不再使用位置条件化MLP，
非局部方向端口仍是学习近似，不能据此宣称完整物理分解。
[方法来源和适用边界](../architecture/modules/neural_material.md)。
[算子、诊断与实验](../architecture/modules/attention.md)记录其实际适用边界。
公共训练按representation构造方法，渲染按geometry选择3DGS或2DGS。
详细接口见 [方法架构](../architecture/modules/methods.md)。
以下日期更早的内容保留作为历史方案记录，新架构不继承超出各方法适用范围的物理声明。

## Current architecture — 2026-09-15

The active model is directional_port_v1: 512 spatial ports, four direction
channels and a shared material direction MLP. Direct and nonlocal radiance are
combined at pixel receivers. See [directional architecture](../architecture/modules/directional_transport.md).
The earlier architecture below is historical.

> Active representation restored on 2026-09-15: 512 learned spatial anchors,
> source irradiance pooling and the original material-response MLP (Git `9e9596a`).
> HashGrid and residual decoders are archived experiments. Current training
> also retains weighted loss logs/plots and the six-scene JSON launcher.

# Continuous receivers over conservative Gaussian-source exchange

The accepted second-round representation separates the source integral from the location where
material response is evaluated. Every Gaussian remains a source with geometry,
opacity, base appearance and learned features. The camera rasterizes base,
features, direct-light visibility and expected camera-Z depth. Covered pixels
recover alpha-normalized attributes and a world-space receiver point from depth,
pixel center and camera intrinsics.

## Source integral and receiver query

For each RGB channel, Gaussian source `j` supplies visible inverse-square
point-light irradiance `E_j`, quadrature `m_j > 0`, material-conditioned exchange
fraction `a_j`, and softmax spatial partition `f_jr`. Pool over all sources:

```text
z_r = sum_j m_j a_j f_jr
u_r = sum_j m_j a_j f_jr E_j / z_r.
```

At receiver `x`, interpolate material features through rasterization, predict its
exchange fraction `a(x)`, and evaluate the same anchor partition `f_r(x)`:

```text
E'(x) = (1-a(x)) E(x) + a(x) sum_r f_r(x) u_r
L(x) = softplus(base(x) + angular_material_network(x)) * E'(x).
```

The network consumes features, encoded light/view/half-vector directions,
light-view dot product and an eight-band encoding of normalized world position
(51 spatial channels). Foreground radiance is shaded per covered pixel, then
linearly alpha-composited before the shared observation transform. Spatial
encoding is intended to represent detail within projected Gaussian support;
GPU receiver/gradient audits passed, and fixed-last validation supports a
perceptual/detail gain with lower PSNR than round one. The acceptance tradeoff is
recorded in [results](../experiments/results.md).

## Exact operator properties and their scope

When receivers are the original source nodes with the same `m`, `a` and `f`,

```text
T_ij = (1-a_i) delta_ij + a_i a_j m_j sum_r f_ir f_jr / z_r.
```

This positive row-normalized operator preserves constant irradiance and
`sum_i m_i E_i`, and satisfies detailed balance `m_i T_ij = m_j T_ji`.
Arbitrary pixel queries are a continuous extension of the same source integral.
They have their own query fractions and partitions; their rendered image has
no discrete source-mass conservation or reversibility guarantee. Both queries
remain linear in supplied light intensity before the observation transform.

Quadrature is detached opacity-times-projected-area, a representational measure.
The angular material response, approximate visibility and alpha composition add
separate modeling assumptions. Expected depth merges contributors along a ray,
so the reconstructed receiver may lie between surfaces. RGB pooling compresses
incoming direction, and the response uses the original point-light direction.
Anchor connectivity approximates source-to-receiver transport. These limits
remain explicit in the candidate's physical interpretation.

The source operator's analytical review is in
[review_20260912.md](../project/review_20260912.md). First-round results and the
motivation for receiver shading are in [results](../experiments/results.md).
Pixel/deferred shading has established related work; this candidate's complete
transport formulation and empirical value require evidence before any novelty
claim. See [external research](../research/related_work_20260912.md).


物理材质对照可以用`material_model=ggx`直接优化六个物理参数，保持同一渲染和传输管线；
这一受控候选目前未超过神经材质，不替换默认先验。

当前点光由训练照度中位数归一化，归一化数值并不构成相机曝光的绝对标定。
可选的共享light_scale校准只改变一个全场景正值，检验有界反照率是否受亮度尺度限制。
它不能单独识别真实曝光、反照率和间接能量，也不能校正未知的非线性相机响应。


## 2026-09-24: PORT-DNA-2DGS

Implemented 8DNA-inspired photo-supervised distribution material on 2DGS.
No SDF or pretrained geometry supervision. See [module](../architecture/modules/distribution_material.md) and the distribution-material experiment report.
