# GS³角度高斯：带宽、朝向与可复用范围

2026-09-23，只读核查；不改变当前add/multiply训练、参数、门槛或测试安排。
[项目PDF](https://gsrelight.github.io/pdfs/GS3.pdf)已取得，但浏览工具因25.8MB大小拒绝解析；
下列论文内容依据同作者[arXiv:2410.11419v1全文](https://arxiv.org/html/2410.11419v1)核实，未比较两个排版版本。
官方仓库可读，固定提交`83c8daa41284cc310855dec6db3158106c065d11`（2024-11-19）。

## 表示及论文/代码差别

[§4.1式(6)(7)](https://arxiv.org/html/2410.11419v1#S4.SS1)在局部half vector上混合8个共享角度高斯，
每个空间高斯拥有混合权重和独立于几何椭球的可学习着色坐标系。
记`h=normalize(wi+wo)`，θ为h与lobe中心z的夹角，s为h在lobe切平面的单位投影：
`G(h) ∝ exp[-0.5*(θ*sqrt((s·x/λ)²+(s·y/μ)²)/σ)²] / σ`。
因此沿x/y主轴的角度标准差是`σλ / σμ`；减小带宽直接提供窄峰，区别于由PE+MLP隐式合成峰形。
它不是BiGS的全双向SH矩阵，也不是当前空间特征×混合方向特征的双线性交互。

[论文§4.4](https://arxiv.org/html/2410.11419v1#S4.SS4)写随机σz∈[.13,.69]、σx=.5、σy=1、α=.5，坐标系与世界轴对齐；
实际[mixture_ASG.py:18–43](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/scene/mixture_ASG.py#L18)为：

- `ratio=softplus(linspace(-2,0,8))`；raw σ=0，所以初始σ=`.5*ratio`，确定性范围`.063464–.346574`。
- λ/μ=`10*sigmoid(raw scales)`，raw值`(-1.0986,-2.1972)`，实际约`(2.500023,1.000022)`。
- lobe quaternion全为`(1,0,0,0)`；八个lobe初始同朝向、不同宽度，并非八个分散中心。
- 据上述公式计算初始主轴标准差：x约`9.09°–49.64°`、y约`3.64°–19.86°`，是half-vector角度，不能换称像素尺寸。

[forward:55–82](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/scene/mixture_ASG.py#L55)
还乘Schlick Fresnel（F0=.04），并除额外常数`√2*π^(2/3)`；[空间权重raw初始化0](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/scene/gaussian_model.py#L249)
经过[Softplus](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/scene/gaussian_model.py#L197)后为`.693147`。不可把论文初值直接当成运行代码。

## 实际可训练参数与几何边界

[优化器](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/scene/gaussian_model.py#L286)
包含混合权重、σ、λ/μ、共享lobe旋转及每高斯local_q；
[训练:91–98](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/train.py#L91)
只把λ/μ与lobe旋转冻结至`asg_freeze_step`（默认22k，场景脚本可覆盖），随后解冻；σ不在该冻结块。
`asg_lr_freeze_step=40k`只是[LR日程偏移](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/scene/gaussian_model.py#L317)，不是所有ASG参数40k内无梯度；早期Lambert阶段也会使specular支路不参与图像。
[渲染:137–159](https://github.com/gsrelight/gs-relight/blob/83c8daa41284cc310855dec6db3158106c065d11/gaussian_renderer/__init__.py#L137)
使用实际近场wi/wo，先转local_q坐标系，再乘cosine项与距离平方衰减。
着色框架可独立于真实几何法线，且方法同时优化主几何；不能把它的效果归因于“只增加窄角度函数”。
论文[§6](https://arxiv.org/html/2410.11419v1#S6)仍报告极高频各向异性高光闪烁，说明角度基不消除空间采样限制。

## SSD-GS现有实现与一个小型控制假设

只读基线HEAD=`26e42517e2c56cb2b943a18116560f292a06cd2c`，所查文件无局部修改。
[SSD-GS/scene/mixture_ASG.py](../../../SSD-GS/scene/mixture_ASG.py)保留上述初值/公式；相比官方版本显式传入σ并下限裁剪宽度。
[SSD-GS/train.py:96](../../../SSD-GS/train.py#L96)保留相同形状/旋转冻结逻辑；
[utils/general_utils.py:106](../../../SSD-GS/utils/general_utils.py#L106)的四元数转矩阵也是纯Torch。
可在PORT内部重写独立的角度函数，复用数学约定和已有Torch算子；不导入或修改baseline模块。
原模块硬编码CUDA设备，整套GS³还依赖专用rasterizer/tiny-cuda-nn；仅角度helper不需要这些依赖或CUDA/PyTorch升级。

**一个未执行的假设**：在相同残差头加入8个固定、单位峰值的isotropic half-vector基，
`K_j=exp(-θ²/(2τ_j²))`、`θ=acos(clamp(n·h))`，中心固定在现有geometry normal；
窄组τ按对数取`2°–32°`，宽控制组τ取`8°–128°`，两组同数量/投影参数、相同随机权重、末RGB仍零。
保留现有PE、log hints、空间网格、真实灯位及`I/(scale*d²)`，固定几何/法线，唯一对照量为基函数带宽。
单位峰值避免`1/σ`把幅度同时放大；不额外移植Fresnel、正值混合约束或可学习朝向。
它检验显式窄峰特征能否改善峰形拟合，而非复现GS³；已有log hints含half-vector信息，结果不能被解释为此前缺少角度输入。
中心受冻结法线误差影响，负结果也不能排除可学习方向的ASG；正结果须同时检查错位/邻环误亮与未训练帧，
不能证明物理BRDF、真实法线或GT几何恢复。此项仅为候选机制，不依据当前中间loss自动选作下一实验。
