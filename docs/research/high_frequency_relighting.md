# 窄高光的方向表示与空间—方向交互：有界研究备忘

查阅日：2026-09-23。仅补充两项原始研究；不改变正在运行的单帧3k／四帧12k拟合诊断，
不新增模型实现。已有NRHints、GOGS和RGS-DR边界见[sdf_geometry_research.md](sdf_geometry_research.md)。

## 1. Ref-GS：可以借交互结构，不能继承换光结论

[Ref-GS原文v2，§5.1–5.3、式(5)(6)](https://arxiv.org/html/2412.00905v2#S5.SS1)
使用 `I = I_d + f(S, K ⊗ S)`，其中K为空间特征，S由反射方向与粗糙度查询球面Mip网格。
可借鉴的是先分别表示空间与方向，再显式形成乘积特征，而非仅拼接后交给MLP学习交互。
但该方向网格承载当前场景照明，其网络没有本项目的可变目标灯位输入；文中的近场反射也
不等于移动点光源重光照。本文报告的是反光场景新视角合成，不能据其图像或分数声称
PORT-GS换光高光会改善。当前主GS已经逐像素延迟着色，再移植此名称不是新变量。

## 2. NRF-RTI：确实改变照明，但公开例子不是近场点光

查阅[A Neural Reflectance Field Model for Accurate Relighting in RTI Applications作者仓库](https://github.com/DAISCVprojects/NRF-RTI)，
固定版本`bdd7c025caad76022df730ab6897fb9f08a33802`。
[data_preprocess.py的expand_light／expand_coordinate](https://github.com/DAISCVprojects/NRF-RTI/blob/bdd7c025caad76022df730ab6897fb9f08a33802/data_preprocess.py#L18)
分别产生9维半球谐波光方向编码和20维二维坐标Fourier编码；
[model.py的RTINetwork.forward](https://github.com/DAISCVprojects/NRF-RTI/blob/bdd7c025caad76022df730ab6897fb9f08a33802/model.py#L39)
再拼接图像邻域编码出的10维局部潜码并预测RGB。
这提供“局部空间信息与光方向分开编码”的实现例子，不证明应无限增加角频率。

固定相机的官方合成配置明确使用
[`emitter type="directional"`](https://github.com/DAISCVprojects/NRF-RTI/blob/bdd7c025caad76022df730ab6897fb9f08a33802/Render/sceneFile.xml#L42)，
渲染循环更新的是[DirectionalEmitter.to_world](https://github.com/DAISCVprojects/NRF-RTI/blob/bdd7c025caad76022df730ab6897fb9f08a33802/Render/render.py#L27)。
因此它有真实的照明方向变化，却不能直接验证自由视角、近场距离衰减或点光位置泛化。
不移植其屏幕坐标潜码、图像patch输入或Sigmoid RGB输出；PORT-GS已有世界位置网格。
作者机构提供的论文PDF本次返回403，未核实论文消融及高光数值；上述判断仅依据可读官方代码。

## 3. 待拟合诊断结束后可考虑的一个最小变化

当前94维输入已含位置／wi／wo的四频PE、法线、余弦、log提示及目标灯位置；
空间网格也已存在。因此以下是**本项目推导出的交互假设，不是两篇方法的复现**：

令 `k(x)∈R²⁴` 为现有网格原始特征，`a∈R⁶⁷` 为现有非位置输入，试验在第一层增加
`W[(A k) ⊙ (B a)]`；固定交互宽度16，A/B/W均无bias。
用同参数量的 `W[(A k) + (B a)]` 作加性交互对照；两组均从相同原source、新零输出头开始。
原输入、空间分辨率、主GS／相机／光强、采样和预算全部相同，不同时增加方向频带或新损失。
这样检验的是显式乘性交互是否更易拟合，不是用多加参数的结果证明旧头容量不足。
两支各新增3504个权重；应同时记录分支特征尺度与梯度。参数数相同不意味着函数类、
激活尺度或优化难度相同，结果仍只能支持这种参数化在该协议下的作用。

`a`必须继续包含实际wi、wo与归一化目标灯位置；最后仍乘已知`I/(light_scale*d²)`，
不能退化为只依赖反射视线的固定照明残差。此建议也不保证BRDF互易、能量守恒或跨灯位泛化。
只有当前单／四帧结果提供下一步依据时才选择是否实施；比较小亮核对比、位置召回、邻环
误亮及整体误差，先作训练诊断，不因一张拟合图变尖便宣布重光照改善。


## 预算诊断后的落实约束（2026-09-23）

完整562/30k已完成，训练小桶对比有限提高而test仍近零，SSIM/LPIPS全部71帧更差；
直接clip仅涉及test最小桶2.386%的可覆盖像素，不能解释大多数缺峰。
因此下一轮选择本节add/multiply匹配控制，具体预算/先fit门槛在[预算结果末节](../experiments/gs_residual_budget.md)。
A/B/W常规随机初始化并在两组中逐张量一致，最后RGB层仍零；不要多设一个零W延迟支路学习。
同时记录因子/交互输出及梯度尺度，初始网格±1e−4会让两种运算的激活尺度不同，这是需报告的参数化差异。

需明确：add支路数学上可并入原第一层和网格投影（WB合并到非位置输入权重、WA合并到网格投影），
故它是同参数数目的加性重参数化控制，不是与乘法完全相同的函数类/有效容量。
multiply增加低秩双线性交互；旧非线性MLP也可能近似这种关系，所以不能据实验结果断言旧头绝对不具备表达能力。
末层零保证两组初始渲染相同，不代表两组初始隐藏特征或优化动力学相同。
