# BiGS：入射—出射方向耦合的实现核查

查阅日：2026-09-23。本备忘与正在运行的add/multiply控制独立；不改变其预算、门槛或测试安排。
仅阅读论文和小型源码文件；没有安装、编译、运行BiGS，也没有下载数据或权重。

## 来源与版本

[作者项目页](https://desmondlzy.github.io/publications/bigs/)标为3DV2025，链接的稿件是
[arXiv:2408.13370v1，2024-08-23](https://arxiv.org/html/2408.13370v1)。本次没有另核会议终稿差异。
[官方仓库](https://github.com/desmondlzy/bigs)固定至`8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613`
（GitHub API查得main提交日2025-03-16）；下列代码链接均固定此版本。

## 论文中的角度表示

每个高斯独立保存散射系数，使用两组球谐的乘积；式(9)可写为

`s(ωi,ωo) = Y(ωi)ᵀ C Y(ωo), C = Cᵀ`。

RGB各一个对称矩阵；25维Y对应每通道325个独立系数。式(6)(7)(11)的单灯形式为
`L = [Tdir(ωi) (ρ+s(ωi,ωo)) + Tind(ωi)] Le`。
§4.3点光用高斯中心到灯的方向及`Le = I / distance²`；这是真实换灯条件，非固定照明新视角。
§7明确承认SH对高频光传输的限制；文中没有本项目1–4px高光指标，不能据此宣称小峰恢复。
[依据：§4.1–4.3、§7](https://arxiv.org/html/2408.13370v1#S4.SS2)

## 官方代码实际计算

- [bigs.py:31–60](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/relight/bigs.py#L31)
  为每个高斯建立`direct_light_shs`、`indirect_light_shs`和`phase_shs_upper`，没有共享空间MLP或网格。
  `basis_dim = sh_degree**2`；[默认sh_degree=5](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/scripts/training.py#L34)
  是25个基（通常记为最高阶l=4），不能误读成36个基。
- [upper_to_symmetric.py](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/relight/upper_to_symmetric.py#L4)
  将同一上三角参数写入矩阵两侧，直接保证散射双线性项的交换对称性。
- [double_shs_single_eval.py:3–12](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/relight/double_shs_single_eval.py#L3)
  先对入射方向求值，得到出射SH系数；再由
  [render_bigs_with_point_light.py:53–60](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/relight/render_bigs_with_point_light.py#L53)
  在相机方向求值，实际落实`Yᵢᵀ C Yₒ`，不是先拼接两个方向再交给MLP。
- [同一渲染文件:28–43、62–67、91–93](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/relight/render_bigs_with_point_light.py#L28)
  实际方向是`μ−light_pos`和`μ−camera_pos`，均与当前PORT的wi/wo相反；移植时必须统一符号。
  直接传输为SH值加0.5；默认距离平方衰减同时作用于直接与间接项。
  训练使用`compute_out_sh=False`：`clamp(incident) * clamp(phase+albedo) + clamp(indirect)`，
  即裁剪的是`phase+albedo`，不可把论文的分量叙述当成此路径逐项相同的实现。
- [training.py:131–179](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/scripts/training.py#L131)
  额外抽随机入射/出射方向，惩罚负散射、负直接传输及散射积分超出[0,1]。
  这是软正则；积分约束针对phase，没有证明整个含ρ与间接项的模型严格能量守恒。
  [优化器:208–221](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/scripts/training.py#L208)
  还更新位置、旋转、尺度和opacity；其结果不等于我们冻结全部主GS的控制。

## 与当前PORT残差的差异及一个可检验机制

当前[残差头](../../materials/radiance_residual.py)在像素GS接收点上计算actual wi/wo，
输入包含各自四频PE、位置PE、法线、余弦、half-vector log提示和目标灯位，最后乘`I/(scale*d²)`。
当前multiply是`W[(A k(x)) ⊙ (B a(wi,wo,n,light_pos))]`：显式耦合**空间与混合方向特征**。
BiGS显式耦合的是**入射与出射方向**，空间变化来自每个高斯自己的C；两种乘法不等价。
当前MLP、余弦和half-vector已经包含方向关系，不能称旧头完全没有wi/wo交互。

一个独立假设是将方向支路改为低秩对称乘积，并让现有空间网格给出局部系数：

`z_r = [(u_rᵀ E(wi))(v_rᵀ E(wo)) + (v_rᵀ E(wi))(u_rᵀ E(wo))] / 2`

`δ = [Σ_r c_r(k(x)) z_r] * I/(scale*d²)`，其中E可先复用当前方向PE。

这是本项目推导的候选机制：把相机—灯光成对关系直接提供给空间条件系数，
而不是要求共享MLP从混合输入中组织该关系；它不是BiGS复现，也不自动增加角频率。
局部项的交换对称性不保证整幅图互易：可见性、距离、主shader、最终截断仍在别处。
对有符号误差修正强加散射互易是否合适也尚未验证；不把它直接称作物理BRDF。
BiGS的方向传输在固定高斯上没有独立灯距输入，仅有显式距离衰减；当前PORT额外含灯位。
若以后检查此假设，应保留实际近场方向与衰减，并独立控制角度乘积、共享权重/对称化与参数预算。
本备忘不依据当前训练中间loss选择下一实验，不推断训练拟合、真实几何或未见灯光质量。

## 依赖与实施边界

[environment.yml](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/environment.yml)
指定Python3.11、PyTorch2.5、CUDA12.2，并使用nerfstudio、tiny-cuda-nn及本地扩展。
[relight/c/__init__.py](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/relight/c/__init__.py#L18)
在导入时JIT编译；[SH autograd:39–49](https://github.com/desmondlzy/bigs/blob/8bdc7d24edb3a3d957e4aa6589f8022b7f0c7613/relight/eval_spherical_harmonics.py#L39)
只返回系数梯度，不返回方向梯度。故不直接导入官方模块到共享ssd-gs环境。
上面的PE、线性映射、逐元素乘法和对称平均均可用现有PyTorch2.4.1实现，无需依赖升级或新CUDA扩展；
如只做纯Torch SH乘积helper也无此依赖需求。是否值得实施须由本轮完整受控结果另行判断。
