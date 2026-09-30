# Surface cross-attention

2026-09-22，研究候选 `surface_attention`，不改变原默认方法。
用户要求调查2DGS退化、简化间接光并用三个GPU训练到结果完成。

六场景30k/seed0与1937张official test已全部完成：PSNR28.622846、SSIM0.918468、LPIPS0.090192。
相对首轮2DGS，PSNR提高0.150060dB、训练时间合计约减半；相对原默认3DGS仍低0.467129dB。
这是简化与效率收益的证据，尚不是整体质量或论文贡献验收。
[完整结果与固定首帧图像对比](../../experiments/results.md)。

## 诊断依据

已有三场景各4个fit帧的检查位于 `runs/surfel_diagnosis/`。
覆盖前景像素的非局部出射光占比约6%–23%（未按alpha加权，非全图能量占比），
不足以声称端口完全失效。
法线先验梯度/图像梯度范数比：dragon位置约0.11–0.35、旋转0.35–0.58；
bunny位置约0.23–0.25。局部存在负梯度余弦，但整体不能概括为所有先验均冲突。
同样每场景2帧的DA3梯度/图像梯度范数比均值较小：位置约0.004/0.014/0.014，
旋转约0.005/0.021/0.020（AnisoMetal/bunny/dragon）。因此先单独检验法线强度，
不同时调低所有表面损失；这些局部梯度检查不能替代完整消融。
每场景24对**完全相同camera transform**、不同灯光的训练图像，StableNormal
前景法线平均差为bunny18.5807°、dragon37.0528°。DA3去尺度log深度平均差为
0.02560/0.04171。每场景24对分别来自24个不同相机位姿。
这支持检验减弱逐图法线监督，而不是将预训练预测视为真值。
此前5k固定RGB预热+颜色重置整体退化；它还减少了重光照更新次数，不是纯几何消融。

已有完整test的alpha L1也说明退化不是一个统一现象：AnisoMetal的轮廓误差
从3DGS的0.000784降至首轮2DGS的0.000543，400/400帧改善，但PSNR下降0.274dB；
bunny则从0.001341升至0.002103，仅1/500帧改善，PSNR下降2.909dB。
dragon从0.002377升至0.003972，仅5/500帧改善。alpha指标只反映投影覆盖，不能代替3D几何真值。
逐帧对比按(frame_index,name)对齐，SSS数据存在重复name，不能只用文件名去重。

已有模型在三个场景的首个训练灯光下，64层阴影的单层深度跨度/场景半径为：

| 场景 | 原3DGS | 首轮2DGS | 5k RGB预热2DGS |
| --- | ---: | ---: | ---: |
| AnisoMetal | 0.02435 | 0.02378 | 0.02429 |
| bunny_small | 0.01364 | 0.01558 | 0.01528 |
| dragon_small | 0.02098 | 0.01918 | 0.02135 |

这次有限检查没有发现2DGS独有的大幅深度范围膨胀；不能据此排除所有阴影近似误差。
精确数值和几何范围保存在同一diagnosis目录的geometry_extent_comparison中。

## 单个attention算子

共享原直接光MLP、点光源衰减和阴影。删除方向端口的学习中心/宽度、三球面高斯瓣
及逐端口4×4矩阵，替换为源token到接收像素的单次cross-attention：

```
Q = query(feature_x, PE(x), PE(view_direction))
K = key(pooled_feature, PE(pooled_xyz), PE(pooled_light_direction))
V = pooled(softplus(source_response(feature_j)) * E_j)
A = softmax(Q K^T / sqrt(d) + log(cell_mass))
L = (1-a_x) * L_direct + a_x * A V
```

直接光仍是位置/特征/光向/视向条件化的神经响应，没有改为解析BRDF，
也没有将监督法线直接作为新的着色输入。法线和深度教师约束的是几何；
因此换用2DGS本身不保证材质或直接/间接光的解耦变好。

源数可达40万，不能构造所有像素×所有源的矩阵。先在固定8×8×8场景网格中，
按原面积测度聚合位置、特征、入光方向和受光RGB，只对非空格构造token。
网格只作计算上的源压缩，没有学习的空间端口、方向传输矩阵或额外缓存模型。
边界格吸收场景归一化范围之外的源；分格是离散的，格内聚合仍对源属性可微。
该压缩会损失格内变化，不等同于逐Gaussian的精确attention。

Q/K不输入光强，只有V输入光强；固定光源位置/几何时保留零光零输出和光强线性。
不同光源仍需分别计算再在线性辐射域相加。没有在AV后加带偏置的RGB MLP。
这是光照响应的结构约束，不代表物理能量守恒、互易性或严格的直接/间接分离。

默认attention_dim=32，使用PyTorch原生SDPA；V补零至32维以使用CUDA高效内核。
共享直接光不变，总网络参数100402→79049；非局部+gate参数29231→7878。
上述计数不含每个Gaussian的几何、RGB和32维特征，不代表整个checkpoint缩小相同比例。
单次真实512px、398161圆盘前后向检查通过，峰值已分配显存1.201GiB（不含整个训练集缓存），
首次测量0.879秒，不能视为完整训练速度基准。

## 对照与选择

复用configs/validation.json的variants与原队列。三个GPU，AnisoMetal/bunny/dragon，
每个条件30k/seed0，使用train内部灯光留出集，完整validation评价：

1. directional_control：原方向端口，normal_weight=0.05。
2. directional_weak_normal：只将normal_weight降为0.005。
3. attention_weak_normal：在2基础上只更换非局部表示。

各条件均不使用固定RGB预热，shadow/port在5000步启用，depth_weight=0.05不变。
这使1→2检查先验强度，2→3检查attention。重用的教师文件含完整train预测，
但训练仅访问fit索引，validation图像不参与优化。
三个场景的fit/validation帧数分别为1775/225、449/51、450/50；
教师均逐图推理，DA3每次输入一个图像，没有跨validation图像的多视角特征融合。
以场景等权validation PSNR为主要选择依据，同时报告SSIM/LPIPS与逐场景退化。
之后进行六场景全train→official test，不把小样本短测当质量结论。

## 参考与贡献边界

### 首轮消融触发的补充验证

减弱法线约束的validation PSNR变化为AnisoMetal +0.055374、bunny -1.855723、
dragon -0.099849dB，不能将减弱先验作为全局改进。bunny最后5k训练图像损失均值
0.004650→0.004149，但validation变差；这支持过拟合/泛化方面的检查，而不是只看训练loss。
AnisoMetal attention在弱约束下达到28.759203dB，比同弱约束方向端口提高0.543756dB。

因此追加attention_normal_control（normal-weight=0.05）三场景，以补齐2×2比较。
原dragon attention发生CUDA错误，排查确认gsplat 1.5.3未初始化通道补齐会污染梯度；
修复仅在renderer适配层显式补零，失败批次前向输出不变、梯度恢复有限。
原失败日志保留。后续队列surface_attention_followup_20260922包含该dragon弱约束重跑
和三个原法线权重attention任务；继续用同一个配置与启动入口。
这是查看首轮validation后增加的对照，不能描述为预先设定的四条件实验。

### dragon注意力饱和与归一化对照

修复数值错误后的dragon弱约束attention完成30k，但validation PSNR31.003466，
比同弱约束方向端口34.486891下降3.483425dB；50帧中47帧PSNR下降，全部alpha误差变差。
最后5k记录的图像损失也更高（0.008563对0.007325），不能只归因于验证泛化。
在2个fit+2个validation帧上，45个非空源格的attention最大权重均值为1，
熵对应的有效源数为1；QK分数最高约161–212。多个帧所选格几乎未受光，
对应非局部项接近零。相同诊断下AnisoMetal/bunny分别读取约22–28/6–8个有效源。
证据写入各模型diagnosis/metrics.json的attention_kernel字段，属于有限帧诊断。

据此增加一个无新增网络参数的稳定性对照：`--attention-score cosine`。
Q/K各自L2归一化，以固定sqrt(attention_dim)乘余弦相似度，再加原log(cell_mass)。
默认32维时，角度分数被限制在[-sqrt(32), sqrt(32)]；面积先验保持原定义。
仅V携带光强，所以光强线性不变。`dot`用于已完成/在跑的未归一化对照，
其公式、参数键及已有checkpoint行为保持原样。这个选项是实际消融，不新增方法类。
先在空闲GPU1从头跑dragon、30k/seed0、原法线权重0.05；再与在跑的同权重dot对齐。
若保留此候选，完整三场景validation和六场景official test仍须实际完成。

这项稳定化参考[Query-Key Normalization for Transformers (Findings EMNLP2020)](https://aclanthology.org/2020.findings-emnlp.379/)：
该工作使用L2归一化与可学习缩放来减轻softmax饱和。这里采用固定sqrt(d)缩放，
不增加可学习温度，也不将Q/K归一化本身作为创新点。10项CPU测试及真实Cat三步训练、
checkpoint重载/评价已通过；缩放Q/K末层1000倍不改变cosine模式输出，光强线性保持。

cosine dragon已完成30k，validation PSNR32.437177，相对同权重dot增加2.159894dB，
仍比同权重方向端口低2.149564dB。抽查2个fit+2个validation帧的有效源约3.5–12.2，
角度分数约[-5.51,5.49]，未再出现此前所有像素的单源饱和；4个validation帧非局部占比均值约22.8%。
这些证据支持稳定化起作用，但不能把它描述为已解决全部质量问题。
全部50帧alpha_L1均值0.012791，中位数0.004252；其中frame150/r_50_l_57为0.419713，
是显著的单视角覆盖失败，不能因其异常而从正式指标中删除。其余49帧也仍有误差。
cosine三场景已完整完成：PSNR均值31.816408，比原权重dot的31.235874高，
仍低于方向端口原权重32.457148。选择cosine/normal_weight=0.05做六场景完整测试，
用于完成最佳attention候选的质量/效率评估，不将其称为全体消融的质量赢家。
[完整数据与冻结设置](../../experiments/results.md)。

[原生SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
定义了标准attention；[Perceiver](https://arxiv.org/abs/2103.03206)是压缩大量输入后再查询的相关工作，
本方法没有复现其学习latent瓶颈。当前是固定空间聚合加单次attention。
attention、线性光传输、空间聚合均非单独的新概念；是否有论文贡献需要更广泛相关工作、
匹配预算消融和跨场景/跨灯光证据，不以“用了attention”宣称创新。

补充：同位姿预测差异也可能包含扩散采样方差，不能只归因于光照；该检查证明的是
训练目标不一致，尚未证明每个目标相对真实法线的误差。

## 与近期attention重光照工作的关系

- [LightSwitch, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Litman_LightSwitch_Multi-view_Relighting_with_Material-guided_Diffusion_ICCV_2025_paper.pdf)：
  以环境光latent cross-attention条件化扩散模型，并使用材质提示；不是这里的显式线性AV光传输。
- [Relightable Holoported Characters](https://vcai.mpi-inf.mpg.de/projects/RHC/pdfs/Relightable_Holoported_Characters__Capturing_and_Relighting_Dynamic_Human_Performance_from_Sparse_Views.pdf)：
  UV特征和环境图之间的cross-attention也用于重光照及复杂传输，不能宣称首次使用attention。
- [LightBridge, 2026预印本](https://arxiv.org/abs/2609.02543)：使用图像/点attention向3DGS传播生成式重光照信息。
- [RelightFormer, 2026](https://arxiv.org/abs/2609.07414)：由视频基础模型改造的生成式Transformer，
  通过cross-attention注入环境光，面向单/多视图直接重光照；不是本项目的逐场景OLAT拟合协议。
- [LiNO, 2026预印本](https://arxiv.org/abs/2606.03262)：面向PDE的光学启发神经算子，
  已将非局部传播表述为归一化成对核，并讨论正特征的线性复杂度形式；不能将这种一般核解释视为首创。

本候选的研究问题是：是否能以少量参数的、光强线性的源—接收点attention核替代方向端口，
在照片监督下保留非局部重光照能力。与这些工作任务、训练数据和算子约束不同；
尚不足以证明创新性或顶会竞争力，需实验后再决定论文叙事。

更直接的光传输参照包括[PRTGaussian](https://arxiv.org/abs/2408.05631)，其从多视角OLAT学习
逐Gaussian高阶球谐传输；[NRTF](https://people.mpi-inf.mpg.de/~llyu/projects/2022-NRTF/index.htm)
则结合神经PRT和路径追踪产生的OLAT监督。
[Neural PRT](https://repo-sam.inria.fr/fungraph/neural-prt/)研究相同参数预算下的小型神经着色器结构。
这些工作已覆盖学习传输、分离光照输入与传输描述等思想，不能将这些一般性质单列为新贡献。

## 简洁方案的验收条件

只保留一个非局部聚合算子；表面模型与教师约束是几何支撑，不打包为多个创新点。
先检查同预算下的质量与速度，尤其是间接光明显的区域，再决定attention是否值得保留。
单次seed0、三个场景的筛选只能支持候选选择，无法证明多场景稳健性、物理解耦或论文竞争力。
最终六场景测试之后仍需匹配协议的强基线和重复种子，才适合评价投稿贡献。

本次测试已完成，证据支持将研究问题收敛到“紧凑、光强线性的表面间传输表示”的质量/效率折中。
2DGS、两个预训练教师、attention与Q/K归一化各自都有已有工作依据，不应打包成多个独立创新点。
后续论文判断需要固定几何的传输消融、重复种子和同协议强基线，特别检查次表面散射场景；
目前单seed、部分场景退化的结果不足以宣称顶会竞争力。现实现的RGB共用attention权重，
而source_response与gate按RGB区分；它也未显式恢复可验证的BRDF或真实间接光分量。
