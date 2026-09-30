# 高光监督与采样：已做工作核查

2026-09-23，配对损失实现前只读代码、历史配置和终端日志。下表描述该次核查时点，
用于避免将旧损失重新包装成新实验；随后新增实现和实验见[成对RGB协议](../experiments/gs_residual_paired_loss.md)。

| 机制 | 已有实现与证据 |
|---|---|
| GT峰25%采样 | sdf_volume.select_rays；sdf_volume_peak_sampling_pilot各3k，峰MAE下降但PSNR下降且宽斑，见[结果](../experiments/sdf_peak_sampling.md) |
| GT峰25%+全局邻环25% | train.py的11px膨胀环排全峰、GTalpha>.9；sdf_peak_shape_pilot改善部分误亮，但未准确恢复，见[结果](../experiments/sdf_peak_shape.md) |
| 当前GS残差 | 2048 draws=512峰+512全局环+512前景+512有效域，独立抽样、重复计权、仅RGB L1；所有已完成残差实验均如此 |
| 显式高光RGB+局部对比 | train.py已有highlight_weight*(峰RGB L1+.5*neutral局部对比L1)，对比为minRGB减11px均值；w=.05曾与着色法线同时试2k，随后Cat/Pixiu8k仍无准确test高光，见[历史](../experiments/highlight_recovery.md) |
| GT峰法线对齐half | neural_material_specular_normal_pilot的w=.1、1k及联合细化尝试均无收益；已从canonical移除，源码归档保留 |
| BRDF先验改监督 | 程序化GGX方向采样10%→50%、log反射及透射/反射L1组合已试；不是场景峰/环损失 |
| 局部成对RGB差分 | 当前代码和所查记录没有此实验；全局环抽样没有峰与环对应关系，11px均值contrast也不是排峰ring差分 |
| 无色残差限制 | 没有；现有头为自由3通道有符号RGB。neutral=minRGB只是掩码/旧contrast/指标，不是achromatic网络限制 |
| 逆PDF补偿 | 没有；当前配额采样重加权目标，没有返回PDF或做无偏补偿。沿射线SDF重要性采样是另一问题 |

现有RGB峰+局部对比损失不能称全新想法，也没有被单独验证为残差头的有效改进；当前residual模式明确禁止highlight_weight。
不能直接在仅有2048像素更新的全图上套旧avgpool损失：邻居可能仍是未修正的原GS，形成不一致目标。
可复用neutral_peak_mask、silhouette_rays的valid/target/图像形状、select_rays及膨胀写法；
没有现成训练paired-neighbor sampler。angular_cues.py逐组件环是CPU后验诊断，不是已经实现的训练采样器。
实际代码：[train](../../train.py)、[采样](../../sdf_volume.py)、[掩码和指标](../../evaluate.py)、
[程序化先验](../../pretrain_material.py)、[头](../../materials/radiance_residual.py)。

## 新成对项的作用边界

令某通道两端误差为e_p=P_p−T_p、e_q=P_q−T_q。新项是|e_p−e_q|，不是约束梯度范数的锐化先验。
若峰偏暗、环偏亮（e_p与e_q异号），它等于|e_p|+|e_q|，按固定权重加强这对像素的修正；
若两端有相同偏移，差分误差可为零，因此仍需要普通RGB项确定绝对亮度与颜色。
这解释了选择成对监督的动机，也表明它可能只改变优化权重，不能据损失定义直接宣称窄峰会恢复。
已通过短测的首批raw pair=.182617、RGB=.076103，lambda=.25时额外项约为RGB项的60%；
这是初始损失数值比，不是参数梯度范数比，不用它动态改变权重或训练预算。


## Closing proposal study: preserved expected loss

The surface-reflectance patch proposal in
[the controlled protocol](../experiments/surface_patch_proposal.md) differs from
the earlier uncorrected residual/SDF GT quotas. With M valid origins and c_o tiny
GT pixels in a patch, q_o=.5/M+.5*c_o/sum(c); use q_o=1/M if empty.
The estimator is (1/P)sum_p loss(o_p)/(M*q(o_p)), including the entire SSIM
loss1-SSIM. Then E[estimator]=(1/M)sum_o loss(o), and since q is GT-defined,
the same identity holds for parameter gradients under this finite sum.
Weights are bounded by2, but this does not guarantee lower gradient variance.
It preserves the old origin-uniform objective's border bias and does not make
it a pixel-uniform full-image loss. Increased foreground concentration also
changes actual fragment work despite constant nominal patch count.
