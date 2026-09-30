> 2026-09-27更新：后续修复和重训已完成，见[最终修复报告](native_2dgs_topology_repair.md)。本文涉及的前轮native_2dgs_normalized_geometry已归档至最终run的`controls/prior_native/`；旧模型权重已按final-only要求清理。

# 原生2DGS训练配置与人口管理复核

2026-09-27，用户指出新灰模明显劣于旧方案，要求优先核查训练参数。本轮仅做源码/终端权重诊断，没有修改模型、启动重训或宣称已修复。继续不用SDF或预训练几何。

## 已核对的默认项与偏离

所用作者代码的默认步数就是30000。位置LR1.6e-4→1.6e-6（乘camera extent）、feature LR.0025（非DC项再除20）、opacity LR.05、scale LR.005、rotation LR.001，以及500–15000之间每100步增密、每3000步opacity reset，均与当前适配循环/已存配置吻合。没有查到这些数值的录入错误。法线权重.05及7000步后启用也一致。

但三组都采用了`depth_ratio=1`，而作者通用默认是0。1对应中值深度，0对应期望深度，影响从深度计算的法线监督，不是显示选项。作者DTU脚本确实采用1及distortion1000，因此不能称1本身是错误；问题是本项目没有先比较通用默认与DTU设置。第一组应称“使用中值深度的原生控制”，不能称完整通用默认复现。

另外，40k轮廓表面初值/统一灰色并非作者COLMAP输入流程的SfM点和彩色初始化；作者合成数据入口也有自己的随机100k初始化。此前三组未隔离初始化因素。

## 可复现的屏幕尺寸剪枝问题

所用作者`scene/gaussian_model.py`第393–394行先调用clone/split；两者的`densification_postfix`在346行把**所有**`max_radii2D`清零。之后398行才检查`max_radii2D > max_screen_size`，因此这一屏幕尺寸条件无法看到之前积累的半径。其他opacity/world-scale剪枝仍可运行。

GPU最小复现明确固定两个高斯的sigma=.001、opacity=.5、extent=1、增密梯度0，历史屏幕半径为[64,1]，屏幕剪枝阈值20，排除其他剪枝条件。调用后两个高斯都保留，历史半径变成[0,0]。这确认该版本这条剪枝规则未生效；没有证明它单独造成全部视觉失败。该行为来自所用作者版本，未被本项目引入；项目在声称流程已验证时遗漏了实际剪枝行为的检查。

最初两点测试未显式控制kNN初始化尺度，被判为不合格隔离测试；其结果及原因随正确复现保留，不能用作结论。

## 参数大小与实际梯度

使用最终checkpoint原训练坐标，对每个场景4个确定性fit帧，分别计算photo、normal、distortion、mask对位置/log-scale/quaternion/opacity-logit的全参数L2梯度范数。没有用验证帧训练或优化任何参数。

Pixiu帧0/184/369/561，distortion位置梯度与photo之比分别约1.30/3.05/5.58/4.86；Cat四帧约3.26/.021/.028/.212。因此不能根据distortion标量loss约1e-4就称其几何作用弱。法线位置梯度约为photo的.19–.44倍，方向参数约.32–.79倍（Pixiu）；这些比值也不能单独证明正则过强。必须通过受控训练验证因果。

## 尺寸门槛的场景适配风险

分裂的尺寸门槛是`.01 * camera_extent`。本数据camera extent与物体包围半径之比为Cat12.22、Pixiu10.46，最终存储高斯中超过该尺寸门槛的比例仅.144%/.070%。这提示狭窄视场数据的分裂尺度选择需要单独检查，但终端状态不能还原训练期间的clone/split次数。

抽查帧中预处理可见高斯的投影包围半径中位数为Cat19–20px、Pixiu9–12px。这些是保守包围半径，含遮挡点，不是实际贡献加权的纹理分辨率。点数增加不能直接作为有效细分或恢复细节的证明。

## 修正此前结论与下一步

当前证据支持：本次实现版本存在未生效的屏幕尺寸剪枝，训练配置也没有充分分离通用默认、DTU几何设置和自定义初始化。不能据这轮结果推断2DGS本身重建差，也不能将数据/标定作为首要原因定论。移动光源问题仍是真实的建模风险，但应与训练链路问题分别验证。

后续应先验证并修正人口管理，再以完整通用默认深度配置建立对照，分别检查正则及面向物体尺度的split门槛。当前尚未修复或重训，质量改善未验证。不直接叠加多个参数变化，也不以增加训练步数替代这些检查。

证据：

- [各损失梯度与投影半径](../../runs/native_2dgs_repaired_geometry/controls/prior_native/parameter_audit/loss_gradient_audit.json)
- [屏幕剪枝复现及终端尺度统计](../../runs/native_2dgs_repaired_geometry/controls/prior_native/parameter_audit/topology_audit.json)
- 作者源码：`third_party/2d-gaussian-splatting/arguments/__init__.py:65`、`scripts/dtu_eval.py:23`、`scene/gaussian_model.py:327–401`。版本仍为f3e3b9fa67bbd1c75e05167ff37391d8dab2a678，模型和光栅源码未更改。
