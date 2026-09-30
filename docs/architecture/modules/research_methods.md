> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

> 2026-09-27：paired_port和local_frame已按用户要求删除活动代码。下文仅为历史设计记录，复现须使用原实验源码归档。

# 研究方案A/B的实现

2026-09-17：[原研究方案](../../research/cache_material_proposals_20260916.md)中的A1/B1已分别实现。
它们是可选择的方法，不是已验证有效的最终模型，也不把两项变化默认组合。

## paired_port：方案A

继承DirectionalTransport，源端继续使用anchor_centers/log_width，
接收端改用output_centers/output_log_width。后者初始化为前者的独立clone，
不是共享Parameter，因此起点精确退回方向端口基线，训练后两端能独立移动。
仍通过相同编号r的4×4方向矩阵联系两端，无R×R图或全Gaussian—像素配对。
R=512时只增加2048个可学习标量，计算阶数保持不变，实际耗时须另测。

源池化、共享方向基、RGB exchange及直接/非局部合并公式沿用方向基线。
非局部项仍是出射RGB，不再乘接收点的直接可见性或另一个材质函数。
可解释为近场传输的低秩因子化；不承诺物理守恒或唯一恢复真实散射路径。

## local_frame：方案B

继承DirectionalTransport，在material_directions hook中变换直接光方向。
frame_net为 `feature_dim→frame_width→6`，隐藏激活SiLU；默认32→32→6。
最后一层权重为0，偏置为(1,0,0,0,1,0)，从单位坐标系开始。
对两个预测向量作Gram–Schmidt正交化，以叉积形成第三轴，组成列向量矩阵Q。
直接光MLP使用Q转置乘光方向/视方向，半角方向在同一坐标系中生成；
其余网络输入与架构保留。非局部方向基不受Q影响。

不输入当前光强、图像编号或端口光状态。坐标系属于外观表示，不等于真实表面法线。
现有像素属性混合仍限制严格材质解释。未增加解析GGX、材质大模型或BSSRDF。
零初始化最后一层意味着第一步frame_net隐藏层梯度为0，输出层更新后才向隐藏层传梯度；
测试检查多步优化后两层实际更新，不以第一步所有参数都必须非零作为错误标准。

## 配置与加载

两者均保留directional的全部构造参数，A不增加超参数；B增加frame_width。
完整配置保存于config.json和checkpoint。evaluate.py按representation自动构造子类，
严格加载其新增参数，不对旧模型做隐式部分加载。
与其他方法共用的几何初始化由train.py的init-geometry入口完成。

## 自审边界

A首先只改空间支持，B首先只改直接光坐标系；不同时改变方向秩和训练日程。
算子测试检查独立公式、梯度、非负性相关路径和零光/强度缩放性质；
真实图像短测检查训练/加载/评估接通。跨光照泛化与漏光风险仍需正式实验。
