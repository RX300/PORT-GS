# 三方法清理：2026-09-30

用户指令：保留 directional_port_v1、surface_attention、neural_material，其它方法全部删除。

## 活动代码

注册表、训练、评价和 manifest 只接受三个保留方法。删除：

- methods/anchor.py、methods/distribution_material.py、methods/local_transport.py。
- local_relighting.py、local_fragments.py、csrc/local_fragments.cu 及 local_fragments_build。
- wrapping_reconstruction.py、native_topology.py、surface_initialization.py。
- 三个几何后端的训练入口与专用参数、旧方法评价/预览入口、DNA 损失/初始化及 mask-surface 预处理。
- 已退役方法的专用测试；保留 Neural Material 三维渲染、GGGS 几何导入及共享边界数学测试。

native_reconstruction.py 只保留相机投影/坐标变换、深度法线、边界指标。
gggs_reconstruction.py 只保留已有几何转换和主线连续深度诊断，转换已解除对 local_fragments 的依赖。
shared PortTransport、2DGS、GGX/神经材质、相机模块和现有 Neural 辅助分支保留，不新增方法。

## 保存范围

- runs/、数据、模型和已有实验指标未删除或覆盖。
- configs/validation.json 未改，仍是上次 no-ports 消融，启动新实验前需改 name/配置。
- 删除前的项目 Python/配置/脚本/CUDA 源码：
  `retired/methods_before_three_method_cleanup_20260930.tar.gz`。
- GaussianWrapping 作者 checkout（含本地修改）先完整归档至
  `retired/gaussian_wrapping_dependency_20260930.tar.gz`，再从 third_party 删除。
- GGGS 作者依赖与其 simple-knn 路径保留，以便加载已有初始化并做主线诊断。
- 当前源码不再直接加载已删除方法的 checkpoint；历史复现使用各 run 或本次 retired 源码归档。

## 验证

已完成，无正式训练或整套数据重评价：

- 18 项针对性检查通过：注册/配置/权重往返、光强线性、attention、冻结神经材质、manifest、相机旋转/平移规范、边界数学、GGGS 导入与 Neural 三维梯度/加载。
- 六个已删除名称在 CLI、模型工厂和 manifest 均被拒绝。
- 三种保留方法均完成真实 Cat 四帧、3 步 CLI 训练、独立权重加载及 CLI fit 评价（带训练相机校正）。共享 2DGS 通道、阴影、增密和预热检查通过。
- 实际 Cat/Pixiu GGGS 源检查点转换与归档旧实现逐位一致。
- 核对 directional/attention/neural/base、renderer、cameras、gaussians、refinement 和 configs/validation.json 与清理前逐字节相同。
- 活动 Python 源码语法及 git diff 空白检查通过。

首次烟雾检查指定两帧，触发原测试固定要求四个 corrected_fit_frames 的断言；训练/加载已成功，改用脚本原有四帧协议后全部通过，未降低断言。
初次及成功日志都保留；新增临时模型/图像已清理，命令/日志/指标归档至 logs/three_method_retirement_smoke_evidence.tar.gz。
[机器可读验证](method_retirement_20260930_validation.json)。
