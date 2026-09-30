# GGGS core：通用表面深度候选实验

2026-09-27。研究入口：[通用方法调研](../research/general_surface_reconstruction_20260927.md)。

## 目的与协议

验证新的通用高斯深度定义能否改善 Cat/Pixiu 的片层、尖刺及不连续表面。
采用作者 GGGS 的模型、GOF 增密和连续深度 CUDA，适配 NRHints 离心主点；
这不是原生 2DGS，也不是 GGGS 完整论文复现。不使用 SDF 或预训练几何监督。

- 环境：既有 ssd-gs，CUDA12.1、PyTorch2.4.1，RTX6000 Ada。
- 新依赖：项目内 trimesh4.6.8；CUDA扩展本地编译，C++17构建补丁。
- Cat fit470/validation52；Pixiu fit506/validation56；原始相机/光源，official test 未使用。
- 每场景 fresh30000，512px、seed0、40000个轮廓占据初始化点，增密到15000。
- 训练坐标按相机 extent 归一化；保存/评价使用世界坐标与原始完整 K。
- 颜色 .8L1+.2SSIM；alpha L1 权重 .2；7000步开始前景深度法线权重 .05。
- 作者默认 SH3、学习率、GOF 增密、3D Mip filter；无曝光拟合、多视图 NCC 或教师深度。
- `percent_dense=0` 是该作者版本实际模型状态，明确记录，没有静默替换为 native 分裂规则。

运行由 `configs/validation.json` → `launch_validation.sh` → `make_validation_manifest.py`
→ `run_benchmark.py` 管理。实际完整 argv、GPU、日志和源代码见
`runs/gggs_core_geometry/manifest.json`、`source.tar`、`status.json`。
评价为完整 validation 与 fit；不得用训练帧解释为未见验证。

## 已完成的关键验证

`test_methods.GroundedGeometryTests`：离心 K 虚拟传感器射线一致、已知平面连续深度、
法线方向、有限非零梯度、坐标缩放一致，以及 Mip filter 的世界坐标保存/重新加载。
日志 `logs/gggs_geometry_checks.log`。

800步、128px、2000点短训完成，增密到3515点，重新加载两张验证帧评价通过；
数值见 [preflight](gggs_preflight.json)。初次绘图缺少 `loss_terms` 字段而退出，
修正日志生产端并重建短训曲线。短训权重已删，不作为最终效果。

## 失败记录

首轮关闭3D Mip filter，Pixiu在3600步、Cat在8000步增密时发生尺度异常，
无最终模型。记录在 `retired/gggs_unfiltered_failure.tar.gz`。
随后恢复作者默认滤波，并检查每次反向传播的参数梯度，首个非有限梯度会保存诊断状态并退出。
不使用 `nan_to_num` 掩盖优化器异常。恢复滤波是否足够，依据正式重跑结果判断。

## 结果与几何验收

两场景各完成 fresh30000、完整 fit/validation；所有最终几何参数有限，世界坐标导出与 Mip filter 保存核验通过。
[完整数值与审计](gggs_core_results.json)。未评价 official test，未训练 relighting。

| 场景/方法 | val PSNR | val SSIM | val LPIPS↓ | val IoU↑ | 高斯数 |
|---|---:|---:|---:|---:|---:|
| Cat / native 2DGS | 14.57945 | .719348 | .300674 | .948379 | 81018 |
| Cat / GGGS core | 14.58354 | .719240 | .291484 | .950858 | 153718 |
| Pixiu / native 2DGS | 18.29994 | .845859 | .163773 | .899376 | 37257 |
| Pixiu / GGGS core | 18.33461 | .846231 | .161102 | .910584 | 65760 |

Cat完整fit470：15.48369/.731335/.282797，IoU .944827。
Pixiu完整fit506：20.15872/.864548/.150748，IoU .913980。
重跑所有步的几何/SH梯度均有限；首轮未保留首个异常梯度，因此只能说恢复默认滤波后本次完整跑通，
不把滤波之外的数值机制断言为已定位。

**人工几何门槛：两场景均未通过。**
Cat表面更连续但大块错误/过平滑区域及局部形状缺失仍在；Pixiu部分尖刺减弱，大片层状结构和变形仍明显。
增加高斯数与改善LPIPS没有转化成可接受的几何恢复。
[Cat四视角灰模](../../runs/gggs_core_geometry/review/Cat/geometry_comparison.png) ·
[Pixiu四视角灰模](../../runs/gggs_core_geometry/review/Pixiu/geometry_comparison.png) ·
[简图](../../runs/gggs_core_geometry/review/summary.png)。

本实验只否定“当前适配下，单靠连续深度核心替换就足够”这个假设。
没有复现完整GGGS的NCC约束或Gaussian Wrapping，不能宣称这些完整方法已经被本实验否定。
下一优先级是适应移动光源的几何观测约束与外观解耦，再评估有向高斯闭合表面；
不把新灰模作为已通过几何教师，也不启动relighting。
原生失败几何对照：`runs/native_2dgs_repaired_geometry/Real_NRHints/{Cat,Pixiu}/validation/`。
灰模使用同一组固定验证视角、同一固定光照公式和各自预测 alpha，不做几何对齐或 GT 遮罩裁剪。
没有真实参考网格，轮廓和自身多视图一致性不等价于真实几何精度；RGB分数改善也不能单独通过。
