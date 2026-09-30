> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

# Geometry-Grounded GS core adapter

`gggs_reconstruction.py` 由统一 `train.py --representation gggs_core` 调用。
评价使用 `evaluate.py --gggs`，复用 `native_reconstruction.evaluate_native` 的量化、
固定几何视角、深度一致性和输出协议，渲染后端通过局部函数选择，不改旧模型的解释。

官方源代码：`third_party/Geometry-Grounded-Gaussian-Splatting`，提交
`23b418c2f0f19b30b251acad6fe13445dae968c7`。模型及 CUDA 算法未修改；仅
`submodules/diff-gaussian-rasterization/setup.py` 将 C++20 改为 C++17，兼容 GCC9。
初次原设置构建失败日志保留。扩展 `build_ext --inplace`，不安装到共享环境；
trimesh4.6.8 以 `--no-deps --target third_party/python` 安装，PyTorch/CUDA 不升级。
simple-knn 复用项目内先前编译的作者扩展。

模型导入使用真实 `scene` 子模块命名空间，跳过仅供作者数据读取的 `scene.__init__`，
因此无需安装 Open3D。与原生 2DGS 作者包名称相同，两种作者后端应在独立进程执行。

输入使用共享 SceneDataset。离心主点通过居中虚拟传感器 + 输出双线性采样适配，
同时纠正 RGB 与连续深度的射线约定。虚拟画布扩大，不改变输入图像或标定。
训练坐标按相机 extent 归一化；检查点 capture 保持训练坐标，canonical gaussians
字段保存世界坐标。终端评价从capture恢复训练坐标与对应Mip filter，在原训练数值尺度求解，
最后把深度乘回世界单位。直接搬到世界坐标求解会在极少数像素产生尺度敏感离群值；
2026-09-27检查发现后已统一重评新旧GGGS。`restore_model`仍提供canonical世界高斯导出，
评价实际使用`restore_evaluation_model` / `render_evaluation_geometry`。

无先验对照为连续深度核心候选实验，保留作者默认 3D Mip filter；无多视图 NCC、曝光网络或SDF。
2026-09-27新增可选fit-only法线/深度先验；不开先验时保留原训练行为。
`gaussians.params.scales` 为三维尺度，不能把此模型直接当成原生 2DGS 继续训练。
完整研究依据与边界见 [调研](../../research/general_surface_reconstruction_20260927.md)。

关键检查：`python -m unittest test_methods.GroundedGeometryTests -v`；
经典匹配检查：`python -m unittest test_methods.MultiviewGeometryTests -v`。

每次反向传播检查几何/SH梯度，首次非有限时保存诊断状态并失败退出。
首轮无滤波配置失败归档，随后恢复作者默认滤波，不把失败的部分训练算作完成。


## 可选法线/深度监督

`--surface-priors`读取normal.pt/depth.pt，`--gggs-prior-normal-weight`和
`--gggs-prior-depth-weight`默认0，`--gggs-prior-from`默认1000，warmup默认2000。
加载时校验场景、分辨率、帧元数据和有限性；法线/深度权重与先验目录必须一起启用。
复用surface.surface_losses的OpenCV法线和去log尺度深度损失，仅取prior两项；
原GGGS自一致性独立保留，不重复增加一致性项。先验启用后立即请求连续深度。
配置、逐项loss和线性启用系数写入config/history，evaluate仍用--gggs。
[实验与精确设置](../../experiments/gggs_normal_depth.md)。
