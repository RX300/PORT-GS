# 项目清理：2026-09-27

用户授权删除多余的失败实验和方法。逐文件清单与剩余模型见
[JSON](project_cleanup_20260927.json)。

删除文件逻辑大小合计 12,272,928,846 bytes（约 11.43 GiB），同时保存
51,093,649 bytes（约 48.73 MiB）的压缩追溯记录；不含随后新实验产生的文件。
这不是对底层文件系统可用块数的精确测量。

删除内容：

- 已淘汰的 surface_attention、neural_material、surface_reflectance 三套模型与图片输出。
  源码快照、配置、原始指标、训练日志归档在 `docs/experiments/retired/*.tar.gz`。
- 旧原生 2DGS 各控制实验的 `.pt/.ply/.npz/.npy` 诊断缓冲；最终原生模型及其评估不变。
  控制实验的图片、指标、日志、源码仍可追溯。
- 不被当前训练/推理配置使用的 DA3、StableNormal、YOSO 权重、DINOv2 缓存，
  旧 DA3 专用环境和两个先验仓库。

保留 SSD-GS、共享数据、用户论文文档、默认 directional_port 六场景、PORT-DNA 两场景、
最近一次失败的原生 2DGS 两场景作为必要比较证据。此刻共十个最终模型，后续新实验另计。
保留 LPIPS 使用的 VGG16 评价权重、本地 CUDA 运行时，以及 tiny-cuda-nn。

没有直接删掉仍被保留模型引用的方法模块：例如 distribution_material 导入
neural_material，删掉后会破坏 PORT-DNA 检查点加载。历史 normal/depth 先验生成分支
仍是可追溯代码，但其外部权重/环境已退役，当前配置没有调用这些分支。
需要复现退役模型时应使用归档源码/配置并重新准备其依赖，不能继续把旧路径当作现存结果。

本次调研获取的 GGGS 和 Gaussian Wrapping 为新的候选方法源码；GGGS 仅在项目目录构建，
没有替换共享 PyTorch/CUDA。经典匹配 probe 和 GGGS 短训使用统一诊断/训练入口。

另已删除58个全部pane已退出、启动命令指向本项目的旧tmux server，保留当前活动训练。
逐项记录见 [进程清理](retired_process_cleanup_20260927.json)。

清理过程中另发现源码快照递归包含历史实验归档；已从两份本轮快照中去掉这些嵌套归档，
释放约163MiB，逐文件字节核验确认保留源码/配置未变。之后的manifest归档会跳过retired目录。
记录见 [归档清理](source_archive_cleanup_20260927.json)。
