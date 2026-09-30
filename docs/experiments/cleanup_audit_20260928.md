# PORT-GS 文件清理审计（2026-09-28）

本次是检查和清单整理，未删除文件或改动权重。范围仅PORT-GS，不涉及SSD-GS等其它项目、共享数据集或用户上传论文。
项目 `du` 约20GiB，其中runs约8.7GiB，third_party约12GiB（显示值取整）。
12个带scheduler状态的场景实验均completed，另有独立完成的共享材质先验。
保留28个场景最终模型＋1个共享材质先验，未发现step_*/中间训练checkpoint残留。

## 建议优先清理

| 项目 | 逻辑大小 | 处理及依据 |
|---|---:|---|
| `__pycache__/`，1308个目录 |113.68MiB|可删除，Python可再生成。包含已删除方法的`surfel.cpython-310.pyc`、`surface_reflectance.cpython-310.pyc`残留。|
| 第三方扩展build/dist及tiny-cuda-nn源码目录里的.o |约92MiB|可删除构建副本/对象文件，保留源码与运行时.so。已逐字节校验8份build内.so与现有运行时副本一致。不要误删pip源码中名字叫build的目录，也不要整个删除local_fragments_build，它包含正在使用的运行库。|
| StableNormal/YoSo的重复text_encoder和VAE权重 |808.86MiB|两组权重哈希完全一致但inode不同；可保留两个路径，改为硬链接复用。只删除其中一路径会破坏加载。UNet/controlnet虽然大小相同，内容不同，不能去重。|

以上类别存在少量重叠，按inode/硬链接计数去重后，含文件系统分配块的预计回收量约 **1.01GiB（1033.71MiB）**。
这是可回收估算，不是已释放空间。硬链接应保持权重只读；未来更新其中一个目录时需要留意共享inode。

## 可选清理，不属于无用垃圾

- 26个`geometry_views.pt`，共832.13MiB：保存几何评价中间张量，可由最终模型重算。但`geometry_review.py`直接读取它们，删后对应复查命令需先重新评价。已有最终图像/指标不受删除张量影响。
- `fit_full/pair_*.png`合计约1.20GiB：是最终训练集评价图，不是训练中间checkpoint；可重渲染，但当前用于完整结果和对比，不建议按垃圾删除。
- `.venv/`约132MiB：虽然主训练使用ssd-gs，`test_method_integration.py`的法线先验准备仍引用此环境。要移除需先统一环境入口并验证，不能仅因它旧就删除。

## 应保留

- 所有最终模型、当前几何源和共享神经材质先验。
- StableNormal/DA3权重与训练先验。Wrapping/GGGS的两份normal.pt路径已经是硬链接，删除一个路径不会释放那2.86GiB。
- `third_party/torch_cache`内DINOv2和VGG16分别用于法线估计与LPIPS评价，虽名为cache，仍是有效运行依赖。
- 各实验source.tar、配置、manifest、history、指标与最终对比图，以及小型失败日志/退役追溯归档。source.tar已做过嵌套历史清理，当前每份约15–22MiB，不应只为减少文件数丢掉复现证据。
- 论文、其它方法项目、数据集、CUDA运行时.so。

项目外`/tmp/port*`还有历史日志/脚本/论文截图，本次未计入或处理，不能按通配符一并删除。

完整候选路径、运行库副本核验、重复权重摘要及字节数见[机器可读清单](cleanup_audit_20260928.json)。

后续状态：2026-09-29已按用户指示执行前三类清理，详见[执行记录](cleanup_executed_20260929.md)。可选几何缓存保留。
