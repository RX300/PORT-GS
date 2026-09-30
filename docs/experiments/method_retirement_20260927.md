# 删除旧方法代码：2026-09-27

用户列出surface_reflectance、directional_surfel并要求删除；两者均已从活动代码移除。
删除文件：methods/surface_reflectance.py、methods/surfel.py、surface_fragments.py、surface_sampling.py。
移除注册、渲染分派/交点着色、专用patch训练/采样/统计、参数与失效的专用测试入口。
保留通用2DGS、材质、初始化、深度/法线模块，distribution_material需要这些共享实现。
不修改DNA方法文件、权重和历史实验配置；旧源码已在既有实验source.tar及retired归档中保留。

验证：45项CPU检查通过；真实Cat两帧、3步CLI训练、检查点重新加载、CLI评价、2DGS通道填充/阴影/增密/预热检查通过。
日志：logs/method_retirement_unit_checks.log、logs/method_retirement_dna_integration.log。
[机器可读记录](method_retirement_validation.json)。临时短训模型已删除，未创建新最终实验。

当前代码注册7个重光照方法＋2个几何重建方法，共9个。历史结果仍为四套/十二个模型，
本次被删的两个方法已无保留的最终权重，因此不会改变最终模型数。

## 后续删除：paired_port / local_frame

按用户追加要求删除methods/paired_port.py、methods/local_frame.py、注册及专用测试/分支，
更新当前架构与README；保留历史实验归档。二者无在用依赖，不涉及渲染/训练共享算法改动。
保留方法注册/配置/权重往返、光照线性/优化及DNA分布检查通过；日志为logs/port_variant_retirement_checks.log。
当前5个重光照＋2个几何入口，共7个；已有最终模型不变。
