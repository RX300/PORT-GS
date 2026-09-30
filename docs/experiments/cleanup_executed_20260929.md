# PORT-GS 残留文件清理（2026-09-29）

按用户指示执行此前审计中的前三类清理，已经完成。

- 删除1308个Python缓存目录，包含已退役方法surfel/surface_reflectance的.pyc残留。
- 删除9个经审计的build/dist目录和6个额外.o文件。
- 将StableNormal/YoSo共用的text_encoder、VAE两组重复权重改为硬链接；保留全部4个原路径，内容校验与审计时一致。
- 依文件分配块计回收1083924480字节（约1.01GiB）；du实际减少1089327104字节，含目录块变化与清理记录开销。

全部6675个runs文件的路径、大小、mtime和inode清理前后相同，包括28个场景最终模型、1个共享材质先验、评价图像/指标、先验输入和复现归档。
26个geometry_views.pt仍由几何复查工具使用，予以保留。.venv、有效依赖权重、运行时.so和其它项目均保留。

验证：9个保留的运行时扩展可在ssd-gs中导入；4条权重路径可在原有stablenormal_env中读取safetensors键和张量。全部运行库校验值保持一致，权重路径成对共享inode且内容未变。
首次验证使用ssd-gs读取safetensors时发现该包不在主环境，随后改用权重原本所属的stablenormal_env验证，通过；未安装或更换任何依赖。

清理后的Python缓存可在后续运行中自然再生成。权重路径为硬链接，后续更新一份权重时应以新文件替换，而不是原位写入共享inode。

[完整执行清单与验证记录](cleanup_executed_20260929.json) · [清理前审计](cleanup_audit_20260928.md)
