# runs仅保留方法最终结果（2026-09-24）

用户明确要求清理当前runs，只保留每个方法的最终结果。范围仅PORT-GS/runs。

清理前 32.306 GiB；清理后约 1.517 GiB；释放 30.789 GiB。删除97个非最终run目录及松散短测日志等，共8416个文件。

保留5个最终run、18个last.pt、各方法所有最终场景指标/现有测试图、最终fit评价、训练日志、源码配置、终端比较与审计。删除最终run内初始surface_initial.pt和首次失败审计记录。最终文件名和原方法身份不变。

默认PORT与旧神经材质的Cat/Pixiu全测试图先归并再删除审计容器；归并前对已有同名图逐字节比较相同，最终保有每场景66/71张。原始metrics.json不覆盖，完整导出指标和来源单独保留。可重用comparison plan更新路径，历史来源记录保持原样。

normal/depth等21GiB训练缓存不是最终方法结果，已清除；初始化种子和材质预训练run也删除。神经材质decoder已嵌入最终checkpoint；所有最终权重包含各自几何/材质。保留的原源码负责推理，从头重训需重新生成训练缓存，不声称输入张量仍完整保留。没有删除共享原始数据、源代码、用户研究文档或其他项目的输出。

核验：18个最终checkpoint在删除前成功CPU读取、步数均30000、各有最终test指标；神经材质包含decoder权重。删除后18个checkpoint大小和路径不变、指标仍在，四种方法Cat/Pixiu全137张导出完整。当前runs仅有5个最终目录及README索引。

四种方法learned_anchor_exchange、paired_port、local_frame、directional_surfel的原始最终产物本次清理前已缺失；未将短测/缓存当作其最终结果。此前的消融数字保留在docs作为历史记录，原始实验路径已删除，不能继续宣称原产物可访问。

[最终目录索引](../../runs/README.md) · [完整删除与归并清单](runs_cleanup_final_only_20260924.json)
