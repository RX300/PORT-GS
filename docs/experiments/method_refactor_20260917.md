# 多方法架构与A/B接入验证 — 2026-09-17

本次验证用于接口和数值正确性，不是模型质量实验。未执行六场景完整训练。

## Git与环境

- 重构前完整代码/文档快照：2204cdc。
- 方法接口重构：cbc3545；后续将无端口方法的最小TransportBase接口与公共PortTransport分离。
- 分支：feature/selectable-transport-methods。
- 环境：原ssd-gs，PyTorch2.4.1/CUDA12.1/gsplat，未新增依赖。
- GPU短测使用已确认空闲的GPU1。沙箱内无法访问驱动，经沙箱外执行完成GPU检查和测试。
- 原runs与数据保留，新证据均位于runs/method_refactor_20260917/。

## 1. 与重构前源码的算子比较

从Git读取2204cdc的两个原始实现，严格载入相同state_dict，在CPU float64上比较：

| 方法 | 关闭端口输出最大差 | 开启端口输出最大差 | 参数梯度 | 参数键 |
| --- | ---: | ---: | --- | --- |
| learned_anchor_exchange | 0 | 0 | 匹配 | 一致 |
| directional_port_v1 | 0 | 0 | 匹配 | 一致 |

证据：refactor_parity.json。该比较包含公共直接光与源池化重构，不只是单个helper。

## 2. 五组CPU方法测试

命令：`python test_methods.py -v`，五组通过，覆盖：

- 注册方法CLI选择及完整配置/权重序列化；local_frame使用非默认frame_width检查尺寸恢复。
- paired_port在相同两端支持时、local_frame在单位坐标系时，精确退回directional基线。
- 改动出光空间参数后的独立循环公式，以及入/出两端中心和宽度的有限非零梯度。
- 局部坐标系正交性、右手性、直接光改变及非局部光不变。
- 四种方法的零光、光强缩放、有限梯度；多步优化后新参数实际更新。

局部frame输出层零初始化，因此隐藏层首步梯度为零是预期；测试检查后续更新。
新方法不声明能量守恒、互易性或真实材质恢复。

## 3. 真实Cat的训练/加载/评估链路

命令：`CUDA_VISIBLE_DEVICES=1 python test_method_integration.py --output runs/method_refactor_20260917/integration`。
每种方法各3步，seed0，128个Gaussian，rank8，32px，使用官方train，
第一步开启deep shadow与端口，关闭细化，最后一步保存唯一last.pt。
随后重新加载模型并比较同一帧像素，再通过evaluate.py评估1个fit视图。

| 方法 | 训练步数 | 最终loss有限 | 重载像素最大差 | CLI评估 |
| --- | ---: | --- | ---: | --- |
| learned_anchor_exchange | 3 | 是 | 0 | 通过 |
| directional_port_v1 | 3 | 是 | 0 | 通过 |
| paired_port | 3 | 是 | 0 | 通过 |
| local_frame | 3 | 是 | 0 | 通过 |

完整命令、训练日志、评估日志、config、split、checkpoint和loss曲线位于integration/；
汇总为integration/report.json。低分辨率3步指标不作为四种方法质量的排序。

## 4. 已完成30k checkpoint的真实渲染回归

读取Cat的rank512_validation_20260914和directional_port512_validation_20260915两个
原last.pt（均30000步），使用同一个真实Cat训练帧、32px和deep shadow，
比较重构前Git源码与新工厂加载得到的图像：两者像素最大差均为0。
证据：saved_checkpoint_parity.json。只读既有checkpoint，不修改已有实验结果。

## 5. 最终接口检查

TransportBase/PortTransport分离后，四种方法均再次通过checkpoint加载和真实Cat CUDA渲染反向传播，
所有网络参数梯度有限。证据：final_cuda_check.json。
四种方法的CLI帮助及各自六场景manifest选择检查通过；新文档的本地链接检查通过。
队列不再强制读取rank作为通用协议字段，方法特定设置仅留在train配置中，允许未来无端口方法。

## 6. 自审结果及范围

统一接口允许新方法直接实现forward，或继承PortTransport/DirectionalTransport的局部hook。
训练器、渲染器、评估器不再包含针对方向方案的实现分支；方法特定参数由注册类声明。
原根目录实现已迁入methods包并删除重复直接光计算，无兼容转发文件。
旧state_dict键保留，缺少representation的旧anchor配置保留已有的解释合同。
跨方法init-checkpoint明确拒绝；需要只移植几何时使用已有init-geometry入口。

A1/B1按研究方案最小版本实现：未混入方向秩扩展、材质大模型或物理路径追踪。
默认方法仍为当前方向端口基线。正式收益需按train内留出灯光、匹配预算和多seed评估。
