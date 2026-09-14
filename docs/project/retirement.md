# PORT-GS 终止决策

日期：2026-09-11。用户将判断标准明确为接近 SOTA，并要求两轮后仍不理想时放弃方法。
据此结束 PORT 的方法开发，保留源码、两轮修复、独立审查和全部实验结果。

## 判断依据

| 官方测试 | PORT 最终 PSNR / SSIM | 本地 SSD-GS 参考 PSNR / SSIM | PSNR 差距 |
|---|---:|---:|---:|
| Cat，66 帧 | 22.000 / 0.7675 | 27.243 / 0.9000 | −5.243 dB |
| Pixiu，71 帧 | 20.972 / 0.8477 | 31.170 / 0.9451 | −10.198 dB |

PORT 记录：[Cat](../../runs/cat_refinement_full_s0/test/metrics.json)、
[Pixiu](../../runs/pixiu_refinement_full_s0/test/metrics.json)。
SSD-GS 记录：[参考指标](../baseline_metrics.json)。PORT 为 30k 训练，参考为 100k；
相机/灯光优化设置亦有差别。上表描述最终产物的差距，不能单独证明模型容量上限。

Cat 第二轮训练内验证较历史同预算结果提高 0.503 dB，但细节能量仍仅约 GT 的
27.1%，困难视角仍有大范围亮度及位置误差。Pixiu 的底座纹理和小高光在最终图像中
明显模糊。小幅指标改善没有达到新的研究目标。
详见 [图像诊断](../experiments/cat_image_diagnosis.md)。

## 历史测试图像的对应关系

本次使用 GPU 0/1 对现有测试 GT 做全图像配对，未训练任何模型。
Cat 66/66、Pixiu 71/71 均找到一一对应的历史 GT，最大通道差均为 1 个 uint8 单位。
平均绝对差分别为 0.020773、0.013146 个 uint8 单位，符合量化级差异。
编号顺序不同，例如 Cat 当前 r_41 对应 SSD-GS 00047.png，Pixiu 当前 r_0 对应
00008.png。该结果支持测试图像集合对应，训练来源与相机/灯光优化协议仍须另行核对。

[完整映射](../../runs/reference_pixel_audit_20260911.json) ·
[审计程序](../../runs/reference_pixel_audit_20260911.py)

执行环境为现有 `ssd-gs`，实际执行命令如下；程序完成后归档到上述审计路径。

```bash
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python /tmp/port_reference_pixel_audit.py
```

## 研究转向

新的研究方案在独立的 `MATE-GS/` 中维护，核心是多光照误差归因与材质/几何分辨率
分配。其方法设计以 SSD-GS 的物理分解和工程协议为参考，重新建立表示与细化逻辑。
PORT 的 paired-origin 传输结构至此终止。

## 后续协议审计纠正（新方法实验阶段）

后续读取 SSD-GS/train.py、实际 checkpoint 与完整重新评价发现：上表 SSD-GS 数字
使用 test GT 优化后的相机/灯光，而 PORT 使用原始测试标定，二者不构成同协议的
方法差距。相同 SSD-GS 权重在原始标定下，Cat 为18.0006 dB、Pixiu为23.4022 dB；
固定历史标定下重新评价为27.2435、31.1704 dB，复现历史输出。

因此撤回将上表差值直接解释为 PORT 方法落后程度的结论。PORT 两轮内的改善仍有效，
其模糊与困难帧诊断也保留。研究方向已按用户要求结束，但后续新方法必须使用分别
标注的协议评价。完整证据见 [新协议审计](../../../MATE-GS/docs/experiments/protocol_audit.md)。
