# LiSA 当前状态

2026-10-03：三类数据集全部18/18场景完成，四方法合计72/72有效结果。最后任务于17:31:31 JST完成，完整测试图评价22764次，中央collector审计通过。

[完整实验记录](../experiments/results.md) · [环境与协议](../experiments/setup.md) · [逐场景机器记录](../experiments/records.json) · [四方法比较](../../../benchmarks/full_dataset_comparison/RESULTS.md)

本轮没有运行或待跑任务。用户于2026-10-03明确恢复SOTA核查、问题诊断与方案研究；分析现已完成，改进尚未实施。

LiSA总体SSIM/LPIPS与Real/SSS类别平均指标领先本地三条baseline，但总体PSNR比SSD-GS低0.2059 dB，不能宣称全面SOTA。主要问题是Lego/Drums的几何覆盖异常，以及多个场景的局部材质分支退化。72项结果复核通过；18个LiSA权重、62帧冻结探针及统一GT的PSNR敏感性检查完成，没有训练或修改模型/超参数。

[完整分析与证据图](../experiments/full_dataset_analysis.md) · [公开文献SOTA核查](../research/sota_assessment.md) · [优先级改进提案（全部未实施）](../research/improvement_proposal.md)

权重、原始日志、配置和源码留在原目录，派生诊断位于runs/light_atlas_analysis_20261003。架构、适配和恢复决策见[decisions.md](decisions.md)。
