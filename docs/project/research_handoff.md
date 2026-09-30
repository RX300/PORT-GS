## 2026-09-30 当前交接与文件保留

当前参考为R2b（default＋旋转相机校正＋平移规范），见[标定实验](../experiments/calib_camrot.md)和[输出索引](../../runs/README.md)。标定实验已结束；neural gauge_light两项为主动取消。

经用户确认，8组历史run的全量 `fit_full/pair_*.png` 改为均匀抽样、关键帧及明确引用帧；完整指标、所有test图、最终模型、配置、划分、日志、独有源码快照、先验和权重保留，R2b全部保留。完整训练集评价曾完成；现存图片已不是全量。旧文中的“全部fit图保留”和目录/模型总数只描述当时状态。目录路径不变，重渲染需对应快照和GPU时间，不保证逐像素一致。

[精确清理记录](../experiments/cleanup_20260930.json)。以下为历史交接。

## 2026-09-28：100万高斯上限对照已完成

Cat/Pixiu均30k及完整test/fit完成，已核对对照图、目标图像与几何。
Cat最终747578个高斯，PSNR22.52035（40万上限22.62266）；Pixiu248502个，PSNR21.53561（此前21.57110）。
Cat训练耗时增加59%，显存13.27→25.06GiB，质量无明确提升，几何仍粗糙；不推广100万设置。
Pixiu两轮均未触及40万，微小差异不应解释为上限收益。原结果保留，无本轮活动训练。
[完整结果与图像](../experiments/gggs_neural_material_1m.md)。以下是历史记录。

## 2026-09-27 最新交接：两轮联合重光照已完成

本轮 **GGGS初始几何＋Neural Material** 已完成：Cat/Pixiu各30,000步，几何与材质持续联合优化，共享神经BRDF解码器冻结。
完整测试PSNR为22.6227/21.5711dB，SSIM为.779711/.850532，LPIPS为.239291/.157998。
较GGGS＋DNA的PSNR提高.078/.386dB，但细节、高光和表面粗糙化仍未解决，**不替换默认参考**。
[完整结果、对照图与几何/换光预览](../experiments/gggs_neural_material_joint.md)。本轮训练和评价均已结束，无本轮运行任务。

用户要求先GGGS初始化接默认，再独立接DNA，都允许几何继续优化。四个模型各30000步，完整test/fit及固定几何/换光预览结束，无本轮后台任务。
DNA相对新默认PSNR提升但LPIPS更差，几何代理退化；未替换默认。三维适配保留GGGS厚度，原DNA2DGS路径仍可用。runs为10个目录24个最终模型，当前注册6个重光照＋3个几何入口。
[完整报告与模型位置](../experiments/gggs_dna_joint.md)。此前local_transport实验冻结几何；它不是DNA，与本轮两种方法不同。下方为历史记录。

## 2026-09-27 最新交接

当前用户要求的GGGS先验几何＋局部神经光传输已完成。两场景30k材质、完整test/fit、固定换光预览及最终核对结束；无本轮后台任务。代码6个重光照方法＋3个几何入口，runs8目录20终端模型。默认未改变，质量仍有明显缺陷。[当前报告](../experiments/local_transport.md)。下方为历史记录。

## 2026-09-24 用户授权：runs仅留最终结果

清理完成：5套最终run、18个模型保留；97个非最终run、中间权重及训练缓存已删，释放约30.8GiB。
[保留索引](../../runs/README.md) · [范围/限制/删除记录](../experiments/runs_cleanup_final_only_20260924.md)。
以下历史文字中“全部保留”的说法由本次明确清理要求更新；历史路径不保证仍存在。

# 2026-09-24 PORT-DNA-2DGS任务完成

本次8DNA照片监督适配与两场景训练评价已结束，未新增自动任务。
源码/配置/模型/完整137test/固定图块/输入及检查点审计在
`runs/distribution_material_final/`；[结果](../experiments/distribution_material.md)。
PSNR提高但LPIPS/细节未胜出，默认仍directional_port_v1。无SDF或预训练几何。
Canonical配置已经完成，不得同名重启；后续实验用新输出，保留已有记录。
旧研究交接保留如下。

---

# PORT-GS 接续：本轮已收尾，按用户要求暂停

用户要求本次方法研究与改善做完、Cat和Pixiu完整测试出来后，清理多余文件、
统计方法结果并暂停。这些收尾条件已经完成。不要自动启动后续实验或重跑
canonical run；更广泛的细小高光目标仍未达到，不能标complete。

## 权威结果和位置

- canonical `configs/validation.json` = surface_reflectance_final，已完成。
- 最终模型：`runs/surface_reflectance_final/Real_NRHints/{Cat,Pixiu}/last.pt`。
- 每场景`test/`保留全部66/71张图与原始指标，`fit_full/`保留全部522/562帧。
- 共同GT比较、14固定crop/8fullframes：`runs/surface_reflectance_final/analysis/`。
- 主结果/手工复核/来源：`test_gates.json`、`manual_review.json`、`audit/report.json`、
  `analysis_verification.json`、`fit_replay_audit.json`、`research_completion.json`。
- [完整最终报告](../experiments/surface_reflectance_final.md)、
  [全部8注册方法清单](../experiments/method_inventory.md)、
  [实际删除清单](../experiments/output_cleanup_20260924.json)。

Cat/Pixiu PSNR21.119731/21.672545，LPIPS.287500/.173014，tiny召回0/575、6/5324。
较默认Cat PSNR−.416453dB、Pixiu+1.094297dB，但LPIPS均差、小高光没有可靠恢复。
2/8、3/8数值门槛通过，manual false，不推广。默认仍directional_port_v1。

## 完成的本轮路线

新基础是surface_reflectance：真实ray–surfel交点局部GGX着色后合成，固定100k点，
独立几何/材质、固定相机、共同正light_scale；不是PORT或RGB残差微调。
100k交点/聚合对照失败；cutoff恢复5000失败；fit-mask表面初始化显著改善同10k
整体拟合，但tiny仍失败。最后的tiny-aware .5混合proposal+逆PDF补偿保持原loss
期望、监督次数约4倍，但5000步受控质量未过，最终按预先规则选uniform。

最终Cat522/Pixiu562各fresh30000，全train-only表面种子、不用开发checkpoint，
完整officialtest66/71后不调参。所有来源/权重/种子/采样均核验。原始拍摄路径
与test不重叠；这仍是历史开发已观察的联合视角/灯光测试，不是盲测或纯换光。
训练拟合、自一致性、实际3D几何精度和未见灯光泛化必须继续严格区分。

23终端计算条件、417组件比较、GT、固定crop、独立环域、64精确fit重放均通过；
严格跨backend scalar检查622差异(SSIM221/LPIPS401)保留，源evaluate指标优先，
所有门槛判断一致。第一次旧baseline字段读取KeyError已保存，其原报告无高光字段，
baseline组件来自先前统一PNG审计，未捏造旧source指标。

## 清理与今后约束

按授权删除117个冗余文件约2.072GiB，最终模型/全部test/正式失败证据/源代码和
配置/59个训练输入保留。短测checkpoint可按原归档重跑生成，勿假定被删文件仍在。
共享数据/依赖/其他方法项目未动。精简前README与完整docs归档在
`runs/surface_reflectance_final/pre_cleanup_docs.tar.gz`，此前长接续记录不再重复展开。
`source.tar`及各run原始日志为历史实验权威，不用当前worktree冒充旧版本。

用户明确要求暂停；不新增自动工作。若用户以后恢复，仍复用ssd-gs环境、canonical
入口和独立输出，不升级共享栈、不reset脏代码，最多使用2张当时检查空闲的GPU。
用户另要求进度检查每半小时一次，不按分钟轮询/汇报。新研究方向需依据现有负结果，
不能把已训练全部562帧的旧模型反向包装为未见验证。
