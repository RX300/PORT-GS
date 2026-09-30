> 2026-09-26补评：未修改终端模型，已评价完整Cat522/Pixiu562训练帧。train PSNR为22.531665/25.126019，tiny召回.1500%/.5071%；此前16fit不再用于代表全train。新增`fit_full/`属于该最终模型的评价，未创建中间run。[瓶颈分析](quality_diagnosis_20260926.md)。

> 2026-09-24清理更新：runs已按用户要求仅保留5套最终结果。中间实验、训练先验/种子和独立审计导出目录已删除；最终模型/指标/图片/源码仍在。历史路径仅作来源记录，重训缓存须重新生成。见[清理说明](runs_cleanup_final_only_20260924.md)。

# PORT-DNA-2DGS：Cat / Pixiu 首轮结果

已实现并完成两场景各30000步训练及全部137张官方测试图。结论：**比原默认PORT-GS的PSNR/SSIM更高，但LPIPS均变差，细节仍明显模糊；不认为整体质量胜出，不替换默认方法。**

新方法将8DNA的“独立能量 × 条件传输分布”适配为2DGS上的神经材质。直接项为解析GGX，间接项为32分量归一化表面/方向分布，用RGB/SSIM与图像边缘分布KL训练。它不是8DNA的路径监督流复现。

[中文方法提案](../research/port_dna_proposal.md) · [数学与实现](../architecture/modules/distribution_material.md)

## 完整官方测试

原evaluate指标为主，PSNR/SSIM越高越好，LPIPS越低越好。历史方法监督与计算预算不同；以下不是单一因素消融。

| 场景 | 方法 | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| Cat | 原默认 PORT-GS | 21.536184 | 0.766465 | 0.230405 |
| Cat | 旧预训练神经材质 | 22.479174 | 0.780930 | 0.255563 |
| Cat | 旧交点 GGX | 21.119731 | 0.763317 | 0.287500 |
| Cat | **PORT-DNA-2DGS** | **22.141679** | **0.776381** | **0.267714** |
| Pixiu | 原默认 PORT-GS | 20.578249 | 0.845556 | 0.156690 |
| Pixiu | 旧预训练神经材质 | 21.478632 | 0.851217 | 0.161733 |
| Pixiu | 旧交点 GGX | 21.672545 | 0.848500 | 0.173014 |
| Pixiu | **PORT-DNA-2DGS** | **21.502959** | **0.848252** | **0.167022** |

相对原默认：Cat PSNR +0.605494dB、SSIM +0.009916、LPIPS +0.037309；Pixiu +0.924710dB、+0.002696、+0.010332。LPIPS的正增量代表变差。

Cat仍低于旧预训练神经材质的PSNR；Pixiu与其PSNR接近，但LPIPS更差。相比旧交点GGX，Cat整体指标改善，PixiuPSNR略低而LPIPS改善。该结果说明在本预算下可以无预训练几何监督训练出可重光照表示，不证明优于所有历史方法。

## 高光与定性表现

| 场景 | 新方法 tiny 召回 | 原默认 tiny 召回 | 新方法 tiny 对比度 / GT |
| --- | ---: | ---: | ---: |
| Cat | 0/575（0.0000%） | 0/575 | 0.054319 |
| Pixiu | 22/5324（0.4132%） | 46/5324 | -0.001013 |

tiny为GT中1–4像素的中性亮峰组件，2px匹配容差，是图像代理指标。Pixiu的负平均对比度意味着这些目标峰位置未形成对应的局部亮峰。Cat毛发、面部和纸褶细节被平滑；Pixiu细小白色高光、窄亮条及底座纹理明显缺失。PSNR提升主要不能解释为高频外观恢复。

固定视图：Cat [0,16,33,49]、Pixiu [0,17,35,53]；GT组件规则选取14个64px裁剪，不根据预测挑选。对比图包含GT、原默认、旧神经材质、旧交点GGX和新方法。

## 神经传输是否被使用

对终端模型的固定16张训练帧关闭间接分支；这是推理干预，不是重新训练的受控消融，也不是测试泛化结果。

| 场景 | 完整 PSNR | 仅直接项 PSNR | 间接项占前景线性能量（逐帧均值） |
| --- | ---: | ---: | ---: |
| Cat | 21.096814 | 18.444299 | 22.30% |
| Pixiu | 26.138993 | 19.323036 | 55.47% |

间接神经分支确实参与拟合。其贡献大不等于学到了真实多次散射；照片监督下，材质、几何和传输存在歧义。共享拟合光强归一化也不能当作绝对辐射标定。

## 训练与数据协议

- 复用ssd-gs：PyTorch2.4.1、gsplat1.5.3、CUDA12.1，无依赖升级；GPU0/1为RTX6000 Ada。
- seed0，512px，固定40000个2D surfels、24维特征、32分量、64宽网络（17767个传输参数）。
- 每场景fresh30000，Cat全部522/Pixiu全部562训练帧；test全部66/71帧。16fit仅为拟合诊断。
- 训练轮廓95%共识、2px扩张、192网格占据概率表面初始化。无距离场，无SDF，无预训练几何/材质，无旧权重。
- RGB .8L1+.2(1-SSIM)，mask .05，特征L2 1e-5，图像分布KL .03；1000步后启用神经间接传输、deep阴影和几何自一致性/畸变（各.01）。
- 相机固定，仅拟合一个正的场景光强归一化。无增密，无test调参或测试后预算延长。
- 测试集曾被历史开发观察，且视角/灯光共同变化；不能称全新盲测或纯换光泛化。LPIPS的VGG只用于评价。

| 场景 | 训练循环秒数 | 峰值 PyTorch allocated GiB | surfels |
| --- | ---: | ---: | ---: |
| Cat | 1064.01 | 3.116 | 40000 |
| Pixiu | 1071.51 | 3.092 | 40000 |

两场景并行；训练时间不含轮廓预处理、初始数据加载与终端评价。峰值为PyTorch已分配张量显存，不是nvidia-smi总占用。

## 检查与复现

五项算子/注册接口检查、Cat512/40000的8步真实前后向、检查点加载渲染已通过。训练启动后验证实际子进程、日志和GPU活动；终端检查两模型30000步、有限参数、二维尺度、禁止的辅助状态缺失。输入审计确认训练/测试路径无交集、种子元数据精确匹配全部训练帧。运行归档内训练/渲染/材质源码与实际代码逐字节一致。

预处理的两项已解决问题保留记录：首次因可选skimage缺失取消，改用已有占据等值面采样器；重启成功导出种子后，报告器读取不存在的距离场字段报错，已修复并从未改变的张量恢复报告。精确预处理时间不可得，不编造。训练归档含报告器修复前版本，旁附报告修复说明；不影响种子或训练。

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
conda activate /workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs
# 每场景准备：更改scene和output；重复实验使用新目录
python prepare_surface_priors.py --kind surfel_init \
 --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Cat \
 --output runs/new_seeds/Cat --device cpu --fit-all \
 --surfel-points 40000 --surfel-surface-mode occupancy
# 修改canonical config的name和init-surfels后运行
bash launch_validation.sh
```

本次固定配置在`runs/distribution_material_final/validation.json`；所有具体命令在`manifest.json`，无需推测参数。

## 输出与证据

根目录：`runs/distribution_material_final/`。

- `Real_NRHints/{Cat,Pixiu}/last.pt`：两场景模型；`test/metrics.json`与66/71张`pair_*.png`：完整测试，左GT右预测。
- `results.json`：原始指标、所有参考来源、差值、训练开销；`transport_usage.json`：神经分支诊断。
- `analysis/{Cat,Pixiu}/full_images.png`、`crops.png`：固定整图/裁剪对比；GT一致性与帧映射记录同目录。
- `input_contract_audit.json`、`checkpoint_audit.json`、`startup_audit.json`、`status.json`：来源/模型/执行证据。
- `source.tar`、`validation.json`、`manifest.json`、训练日志和`history.jsonl`：复现记录。
- `runs/distribution_material_seeds/{Cat,Pixiu}/`：输入种子、生成源码与预处理失败/恢复记录。

本轮请求的实现、训练与评价完成；默认仍directional_port_v1。若继续研究，优先做相同初始化/预算下的归一化分布与KL消融，以及无教师的几何和高频着色改进；这些是后续建议，本轮未声称已验证。

最终共同PNG核验：四种模型在137张test上的已存GT全部逐字节等于当前数据集GT，帧映射一致。固定8张整图/14个裁剪人工复核完成，未发现可靠高光恢复。见`manual_review.json`与`analysis/summary.json`。
