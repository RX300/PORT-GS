# LiSA 实验环境与协议

共享 ssd-gs：Python 3.10 / PyTorch 2.4.1 / CUDA 12.1 / gsplat 1.5.3。

当前LiSA-staged全量使用30k+8k（38k），见下文分阶段协议；以下原LiSA基线配置为 light_atlas + svbrdf 材质头，每场景30k步、seed0、原生3DGS从头初始化，无外部预训练或几何先验。真实场景仅训练相机旋转优化及平移规范，测试使用原始标定。此前六场景参与过研发观察，不能把整个研发流程描述为独立盲测。

## 全18场景共同协议

完整结果与每场景来源统一见[results.md](results.md)，原始命令、配置、源码、GPU与日志索引见[records.json](records.json)。全部official train、完整official test；固定最终checkpoint；测试相机和灯位采用原始标定，不做测试拟合或曝光对齐。Real_NRHints与Synthetic_GS3最长边512px，Synthetic_SSS-GS为256px；Synthetic_GS3线性HDR训练、白背景、gamma2.2展示，其余黑背景。PSNR单位RGB、11×11 sigma1.5零padding SSIM、标准VGG LPIPS[-1,1]；先平均每场景帧，再等权平均场景。

单种子seed0，各方法训练预算不同；全量表不构成等计算量对照。历史命令应从对应源码快照重现，并将输出改为新目录，避免覆盖已完成结果。原始运行文件中的GPU分配仅用于追溯，不是新的资源使用授权。

## 原LiSA初始化与固定参数

原生3DGS：20k初始点、400k点上限，细化至25k；训练30k步。shadow-start与port-start均2000。svbrdf为最终材质头，训练所有official train帧；真实数据仅训练相机旋转优化（camera-mode rotation、camera-gauge translation）。精确解析参数以各场景config.json/命令清单为准，不从当前validation.json推定历史设置。

## 原LiSA训练与评价入口

```bash
python train.py --scene SCENE --output NEW_OUTPUT --representation light_atlas --material-head svbrdf --steps 30000 --seed 0 --fit-all --shadow-start 2000 --port-start 2000
python evaluate.py NEW_OUTPUT/last.pt --split test --output NEW_OUTPUT/test --lpips --highlights --save-all
```

上述为入口示意；分辨率、背景、真实场景相机参数等必须按records.json中的对应场景命令补齐。其他PORT方法的历史协议分别保留在各自实验文档，不用来解释LiSA结果。

## LiSA-v2协议（2026-10-04）

根因与选型：[lisa_v2_root_causes_20261004.md](lisa_v2_root_causes_20261004.md)。与上文原LiSA协议相同（30k、seed0、20k随机初始点、400k上限、
2D足迹分裂与梯度增密至25k、光源通道2000步、svbrdf、真实场景训练相机旋转校正、全official train、完整official test原始标定、最终权重），
只增加三个显式选项：`--mask-weight 0.5`、`--foreground-appearance-until 2000`、`--specular lobes`。

```bash
# 单场景（真实场景另加 --optimize-cameras --camera-mode rotation --camera-gauge translation --camera-start 2000 --camera-lr 0.001 --camera-lr-final 1e-05；
# GS3 加 --background 1.0；SSS 加 --resolution 256 --unit-light-intensity 1.0）
python train.py --scene SCENE --output NEW_OUTPUT --representation light_atlas --material-head svbrdf \
  --steps 30000 --seed 0 --fit-all --shadow-start 2000 --port-start 2000 \
  --mask-weight 0.5 --foreground-appearance-until 2000 --specular lobes
python evaluate.py NEW_OUTPUT/last.pt --split test --output NEW_OUTPUT/test --lpips --highlights --save-all
```

原LiSA-v2全18场景配置快照在`runs/lisa_v2_full_dataset_20261004/validation.json`，
输出 `runs/lisa_v2_full_dataset_20261004/lisa_v2/<family>/<scene>/`，`source.tar` 与 `manifest.json` 记录源码、diff哈希与精确argv。
历史调度曾每GPU两个worker；当前用户约束为全任务至多两张空闲GPU、每张一个worker。

## LiSA-staged分阶段协议（2026-10-04）

当前规范配置`configs/validation.json`为`lisa_staged_full_dataset_20261004`，仍用`bash launch_validation.sh`。
每场景从全部官方train独立初始化，30k辐射域几何/联合阶段→8k固定几何的观测域像素外观阶段→完整official test。
模型分别位于`<family>/<scene>/geometry/last.pt`与`appearance/last.pt`，test指标与全部成对图在`appearance/test/`。
总预算38k（8000几何预热+22000联合+8000外观），明确区分v2 30k及GS³100k；不根据test选择阶段或checkpoint。

选型只用train内留出：光照组留出（`split_train_lights`，外推）与插值留出（`--holdout-every 10`，近似官方test），均不渲染官方test帧。
插值留出中真实场景的留出帧保持原始标定（同官方test），Cat等标定误差较大的场景分数偏低，但对各变体相同。

## 原六场景数据泄漏审计（2026-10-02）


## 结论

最终svbrdf六场景实验未发现官方test图直接混入训练、渲染使用测试RGB/GT mask、测试时拟合或按测试分数选checkpoint。存在研发期间观察官方test结果和图像后继续改进方法的记录，因此不能称为整个研发流程未接触test的盲测。此结论限定于当前可核查的源码快照、数据、权重和日志，不能证明所有历史操作。

## 数据与权重核查

| 场景 | train | test | 路径/同一文件交叉 | 字节完全相同文件交叉 | 权重fit索引 |
|---|---:|---:|---:|---:|---|
| Cat | 522 | 66 | 0 | 0 | 全部且仅官方train |
| Pixiu | 562 | 71 | 0 | 0 | 全部且仅官方train |
| AnisoMetal | 2000 | 400 | 0 | 0 | 全部且仅官方train |
| Translucent | 2000 | 400 | 0 | 0 | 全部且仅官方train |
| bunny_small | 500 | 500 | 0 | 0 | 全部且仅官方train |
| dragon_small | 500 | 500 | 0 | 0 | 全部且仅官方train |

核对实际解析路径、设备/inode、SHA-256（仅尺寸相同文件可能相同，先按尺寸筛选再计算）。相机+灯位组合按六位小数检查也无交叉。没有进行重编码后像素去重或近重复图检索。六个last.pt均为30k，fit_indices恰好覆盖transforms_train.json，val_indices为空；init_checkpoint/init_geometry/surface_priors/material_decoder均为空，无外部几何或先验混入途径。训练日志中已记录的抽样索引都属于fit；日志不是每步全量记录。

## 实际代码与推理核查

以本次source.tar为依据：train.py只构造train数据集；fit-all表示全部官方train，不包含test。training/initialization.py从fit_indices建立采样池、归一化和相机范围。官方train内部留出验证时虽然会加载验证图到内存，优化采样池仍只取fit索引。

renderer.py读取sample image的形状来确定输出尺寸，不读取RGB数值；LiSA atlas来自已训练高斯和目标灯位，无GT输入。evaluate.py先渲染再读取目标图计算分数；主PSNR/SSIM/LPIPS在全图计算，没有用GT mask裁去渲染错误。GT alpha用于目标图合成以及另外报告的mask/高光指标。

关键运行模块当前文件与source.tar字节一致。在Cat最终权重、test第0帧64px上，将RGB、GT alpha改为随机值，同时改变name/frame_index，渲染RGB与预测alpha最大差均为0。这是一次有针对性的推理依赖检查，并非全部帧重跑。

## 评价与选型

六组评估命令均为last.pt --split test --lpips --highlights --save-all。未启用calibrate-test-views或shift-align，corrected_fit_frames均为0。所有1937帧索引完整且无重复；从逐帧PSNR/SSIM/LPIPS重新取均值，与报告相符。训练相机rotation优化只用于train，不传递给test。

检查head_selection的6项和light_holdout_generalization的15项：fit与validation索引无交叉，已记录训练索引均在fit，评估索引恰为validation；均从头初始化，不使用完整train模型作为起点。svbrdf选型规则已出现在该验证队列的源码快照中，该快照尚未包含adopted decision；支持其在本轮结果前设定的本地记录，但不是外部不可篡改预注册。

## 必须披露的局限

实验记录明确分析过compact/spatial版本的AnisoMetal测试裁图、测试逐视图误差、Pixiu测试视图的可见度分解，随后继续迭代材质头。最终头按train内留出验证选定，并不能消除前期对官方test的反复观察。应将这些结果视为开发过程中反复评估的benchmark结果，而不是完全独立的最终盲测；这与将测试图用于梯度训练是不同问题。

官方测试光照很接近训练光照，SSS两个场景还复用了训练灯位（相机与灯的组合不同）。这不是文件泄漏，但当前高分不能单独证明大幅度未知光照外推。已有train内光照留出评估又参与了模型选择，也不是额外独立最终test。

若需要严格盲测结论，先冻结方法和超参数，再用未参与研发的新场景/独立留出集做一次最终评价；仅在已经用于开发的六场景里重新划分不足以彻底消除开发选择偏差。此次未改训练代码、权重或既有结果，也未重新训练。

详细机器可读证据：[data_leakage_audit.json](../../runs/light_atlas_svbrdf_six_scene_20261002/data_leakage_audit.json)。
