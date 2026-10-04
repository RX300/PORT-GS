# 共享重光照数据盘点

日期：2026-09-12。读取 `/workspace/datasets/` 目录和场景 transforms JSON，
逐项检查其图像路径存在性。未解码全量图像；“缺失 0”表示 JSON 引用完整，
不代表已验证全部图像内容或与远端发布逐字节一致。数据均保留在共享 datasets 中。

根目录：`/workspace/datasets/SSD-GS/data/`。这个存储目录名表示现有共享位置；
GS³、SSS-GS 和 NRHints 的原始数据归属与引用分别记录。

## Synthetic_GS3

| 场景 | train | test | 引用缺失 | 格式 |
| --- | ---: | ---: | ---: | --- |
| AnisoMetal | 2000 | 400 | 0 | EXR |
| Drums | 2000 | 400 | 0 | EXR |
| FurBall | 2000 | 400 | 0 | EXR |
| Hotdog | 2000 | 400 | 0 | EXR |
| Lego | 2000 | 400 | 0 | EXR |
| Translucent | 2000 | 400 | 0 | EXR |

每个 frame 提供 `file_ext`、`file_path`、`pl_pos`、`pl_intensity` 与
`transform_matrix`。没有独立 validation JSON，PORT 使用训练灯光分组留出。
官方来源：[gsrelight-data](https://huggingface.co/datasets/gsrelight/gsrelight-data)。

## Synthetic_SSS-GS

| 场景 | train | val | test | 引用缺失 | 格式 |
| --- | ---: | ---: | ---: | ---: | --- |
| bunny_small | 500 | 500 | 500 | 0 | PNG |
| candle_small | 500 | 500 | 500 | 0 | PNG |
| dragon_small | 500 | 500 | 500 | 0 | PNG |
| soap_small | 500 | 500 | 500 | 0 | PNG |
| statue_small | 500 | 500 | 500 | 0 | PNG |

每个 frame 提供 `file_paths`、`light_positions`、`transform_matrix`、尺寸与
相机内参。当前 loader 每 frame 使用第一个图像和灯，场景所属父目录以
`Synthetic_SSS-GS` 识别其 split 与光强处理。官方真实数据可能按相机聚合多个灯，
后续接入时需完整展开 frame，保留全部观测。官方来源：
[CGTuebingen/SSS-GS](https://huggingface.co/datasets/CGTuebingen/SSS-GS)。

## Real_NRHints

| 场景 | train | valid | test | 引用缺失 |
| --- | ---: | ---: | ---: | ---: |
| Cat | 522 | 400 | 66 | 0 |
| CatSmall | 1258 | 400 | 158 | 0 |
| CupFabric | 1153 | 400 | 145 | 0 |
| Fish | 526 | 400 | 66 | 0 |
| FurScene | 676 | 400 | 85 | 0 |
| Pikachu | 1500 | 400 | 200 | 0 |
| Pixiu | 562 | 400 | 71 | 0 |

`valid` 对应 `transforms_valid.json`，与 PORT 的 train 内灯光留出集分开。
Cat 的最终评价使用 `transforms_test.json` 的 66 帧。
原始来源：[NRHints](https://github.com/iamNCJ/NRHints)。

## 可复用范围

GS³ 6 场景与 SSS-GS 5 个 small 场景均可复用现有 PORT loader。
SSS-GS 需要显式 `--unit-light-intensity 1`，其物理含义限于等功率假设。
ReCap、SSS-GS 真实子集未出现在共享根目录的已识别重光照数据目录中。
本次未修改、移动或下载任何数据；官方下载入口和推荐场景见
[调研记录](related_work_20260912.md)。

## 2026-10-03：数据血缘与源包核验

当前18场景是沿用SSD-GS论文的三类评测组合，不是SSD-GS原创的一套数据。真实7场景是NRHints发布/使用的数据集合（其中4个继承DNL、3个为NRHints新增）；合成6场景采用GS³公开的2000/400图像版本；SSS合成5场景采用作者的256px small版本。

本地直接源包分别保留在/workspace/datasets/SSD-GS/_sources/gsrelight-data/NRHints、gsrelight-data/Synthetic和SSS-GS/synthetic。对18个zip/tar对应的48份transforms JSON作解析后比较，全部与data/现有元数据一致；没有重划分或改动数据。本次未重新下载并认证整套远端图像字节。

别名：Cat_on_Decor → CatSmall、Cup-Fabric → CupFabric、Cluttered → FurScene。CatSmall不是Cat的下采样版。相同Lego/Hotdog等资产在NRHints和GS³中存在不同渲染与划分版本，不能仅凭同名认定评测相同。

[完整数据出处与论文筛选](per_scene_olat_literature_202503_202610.md) · [18场景源包核验记录](dataset_provenance_audit_20261003.json)
