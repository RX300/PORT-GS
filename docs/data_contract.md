# PORT-GS 数据合同

本合同覆盖 `Real_NRHints`、`Synthetic_GS3` 与 `Synthetic_SSS-GS` 的训练 split，
并规定显式评估 split 的读取方式。实现位于
[PORT-GS/data.py](/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/data.py)。
数据根目录只读；加载器不读取 `points3d.ply`，也不会自动打开其他 split。

## 构造与元数据

```python
dataset = SceneDataset(scene_path, split="train", resolution=256)
sample = dataset[index]
```

SSS-GS 的光强不是数据集已知的辐射定标。读取 SSS-GS 时必须显式提供正的
`unit_light_intensity`，例如 `SceneDataset(scene_path, unit_light_intensity=1.0)`；
该标量只会扩展成 RGB 三通道。`1.0` 是待验证的等功率 unit-radiance 假设，不能
写成真实灯功率或已知光强，也不能用于声称与有辐射定标数据的无条件公平比较。

`scene_path` 是单个场景目录，例如
`/workspace/datasets/SSD-GS/data/Real_NRHints/Cat`。构造函数只读取下表指定的
一个 JSON，并把 JSON 中原始的 `frames` 列表暴露为 `dataset.frames`。
`dataset.metadata_path` 是实际读取的 JSON 路径，可用于在运行记录中核对 split。
构造阶段不解码图像；图像在
`dataset[index]` 时按需读取。训练器固定传 `split="train"`；评估器若要读
测试相机，必须显式构造 `split="test"`，不能让训练数据集自动混入。

| 场景 | `split` | 实际 metadata 文件 |
| --- | --- | --- |
| `Real_NRHints` | `train` / `val` / `test` | `transforms_train.json` / `transforms_valid.json` / `transforms_test.json` |
| `Synthetic_GS3` | `train` / `test` | `transforms_train.json` / `transforms_test.json` |
| `Synthetic_GS3` | `val` | 不存在 `transforms_val.json`，请求会报 `FileNotFoundError` |
| `Synthetic_SSS-GS` | `train` / `val` / `test` | `transforms_train.json` / `transforms_val.json` / `transforms_test.json` |

`split` 保持显式。`resolution` 是输出图像最长边的像素数，默认 256，保持长
宽比。

每个样本的数值字段都是 CPU `torch.float32` tensor；`name` 是不带扩展名的
`file_path` 或 `file_paths[0]` 文件名（如 `r_0`），`frame_index` 是所选 metadata
JSON 中原始 frame 的列表下标。

| 字段 | 形状 | 约定 |
| --- | --- | --- |
| `image` | `H x W x 3` | RGB，float32，保留文件数值空间 |
| `alpha` | `H x W x 1` 或 `None` | 独立 alpha，不做背景合成 |
| `c2w` | `4 x 4` | JSON 中原始 OpenGL camera-to-world |
| `viewmat` | `4 x 4` | OpenCV world-to-camera |
| `K` | `3 x 3` | 按最长边缩放后的相机内参 |
| `light_pos` | `3` | Real/GS3 使用 frame 的 `pl_pos`；SSS-GS 使用官方端到端 reader 输出的 `light_positions[0]` 原始 world 坐标 |
| `light_intensity` | `3` | Real/GS3 使用 frame 的 `pl_intensity`；SSS-GS 使用显式 `unit_light_intensity` 扩展的 RGB 标量 |
| `name` | 字符串 | `file_path` 或 `file_paths[0]` stem |
| `frame_index` | Python `int` | 原始 frame 下标 |

## 图像数值空间

图像用 OpenCV 的 unchanged 模式读取，随后把 BGR(A) 排列转换为 RGB(A)。

- PNG 的 `uint8` 通道除以 255，`uint16` 通道除以 65535；不会因为 PNG
  元数据中的 `sRGB` 或 `gamma` 标签而逆 gamma。
- EXR 通道直接转为 float32；不做 gamma、曝光、tone mapping、裁剪或归一化。
- 第四通道单独返回为 `alpha`。数据集不把 RGB 与 alpha 合成，也不填充白色
  背景；背景策略由训练器显式决定。为避免透明边缘在缩小时被背景颜色污染，
  有 alpha 时实现先分别 resize `RGB*alpha` 与 alpha，再用 alpha 恢复
  straight RGB；这仍然不改变通道的数值空间。

对训练文件的抽样审计结果：Real_NRHints 的 Cat、CatSmall、CupFabric、Fish、
FurScene、Pikachu、Pixiu 为 RGBA PNG，空间尺寸分别为 512 或 1024 的正方形；
抽查的透明像素 RGB 为 0，alpha 有连续的 0--255 值。Synthetic_GS3 的训练
文件为 512 x 512 RGBA float32 EXR，alpha 为 0--1，RGB 在不同场景中可超过
1（抽查 AnisoMetal 与 Hotdog 可见远大于 1 的值）。这些事实不改变 loader
对每个文件的原值处理。

## 相机与缩放

令原图宽高为 `W, H`，缩放因子为

```text
s = resolution / max(W, H)
```

输出尺寸为四舍五入后的 `(round(H*s), round(W*s))`。由于取整，`K` 的第一行
乘以 `output_width/W`，第二行乘以 `output_height/H`；方形数据上两者都等于
`s`。

Real_NRHints 的 `camera_intrinsics` 顺序是 `[cx, cy, fx, fy]`，因此未缩放
矩阵为

```text
[[fx, 0, cx],
 [0, fy, cy],
 [0,  0,  1]]
```

Synthetic_GS3 没有该数组，使用 `camera_angle_x` 推出
`fx = fy = 0.5*W/tan(0.5*camera_angle_x)`，并取
`cx=W/2, cy=H/2`，再按上面的 `s` 缩放。

JSON 的 `transform_matrix` 原样作为 OpenGL `c2w`。本实现沿用官方 SSS-GS
输入约定对相机局部轴做 y、z 变换，令

```text
F = diag(1, -1, -1, 1)
viewmat = inverse(c2w @ F)
```

因此 `c2w` 保留原坐标约定，而 `viewmat` 是供 OpenCV/栅格化路径使用的
world-to-camera 矩阵。官方 SSS-GS 的端到端数据路径在构造 `CameraInfo` 时先对
灯坐标翻转 Y/Z，随后 `CameraDataset` 再翻转一次；最终 `light_pos` 为原始
`frame["light_positions"][0]` world 坐标。当前 loader 与该端点一致，不在
`SceneDataset.__getitem__` 中再做一次单独翻转。这个代码路径不能独立证明物理
世界轴真值；本地 `anno.RT`、`K`、`light_pos` 与 transform 的对照只能核对字段
关系和数值实现。

`split_train_lights` 为保持既有、确定性的角度分组，仍对 SSS 灯位置使用一次
Y/Z 变换后再分箱。这里的 180° X 基底只属于划分坐标系，不是 renderer 的
`light_pos` 坐标变换，也不改变样本输出。

## 训练内灯光验证划分

```python
fit_indices, val_indices = split_train_lights(dataset.frames, fraction=0.1)
```

该函数只消费训练 JSON 的 frame 元数据，不读取任何图像、测试元数据或
点云。Real/GS3 使用 `pl_pos`；SSS-GS 仅在固定的角度分组基底中对
`light_positions[0]` 翻转 Y/Z。得到的灯位置先相对场景坐标原点归一化为方向，再按固定的 30° 方位角
和 30° 仰角网格分组。网格 key 按 `(elevation_bin, azimuth_bin)` 排序；函数
逐次选择能让验证样本数最接近目标 `round(N*fraction)` 的完整角度组，并保留
至少一个组在 fit 中。这样 holdout 是灯光角度组，且返回的两个列表仍按原始
frame 顺序排列。

没有独立 source ID 时，无法从元数据证明两个近邻位置属于同一物理灯。实现会
按规范化后的灯位置保留 6 位小数做审计，并通过模块 logger 记录
`same_light_cross_group` 与 crossing 数；同一位置若跨角度组会被明确标记。
精确重复位置在该固定分组中会一起进入同一组。训练器应保存这条日志和两个
index 列表，不能用测试图像来修正划分。

## 来源与边界

字段名及图像处理约定与原始数据布局一致；Real/GS3 相关读取约定可对照
[NRHints 的 data parser](https://github.com/iamNCJ/NRHints/blob/main/data/data_parser.py)
和 [GS3 的 dataset reader](https://github.com/gsrelight/gs-relight/blob/main/scene/dataset_readers.py)。
SSS-GS 的端到端相机/灯坐标路径可对照官方
[dataset reader](https://github.com/cgtuebingen/SSS-GS/blob/main/scene/dataset_readers.py)、
[CameraDataset](https://github.com/cgtuebingen/SSS-GS/blob/main/scene/dataset.py)、
[Scene 装配](https://github.com/cgtuebingen/SSS-GS/blob/main/scene/__init__.py) 和
[相机矩阵工具](https://github.com/cgtuebingen/SSS-GS/blob/main/utils/graphics_utils.py)。
这些链接仅用于数据/相机 I/O 约定，不改变本项目的数据隔离规则。当前 loader
不声明真实物理 radiance、颜色空间线性或光强单位；SSS-GS 的
`unit_light_intensity=1.0` 只能作为显式待验证的等功率假设，训练/评价协议需
另行记录。

旧 `runs/bunny_port` 使用过已修正的 split 映射，实际读到 SSS `transforms_val.json`，
其 23 dB 结果和权重均作废，不能复用。
