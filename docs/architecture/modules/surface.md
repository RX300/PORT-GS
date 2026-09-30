> 2026-09-27：本文描述历史方案。surface_reflectance和directional_surfel的独立方法入口已按用户要求删除；共享2DGS几何、GGX和初始化模块仍保留给DNA等方法。历史复现使用原实验源码归档。

# 2DGS + StableNormal + DA3

2026-09-21：新方法 `directional_surfel`，默认方法仍为 `directional_port_v1`。
这是 PORT 内的几何/表面监督变体，复用原训练器、评估器和512×4方向端口。

## 几何与光照

- `methods/surfel.py` 继承 DirectionalTransport，仅声明使用2DGS几何。
- `gaussians.py` 保存二维对数尺度、三维中心、四元数、不透明度、base和32维特征。
  圆盘随机朝向初始化。gsplat 1.5.3 接口仍接收三维尺度数组，补入的第三项为常数，
  不是可训练厚度，光栅化采用真正的ray/splat intersection。
- `renderer.py` 使用 `rasterization_2dgs`，显式相机维支持属性与64层阴影通道；
  输出期望camera-Z深度、世界坐标圆盘法线、depth distortion以及增密梯度。
  原像素接收点反投影、直接光MLP、方向基、端口矩阵和观察模型继续复用。
- `refinement.py` 使用gsplat 1.5.3的 `means2d.absgrad`（默认absgrad模式）或
  `gradient_2dgs.grad`（关闭absgrad时）增密，圆盘在切平面内分裂；优化器状态与
  所有逐点属性同步调整。复制、删除、不透明度重置沿用既有实现。
- 源端质量使用停止梯度的归一化 `opacity * scale_u * scale_v` 面积代理量。
  不因此宣称能量守恒或恢复物理散射。
- 不隐式把旧3DGS checkpoint压扁。`--init-geometry` 要求相同几何类型、相同划分。
  checkpoint记录geometry与representation，评估按方法恢复圆盘参数。

## 阴影边界

相机和光源视角均使用2DGS光栅化；保留64层deep-shadow的近似流程。
圆盘没有椭球厚度，查询偏移改为 `2 * bin_width + radius * 0.002`，
排除源自身的相邻两层贡献。圆盘不进行基于法线的单面剔除。

**仍按圆盘中心深度分层，不是逐光源像素的精确交点深度分层。**
大圆盘、倾斜圆盘和紧邻表面的阴影可能有误差；本次没有修改gsplat CUDA内核，
不将此实现称为精确光线追踪。像素直接光仍混合源端可见性。
新方法要求 `--shadow-mode deep`；没有更换直接光BRDF/添加法线余弦项。

## 预训练监督与损失

`prepare_surface_priors.py` 是两个预测器共用的唯一预处理入口：

1. 只读取官方train图像，默认使用与训练相同的灯光fit划分；正式全train使用 `--fit-all`。
2. 图像以至少processing resolution读取，采用与训练一致的背景和HDR显示变换。
3. StableNormal使用浮点 `prediction`，不用8位法线可视化；按作者2DGS接入实现，
   **对原始法线三个分量全部取负**，转换到当前OpenCV camera坐标，再归一化。
4. DA3-Large逐帧单目推理，`upper_bound_resize` 保留完整视野，无中心裁剪；深度
   插值回训练分辨率。没有用模型预测相机替换数据相机，也没有生成度量真值。
5. 每类保存一个 `normal.pt` / `depth.pt`，包含帧metadata、浮点预测、场景、分辨率、
   模型名称、源码revision和预处理参数。不在训练时反复执行预测器。
6. 训练加载时检查来源、法线约定与fit帧对应，缺失先验报错，不静默跳过。
   评估只读取checkpoint，不读取先验、不调用教师模型、不拟合测试标定。

监督从 `--surface-start 1000` 开始，默认权重均为起始设置，未经质量调参：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--normal-weight` | 0.05 | StableNormal同时约束渲染圆盘法线和深度反投影导出的法线，余弦损失取两项均值 |
| `--depth-weight` | 0.05 | DA3的camera-Z相对深度，log深度残差去均值后SmoothL1，消除单目尺度不确定性 |
| `--surface-consistency-weight` | 0.01 | 渲染法线与深度导出法线一致性 |
| `--distortion-weight` | 0.01 | gsplat深度畸变，除以scene radius平方进行尺度归一化 |
| `--geometry-warmup-steps` | 0 | 设置为5000时，1–5000步仅RGB圆盘重建，5001步开始重光照 |

训练可见性mask停止梯度，要求alpha>0.05；有GT alpha时要求GT alpha>0.9。
法线监督进一步腐蚀一个像素，避免跨背景差分。深度只使用正值有效区域。
两种prior weight置0即可做不带教师监督的几何消融。教师预测不是GT，在OLAT阴影、
金属、毛发和透明场景可能有偏差；当前未增加置信度模型或多视角先验融合。

## 纯2DGS预热（2026-09-21追加）

用户要求前5000步不训练重光照，并确认使用“2DGS RGB重建+表面约束”。
`--geometry-warmup-steps 5000` 时直接光栅化每个圆盘的softplus(base) RGB（零阶颜色），
不调用transport.forward，不计算灯光照明或阴影。冻结transport全部参数及32维features，
仅优化圆盘几何、opacity和预热颜色；L1/SSIM、mask、法线/深度与几何正则按原设置运行。
surface-start仍为1000，前999步只有RGB/mask，1000步起增加表面约束。

第5001步解除冻结、启用重光照。丢弃预热中含光照的RGB：base重置为初始化的−1.5，
清除对应Adam状态；保留几何、opacity及其优化器状态。网络学习率从重光照阶段起算，
几何学习率仍按总训练步数衰减。第二阶段几何仍继续联合优化，没有冻结表面。
本次配置总30k步，即5k预热+25k重光照；shadow-start/port-start设为5001。
新增history.stage及一次relighting_start事件。checkpoint在预热内保存时，evaluate也
按预热RGB路径渲染；旧checkpoint缺少该参数时仍使用原重光照路径。

## 环境与权重

训练继续使用 `ssd-gs`：PyTorch2.4.1、CUDA12.1、gsplat1.5.3，无升级。
StableNormal旧版diffusers与DA3新版Hub依赖分开安装，两个本地venv通过
`--system-site-packages` 共享基线PyTorch/CUDA，不修改Conda环境：

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
PYTHON=/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python
git clone https://github.com/Stable-X/StableNormal.git third_party/StableNormal
git -C third_party/StableNormal checkout 594b934630ab3bc71f35c77d14ec7feb98480cd0
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/Depth-Anything-3
git -C third_party/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
$PYTHON -m venv --system-site-packages .venv
.venv/bin/pip install diffusers==0.28.0 transformers==4.36.1 huggingface-hub==0.23.0 accelerate==0.30.1 einops==0.7.0
$PYTHON -m venv --system-site-packages third_party/da3_env
third_party/da3_env/bin/pip install 'huggingface-hub>=0.34,<1' 'einops>=0.8' omegaconf evo e3nn plyfile pillow_heif moviepy==1.0.3 addict pycolmap trimesh safetensors
third_party/da3_env/bin/pip install --no-deps -e third_party/Depth-Anything-3
third_party/da3_env/bin/hf download depth-anything/DA3-LARGE --local-dir third_party/weights/DA3-LARGE --include config.json model.safetensors
third_party/da3_env/bin/hf download Stable-X/stable-normal-v0-1 --local-dir third_party/weights/stable-normal-v0-1 --include '*.json' '*.txt' '*.py' '*.fp16.safetensors'
third_party/da3_env/bin/hf download Stable-X/yoso-normal-v0-3 --local-dir third_party/weights/yoso-normal-v0-3 --include '*.json' '*.txt' '*.py' '*.fp16.safetensors'
```

StableNormal需要YOSO初始化网络和DINOv2-L特征骨干；它们是官方流水线内部依赖，
不是额外的训练监督方法。首次推理会下载DINOv2。未安装会替换Torch的xformers，
两个模型已使用现有Torch实际完成推理。源码和权重位于忽略的third_party下。

本次实际权重revision：DA3-Large `c54c26b16ec04d218e8d584ecf4bce082a9fcc20`；
StableNormal `4362d11c636ef42ed117dd76984c7f2d83e6abd9`；
YOSO `2202fcb69960d94b437e06c19c556a12ceeb57c0`。
重新下载复现时可给上述hf命令增加对应的 `--revision`。

## 运行

以下为train灯光留出模式；预处理与训练必须使用相同scene、resolution、划分。
针对SSS场景增加 `--unit-light-intensity 1 --resolution 256`；白底场景增加
`--background 1`。全train实验给两个预处理命令和训练命令都加 `--fit-all`。

```bash
export CUDA_VISIBLE_DEVICES=0  # 先确认该GPU空闲
export HF_HOME="$PWD/third_party/hf_cache"
export TORCH_HOME="$PWD/third_party/torch_cache"
SCENE=/workspace/datasets/SSD-GS/data/Real_NRHints/Cat
PRIORS="$PWD/runs/surface_priors/Real_NRHints/Cat"
.venv/bin/python prepare_surface_priors.py --kind normal --scene "$SCENE" --output "$PRIORS"
third_party/da3_env/bin/python prepare_surface_priors.py --kind depth --scene "$SCENE" --output "$PRIORS"
$PYTHON train.py --scene "$SCENE" --output runs/cat_surfel_s0 --representation directional_surfel --surface-priors "$PRIORS"
$PYTHON evaluate.py runs/cat_surfel_s0/last.pt --split validation --output runs/cat_surfel_s0/validation --lpips
```

继续复用configs/validation.json批量入口：train.representation设置directional_surfel，
train.surface-priors可写绝对路径模板 `.../surface_priors/{family}/{scene}`，构造manifest时展开。
默认由用户提前生成先验。配置包含 `surface_preprocessing` 时，既有队列会在每个
场景训练之前依次执行normal/depth预测：该项指定normal_python、depth_python与
processing_resolution；模型权重需已下载。预处理与训练共享同一GPU工作槽，
不会额外占用第四张GPU。不要覆盖旧实验结果。`eval_split` 默认test，亦可显式设置validation。

## 依据

- [2DGS](https://surfsplatting.github.io/)：圆盘几何、光栅化及几何正则。
- [StableNormal](https://github.com/Stable-X/StableNormal)：冻结法线教师。
- [作者的2DGS接入代码](https://github.com/hugoycj/2d-gaussian-splatting-great-again/blob/main/utils/camera_utils.py)：原始法线取负后转到世界坐标的约定。
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)：冻结单目相对深度教师。

监督组合、权重及PORT圆盘阴影适配属于本项目实现，不是论文效果复现声明。
# 2DGS 多特征通道修复 — 2026-09-22

当前安装的gsplat 1.5.3会用torch.empty把不支持的通道数补到二次幂。
本项目的36维属性+深度共37通道，因此补齐区域可能包含NaN或极大值；
这些通道虽不出现在最终图像中，仍可能通过CUDA反向传播污染几何梯度。
单圆盘、36属性通道、未初始化内存填NaN的检查能稳定复现。

renderer._rasterize在调用原生2DGS前显式补零，并在深度追加前完成填充；
返回时恢复原属性与最后的深度通道。未修改或升级共享gsplat/PyTorch环境。
原失败的step6559批次重放后所有参数梯度有限，前向图像及alpha与归档实现最大差为0。
回归检查加入现有test_method_integration.py，两个2DGS方法的短训/重载/评价通过。
证据在runs/surfel_diagnosis/dragon_attention_failure/和runs/surfel_padding_check/。
该修复处理数值错误，不改变预期光传输公式；不能据此断言它解释了全部历史质量差异。

## 相机条件多视图DA3（2026-09-23开发）

prepare_surface_priors.py增加`--depth-views 8`，复用现有DA3-LARGE权重与da3_env。
每批最多8张train图像：4个通过训练相机中心最远点采样得到的共享锚点，加4个目标帧；
重复项去重。已知OpenCV world-to-camera外参与对应输入尺寸K传入官方API，
使用first参考视角和align_to_input_ext_scale=True。锚点贯穿各批，目标只保存一次。
`--limit`限制输出目标，不把验证/test图像作为上下文；fit-all仍遵循原训练划分。
默认depth-views=1保留单图推理。输出记录每帧上下文、相机元数据和尺度对齐标志。

已有surface_losses仍使用消除全局log尺度的相对形状损失；暂不把模型尺度估计当作绝对几何GT。
新先验另存，保留旧先验与checkpoint。训练采用前需要先完成少量真实帧的相机尺度/法线质量检查。

## 可选逐像素交点深度

`--surface-depth intersection`使用renderer.intersection_depth复用gsplat提供的可见贡献索引，
按原像素内顺序重新计算alpha、透射率和局部射线/圆盘交点Z，再做期望深度合成。
全局分段log透射率扫描用float64保持精度；其余遵循原float32栈，梯度到位置、旋转、尺度、opacity。
2D低通过滤支撑采用中心Z，排序仍是原中心深度；distortion和深阴影仍是原代理，未宣称全渲染物理精确。
新的receiver位置、SDF采样及表面深度监督共用交点深度。颜色属性和alpha仍由原CUDA路径输出。
Gaussians.surface_depth由训练配置与checkpoint恢复；旧记录无该字段明确对应center，默认不变。
本路径是较耗内存的稀疏可微参考实现，尚无质量胜出的结论；不改共享依赖，亦不需要新环境。
