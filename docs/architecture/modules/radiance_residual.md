# 光照条件的GS前景响应残差

状态：2026-09-23，模块、训练和评价接入完成；CPU与真实场景短测通过。
研究依据见 [方案](../../research/gs_radiance_residual.md)，固定实验见
[gs_radiance_residual_pilot](../../experiments/gs_radiance_residual.md)。

## 插入点与冻结边界

模块对当前 `neural_material` 的主GS渲染结果添加单点光源条件的有符号响应修正。
插入点为renderer已有transport前景输出之后、alpha/background合成之前，
复用原接收点反投影、alpha除法、世界坐标混合法线和已有覆盖mask。
正式训练使用renderer显式可选接口；诊断hook不能成为训练接口，不另复制接收点计算。

```text
原GS光栅化与receiver → 原transport前景L_GS
receiver/法线/目标灯光/视角 → 新响应残差R
L_new = max(0, L_GS + I / (light_scale * d²) * R)
原alpha与background合成 → 既有观察变换
```

原GS属性、transport/decoder/PORT、相机、light_scale及源场景center/radius固定。
新模块只训练自己的MLP和位置网格；原alpha和GS覆盖mask不变。
未覆盖的像素不查询残差、不根据GT补出表面。
原foreground参与加法但不是残差网络的输入，不能通过它泄漏teacher RGB特征。

## 特征、法线与输出结构

归一化位置为 `(receiver_world-center)/radius`，wi为接收点指向实际点光源的单位方向，
wo为接收点指向相机的单位方向，光位置使用同center/radius归一化。
94维输入沿用已有辐射特征结构：

| 特征 | 维数 |
|---|---:|
| 位置，4频率sin/cos及原始3维 | 27 |
| wi，4频率sin/cos及原始3维 | 27 |
| wo，4频率sin/cos及原始3维 | 27 |
| 冻结法线n | 3 |
| n·wi、n·wo、n·half、wi·half | 4 |
| 三尺度log高光提示 | 3 |
| 归一化实际光位置 | 3 |

位置/方向采用现有四频 `[1,2,4,8]` 的π编码。
三个高光提示为 `log1p(b*(1-clamp(n·half,0,1)))/log1p(b)`，
`b=[100,1000,10000]`，停止梯度；不启用上一轮失败的额外hint PE。
不加入frame ID、GT、原GS RGB或新材质标签。

新MLP为94输入、三个128维ReLU隐藏层、RGB Linear输出。
使用新的默认 `SpatialDetail(128)`：12层×2通道、分辨率16..256，
纯PyTorch dense/hash三线性网格及24→128投影，在第一层Linear之后、ReLU之前相加。
它复用现有结构和依赖，不迁移SDF头或网格权重。

两种法线模式只影响新头：geometry用原 `receivers['normals']`；
material用原 `NeuralMaterialTransport.material_normal(receivers)`。
后者由已有材质修正码产生有界切向偏移并归一化，没有新法线参数。
原GS着色路径始终保留其原材质法线，不因选择geometry而改成几何法线着色。
接收点、全部法线与原foreground均停止梯度；新头不得反向调整场景或相机。

## 零残差与光照条件契约

隐藏层随机初始化，RGB末层weight和bias全零，输出有符号值，不经Softplus。
SpatialDetail按既有默认初始化；末RGB层为零保证初始响应残差为零。
不要把整个网络置零；第一次反传主要更新末层是预期行为，后续隐层/网格须有实际梯度更新。

原前景加上 `I/(light_scale*d²)` 倍响应残差，再做非负截断。
不增加visibility、余弦、门控或正则项；原transport已有直接光、阴影与PORT响应保持。
同一光位置下，原路径和残差对正标量光强缩放应保持正齐次，零光下前景为零。
由于残差允许负值且总前景截断，不保证任意多光叠加线性、BRDF互易或能量守恒。
这是实际目标灯位置/入射方向条件的有效响应修正，不能等同于仅视角NVS网络。

## 抽样训练与完整评价

训练从原source bbox有效射线域按既有GT峰/邻环/前景规则取得像素ID，
只查询其中GS覆盖的接收点。损失只对抽样像素的最终观察空间RGB做L1。
重复ID必须按原抽样次数计权；即使内部去重计算，也必须恢复重复项参与loss，不能改变目标。
未覆盖ID维持原alpha/background路径，不新增可学习颜色。

评价对全部GS覆盖像素分块查询，使用与训练相同的响应与观察变换。
继承原PNG“前景gamma后alpha合成”的契约，不能对已合成RGB重复gamma或混用HDR路径。
精确学习率及实验预算由canonical训练配置管理；本轮末端lr为.00028，见实验协议。

## 状态保存与诊断

新头以独立state/config保存，不改变原GS/transport参数键。
配置须明确残差启用状态和法线模式；启用却缺少权重、形状不匹配或模式不一致时显式失败，
不能静默忽略残差。评价按保存配置构造并严格载入，报告实际渲染分支名称。
旧checkpoint未启用残差时保持原行为，既有七种方法和默认方法不因新增模块改变。

记录原前景、响应残差与光度修正的幅度，以及clamp前负通道比例，
明确是抽样区域还是完整GS覆盖区域、像素/通道分母，避免把大范围抵消误当精细修复。
质量同时看整体、峰定位/对比、小亮核、邻环和纹理，不只看训练loss或峰幅值。

实际接口：`RadianceResidual(center, radius, normal_source)`，`normalized(points)`与源bbox采样复用；
`renderer.render(..., radiance_residual=None, residual_indices=None)`复用原receivers。
公开纯Torch算子`apply_radiance_residual`用于算子检查；完整查询按≤4096接收点分块。
`evaluate.load_radiance_residual(checkpoint)`严格载入；单独`load_model()`仍只返回基模型。
新checkpoint保存`radiance_residual`、`residual_steps`和`config.residual_normal`；
遗漏启用flag、状态缺失、续训切换normal均失败。仅支持无SDF着色/normal-field的neural_material源。
源surface_depth、shadow_mode、background、display_gamma、unit_light_intensity与已启用的shadow/PORT自动继承。
原相机offset若存在则保留；残差训练强制冻结相机并从首步应用offset。
`--init-checkpoint`恢复权重及累计步数，Adam、随机序列与LR日程重新创建，沿用现有训练契约。

`residual_stats`含unique查询数、覆盖数、请求数、请求中覆盖数（含重复），
`clamp_fraction`、`delta_abs_mean`和`base_abs_mean`均以被查询RGB通道为分母。
完整评价保存逐帧统计，再按queried_pixels加权聚合；训练统计只涵盖当步unique采样覆盖点。
没有统计原始未乘光强的R幅度，不应将光度delta幅度误作该量。
训练bbox有效域与评价全部GS覆盖域可能不同，域外残差缺少本轮直接监督。

5项CPU检查及`runs/gs_radiance_residual_smoke/verification_retry/report.json`通过：
128px真实场景两法线模式的零残差线性/观察RGB和alpha保持、零光/正光强倍增、
强制原模块可训练时梯度仍隔离；native512/2048像素的3+3训练/续训中主状态逐位固定，
新头更新并严格重载，CLI评价使用gaussian_residual。CPU覆盖4097点跨chunk和重复ID损失权重。
初次短测有测试预期张量shape错误，修复记录与失败log保留，生产代码未因此改变。
这些是实现契约检查，不是重光照质量或真实几何验证。


## 训练帧子集诊断

`--residual-frames 0 35 70 105`只限制本stage新残差的实际random.choice池，
唯一索引必须属于源fit集合，顺序保持；不改变源fit_indices/val_indices、归一化、光强或相机映射。
不传参数时（包括resume）使用源全部fit，不隐式继承上一stage的子集。
只允许残差模式，CLI/helper拒绝重复、空或域外索引。
checkpoint保存`residual_sample_indices`及`residual_frame_counts`，计数在成功更新head后递增。
计数只涵盖当前stage，resume从0开始；`residual_steps`仍累积，不能混淆二者。
history同步记完整stage计数。来源仍由原fit成员与init_checkpoint链决定，不把少帧优化改写成少帧从头来源。
真实四帧池3+3检查通过，见`runs/gs_residual_subset_smoke/verification/report.json`；
实际两stage各计数{0:1,35:0,70:0,105:2}，源562项fit及全部相机/主模型逐位保留。
[少帧重复拟合协议](../../experiments/gs_residual_subset.md)。


## 完整小亮核评价

`evaluate.py --highlights`保留原全峰指标，另用量化RGB的uint8字节在CPU重建观测图，
生成GT8连通组件的1–4/5–16/>16px桶与raw sums；这样与保存PNG的离线诊断一致，
避免GPU除255与CPU除255的末位差异。原RGB/PSNR/LPIPS/全峰metric路径不变。
每帧和完整split都有`neutral_peak_components`，先汇总raw sums再派生MAE/对比/像素recall/组件any-hit。
这是光亮点图像代理，宽斑也可能满足any-hit；需与定位precision、邻环和整图质量共同使用。
[完整数据预算协议](../../experiments/gs_residual_budget.md)。

## 可选空间—方向交互

`--residual-interaction none/add/multiply`默认none；旧checkpoint缺该字段按none严格载入。
新头的k=同一次位置网格查询24维、a=94维输入中去除27维位置编码后的67维。
无bias的`interaction_spatial` A24→16、`interaction_angular` B67→16、
`interaction_projection` W16→128在第一层Linear+原网格投影之后/ReLU之前增加
W(Ak+Ba)或W((Ak)⊙(Ba))。两组各3504新参数，初始化顺序和张量完全相同；末RGB保持零。
add可以吸收到原首Linear和网格投影，是重参数化控制；不声称同参数量等于同有效容量。

head的`forward(..., return_stats=True)`显式返回输出及detached统计；默认仍只返回tensor。
renderer为启用交互的头收集`interaction_{grid,spatial,angular,output}_{abs,square}_mean`，
每块按唯一query像素数加权，全图同理；均方开方得到RMS，不能直接平均各块RMS。
空查询统计为零，没有共享可变last_stats。none不增加统计键，也不改旧数值路径。

新建残差在训练前保存`residual_initial.pt`（config、state、residual_steps=0），续训不新建初始化快照。
mode与normal均须在续训时显式匹配；评价按config构造后严格载入，不能把有新权重的头静默当none。
训练日志`residual_optimization_stats`记录A/B/W、table、原投影和首末层参数/梯度RMS；
参数为当步更新后值，梯度为该步反传值。无梯度用null与数值零区分，零末层导致首步内部梯度为零属预期。
[受控协议](../../experiments/gs_residual_interaction.md)。

## 可移动外观中心与球面高斯特征

可选--residual-angular-bank wide/narrow要求multiply交互及geometry输入；默认none不增加权重键或改变旧路径。
center_network以现有27维位置PE和同一次24维grid查询为输入，51→32 ReLU→3，末Linear为零。
它生成d，并以m=normalize(n_geometry.detach()+d)作为新外观核中心；m不传给主shader或几何模块。
8个核K=exp((clamp(m·h,−1,1)−1)/tau_rad²)经无bias的8→128投影，加入首ReLU之前。
中心网络1763参数、核投影1024参数；总head2200784。原位置/方向/法线/cos/log提示及multiply路径均保留。

wide的tau为8°–128°、narrow为2°–32°，都是8个对数均匀尺度，CPU构造后转设备。
这保证GPU保存、CPU重载时固定buffer一致；train/evaluate核对其与配置一致，续训不能静默换bank。
SG不使用可微acos；tau表示局部曲率角度尺度，大角度处与exp(−theta²/(2tau²))不同，不是严格角度标准差。
初始末RGB零使最终图像保持，中心末层零使初始d为0；各模式隐藏特征可不同。

记录每核均值/均方/零值比例、核投影各列梯度RMS、中心两层和投影的参数/梯度RMS。
中心转角仅detach后用atan2(||n×m||,n·m)计算；另外记录offset均方、raw中心范数min及<.1比例。
renderer跨非空chunk取center_raw_norm_min的min；evaluate.aggregate_residual_stats跨非空查询帧取min。
空查询min为0哨兵，不能拉低其它非空帧的min；其它比例/均值继续按查询像素加权。
没有依据这些诊断动态改变带宽、学习率或加入正则。

4项新增CPU和两组真实3+3/重载/CLI检查通过，既有9项残差检查保持。
宽/窄实验已完成，未通过继续门槛，见[协议与结果](../../experiments/gs_residual_movable_centers.md)。

## 可选局部成对RGB监督

`--residual-paired-context`让每个GT峰锚点与自己的11×11环像素配对。候选均在rays.valid中，
环要求GT alpha>.9并排除所有GT峰；从有非空环的峰像素均匀抽样，再从它自己的环抽一点。
`residual_sampling.py`缓存GT候选池，不缓存接收点或渲染结果。组件桶只用于记录覆盖，不改变抽样概率。
N次draw包含N/4峰、N/4对应环、N/4前景、N/4有效域；无合格峰时沿用前景/有效域各半回填。
重复draw仍重复计权，全部pair端点在renderer的residual_indices中；默认关闭保持旧全局环采样路径。

普通RGB L1仍用全部N个draw。额外项在未量化观察RGB上计算两端预测差与GT差的逐通道L1，
仅包含两端冻结GS alpha均>0的pair，始终除以3N/4；不重采样或按有效pair数重新归一化。
`--residual-pair-weight`乘该项，0权重也计算并记录原pair值。无pair/无支持时返回连接预测图的零值。
mode和weight保存到config，续训须显式匹配；它们不改变推理公式或head state。
日志记录原始/加权pair loss、名义/实际/支持pair数、退化次数、组件抽样覆盖。
checkpoint和residual_pair_audit.json保存本stage的累计计数、逐帧资格与组件覆盖；续训stage重新计数。
[预定受控协议](../../experiments/gs_residual_paired_loss.md)明确两组共同配对，仅权重0/.25不同。
