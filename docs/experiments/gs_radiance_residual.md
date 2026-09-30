# 固定主GS的光照条件响应残差对照

日期：2026-09-23。状态：实现、短测、两组正式3k/16fit/71test评价及独立CPU终端审计全部完成。设计依据见 [研究决策](../research/gs_radiance_residual.md)，
模块实现见 [响应残差](../architecture/modules/radiance_residual.md)。

## 目的与共同控制

此前SDF体分支经过细节网格、原生512px、峰/邻环采样、射线预算和固定厚度对照，
仍有宽斑和错位，整体训练拟合明显低于固定主GS；较薄体支撑还降低1–4px小亮核的对比和召回。
本轮保留主GS已有纹理和辐射结果，只学习它的有符号、目标灯光条件误差修正。
它不是把整幅GS颜色替换为SDF辅助头，也不是仅视角条件的固定照明NVS残差。

Run为 `gs_radiance_residual_pilot`，场景 `Real_NRHints/Pixiu`，variants为 `geometry`、`material`。
两组共同源为 `runs/sdf_volume_detail_pilot/geometry_grid/Real_NRHints/Pixiu/last.pt`，
仅使用其中固定的主GS/传输/相机及场景归一化；不继承已训练SDF辐射头或外观网格。
同源原GS是零残差控制，无须重训。源GS已证明等于既有40k主模型，
控制与候选均应按本次固定评价协议渲染，而不是借不同分辨率的历史分数混比。

## 两组唯一差异：新头输入的冻结法线

| 分支 | 新残差头使用的法线 |
|---|---|
| geometry | 原 `receivers['normals']`，世界坐标GS几何混合法线 |
| material | `NeuralMaterialTransport.material_normal(receivers)`，包含已有有界着色法线修正 |

material法线来自原材质features中的修正码（以及源配置若存在的normal residual），
在几何法线切平面内施加既有有界偏移后归一化；没有新增法线自由度。
原GS颜色分支在两组中始终使用它原本的材质着色法线，
只有新头输入改变。两头使用同seed的相同新权重初始化、相同训练帧序列和预算。
本实验检验已有材质拟合法线对残差学习的作用，不把它当真实几何法线优劣比较。

## 残差定义与零初始化

在原transport前景颜色之后、alpha合成之前使用：

```text
x = (receiver_world - center) / radius
L_new = clamp_min(L_GS + light_intensity / (light_scale * distance_squared)
                       * R(x, n, wi, wo, normalized_light_position), 0)
```

新头使用已有94维位置/方向/法线/四余弦/三个log高光提示结构、3×128 ReLU隐藏层，
以及默认12层、每层2通道、分辨率16..256的 `SpatialDetail` 网格。
隐藏层随机初始化，末RGB Linear的weight/bias全部为0，输出有符号值，不使用Softplus。
不能把整个网络置零；首步末层先学习是零输出初始化的正常梯度行为。
新SpatialDetail按现有默认方式初始化；不迁移SDF头的隐层或网格。

原GS全部属性、原transport/decoder/PORT、相机和light_scale完全固定；
原直接光和PORT贡献保持，原alpha保持。只训练新残差头及其新位置网格。
不输入teacher RGB、GT或frame ID；不增加visibility/cosine/gate乘子、新正则或新几何loss。
接收点、原foreground、法线与归一化量停止梯度，沿用既有PNG前景gamma/alpha观察模型。
同一灯位置下正标量光强缩放应保持正齐次；非负截断不保证任意多灯相加线性，
本头是单点光源有效响应残差，不称内在BRDF或能量守恒材质。

## 固定训练协议与准确学习率

两组native512、3000更新步、seed0、每步2048抽样像素。
使用与已有source bbox相同的有效射线域和采样规则：
25% GT峰+25% 11×11非峰邻环+剩余前景/定义域各半，
即512峰+512邻环+512前景+512有效域。
邻环由GT峰11×11 max-pool膨胀减去峰本身、alpha>.9得到，Chebyshev半径5。
专用池与有效域取交集，空池余量按原规则分配；全部有放回采样。

只对所选像素的观察域RGB计算L1，不叠加SSIM、mask、depth/normal、feature或残差正则。
重复像素ID仍按每次抽样计入损失，不以去重平均改变权重。
新头只查询被采样且原GS覆盖的像素；alpha为0的地方保留原背景，不用GT补足未覆盖区域。
评价查询全部原GS覆盖像素并分块，必须与训练共用相同残差着色函数。

采用当前train源码的网络学习率日程，geometry warmup=0、decay steps=3000：

```text
lr(step) = .001 * (.2 + .8 * .1 ** (step / 3000))
lr(3000) = .00028
```

不要将末端值误写为.0001。两组相同optimizer/日程，实际初末lr进入日志和最终审计。
每组总像素抽样数6,144,000，抽样次数不等于唯一像素数。
单点残差头的查询成本远低于先前SDF体分支，但不宣称相同FLOPs、相同训练成本或已收敛。
记录真实时间/显存，而不是用像素数直接推断成本。

## 短测与正式启动条件

复用ssd-gs/CUDA12.1，不升级共享依赖，启动前检查空闲GPU，最多同时两张。
在现有测试/训练入口完成以下最小检查，未通过不声称正式实验成功：

- 零末层下原GS线性RGB、观察RGB和alpha保持，含两种法线模式。
- 零光、同灯位置正标量光强缩放及预合成/合成路径一致。
- 梯度只进入残差；GS/主传输/相机/light_scale逐位固定，残差参数有限且确实更新。
- native512、真实2048像素的3+3步训练、续训、严格保存重载与CLI评价。

独立state/config明确保存头和法线模式；启用残差但缺少权重须显式失败，不能静默回到原GS。
原方法默认行为保持，未启用时不增加残差参数或改变旧推理路径。
每个实验保存配置/源码/命令，启动以实际训练子进程、关键日志和GPU活动共同核验。

## 固定评价：fit与开发期未训练帧

先完成同16个训练帧的native512拟合评价：
`[0,35,70,105,140,175,210,245,280,315,350,385,420,455,490,525]`，
使用保存的训练相机校正，GT峰池5390像素。
固定两组3000步终端checkpoint后，再以原始test相机/灯光完整评价全部71个official-test帧，
同时评价原source GS控制；不按test挑中间checkpoint、调整阈值或选择有利帧。
不做test曝光、相机、灯光、法线或先验拟合。

完整来源审计表明这71帧没有进入当前模型链拟合，但测试集曾被旧研究开发观察，
因此明确标为“开发期完整71帧未训练图像评价”，不是新的独立盲测。
fit使用过全部562训练帧及其上游信息，不能从中事后抽新的held-out集合。
光照条件输入本身不能证明新灯光泛化；训练收益必须与完整test结果分别报告，含全部回归。

两种划分均报告整体PSNR/SSIM/LPIPS/alpha L1和标准峰RGB MAE、局部对比/GT、
2px precision/recall及峰计数，按共同GT定义汇总。
训练诊断另记录固定首4帧0/35/70/105的11×11邻环/远处前景，
以及1–4px、5–16px、>16px连通亮核面积桶；保留相同GT裁剪。
若追加test分桶或邻环，必须单列其实际帧集合和池大小，不把4帧诊断写成完整16/71帧统计。
any-hit组件或recall增加不能代替准确窄峰形状，需共同看误报、对比、邻环和全图纹理。

记录光度修正的平均绝对幅值、原foreground幅值，以及clamp前负通道比例；未统计未乘光强的R幅值。
训练统计注明抽样像素、覆盖像素及通道分母；评价记录全覆盖像素同类统计，
防止将大量抵消、截断或纹理破坏隐藏在平均loss里。

## 入口、输出及当前状态

使用canonical `configs/validation.json` 的新run名称和上述两组参数，复用：

```bash
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS
bash launch_validation.sh configs/validation.json
```

训练/评价仍经已有train/evaluate/manifest/benchmark入口，正式渲染使用显式接口，
不使用临时hook作为训练数据接口，不创建新启动器。
计划输出为 `runs/gs_radiance_residual_pilot/{geometry,material}/Real_NRHints/Pixiu/`，
保留原GS控制、fit/test全部指标、图像/局部裁剪、残差诊断、参数审计、配置与源码快照。

5项CPU测试通过，包括4097点跨chunk、重复ID原权重、strict state。
真实短测位于`runs/gs_radiance_residual_smoke/verification_retry/report.json`：
128px算子契约与native512/2048/.25+.25的3+3真实训练均通过；主GS/传输/相机逐位相同。
初次检查因测试背景预期shape为1通道失败，修为RGB广播后通过；初次日志保留，无生产模型修改。
GPU0/1启动前均14MiB、0%、无compute app，两个真实train子进程及100/300步日志已核实。
原GS控制在`runs/gs_radiance_residual_reference/{fit,test}`，已重新完成同协议16/71帧评价。
manifest的`eval_test:true`只在终端fit后追加完整test，不继承fit limit16，不评价中间checkpoint。
正式结果与参数审计已完成，详见下文。


## 终端质量结果

所有指标均为native512、uint8量化观察空间；fit使用保存的训练相机修正，test使用原相机。
同16fit GT峰池5390像素；完整71test GT峰池26147像素。峰为图像代理，不是物理高光GT分解。

| 同16fit | 原GS | geometry residual | material residual |
|---|---:|---:|---:|
| PSNR |26.078235|25.796121|25.890630|
| SSIM |.898818|.889462|.890696|
| LPIPS |.118444|.125509|.124059|
| GT峰RGB MAE |.158159|.140622|.142565|
| 峰对比/GT |.451671|.447486|.451300|
| precision@2px |.673069|.656343|.662723|
| recall@2px |.603525|.630427|.629685|

| 完整71未训练帧 | 原GS | geometry residual | material residual |
|---|---:|---:|---:|
| PSNR |21.331654|21.376813|21.342007|
| SSIM |.843040|.836821|.837339|
| LPIPS |.152835|.156177|.155676|
| GT峰RGB MAE |.358917|.332993|.335391|
| 峰对比/GT |−.005892|−.004879|−.005549|
| precision@2px |.178755|.166625|.166032|
| recall@2px |.179523|.181512|.180518|

geometry在fit峰MAE下降11.09%，却使PSNR下降.282dB、局部对比略降；material趋势相同。
完整test峰MAE下降7.22%/6.55%，但局部对比仍负，即GT峰位没有形成局部亮核。
PSNR仅+.04516/+.01035dB，SSIM和LPIPS均退化、precision下降；不能称准确高光恢复或全面重光照改善。
因此两个候选都不设为默认，也不把geometry的微小PSNR增益作为成功。
本轮主渲染输出确实改变，终端审计确认源主参数逐位不变；没有新几何自由度或真实几何质量证据。

两组fit全覆盖截断通道比例9.352%/8.237%，test为9.410%/7.417%；
test平均绝对光度delta/base约7.671%/6.946%（先聚合幅值再相除）。
这是全部覆盖点等RGB通道加权统计，包含低alpha边缘/暗区域，不能直接解释为可见图像同等比例变黑。
完整evaluation记录按像素加权，不能用单个训练步的截断率替代。

训练实际179.56/187.00秒，最大allocated约7.628/7.628GiB；每组6,144,000次抽样，
实际采中GT峰1,549,317次、邻环1,769,969次，含基础前景/域采样额外碰到的峰/环。
抽样次数不是唯一覆盖数；两组相同最后帧428及日志帧序列，终端lr .00028。
这只隔离两种冻结法线输入，不是与原GS匹配追加训练预算的因果对照。

## 未训练数据的实际新颖性

`light_camera_overlap.json`核查完整71×562 metadata及float32 loader坐标：
灯位、相机中心和完整pose均无严格或1e−6容差重合；光强则全部相同RGB `[6.644080162]*3`。
最近训练灯位距离按source radius=1.402038288归一化，min/median/max为.121991/.312993/.814789；
灯方向最近夹角.401°/2.028°/5.105°，相机光轴最近夹角.152°/1.694°/5.961°。
可称新点光源位置与新视角共同变化的未训练帧评价；不是新光强、固定视角纯重光照、远分布光照或新盲测。
来源链和先验未使用test的既有审计仍有效；本次metadata重合检查不取代完整训练来源审计。


## 小亮核、邻环、逐帧与视觉审计

以下桶和邻环严格仅4fit帧`[0,35,70,105]`与4test帧`[0,1,2,3]`，不冒充全16/71帧。
面积按同一GT的8连通组件计算，2px匹配与前述主指标一致。

| 1–4px小亮核 | 原GS | geometry | material |
|---|---:|---:|---:|
| 4fit：168组件/307像素，RGB MAE |.204386|.185706|.190939|
| 4fit：对比/GT |.323631|.325096|.330644|
| 4fit：命中像素数 |108|105|116|
| 4fit：至少一个命中组件 |55|53|58|
| 4test：123组件/253像素，RGB MAE |.375851|.334842|.336842|
| 4test：对比/GT |.000564|.008869|.009606|
| 4test：命中像素数 |8|8|8|
| 4test：至少一个命中组件 |7|6|6|

4test最小桶recall均3.162%，准确小峰像素命中没有增加；局部MAE下降不足以证明窄峰恢复。
4fit的material小桶有小幅收益，但不能推广到未训练数据，更不能称几何精度改善。

4fit的11px邻环24920像素：neutral正向误差.063288→.073256/.070189（+15.75%/+10.90%），
RGB MAE .086615→.088985/.088308。4test邻环26014像素：neutral正向误差
.053316→.067805/.070115（+27.17%/+31.51%），尽管RGB MAE由.162282降为.148971/.152413。
4test远处前景196304像素RGB MAE .097757→.099409/.101230；亮度误差降低伴随外溢和纹理代价。
没有把正向误差的均值当成所有像素都更亮；完整带符号、分区域指标在comparison.json。

完整71test：geometry/material的PSNR改善帧数仅21/17，中位变化−.028452/−.042606dB；
LPIPS改善帧数9/8，中位变化+.002610/+.002448。
极少量平均PSNR增益不能概括为多数测试图改善。

`comparison.png`与`comparison_crops.png`使用固定帧和GT最大/中位组件位置，未按预测挑有利裁剪，
无单图对比调整。视觉复核仍见宽斑、错位以及未恢复的小白点，和分桶指标一致。

## 终端冻结审计与可复现产物

`final_audit.json`：两checkpoint的330930个GS、全部transport/decoder/光强buffer、相机offset
均与原source逐字节相同；全部张量有限；residual_steps=3000，末RGB层和detail投影已非零。
原split562fit/0val保留；两配置仅residual_normal/output不同，首次loss与31条日志帧序列相同。
日志仅每100步及首步记录，不把31条日志误称完整逐步帧轨迹；累积峰/邻环抽样数一致。
完整16/71帧的共同GT、帧号、峰计数相同，汇总指标和按query像素加权统计重算一致；
alpha_L1逐帧与原GS相同，test corrected_fit_frames=0。

固定基线`runs/gs_radiance_residual_reference`包含同协议fit/test、commands.json与source.tar。
本run含`manifest.json`、`validation.json`、`source.tar`、两组checkpoint/history/log/fit/test；
根目录另有`comparison.json/png`、`comparison_crops.png`、`small_peak_audit.json`、
`final_audit.json`、`analysis_source.tar`、`light_camera_overlap.json`及其源码归档。
CPU审计首次执行通过；短测的首次shape断言失败和修复记录单独保留在smoke/summary.json。
所有训练、评价与GPU审计均结束，GPU0/1释放。没有设置默认、没有中间test选点或测试适配。

## 下一项：单帧与少帧重复拟合诊断（尚未实施）

完整数据上的峰对比几乎未变，尚不能区分监督覆盖、共同拟合、损失/截断或表达限制。
先检验当前头本身能否在固定少帧产生窄峰；暂不叠加更多频率编码或新几何。
从同一原source重新初始化零头，固定geometry法线，仍512px/2048像素/.25峰+.25邻环/seed0。
实际残差抽样池设`[0]`与`[0,35,70,105]`。选单帧3k与四帧12k，分别decay_steps3k/12k，
大致匹配每帧更新曝光；必须记录各帧实际次数，并明确总步数/成本不匹配。

只新增显式残差抽样子集参数，保留完整fit_indices/val_indices及source相机映射，
另存本轮采样池/计数；禁止把已有562帧来源改写成单/四帧从头来源。
同16fit评价保留，其中前4帧和共同frame0为主要诊断；其它帧只是本轮修改外溢诊断，不是held-out。
本诊断不追加official test。若单帧成功可说明局部拟合能力存在；若单帧失败也不能直接认定容量上限，
仍需检查优化、截断、固定alpha覆盖与接收点。四帧失效也不能单因果归于视图冲突。
