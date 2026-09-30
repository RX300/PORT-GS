> **2026-09-30 方法清理更新：** 当前只保留 `directional_port_v1`、`surface_attention`、`neural_material`。
> 其余方法和独立几何训练入口已删除；已有 GGGS 几何读取/诊断和三种方法的公共依赖保留。
> 以下涉及已删除方法的内容仅作历史记录，不是可运行入口。清理详情见 `docs/experiments/method_retirement_20260930.md`（项目根目录相对路径）。

## 2026-09-27：GGGS初始化＋默认/DNA联合优化两轮完成

2026-09-28：100万上限实验完成，Cat/Pixiu各30k，测试PSNR22.52035/21.53561；增加点数未解决质量瓶颈，不推广。[对照与几何](gggs_neural_material_1m.md)。

本轮 **GGGS初始几何＋Neural Material** 已完成：Cat/Pixiu各30,000步，几何与材质持续联合优化，共享神经BRDF解码器冻结。
完整测试PSNR为22.6227/21.5711dB，SSIM为.779711/.850532，LPIPS为.239291/.157998。
较GGGS＋DNA的PSNR提高.078/.386dB，但细节、高光和表面粗糙化仍未解决，**不替换默认参考**。
[完整结果、对照图与几何/换光预览](gggs_neural_material_joint.md)。本轮训练和评价均已结束，无本轮运行任务。


四个模型各30k，全部test/fit完成，按用户要求均允许继续优化几何。
默认Cat/Pixiu：21.81926/.770223/.228127、20.54358/.845065/.155993；
DNA：22.54456/.774314/.240020、21.18482/.846024/.158111（PSNR/SSIM/LPIPS）。
DNA相对新默认PSNR提高，但LPIPS更差；相对旧DNA也是混合结果，尚未解决毛发/小高光问题。
两种渲染后端下固定几何代理均未保持初始GGGS表现；不推广，不宣称真实3D精度改善。
[完整报告/图片](gggs_dna_joint.md) · [全部DNA数值与核对](gggs_dna_joint_results.json)。

## 2026-09-27：GGGS初始化＋默认重光照联合优化完成

两场景各30k、完整test/fit完成。Cat21.81926/.770223/.228127，Pixiu20.54358/.845065/.155993。
相对旧默认，Cat PSNR+.283dB、LPIPS略好，Pixiu基本持平；灰模更噪、固定视角轮廓指标下降。
未解决整体几何/材质质量，不替换默认参考。随后按用户要求独立启动DNA联合实验。
[完整默认实验结果/图片](gggs_default_joint.md) · [数值/核对](gggs_default_joint_results.json)。

## 2026-09-27：GGGS固定几何＋局部逐贡献神经光传输

复用30k先验几何，Cat/Pixiu各30k材质，完整test66/71与fit522/562已完成。
测试PSNR22.51677/21.90448、SSIM.774560/.847826、LPIPS.263330/.174197。
较DNA PSNR+.37509/+.40152dB，但Pixiu LPIPS变差，两场景细节未恢复；不替换默认。
几何未更新，源几何形状缺陷保留。换光源预览可用，不等于物理材质辨识通过。
[报告/固定对照图/预览](local_transport.md) · [全部数值与核对](local_transport_results.json)。

## 2026-09-27：GGGS + StableNormal/DA3 对照完成

Cat/Pixiu各fresh30000，完整fit/validation；PSNR仅+.00318/+.00149dB，LPIPS略差，未可靠提点。
Pixiu轮廓外扩p95 16.48→13.91px、粗糙度29.09°→13.76°，但IoU/F1下降，形状仍不正确；Cat有新折痕。
统一修正评价坐标数值尺度后，旧GGGS和候选的全部fit/validation均重评，最终比较使用同一路径。
没有SDF、official test或新relighting训练；几何均未通过。
[实验、坐标修正与固定灰模](gggs_normal_depth.md) · [完整数值](gggs_normal_depth_results.json)。

## 2026-09-27：Gaussian Wrapping 与 StableNormal 完整对照

Cat/Pixiu无先验和加先验均各fresh30000，完整fit/validation完成，几何均未通过。
Pixiu加先验后边界粗糙度23.54°→19.67°，但F1 .31331→.29414、IoU .88150→.87920，尖刺和形变仍明显；
Cat无实质改善。未使用SDF/深度教师，未评价official test，未训练新relighting。
[实验、固定灰模与限制](gaussian_wrapping.md) · [全部数值/审计](wrapping_results.json)。

## 2026-09-27：剪枝与统计时序修复后的重训

Cat/Pixiu各四组独立30000完成。最终验证PSNR14.57945/18.29994、SSIM.71935/.84586、LPIPS.30067/.16377、IoU.94838/.89938；实际点数81018/37257。功能修复和真实分裂/剪枝计数验证通过，但灰模仍有过平滑、片层与尖刺，**几何质量未通过**。按用户“好的话再接relighting”的条件，本轮未训练新重光照分支。
[实验、图与保留位置](native_2dgs_topology_repair.md) · [全部数值](native_2dgs_repair_results.json)。

## 2026-09-26：先重建几何的原生2DGS对照

Cat/Pixiu各完成3组fresh30000，作者模型/CUDA、无SDF/预训练几何、内部470/52与506/56划分。原生默认控制产生不透明黑背景曲面，IoU .444/.207；alpha约束+深度畸变后.950/.904；坐标归一化后.950/.905，质量没有实质进一步提升。
最终验证PSNR14.5374/18.2816、SSIM.71802/.84538、LPIPS.31405/.16509；这是SH重建误差，不是重光照指标。灰模检查仍显示Cat过平滑、Pixiu片层/尖刺，**几何验收未通过**。仅保留最终两个模型及完整对照证据，前5套重光照最终模型保留。
[完整结果与图](native_2dgs_geometry.md) · [数值汇总](native_2dgs_geometry_results.json)。

## 2026-09-26：完整训练集诊断与重设计

当前PORT-DNA终端模型的522/562训练帧补评完成：PSNR22.531665/25.126019，tiny召回.1500%/.5071%。
终端40000点中54.94%/56.62%的opacity低于原生相机光栅cutoff；refine-stop=0未补充有效人口。
据此建议先验证几何容量、局部求值与跨条件一致性，再推进新材质。没有新训练或测试调参。
[诊断与来源](quality_diagnosis_20260926.md) · [分阶段重设计](../research/local_transport_redesign_20260926.md)。

> 2026-09-24清理更新：runs已按用户要求仅保留5套最终结果。中间实验、训练先验/种子和独立审计导出目录已删除；最终模型/指标/图片/源码仍在。历史路径仅作来源记录，重训缓存须重新生成。见[清理说明](runs_cleanup_final_only_20260924.md)。

## 2026-09-24：PORT-DNA-2DGS 全部结果已完成

无SDF或预训练几何，Cat/Pixiu各fresh30000和全部137test完成。
Cat PSNR/SSIM/LPIPS=22.141679/.776381/.267714；Pixiu=21.502959/.848252/.167022。
相对默认PSNR+.605494/+.924710dB、SSIM改善，LPIPS均变差，高光仍未可靠恢复。
固定图像、GT一致性、训练输入与权重有限性核验完成。默认不变。
[完整方法与结果](distribution_material.md)。

## 交点反射100k正式对照完成：未恢复小峰，2/8数值门槛通过（2026-09-23）

两组各100000步、全部12阶段于13:33:54UTC完成，CPU终端审计及固定8crop/4fullframe目视检查完成。
intersection仅PSNR、SSIM两项通过；tiny对比/recall/MAE、全峰precision、固定四全帧环域及LPIPS六项失败，人工false。
不追加official71、不延长训练、不推广默认；完整研究目标仍未完成。以下使用原evaluate指标，fit506与validation56分列，recall为比例。

| Split / mode | PSNR | SSIM | LPIPS | Tiny MAE | Tiny contrast/GT | Tiny recall | Peak precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation56 / aggregate |20.710851|0.835748|0.189349|0.412307|0.004432|0.000265|0.050584|
| validation56 / intersection |20.756309|0.841830|0.194711|0.417327|0.001574|0.000265|0.038462|
| fit506 / aggregate |21.517662|0.844556|0.183445|0.375719|0.005352|0.000164|0.042975|
| fit506 / intersection |21.411010|0.849832|0.188207|0.384767|0.003849|0.000109|0.097187|

验证tiny仅双方各命中1/3770px；contrast/GT下降.004432305→.001573793，MAE上升.005020371。
precision下降.012122119，LPIPS增加.005362457；PSNR+.045458dB、SSIM+.006082不足以通过全部条件。
固定4完整帧的19039px环域过亮.008547779742→.024173499343，增加182.804425%，不是只统计显示crop。
8个固定GT图块均未恢复准确窄峰；整图模糊、轮廓条带/面片与细节丢失，intersection在f148底座及f409红色身体增加错误白/青亮斑。

完整来源/RNG/预算/有限状态、1220PNG映射与raw组件池化、fit16/fullfit重放检查通过；初始8张量及16核心文件字节相同。
但严格跨后端标量检查有168项超出原容差：LPIPS106逐帧+2均值，SSIM10逐帧，全峰contrast ratio49逐帧+1均值。
最大LPIPS差1.062453e-5；原source仍为主指标，CPU仅独立复核，未扩大容差，8项gate判断两后端完全一致。

存储100000点不等于有效支持：25k/50k/100k的opacity>=1/255点数为intersection5047/557/511、aggregate6160/947/844。
这些是均匀参数统计，不是可见贡献或几何真值；不能据此单独确定失败原因。训练16630.04/16415.13s，
峰值自身已分配显存13.709668/10.276782GiB；长跑存在已记录的GPU0共享资源观察，不能直接当独占速度对照。

56val未进入该新模型的初始化/优化，但历史开发已观察，且camera/light共同变化；不是盲测、纯固定视角换光或真实几何证据。
本轮否定该固定实现/训练选择的可靠小峰恢复，不能据一seed否定所有表面基础。点生命周期/像素积分解析研究仍与正式结果分开，未启动补救训练。
[完整协议和所有终端证据](intersection_reflectance.md) · [主门槛](../../runs/intersection_reflectance_pilot/validation_gate.json) ·
[人工检查](../../runs/intersection_reflectance_pilot/manual_review.json) · [数值边界](../../runs/intersection_reflectance_pilot/analysis_verification.json)。


Postterminal provenance: after the completed source/metric audits and failed primary/manual gates, root began the separate optional `fragment_alpha_min` / fixed-checkpoint gradient diagnostic implementation (default cutoff remains1/255). The final closeout check failed because it still required the current worktree to equal the old launch source; freeze release had not been explicitly coordinated with archival. The first log/source and `closeout_provenance_issue.json` are retained. This result is governed by immutable `source.tar` and the earlier successful terminal audit, not the subsequently edited worktree. In a later update root reported that the four-fit-frame gradient probe completed (gradients restored, all four initial patch losses worsened), and the separate `surface_opacity_profile` cut/uncut100-step resource stages launched from the same intersection25k source. Root verified their actual processes/GPU activity; this terminal auditor did not perform that live check. The new canonical/core/profile are separate later work and provide no new quality claim here.

以下为历史阶段；旧“运行中/没有终端结果”等保留当时状态，由顶部结案接续。

## 交点反射新基础：算子、真实集成及资源可行性完成（2026-09-23）

新路径逐ray–surfel真实交点计算局部两瓣GGX后合成，对照为同交点先聚合物理属性后着色；新初始化联合学习几何/材质。
不是旧冻结RGB残差的微调；不含PORT/冻结decoder/SDF先验。首轮明确unoccluded，不能代表完整投射阴影重光照。
合成场景中，一个覆盖整幅512图像的surfel可产生3×2px半高宽亮点并随灯位移动29px，单层两模式相同；
重叠表面公式及六组几何/材质梯度检查通过。相同状态CUDA前缀和产生末位差，精确重载断言失败及容差复核均保留。
真实首3步因首次库调用消耗global Python RNG而未过采样控制；改独立frame RNG后，两模式3+3的8项检查通过。
共享light-scale分支的两模式3步6项检查也首次通过；这些均不是重建质量改善证据。

固定100步资源实验于08:39:19 UTC完成，两张预先检查空闲GPU并行，无环境升级。
intersection/aggregate训练时间30.23/25.00s，首步后平均.295051/.241515s，峰值已分配显存13.21697/9.87526GiB。
八个初始张量逐字节相同，配置仅mode/output不同；每组400个patchdraw、93个fit帧、506/56划分和独立RNG完全核对。
全部几何/材质组更新且有限，center/radius/light保持；只评1fit帧作入口检查，不作质量或泛化结论。
证据[资源审计](../../runs/intersection_reflectance_profile/resource_audit.json)与[验证](../../runs/intersection_reflectance_profile/verification.json)。

canonical已固定intersection_reflectance_pilot两组各100k、native512、100000点、4个64²核心patch及5px halo；
仅fit506优化，validation56不进入初始化/先验/标定/训练，但这些帧历史开发中已被观察过，非盲测。
两组共同开放一个正值场景光强标量，保持相机固定，无shadow；25k/50k/100k fit16及终端完整fit506/val56，暂不测official71。
正式质量训练于08:50:31UTC已在GPU0/1启动，没有终端真实质量结果。预定七项小峰/整体门槛加SSIM与GT固定8crop检查，见[完整协议](intersection_reflectance.md)。

## 基础表示同GT审计完成：现成GS³/SSD未恢复可靠细小高光（2026-09-23）

Pixiu71/Cat66个开发已观察test帧、9个模型/场景流、619张预测共同GT审计与固定图块人工检查已完成。
预算、表示和监督不同，这是基础方案观察性比较，不是单一ASG模块的因果消融或新盲测。

| 场景/模型 | PSNR | SSIM | LPIPS | 1–4px MAE | 小桶对比/GT | 小桶recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pixiu GS³100k |23.723631|.864052|.117342|.265749|.000257|.150075|
| Pixiu SSD100k参考 |23.403902|.862117|.119268|.269461|−.003937|.126597|
| Pixiu PORT default30k |20.578249|.845557|.156690|.353612|−.004904|.008640|
| Pixiu PORT neural30k |21.478632|.851218|.161733|.365245|−.003446|.015778|
| Pixiu PORT source控制 |21.331655|.843041|.152835|.318331|−.006346|.126221|
| Cat GS³100k |19.123925|.716881|.231927|.155471|.010593|.036522|
| Cat SSD100k参考 |18.003657|.703663|.248188|.196733|.010436|.033043|
| Cat PORT default30k |21.536184|.766470|.230405|.185985|.031460|.000000|
| Cat PORT neural30k控制 |22.479174|.780935|.255563|.174941|.055782|.013913|

Pixiu GS³相对source的整体分数更好，但小桶对比距1仅改善.006603、recall+2.3854pp，未达.03/3pp，6/8数值条件通过。
全峰precision .178755→.385122、假阳性25521→11307也有改善，仍不能把接近零的小峰对比称为准确恢复。
Cat GS³仅MAE/LPIPS两项通过，2/8条件通过；PSNR/SSIM低于neural控制。固定8个Pixiu及6个Cat GT图块均未确认可靠窄峰恢复。
Cat帧33/49各只有一个合格小组件，没有按预测补选。人工review completed/false，全部gate/summary同步未推广。

GS³保存input.ply与seed0的100000随机点/颜色、零法线逐位匹配；训练相机完整对应Cat522/Pixiu562，
没有缓存点云、SfM、单目先验或test训练初始化。SSD曾用test RGB拟合test相机/灯，但其Gaussian optimizer不在test分支更新；
重置标定不能消除曝光，所以所有主决策中的SSD仍为false，仅列作参考。
五个PORT流使用各自正确历史源码，全部metrics/views及旧前4PNG精确保存，旧neural的features6:9语义没有被当前解释替换。

结构、GT、映射、组件raw池化及全峰/环域计数核查通过，source71组件行与旧reference精确一致。
GS³/SSD的保存GT与canonical最多差1byte，PORT为0。严格跨后端标量核查仍为false：13帧SSIM和57帧LPIPS超出原容差，
最大差9.059906e−6/1.683831e−6；PSNR及三项整组均值在容差内。首失败日志/源码保留，未扩大rtol1e−6/atol1e−8。
单帧18探针用相同PNG复现byte/255输入转换及SSIM后端差异：CPU/GPU转换最大5.96e−8；旧GPU输入分数精确复现旧报告，
CPU除法输入精确复现共同审计LPIPS。共同CPU协议和门槛未改，不能说“全部核查通过”，也不能据这些末位差改写明显未过的峰形门槛。
[完整协议、表格和数值边界](foundation_comparison.md) · [核查报告](../../runs/foundation_comparison_audit/analysis_verification.json) ·
[人工检查](../../runs/foundation_comparison_audit/manual_review.json)。失败的paired/RGB残差头没有新增test。

完整研究目标仍active，停止自由RGB残差微调主线。已选择[逐ray表面交点反射原型](intersection_reflectance.md)，
比较非负GGX先着色后合成与先聚合再求BRDF，从新初始化联合学习几何/材质，不用PORT、冻结decoder或RGB残差。
当前是实现及有限后端检查阶段（6项材质CPU检查通过，尚未注册），没有正式质量训练；计划506fit/56val，不复用562帧source/先验。
当前GPU0可用于该独立原型的有限检查；本结案不声明全机空闲。canonical仍为已完成的foundation诊断配置，不用于训练launcher。

以下为历史阶段；旧“尚未选定”等描述保留当时状态，当前下一主线以上文为准。

## 局部成对RGB差分监督完成：小峰有限改善，整体质量门槛失败（2026-09-23）

匹配RGB/paired各30k、3k/12k/30k的fit16、终端full562于05:40:45 UTC全部完成；CPU审计及8个预定GT图块人工检查完成。
两组共享multiply+wide头、相同初值和局部峰环采样，只改变pair权重0/.25。以下562帧均为源链及头已见的训练拟合。

| 完整562fit | 原GS | 历史multiply | 历史wide | RGB控制 | paired |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSNR |25.259190|25.242683|25.365816|25.173248|24.732770|
| SSIM |.891718|.876765|.878177|.876975|.871918|
| LPIPS |.129588|.137921|.137976|.138497|.141648|
| 1–4px小桶MAE |.202598|.140797|.144804|.139224|.138402|
| 小桶对比/GT |.294762|.410018|.406718|.414460|.456402|
| 小桶recall |.358357|.503277|.498231|.517450|.571768|
| 全峰precision |.631668|.566809|.575195|.536719|.469261|

paired相对匹配RGB通过4/7预定数值条件：小桶对比距1改善.041941（要求≥.03）、recall+5.4317pp（要求≥3pp）、
MAE下降.000823；固定4fit的11px环域neutral过亮增加3.32047%，未超过5%。
全峰precision下降.067459、PSNR下降.440478dB、LPIPS增加.003151，三项失败。
paired相对原GS的PSNR≥25.159190、SSIM≥.889718、LPIPS≤.131588三项质量条件也全部失败（判定用未舍入精确值）。
人工检查false：有限峰对比/命中改善伴随额外亮粒、合并/错位亮斑、色块与暗斑，未证明可靠窄峰恢复。
没有追加71test，没有默认推广；三个gate副本及manual_review一致，保留所有负结果。

| 增量残差代价 | 原GS | 历史multiply | 历史wide | RGB控制 | paired |
| --- | ---: | ---: | ---: | ---: | ---: |
| 更新步数 |0|30000|30000|30000|30000|
| 参数量 |0|2197997|2200784|2200784|2200784|
| 训练秒数 |—|1862.30|1972.27|2049.78|2060.71|
| 峰值已分配GiB |—|7.63236|7.63241|7.80154|7.80154|

共同已拟合源链代价不计入各头的增量；历史multiply/wide只作上下文，使用旧采样，不是本轮匹配控制或严格速度基准。
paired训练耗时较匹配RGB增加0.5332%。本轮25个初始tensor逐位相同，六个checkpoint的源GS/transport/camera/radius固定，
计数、LR、有限值、GT组件池化与归档源码审计通过，CPU最终审计首跑通过。
两组每组15360000对中15355503有双端支持、4497排除，无退化帧；资格/抽样/组件覆盖计数相同，所有eligible组件至少采到一次。
浮点训练mask有212982峰像素，量化评价有212877峰像素及21237个1–4px组件；此差异不是遗漏帧，也不能当质量改善。
旧进程退出、GPU0/1释放（07:09 UTC和结案快照确认）；没有重跑审计或启动新训练。
[完整协议与结果](gs_residual_paired_loss.md) · [结案核查](../../runs/gs_residual_paired_loss_pilot/verification.json) ·
[固定图块](../../runs/gs_residual_paired_loss_pilot/comparison_crops.png)。`source.tar`、`analysis_source.tar`及四份结案文档`docs.tar`保留来源。

完整目标仍active。根据用户对核心方案的质疑，停止将残差头带宽、采样、损失微调作为下一主线，
转入基础表示/光传输/监督目标复核；替代方案尚未选定或实现，不把本轮结案当研究目标完成。

以下为历史阶段，旧运行中描述仅说明当时状态。

## 历史启动记录：局部成对RGB差分监督正式对照（现已完成）

已完成7项CPU与两组真实3+3检查；共有25个初始张量完全相同、源状态冻结，只有差分权重0/.25不同。
canonical gs_residual_paired_loss_pilot已在GPU0/1启动，各30k；同一multiply+wide头与局部峰环配对采样。
固定三个fit16节点及完整562fit，先不评test。当前无正式质量结论，不能把短测通过当作高光改善。
详细协议与最新状态见gs_residual_paired_loss实验文档及research_handoff；完整目标active。

以下为历史阶段，顶部状态优先。

## 可移动外观中心对照完成：窄核未通过继续门槛（2026-09-23）

两组各30k、三个节点fit16、终端562fit、最终审计及固定8个GT图块人工检查均已完成。以下是训练拟合，不是held-out结果。

| 完整562fit | 原GS | 历史multiply | wide | narrow |
| --- | ---: | ---: | ---: | ---: |
| PSNR |25.259190|25.242683|25.365816|25.183454|
| SSIM |.891718|.876765|.878177|.876021|
| LPIPS |.129588|.137921|.137976|.139161|
| 1–4px小桶MAE |.202598|.140797|.144804|.137793|
| 小桶对比/GT |.294762|.410018|.406718|.416703|
| 小桶recall |.358357|.503277|.498231|.508323|
| 全峰precision |.631668|.566809|.575195|.546558|

narrow相对wide的小桶对比改善.009985、recall增加1.009pp，低于.03/3pp要求；
全峰precision下降.028638、固定4fit邻环neutral过亮增加11.7645%、PSNR下降.182362dB，故5/7数值门槛失败。
小桶MAE和LPIPS增幅通过各自条件；固定图块仍有缺点、错位/合并亮斑和错误色块，人工门槛也不通过。
未追加71test、未改默认。历史multiply不是本轮匹配控制，不能将对它的差异都解释为中心学习的因果收益。
wide虽略增原GS的PSNR，SSIM561/562、LPIPS551/562帧变差；narrow对应562/562、549/562帧变差，两者均非整体质量赢家。

六个checkpoint原GS/transport/camera/radius逐位固定；24个公共初值相同，仅尺度不同；D/Q确实更新且全部有限。
中心平均转角wide14.572866°、narrow27.844391°，只表明外观特征改变，不能证明物理法线恢复或未见光照泛化。
所有训练/评价/审计结束，GPU已释放。完整证据、图像与来源见[实验记录](gs_residual_movable_centers.md)及
[runs/gs_residual_movable_center_pilot](../../runs/gs_residual_movable_center_pilot)。
下一项[同预算局部成对RGB差分监督](gs_residual_paired_loss.md)正在实现，尚无新训练或质量结果；完整目标仍active。

以下为历史记录；顶部结果为当前已完成状态。

## 历史启动记录：可移动外观中心宽／窄SG正式对照（现已完成）

结构、接口与CPU13项（旧9+新4）、两组真实3+3训练/重载/CLI检查通过。
canonical新name gs_residual_movable_center_pilot已在GPU0/1启动，各固定30k；初始24个公共张量逐位相同，
只有尺度buffer不同，源码与启动归档一致。原GS/主着色法线/传输/相机固定，中心只控制残差新特征。
先3k/12k/30k fit16及终端full562，eval_test:false，按预定narrow相对wide门槛再决定成对test。
当前没有正式质量结论，不将中心转动当作高光恢复或物理法线改善。实际进度以run/status.json和research_handoff为准。

以下为已完成历史阶段，顶部状态优先。

## 局部角度提示诊断完成：不启动固定法线窄核训练（2026-09-23）

当前multiply30k的完整562train前向、CPU6项/真实16帧检查及独立复核全部完成，没有新训练或test评价。
全部21237个1–4px组件/40429px复现原评价；完全未命中11177组件/19169px，只有3组件/4px无GS覆盖。
geometry提示极值偏离>2px者9748（87.21%），非正最佳窄核对比或错位的并集9870（88.31%）。
仅62（0.55%）满足预定定位/对比/增益条件，远低于60%；material参考仅76（0.68%）。
固定crop也不支持直接缩窄固定法线核。此结果不是几何真值、容量上界或未见灯光评价。
所有state/radius保持，GPU0已释放，暂无运行中任务；不要重跑完成的诊断。

下一项已固定为可移动外观中心的宽／窄球面高斯对照，尚未实现或启动。
两组均学习仅供残差使用的中心；原几何、主着色法线、相机、灯位和PORT保持。
不把灯随相机offset搬动：训练元数据不支持刚性相机灯架假设。
[诊断结果](gs_residual_angular_cues.md)。

以下为历史阶段，顶部状态优先。

## 空间—方向交互对照完成：有限训练收益，未通过继续门槛（2026-09-23）

add/multiply各30k、三个节点fit16和终端562fit全部完成，CPU9项/真实3+3/最终审计与人工图块检查完成。
19个初始状态tensor逐位相同，源GS/传输/相机/radius全程固定，预算/抽样/LR一致，全部tensor有限。
multiply相对add完整562fit：PSNR25.061057→25.242683，LPIPS .140676→.137921；
1–4px小峰MAE .145893→.140797，对比/GT .383538→.410018，recall .478196→.503277。
对比改善.026480、recall+2.508pp，低于预定.03/3pp；其他五项数值门槛通过。
局部亮核改善存在，但固定图块仍缺点、错位和宽亮斑，人工检查未确认可靠恢复。
不降低门槛、不追加71test、不采用默认。相对原GS，multiply的SSIM有561/562退化、LPIPS547/562退化。
这是有限正向拟合证据，不能宣称乘法无效、旧头达到容量上限或未见灯光改善；完整研究目标仍active。

所有任务结束、GPU0/1释放，无训练/评价在运行；canonical仍是已完成gs_residual_interaction_pilot，勿同名重启。
证据：runs/gs_residual_interaction_pilot/{comparison.json,comparison_crops.png,training_gate.json,manual_review.json,final_audit.json}。
[完整协议与结果](gs_residual_interaction.md)。

以下为历史阶段，顶部状态优先。

## 空间—方向交互正式对照运行中（2026-09-23）

新增可选add/multiply第一层交互，各3504权重，原结构/损失/采样/合成保持；none兼容旧模型。
CPU9项和真实两组3+3训练/续训/重载/CLI检查通过。初始19个状态张量完全相同，
正式canonical新name gs_residual_interaction_pilot已在GPU0/1启动，均固定30k、完整562fit、512px/2048rays。
先3k/12k/30k fit16及终端562fit，通过预定训练门槛才成对追加71test；当前没有正式质量结果。
初始交互输出尺度差已记录，不中途改归一化。目标active；实际状态以research_handoff和run/status.json为准。

## 完整562帧残差30k预算实验完成（2026-09-23）

同结构单链3k/12k/30k与16fit、终端562fit/71test全部完成。训练小峰随预算有限改善，
16fit最小桶对比/GT .3203→.3430→.3864，recall39.32%→42.96%→46.12%；并未接近少帧拟合能力。
完整71test PSNR21.331654→21.080794，LPIPS .152835→.158758，SSIM/LPIPS全部71帧退化。
完整最小桶5324像素，recall12.62%→15.31%但对比仍−.004928，全峰precision下降；不是准确高光恢复。
不采用30k为质量赢家，不继续只凭训练MAE下降增加同结构预算；目标仍active。

完整组件评价已接入--highlights并通过CPU/真实CLI检查，原指标保持；
三个checkpoint主GS/传输/相机/radius与源完全相同，全部有限、采样/LR/指标审计通过。
当前模型完整test截断诊断确认仅2.386%可覆盖小峰像素有任一通道clip，不能解释绝大多数缺峰。
所有训练/评价/诊断已结束、GPU0/1释放。canonical仍为已完成gs_residual_budget_pilot，勿同名重启。
下一项为相同参数量的add/multiply空间—方向交互，两组30k，先train门槛再成对test；尚未实现/启动。
[结果与下一固定协议](gs_residual_budget.md) · [接续](../project/research_handoff.md)。

以下为历史记录；旧“运行中”描述已结束。

## GS残差少帧拟合诊断完成（2026-09-23）

实现了保留完整来源的--residual-frames采样池，CPU与真实3+3续训检查通过。
从原geometry_grid主模型新建零头，single帧0训练3k、four帧0/35/70/105训练12k，
各帧实际曝光约3000、总预算四倍；512px/2048像素，主GS/传输/相机完全冻结。
共同frame0最小1–4px高光：命中34/75→single72/75、four68/75，
对比/GT .3753→.9708/.9522，邻环与远处前景误差同步下降；four四帧小桶合池命中295/307。
这证明现有表示具有这些已见条件的局部拟合能力，不代表物理材质或未见灯光泛化。
其余12个source已见帧PSNR从26.058降至19.432/21.434，因此不采用这两个模型为质量赢家。

所有状态冻结/有限值/采样计数/指标审计通过；覆盖域检查只发现1个未覆盖GT峰，bbox未排除任何峰。
截断分区确认大幅残差主要在低alpha背景，但仍有2.8%有色前景通道被截断；GT峰自身无负截断。
全部训练/评价/诊断结束，GPU0/1释放；本轮无test评价。完整研究目标active。
下一项优先旧头完整562帧单链30k的预算曲线，3k/12k/30k fit、只终端完整fit/test，尚未启动。
[详细结果](gs_residual_subset.md) · [接续](../project/research_handoff.md)。

以下为历史里程碑。

## 主GS光照残差对照完成（2026-09-23）

零初始化有符号光照残差已实现；两法线输入geometry/material各3k、512px、2048采样像素，
全部562train，原GS/传输/相机完全固定。CPU与真实3+3短测、16fit/完整71test及终端审计完成。
原GS/geometry/material的71test PSNR为21.331654/21.376813/21.342007，
LPIPS .152835/.156177/.155676；GT峰对比/GT仍−.005892/−.004879/−.005549。
峰MAE稍降但未恢复窄峰，4test的1–4px桶命中像素均8/253，邻环过亮增加27%/32%。
候选不设默认，不能称全面改善。两checkpoint源GS/transport/camera逐字节不变、全有限。
71帧未参与拟合且灯位/相机均新，但曾被开发观察、光强相同、视角与光位置同时变化。
GPU0/1全部结束。完整高光目标仍active，下一项是单/四训练帧重复拟合诊断，尚未启动。
[本轮协议/结果/边界](gs_radiance_residual.md) · [接续](../project/research_handoff.md)。

以下为历史里程碑；旧“缺少创建新主任务接口”状态已由当前active目标接续，不是当前阻塞。

## SDF法线直接着色对照为负结果 — 2026-09-23

Pixiu相同已拟合SDF初始化，两组各2000步，仅sdf_shading开关不同。
直接着色全fit PSNR23.46924，对照23.58488；16帧训练亮点对比度/GT .23227，对照.29380。
SDF/GS几何自一致性仍改善，但亮点法线方向更差，未扩展该候选的official test评价。
[具体指标、图像和单Gaussian审计](sdf_geometry.md)。

## SDF后续训练集对照：小步续训与多视图先验均未恢复高光 — 2026-09-23

Pixiu小步位置更新control/SDF各2k完成。随后复用SDF对照、只改相机条件多视图DA3深度，
再做同初始化/预算2k；全562帧fit PSNR仅23.573149→23.573587。
固定16帧的高光对比度/GT .27979→.28105，均低于38k初始化.29886。
新深度先验与独立法线先验更一致，却未转化为渲染收益；不据此扩展test或替换默认。
[全部指标及边界](sdf_geometry.md)。

## SDF双向几何首轮：一致性改善，高光仍未恢复 — 2026-09-23

同一38k初始化，control/SDF各追加3000步，Cat/Pixiu全137帧test。
SDF PSNR为21.517097/21.062508，对照为21.517294/21.019045；两者均低于初始化。
法线/深度夹角由control 15.742°/16.477°降至14.351°/15.194°，但无几何GT，不能等同于精度提升。
Pixiu亮点对比度/GT约−0.0051，未恢复准确小高光；默认方法不变。
[协议、完整指标与后续](sdf_geometry.md) · [图像](../../runs/neural_material_sdf_geometry/comparison.png)。

## 高光修复试验：完整测试未恢复准确小高光

Cat/Pixiu从原30k模型固定几何各追加8000步，完整137帧official test已完成。
共享先验窄峰拟合改善，图像中出现宽反光，但GT亮点处的局部对比度没有改善。

| 场景 | 原神经材质PSNR | 修复PSNR | 原LPIPS | 修复LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Cat | 22.479174 | 22.333152 | 0.255563 | 0.257210 |
| Pixiu | 21.478632 | 21.205147 | 0.161733 | 0.160396 |

Pixiu全部test的26147个固定GT亮点代理像素，RGB MAE由0.416603降至0.369330，
但局部对比度/GT由0.002218变为-0.003414，说明更亮不等于准确恢复高光。
后续三项训练帧对照也未带来亮点收益，未采用；默认方法不变。
[完整诊断、额外预算与负结果](highlight_recovery.md) ·
[固定首帧对比](../../runs/neural_material_highlight_recovery/comparison.png)。

以下为原版神经材质及此前方法的历史结果，复现须配套各run的源码快照。

## 共享神经材质Cat/Pixiu正式实验已完成 — 2026-09-22

2/2场景完成从头30k/seed0训练与全部137帧official test，0失败。
Cat/Pixiu分别使用522/562张完整train和66/71张test，512px黑底、原始测试标定，
没有相机优化、RGB预热或test选模型。固定末端last.pt，默认方法仍directional_port_v1。
完成于2026-09-22 19:01:09 JST，两张RTX6000 Ada上的队列耗时1:23:52，GPU已释放。

| 场景 | 帧数 | PSNR↑ | SSIM↑ | LPIPS↓ | 相对默认PSNR dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cat | 66 | 22.479174 | 0.780930 | 0.255563 | +0.942990 |
| Pixiu | 71 | 21.478632 | 0.851217 | 0.161733 | +0.900383 |
| 两场景等权 | 137 | 21.978903 | 0.816073 | 0.208648 | +0.921686 |

下表只比较同一Cat/Pixiu子集，不能与其余章节的六场景均值直接比较。

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练合计s | 最大已分配GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原默认3DGS方向端口 | 21.057217 | 0.806011 | 0.193548 | 6723.11 | 13.450 |
| 历史2DGS方向端口 | 21.298620 | 0.812615 | 0.195302 | 9143.12 | 13.583 |
| 2DGS cosine attention | 21.223046 | 0.812377 | 0.196935 | 4254.81 | 4.156 |
| 共享神经材质 | 21.978903 | 0.816073 | 0.208648 | 9433.02 | 13.159 |

新方法的PSNR/SSIM在两个场景都高于以上三个对照，但LPIPS在两个场景都更差。
相对默认，两场景均值变化为PSNR +0.921686dB、SSIM +0.010063、LPIPS +0.015101。
相对历史2DGS方向端口，PSNR +0.680283dB、LPIPS +0.013346；没有全面画质提升的证据。
固定各场景test首帧的图像中仍可见毛发、高光和底座纹理模糊，不能由PSNR提升推断细节恢复。
实验只覆盖两个场景、单seed；没有证明真实材质恢复、直接/间接光物理解耦或统计显著性。

默认与attention的全部(frame_index,name)及评价协议已逐帧核对；对比图的GT也逐像素一致。
历史surfel取自仍保留的attention/comparison.json汇总，原模型/逐帧输出已清理，
不是本次重新训练或重新评价的消融。新方法同时使用2DGS与材质先验，
相对默认3DGS的提升不能全部归因于共享decoder。

Cat/Pixiu训练分别5015.05/4417.97s，最终399484/399399个Gaussian，
峰值已分配显存13.145/13.159GiB。时间不含先验生成与test；显存包含训练图像缓存，
并非NVML峰值或硬件容量承诺。各对照实际点数不同，运行时间仅为单次测量。
网络参数55926，其中26695为冻结decoder、29231可训练，不包含逐Gaussian属性。
两份checkpoint的20个decoder张量均与先验完全相同，全部浮点张量有限；
将先验路径改为不存在的文件后仍能严格重载并保持冻结。

后验诊断：Cat/Pixiu的Gaussian漫反射RGB通道中，超过0.98的比例约49.05%/29.46%；
这是未加权的参数计数，不是像素比例。Pixiu固定8张fit图的非局部radiance占比均值约0.735，
该统计未作alpha加权，也不代表物理能量分解或退化因果证据。
继承的light_scale是场景归一化尺度，拟合颜色不能解释为绝对校准反照率。

[机器可读对比](../../runs/neural_material_real_validation_20260922/comparison.json) ·
[固定首帧图像](../../runs/neural_material_real_validation_20260922/comparison.png) ·
[完成状态](../../runs/neural_material_real_validation_20260922/status.json) ·
[Cat权重核验](../../runs/neural_material_real_validation_20260922/Real_NRHints/Cat/checkpoint_audit.json) ·
[Pixiu权重核验](../../runs/neural_material_real_validation_20260922/Real_NRHints/Pixiu/checkpoint_audit.json)。
运行目录保存validation.json、manifest.json、source.tar、每场景config/split/history与last.pt；
Git HEAD为47028ea0ff77f427aa6a016cb64e039432935a5e并含工作区修改，复现以source.tar为准。
7个材质/训练/渲染/评价核心文件与冻结源码逐字节一致，完成后仅补充结果文档。

### 共享先验与接口验证

新方法neural_material，方法与预算在测试前固定，见[实现说明](../architecture/modules/neural_material.md)。
程序化先验50k/seed0、batch4096、单RTX6000 Ada：恒定学习率首轮末端不稳定，
改为0.001→0.00001余弦衰减后重新训练，不改变模型或数据。

| 共享先验 | 时间s | 独立BRDF相对L2 | 透射MAE | 反射亮度MAE |
| --- | ---: | ---: | ---: | ---: |
| 首轮恒定LR（未采用） | 393.29 | 0.229082 | 0.218567 | 0.159462 |
| 余弦LR（正式采用） | 363.57 | 0.062001 | 0.027582 | 0.037121 |

验证为seed1的32768个程序化样本，不含Cat/Pixiu图像；这些误差不能直接当作重光照指标。
正式先验包含26695个decoder参数，两个场景共用且冻结，encoder仅用于预训练/初始化。
[训练曲线](../../runs/neural_material_prior/training.png) ·
[固定seed每类首个材质预览](../../runs/neural_material_prior/decoder_preview.png) ·
[完整先验指标](../../runs/neural_material_prior/metrics.json)。
预览仍有彩色高光和宽散射误差，不等同于官方实现的材质重建精度。

真实Cat三步GPU集成检查通过，重载像素最大误差0，decoder权重未变化。
正式输出runs/neural_material_real_validation_20260922，结果见上表。
正式先验额外用时363.57s；包含未采用的首轮预训练，开发预训练总计756.86s，
均未计入上表的场景训练时间。

> 文件清理说明（2026-09-22）：已按用户要求移除两个候选及其历史指标。
> 本文保留其余方法与SSD-GS参考结果；旧路径仅表示历史来源。
> 当前可用文件以[清理清单](output_cleanup_20260914.md)为准。

## SSD-GS已有预测的统一指标参考（2026-09-22）

全部1937帧用PORT-GS共同目标和标准指标重算，未改任何baseline文件。
预算与本项目30k不同；两项Real的旧权重有test校准优化来源，不能当作严格干净的同协议基线。

| 场景 | 步数 | PSNR | SSIM | 标准LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Cat（来源受限） | 100k | 18.003657 | .703654 | .248188 |
| Pixiu（来源受限） | 100k | 23.403902 | .862116 | .119268 |
| AnisoMetal | 100k | 28.000640 | .956571 | .046991 |
| Translucent | 100k | 31.986258 | .975091 | .038364 |
| bunny_small | 60k | 38.486211 | .989130 | .019395 |
| dragon_small | 60k | 37.376969 | .981792 | .026607 |
| 合成四场景等权 | — | 33.962519 | .975646 | .032839 |

完整指标与逐帧证据见[research_summary.json](research_summary.json)的ssdgs_reference；协议见setup.md。
这里的标准LPIPS与旧SSD-GS results_test.json不可混用，旧入口直接传[0,1]给其LPIPS实现。

## surface_attention六场景正式结果 — 2026-09-22

6/6场景完成从头30k/seed0训练及全部1937张official test，0失败。GPU0/1/2已释放。
完成时间2026-09-22 11:44:30 JST；三卡队列耗时1:38:15，不含此前消融与已复用的教师预测。
固定score=cosine、normal_weight=depth_weight=0.05，surface_start1000、shadow/port5000、
refine_stop25000、geometry_warmup_steps=0；全train拟合，固定last.pt，原始测试标定。
代码/配置冻结于测试前，源码和命令在该run/source.tar、validation.json、manifest.json。

六场景等权均值（每场景先平均全部帧）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: |
| 原默认3DGS方向端口 | 29.089976 | 0.916603 | 0.089701 |
| 首轮2DGS方向端口 | 28.472787 | 0.917875 | 0.090103 |
| 2DGS纯RGB预热5k | 28.102730 | 0.915623 | 0.091280 |
| 本轮2DGS cosine attention | 28.622846 | 0.918468 | 0.090192 |

相对首轮2DGS，PSNR +0.150060dB、SSIM +0.000592，但LPIPS略差0.000089。
相对5k RGB预热实验，PSNR +0.520116dB，SSIM/LPIPS也改善。
相对原默认3DGS，PSNR仍低0.467129dB，LPIPS也略差0.000491，SSIM高0.001865。
本轮支持简化与效率收益，尚不支持整体质量优于原默认；不改变原默认方法。

| 场景 | 原3DGS | 首轮2DGS | 5k预热2DGS | 本轮attention | 相对原3DGS dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cat | 21.536 | 21.940 | 21.861 | 21.806 | +0.270 |
| Pixiu | 20.578 | 20.657 | 20.684 | 20.640 | +0.061 |
| AnisoMetal | 28.297 | 28.023 | 27.028 | 28.537 | +0.240 |
| Translucent | 29.012 | 28.676 | 27.629 | 28.505 | -0.507 |
| bunny_small | 39.045 | 36.137 | 37.284 | 37.183 | -1.862 |
| dragon_small | 36.071 | 35.404 | 34.130 | 35.066 | -1.005 |

Cat/Pixiu/AnisoMetal相对原默认提高；Translucent/bunny/dragon下降，bunny差距最大（1.862108dB）。
全部帧按(frame_index,name)对齐，保留SSS的重复文件名，未排除困难视角。
本轮bunny frame332仅12.108dB，dragon frame130仅18.687dB；均值不能替代对失败视角的检查。
这些视角在历史方法中也有错误，不能仅凭该现象判定唯一成因。

### 实际训练资源

下表为每次训练最后一条history记录；时间包含训练循环，不包含先验预测及test。
显存为PyTorch峰值已分配量（含训练数据缓存），不是NVML峰值或最低硬件容量承诺。

| 场景 | 首轮2DGS分钟 | attention分钟 | 首轮峰值GiB | attention峰值GiB |
| --- | ---: | ---: | ---: | ---: |
| Cat | 83.27 | 39.03 | 13.583 | 4.156 |
| Pixiu | 69.12 | 31.89 | 12.838 | 3.965 |
| AnisoMetal | 76.92 | 36.11 | 19.450 | 9.976 |
| Translucent | 78.57 | 37.98 | 19.467 | 9.990 |
| bunny_small | 100.80 | 55.02 | 11.088 | 1.614 |
| dragon_small | 86.84 | 58.40 | 11.145 | 1.612 |

六场景训练时间合计29731.06s → 15504.78s，约1.92倍速度；相对原3DGS约1.52倍。
全套最大已分配显存19.467GiB → 9.990GiB。各场景实际Gaussian数量不同，均使用同一40万上限；
单次硬件运行时间不是多次重复的严格性能统计。
非局部网络+gate参数29231 → 7878；整个网络100402 → 79049，不包含每个Gaussian的属性。

[完整机器可读对比](../../runs/surface_attention_validation_20260922/comparison.json) ·
[固定各场景首帧的渲染对比](../../runs/surface_attention_validation_20260922/comparison.png) ·
[完成状态](../../runs/surface_attention_validation_20260922/status.json)

以下为用于选型的train内部validation及历史实验，不能与上面的official test混为同一划分。

## 2DGS退化诊断与attention完整消融 — 2026-09-22

下表是**train内部灯光留出的validation**，与六场景official test是不同划分。
AnisoMetal/bunny/dragon分别1775/225、449/51、450/50 fit/validation帧；
各条件从头30k、seed0，未使用固定RGB预热，表面监督从1000步启用，阴影/非局部从5000步启用。
depth_weight=0.05、表面一致性与distortion各0.01、其余协议固定。

| 非局部表示 / 法线权重 | AnisoMetal PSNR | bunny PSNR | dragon PSNR | 场景均值 PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 方向端口 / 0.05 | 28.160072 | 34.624631 | 34.586740 | 32.457148 | 0.970710 | 0.036184 |
| 方向端口 / 0.005 | 28.215447 | 32.768908 | 34.486891 | 31.823749 | 0.969839 | 0.035969 |
| dot attention / 0.005 | 28.759203 | 33.845246 | 31.003466 | 31.202638 | 0.962760 | 0.043880 |
| dot attention / 0.05 | 28.958801 | 34.471539 | 30.277283 | 31.235874 | 0.966578 | 0.038639 |
| cosine attention / 0.05 | 29.321029 | 33.691018 | 32.437177 | 31.816408 | 0.967735 | 0.039763 |

结果来自runs/surface_attention_ablation_20260922各条件；attention dragon的成功重跑位于
runs/surface_attention_followup_20260922/attention_weak_normal/Synthetic_SSS-GS/dragon_small。
原失败日志保留：gsplat未初始化padding导致梯度NaN，已在适配层显式补零修复；
真实失败批次前向差为0、梯度恢复有限，详见[表面实现](../architecture/modules/surface.md)。

弱法线权重没有全局收益；attention对AnisoMetal/bunny同约束端口改善0.544/1.076dB，
但dragon退化3.483dB，整体均值更低。原权重dot组三场景也已完成，未解决dragon退化。
该组bunny训练已完成后，评估曾因新增attention_score字段缺省读取失败；工厂统一调用既有
resolve_config并补跑eval，未重复训练。旧checkpoint保持dot行为，真实权重重载和11项CPU测试通过。
dragon弱约束抽查注意力出现单源格饱和；原权重组抽查有效源约1.7–6.7，不能将其全部概括为单源。
cosine组三场景也已完成，结果位于runs/surface_attention_cosine_{anisometal,bunny,dragon}_20260922。
它的PSNR均值是attention候选中最高，但比方向端口原权重低0.640740dB，不能称为本消融的质量赢家。
其三场景日志训练时间合计8797.75s，对照16104.71s（约1.83倍速度）；单次运行受硬件状态影响。
对应峰值已分配显存分别9.982/1.616/2.319GiB，对照19.451/11.098/11.135GiB，包含训练数据缓存。
网络参数不含Gaussian属性：非局部+gate为7878，对照29231；整个网络79049，对照100402。

为完成用户要求的新方法六场景比较，固定score=cosine、normal_weight=0.05、其余原协议，
进行从头全train→official test。此选择用于评估最佳attention候选的质量/效率折中，
没有宣布它优于端口或更改原默认方法。固定配置时间2026-09-22 01:04:59 UTC，正式输出
runs/surface_attention_validation_20260922；冻结时未读取该run的test指标，随后按该配置完成正式测试。
完整诊断、后验追加对照及论文边界见
[attention方法](../architecture/modules/attention.md)。所有已完成比较按(frame_index,name)核对同一组帧；
CUDA训练和不同结构的随机数消耗不保证位级相同，单seed不等于显著性检验。

## 纯2DGS预热重跑结果 — 2026-09-22核查

6/6场景完成30k/seed0及1937帧official test，无失败；日志逐场景确认在5001步发生
relighting_start并重置预热RGB。总耗时2:51:59，完成时间2026-09-21 23:56:12 JST。
本次使用GPU0/1/2，复用上轮StableNormal/DA3先验，未重新预测。
结果目录：`runs/directional_surfel512_warmup5000_20260921/`。

| 场景 | 本轮PSNR | 上轮2DGS PSNR | 差值dB | 原3DGS默认PSNR |
| --- | ---: | ---: | ---: | ---: |
| Cat | 21.861 | 21.940 | -0.079 | 21.536 |
| Pixiu | 20.684 | 20.657 | +0.027 | 20.578 |
| AnisoMetal | 27.028 | 28.023 | -0.994 | 28.297 |
| Translucent | 27.629 | 28.676 | -1.047 | 29.012 |
| bunny_small | 37.284 | 36.137 | +1.147 | 39.045 |
| dragon_small | 34.130 | 35.404 | -1.274 | 36.071 |

六场景等权均值：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: |
| 本轮2DGS纯预热 | 28.102730 | 0.915623 | 0.091280 |
| 上轮2DGS | 28.472787 | 0.917875 | 0.090103 |
| 原默认3DGS方向端口 | 29.089976 | 0.916603 | 0.089701 |

相比上轮，PSNR下降0.370056dB；相比原默认下降0.987245dB。bunny改善1.147384dB，
但AnisoMetal/Translucent/dragon分别退化0.994437/1.046868/1.274360dB。
当前不支持纯RGB预热的整体收益。该改动同时推迟了重光照学习、在切换时重置RGB、
按第二阶段重新计算网络LR，且重光照网络只有25k更新；单seed结果不能单独解释每项因果。
所有数据保持完整test、limit=0、original calibration，评价帧数与两轮对照一致。

## directional_surfel 六场景完成 — 2026-09-21

六场景全部完成30k/seed0训练及1937帧官方test，全部limit=0、原始标定，无失败。
GPU0/1/2队列含先验、训练和评估总耗时3:15:40，于2026-09-21 20:38:36 JST完成。
原始结果：`runs/directional_surfel512_validation_20260921/{family}/{scene}/test/metrics.json`。
对照：`runs/directional_port512_validation_20260915/`；下表均为完整official test。

| 场景 | 新方法 PSNR | 原默认 PSNR | 差值 dB | 新方法 SSIM | 新方法 LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cat | 21.940 | 21.536 | +0.404 | 0.77798 | 0.23380 |
| Pixiu | 20.657 | 20.578 | +0.079 | 0.84725 | 0.15681 |
| AnisoMetal | 28.023 | 28.297 | -0.274 | 0.95787 | 0.04150 |
| Translucent | 28.676 | 29.012 | -0.336 | 0.96554 | 0.04769 |
| bunny_small | 36.137 | 39.045 | -2.909 | 0.98496 | 0.02290 |
| dragon_small | 35.404 | 36.071 | -0.667 | 0.97366 | 0.03792 |

六场景等权均值：PSNR 28.472787（对照29.089976，−0.617189 dB），
SSIM 0.917875（对照0.916603，+0.001273），
LPIPS 0.090103（对照0.089701，+0.000402；越低越好）。

Cat/Pixiu的PSNR改善，其余四场景下降，bunny下降2.908524 dB最明显。
AnisoMetal/Translucent的LPIPS改善，但PSNR下降。当前没有整体优于原默认方法的证据。
这是2DGS+StableNormal+DA3的组合结果，只有seed0；不能将差异单独归因于2DGS、
某一种先验、阴影近似或5000步预热。未据此改变默认方法或中途修改实验配置。

## Directional rank512 outcome — 2026-09-15

Completed6/6 fresh30k/seed0 experiments and1937 test frames. Mean PSNR29.089976, SSIM0.916603, LPIPS0.089701. Versus directional rank64: +0.850820dB, +0.002530, -0.000992.

[Full comparison](../experiments/directional_port512_validation_20260915.md).

## Directional port results — 2026-09-15

Completed6/6 fresh30k seed0 fits and all1937 official test frames.
PSNR28.239156 / SSIM0.914073 / LPIPS0.090693 (equal scene mean).
Versus legacy rank512: −0.185288dB / −0.000540 / +0.001729.
AnisoMetal improves+0.533654dB; bunny_small drops−1.625855dB; overall quality
is not improved in this run. New architecture and preflight/checkpoint checks
are complete. Full protocol, per-scene metrics and evidence:
[directional experiment](../experiments/directional_port_validation_20260915.md).

# Current research results — 2026-09-12–13

## Round 1: fixed-last Cat validation, followed by a second-round decision

`runs/research_20260912/cat_r1_s0/last.pt`, 30,000 steps, seed 0, GPU 0,
512px black background, original held-out calibration, 470 fit / 52 validation.
Final metrics use uint8 observation RGB, standard VGG LPIPS input [-1,1].
Training took 1,470.625 seconds. First-round weights are reproduced with
`runs/research_20260912/cat_r1_source.tar`.

| Checkpoint | Validation PSNR | SSIM | Standard LPIPS | Gaussians |
| --- | ---: | ---: | ---: | ---: |
| Historical `cat_shadow_gradient_r2_s0`, 30k | 22.201852 | .770023 | .253591 | 156667 |
| Current `cat_r1_s0`, 30k | 22.540632 | .771293 | .247135 | 399358 |

PSNR increases .338780 dB; this combined representation/geometry change does not
isolate either component's effect. Quantized fit32 PSNR is 25.856165 and SSIM
.845761. The intermediate 20k unquantized validation value is 22.682633 dB;
30k training validation is 22.541187. Both monitor values are distinct from the
fixed-last quantized score used in the table.

The shared diagnosis covers 32 fit and all 52 validation frames. Validation fine
detail energy is .268290 of GT versus approximately .271 historically; fit32 is
.279299. Mean visible support radius falls from about 21.81px to 17.338px on
validation, while 78.62% of squared error remains in the object interior.
The increased point count and smaller support have not resolved blur. Mean
source exchange fraction is .154824; this statistic describes learned parameters
and does not measure a causal image-quality contribution.

Difficult validation [frame 233](../../runs/research_20260912/cat_r1_s0/diagnosis/hard_233.png)
reaches 14.698247 dB and
[frame 158](../../runs/research_20260912/cat_r1_s0/diagnosis/hard_158.png) reaches
17.055780 dB. The first round improves average metrics, while the remaining
texture and difficult-view errors motivate a second round instead of acceptance.

Evidence: [analysis summary](../../runs/research_20260912/cat_r1_s0/analysis_summary.json),
[validation metrics](../../runs/research_20260912/cat_r1_s0/validation/metrics.json),
[fit32 metrics](../../runs/research_20260912/cat_r1_s0/fit32/metrics.json),
[diagnosis](../../runs/research_20260912/cat_r1_s0/diagnosis/metrics.json),
[training history](../../runs/research_20260912/cat_r1_s0/history.jsonl).

## Round 2: accepted for perceptual and detail gains

`cat_r2_s0/last.pt`, fresh 30k, seed 0, GPU 0, the same 470/52 split and standard
uint8 evaluation. Training completed in 1,612.392 seconds with 398,791 Gaussians.
GPU operator and receiver-rendering audits passed their documented checks.

| Validation measure | Round one | Accepted round two |
| --- | ---: | ---: |
| PSNR | 22.540632 | 22.316405 |
| SSIM | .771293 | .778143 |
| Standard LPIPS | .247135 | .228932 |
| Fine-detail energy / GT | .268290 | .339026 |
| Mean visible support radius | 17.338px | 12.646px |
| Silhouette IoU | .957555 | .954375 |

Detail energy increases 26.4%; PSNR decreases .224227 dB and silhouette IoU
also decreases. Compared with historical 30k matched validation
22.201852/.770023/.253591, the accepted model remains +.114553 dB in PSNR with
better SSIM/LPIPS; historical detail energy was about .2710 of GT. Acceptance
prioritizes perceptual/detail gains while retaining the PSNR and outline tradeoff.
The remaining detail energy is still well below GT. Independent paired review
finds LPIPS improvement on 52/52 validation frames, SSIM improvement on 35/52,
and PSNR improvement on 22/52. Alpha L1 increases .020662 → .023228 and mean
silhouette distance increases 3.840 → 4.217px; these costs accompany acceptance.

Fit32 is 25.819267 dB / .847134 SSIM, with detail energy .350311 of GT and
12.755px support. Validation hard
[frame 233](../../runs/research_20260912/cat_r2_s0/diagnosis/hard_233.png) reaches
14.948138 dB (+.249891 over round one), and
[frame 158](../../runs/research_20260912/cat_r2_s0/diagnosis/hard_158.png) reaches
17.324608 dB (+.268827). These remain challenging views.

Structural iteration ends after two rounds; first-round startup corrections are
engineering restarts. The selected source is frozen in `cat_r2_source.tar`, and
round-one weights remain reproducible with `cat_r1_source.tar`.
Fresh Cat full fit and then Translucent/Bunny full fits are authorized on GPU 0
in sequence, using frozen settings. Cat full fitting and official testing have completed; results follow below.
Selection was completed before official test evaluation. See
[frozen cross-data protocol](comparison_20260912.md).

Evidence: [round-two summary](../../runs/research_20260912/cat_r2_s0/analysis_summary.json),
[validation](../../runs/research_20260912/cat_r2_s0/validation/metrics.json),
[fit32](../../runs/research_20260912/cat_r2_s0/fit32/metrics.json),
[receiver operator audit](../../runs/research_20260912/receiver_operator_audit.json),
[receiver rendering audit](../../runs/research_20260912/receiver_rendering_audit.json).

## Frozen full Cat: complete official test, above 20 dB

Fresh training used all 522 train frames for 30k steps on GPU 0, seed 0, 512px
black background; the fixed last checkpoint contains 398,851 Gaussians.
All 66 official test frames were evaluated once with original camera/light
calibration and zero test fitting. The second-round selection and source freeze
preceded test evaluation. Recorded training time is 1579.640 seconds, launcher
wall time 1621.451 seconds, and full test evaluation 6.326 seconds.

| Complete Cat official test | PSNR | SSIM | Standard LPIPS |
| --- | ---: | ---: | ---: |
| Historical September 11 full fit | 22.000071 | .767526 | .249252 |
| Frozen September 12 second-round full fit | 21.501174 | .766281 | .227281 |
| Current minus historical | -.498897 | -.001245 | -.021971 |

The >20 dB real-Cat target is met. LPIPS improves about 8.81%; PSNR is lower
than historical full fitting. Independent paired review finds LPIPS improved
on 65/66 frames, PSNR improved on 22/66 and SSIM improved on 31/66. Alpha L1
increases .013970 → .019071 (+.005101), with all 66 frames worse in alpha error.
These paired statistics preserve the perception/pixel/outline tradeoff.
This is consistent with the perceptual/detail
selection priority, while broad lighting and position errors remain. Inspection
of [current frame 41](../../runs/research_20260912/cat_full_s0/test/hard_41.png)
and [historical frame 41](../../runs/cat_refinement_full_s0/test/inspection_041.png)
shows clearer texture alongside unresolved broad errors.

The two lowest-PSNR official frames and the retained historical difficult frame are
[33](../../runs/research_20260912/cat_full_s0/test/hard_33.png): 17.230951 dB,
[41](../../runs/research_20260912/cat_full_s0/test/hard_41.png): 18.207802 dB, and
[64](../../runs/research_20260912/cat_full_s0/test/hard_64.png): 20.542339 dB
(the historical difficult frame, not the third-lowest frame).
All 66 frames contribute to the reported aggregate.

Evidence: [run result](../../runs/research_20260912/cat_full_s0/result.json),
[full test metrics](../../runs/research_20260912/cat_full_s0/test/metrics.json),
[aggregate full results](../../runs/research_20260912/full_results.json).
The final aggregate is **3/3 complete**. External results follow below; test
scores triggered no method changes or further tuning.

## Frozen Translucent: complete 400-frame official test

The accepted second-round source `cat_r2_source.tar` trained a fresh model on
all 2,000 GS³ Translucent train frames for 30k steps, seed 0, GPU 0, 512px white
background and gamma 2.2. The fixed-last checkpoint has 399,830 Gaussians.
Training took 1539.787 seconds (1576.255 seconds launcher wall time), and the
complete 400-frame test took 26.346 seconds. Original test metadata and the
frozen settings were retained throughout.

| Translucent result | PSNR | SSIM | LPIPS | Training/protocol scope |
| --- | ---: | ---: | ---: | --- |
| PORT frozen round two | 28.304924 | .960368 | .051791 | 30k, complete 400 test, standard VGG LPIPS |
| GS³ paper reference | 32.34 | .9740 | .0318 | Official script 100k; final paper metric implementation incompletely specified |

The GS³ values are published references. The HDR white-background/gamma export
path matches, but budget and metric evidence differ; this table is not a matched
SOTA ranking. No external-data tuning, branch ablations or other-method training
were performed. See [protocol audit](../research/related_work_20260912.md).

Difficult-view outputs include
[frame 330](../../runs/research_20260912/translucent_full_s0/test/hard_330.png)
(25.081852 dB) and
[frame 118](../../runs/research_20260912/translucent_full_s0/test/hard_118.png)
(25.106934 dB). Main-task inspection of frame 330 finds the checkerboard and
overall object shape retained, with softer predicted cast shadows, some
boundary/position mismatch and local material-brightness differences. This
supports approximate visibility as a remaining limitation; one image does not
identify a unique cause. The frozen experiment protocol remains unchanged.
Evidence: [run result](../../runs/research_20260912/translucent_full_s0/result.json),
[complete test metrics](../../runs/research_20260912/translucent_full_s0/test/metrics.json).

## Frozen Bunny small: complete 500-frame test with failure views

The final frozen run used all 500 `bunny_small` training images, a fresh 30k
model, seed 0 and GPU 0 at 256px black background with gamma 2.2 and explicit
`unit_light_intensity=1`. The final checkpoint has 354,592 Gaussians. Training
took 2328.689 seconds (2341.946 seconds launcher wall time), and all 500 official
test frames were evaluated in 19.113 seconds.

| Complete Bunny small test | PSNR | SSIM | Standard LPIPS |
| --- | ---: | ---: | ---: |
| Frozen PORT round two | 37.600658 | .986484 | .019684 |

The average hides severe failures: [frame 332](../../runs/research_20260912/bunny_full_s0/test/hard_332.png)
is 12.143550 dB and [frame 98](../../runs/research_20260912/bunny_full_s0/test/hard_98.png)
is 12.436795 dB. Both remain in the 500-frame aggregate. Median PSNR is
38.047665 dB, with 7/500 frames below 20 dB and 9/500 below 25 dB. Main-task
inspection of frame 332 finds a clear side-view bunny in GT but severe predicted
view/outline and shape errors. This is a substantial view-specific failure;
the image alone does not establish a unique cause or a dataset bug. The SSS-GS paper's
35.01 dB small-data result is a synthetic-scene aggregate, not a verified Bunny
single-scene reference, and its metric defaults differ. The present 37.60 dB
therefore does not establish a win over SSS-GS. The unit-light setting remains
an explicit equal-power assumption.

Evidence: [Bunny result](../../runs/research_20260912/bunny_full_s0/result.json),
[complete test metrics](../../runs/research_20260912/bunny_full_s0/test/metrics.json),
[protocol boundaries](../research/related_work_20260912.md).

## Completed full-fit summary — September 13 JST

| Frozen 30k full fit | Train / test | PSNR | SSIM | LPIPS | Points | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cat | 522 / 66 | 21.501174 | .766281 | .227281 | 398851 | 1579.640 |
| Translucent | 2000 / 400 | 28.304924 | .960368 | .051791 | 399830 | 1539.787 |
| Bunny small | 500 / 500 | 37.600658 | .986484 | .019684 | 354592 | 2328.689 |

[Final aggregate](../../runs/research_20260912/full_results.json) records **3/3
complete**. Two structural rounds preceded the three fresh full fits. All ran
serially on GPU 0, which is released. The accepted source and parameters stayed
fixed; there was no external-data tuning, ablation or other-method training.
Complete existing data were reused under `/workspace/datasets/`, with no duplicate
downloads. Source archives and all earlier outputs remain intact.

Final code accounting: eight production modules 2009 → 1531 lines (23.79%
reduction); all top-level Python including tests 4133 → 2242 (45.75%). These are
physical-line counts with matched scope, recorded in
[cleanup statistics](../../runs/research_20260912/cleanup_files.json).
Folders retain the September 12 launch date; completion is September 13 JST.

The following results are the completed September 11 research cycle.

# Real-scene repair results — 2026-09-11

Two structural rounds are complete. The accepted implementation fixes opacity
reset scheduling and propagates shadow derivatives to geometry/opacity at fixed
per-forward sampling settings. Independent source, derivative and result reviews
are complete. Both frozen full fits and their complete official tests are
finished. All training/evaluation processes have exited. [Setup](setup.md) records budgets, exact commands and data membership.

## Cat: fixed 30k training-derived validation

470 fit / 52 validation frames, seed 0, 512px, uint8 metrics, fixed last checkpoint:

| Run | PSNR | SSIM | Standard LPIPS | Gaussians |
|---|---:|---:|---:|---:|
| Historical `cat_localized_s0` | 21.699086 | .770378 | .255818 | 318381 |
| `cat_refinement_r1_s0` | 21.646310 | .770942 | .257096 | 216456 |
| `cat_shadow_gradient_r2_s0` | **22.201852** | .770023 | **.253591** | **156667** |

Round 1 corrects a verified schedule bug and reduces model size, but leaves Cat
image quality slightly lower. After actual image/data diagnosis, round 2 adds
conditional shadow derivatives. Its configuration matches round 1 except for
the output directory; the forward rendering function remains numerically equal
for fixed weights. R2−R1 PSNR is +.555542 dB, with median +.391666 dB and 33/52
frames improved. LPIPS improves .003506 and 31/52 frames; SSIM improves on 23/52.
Alpha L1 increases .019837 → .020715.

Relative to the historical localized run, final PSNR improves **.502767 dB** and
point count falls **50.79%**. Fit32 improves from R1's 24.503358 / .822503 /
.229579 to R2's **25.575004 / .833277 / .221392** (PSNR / SSIM / standard LPIPS).

### What the pictures and error data establish

The same 84-frame diagnostic covers fit32 and validation52. Validation
quantized MSE falls .00801867 → .00704586 (**12.13%**); 16x16 block-averaged
error falls .00453249 → .00362905 (**19.93%**). Two-pixel detail energy grows
26.20% → 27.10% of GT, while substantial blur remains. Silhouette IoU slightly
falls .95787 → .95647. Mean camera-weighted support radius falls 23.61 → 21.81px.

Hard frames 233 and 158 remain poor and slightly worsen: 14.8659 → 14.7876 and
16.8408 → 16.3586 dB. Their broad lighting and alignment errors remain visible.
These results support an overall RGB/LPIPS improvement, with clear unresolved
failure modes. Model visibility-bin membership changes between runs; bucket
averages describe each model and do not identify a same-pixel causal effect.

See [image diagnosis](cat_image_diagnosis.md),
[paired frame changes](../../runs/cat_round2_paired_validation.json), and the
[R2 diagnostic data](../../runs/cat_shadow_gradient_r2_image_diagnosis/metrics.json).

## Pixiu: first-round validation and full fitting

Same historical 32 validation frames, 30k steps, unquantized training metrics:

| Metric | Historical `pixiu_radiometric` | `pixiu_refinement_r1_s0` |
|---|---:|---:|
| PSNR | 23.465475 | **23.998248** |
| SSIM | .861398 | **.864636** |
| Alpha L1 | .025812 | .028809 |
| Gaussians | 93443 | **38413** |

PSNR improves .532774 dB and point count falls 58.89%; alpha error grows slightly.
The old config omits later-added defaults, including transport measure. The
rerun resolves them with the current parser and records the resulting values,
so this is a repaired-project outcome with limited single-factor attribution.

Full 56-frame uint8 validation is 23.634690 / .868399 / .156508; fit32 is
25.969370 / .894024 / .138293. These use a different metric/sample protocol from
the 32-frame unquantized table above.

## Frozen full fits and official tests

Each full fit is fresh, uses all official train frames, has empty validation,
and evaluates its fixed final checkpoint once. No test-time parameter fitting.

| Scene | Steps / train frames | Official test | PSNR | SSIM | Standard LPIPS |
|---|---|---|---:|---:|---:|
| Pixiu | 30k / 562 | 71 frames complete | **20.972036** | **.847689** | **.163980** |
| Cat | 30k / 522 | 66 frames complete | **22.000071** | **.767526** | **.249252** |

Pixiu final point count is 37965, test alpha L1 .025962 and separately labeled
LPIPS[0,1] .159615. Its output is
[`pixiu_refinement_full_s0`](../../runs/pixiu_refinement_full_s0/test/metrics.json).
Cat used the accepted r2 configuration in `runs/cat_refinement_full_s0`, GPU 1.
Its final point count is 167692, test alpha L1 .013970, and LPIPS[0,1] .226048.
The complete report is [Cat test metrics](../../runs/cat_refinement_full_s0/test/metrics.json).
The new Cat full fit uses 30k steps; the historical 100k candidate also has
different optional angular/compositing settings, so its 18.8050 dB test score is
a historical project reference rather than a single-factor comparison.

## Verification and attribution

- Independent review identified the installed gsplat expression
  `step % self.reset_every == 0 & step > 0` as always false. PORT owns its corrected
  callback while preserving the shared dependency. Actual runs log opacity resets
  at 3000, 6000, 9000 and 12000.
- The point budget constrains growth while scheduled pruning/reset continue.
  Real-frame regression covered reset boundaries and opacity Adam-state clearing.
- Existing GPU checks cover transport linearity, positivity, normalization,
  reciprocity, coordinate rotations and deferred source subsets. Cleanup preserves
  RGB/alpha exactly on the tested existing Cat/Pixiu frames.
- Shadow opacity finite differences converge: at step .001, weighted 128-logit
  relative error is .551%, single-logit error .0163%; a real receiver's checks
  stay below .066%. Geometry gradients pass finite-value checks. This does not
  constitute exhaustive geometric finite differences or exact physical visibility.
- Fixed-camera image diagnosis and direction statistics preceded choosing round 2.
  The accepted gain does not explain every calibration, silhouette or material
  error. A third structural round is unnecessary for this cycle's measured gain.

Review evidence: [code review](../project/code_review.md),
[cleanup parity](../../runs/cleanup_verification.log),
[derivative audit](../../runs/shadow_derivative_audit.json).
All old outputs, failed-diagnostic evidence and user research documents remain
at their original paths. Commands and outputs are recorded within PORT-GS.


### Final Cat project outcome

Historical official-test PSNR 18.805021 → **22.000071 dB**, a **3.195050 dB**
increase; 61/66 corresponding frames improve. SSIM .742850 → .767526 and
standard LPIPS .295151 → .249252. The historical model uses 100k steps and
different optional Gaussian-frame/alpha settings; this reports project progress,
not a single-factor attribution. The same-config r1/r2 internal validation above
provides the narrower evidence for shadow-gradient utility.

The previous worst test frame 64 improves from 10.9224 to 21.4234 dB. The current
worst is frame 41 at 18.4222 dB. Final image inspection still shows blurred fur,
soft paper folds and some alignment/brightness errors. Both the worst current
frame and the historical failure frame were inspected with original cameras and
fixed weights; inspection did not modify metrics or select another model.

[Final frame changes](../../runs/cat_final_test_summary.json) ·
[Current difficult test frame](../../runs/cat_refinement_full_s0/test/inspection_041.png) ·
[Historical failure frame, current prediction](../../runs/cat_refinement_full_s0/test/inspection_064.png)

Independent final review confirmed fresh full fitting, empty validation,
`best_validation_psnr=null`, completed 30k logs, fixed `last.pt`, `limit=0`, and
complete 66/71-frame test membership. The cycle is complete after two structural
rounds, with measured improvements and the remaining limitations stated above.
# 2026-09-21：directional_surfel实现验证

本次是接口/梯度验证，没有完整质量实验。基于工作树47028ea继续修改，
保留此前用户清理改动；训练环境ssd-gs、Torch2.4.1、gsplat1.5.3、CUDA12.1，
GPU0为RTX6000 Ada，seed0。

- `python test_methods.py -v`：8项通过，包括现有方法、圆盘面积与切平面分裂、
  优化器状态、camera-Z法线方向与梯度、DA3深度损失的尺度不变性及形状敏感性。
- 原始Cat train中等距选择4帧（0、173、347、521），保持真实图像与标定。
  教师以512处理分辨率推理，监督缩至32；小模型128点、8端口、3步，
  从第1步启用阴影和端口，测试中从第1步启用全部几何监督。
- StableNormal（含YOSO及DINOv2）与DA3-Large真实预测成功，保存到
  `runs/surfel_supervised/surface_priors/`。
- `runs/surfel_supervised/check/`：带法线/深度监督训练、严格checkpoint重载及CLI评估通过；
  重载前后像素差为0，两类先验损失对几何位置的梯度非零且有限，法线先验对四元数亦如此。
- 两个平行圆盘的shadow检查：前盘V=1，后盘V=0.300350（前盘opacity=0.7）；
  遮挡梯度方向正确。修复了两层插值造成的轻微自遮挡。
- 真实图像的native absgrad增密触发切平面分裂，增加8个圆盘，优化器状态通过检查。
- `runs/surfel_integration/`：新方法不带教师的接通检查和原默认方法三步训练/重载/评估通过。
- 六场景surface-priors路径模板、源码归档包含新模块且排除环境/权重、语法和diff检查通过。

重复验证使用既有test_method_integration.py的--methods、--prepare-priors或
--surface-priors参数；没有新增一次性实验脚本。小规模fit指标不用于声称提升重光照质量。

## 2026-09-23：修正交点深度后的30k SDF对照（Pixiu）

两组固定30k最终模型，562 fit与全部71 official test完成。详见[sdf_geometry](sdf_geometry.md)。

|模型|test PSNR|SSIM|LPIPS|亮点MAE|亮点召回@2px|
|---|---:|---:|---:|---:|---:|
|2DGS|21.248259|.843846|.161047|.362038|.160898|
|2DGS+SDF|21.315856|.844464|.160689|.361996|.151375|

SDF改善法线/深度一致性，但未恢复高光位置/对比度；不提升为默认方法。数据、图像和权重审计在
runs/neural_material_sdf_fresh_validation。输入相机与图像特征匹配有独立偏差证据，下一阶段单独检验训练相机修正。

## 2026-09-23：训练相机修正对照（Pixiu，30k+5k）

|精修方式|test PSNR|SSIM|LPIPS|测试亮点MAE|平台极线误差px|
|---|---:|---:|---:|---:|---:|
|原训练相机|21.059702|.842716|.161373|.360458|1.775818|
|优化训练相机|21.322745|.843663|.155105|.357679|.682851|

完整71帧test保持原始标定。整体画质与训练几何对齐改善，但测试高光位置/对比度仍不准确。
输出runs/neural_material_camera_refine_pilot；两组decoder冻结、点数一致、权重有限。


## 固定校正相机后的SDF互监督（2026-09-23）

Pixiu同35k初始化各精修5k，完整71 test：control/sdf PSNR21.308830/21.331654，
LPIPS .152874/.152835，高光MAE .359883/.358917、召回@2px .178108/.179523。
法线/深度夹角17.734°/16.569°。SDF仍未恢复准确亮点；相机、decoder逐位不变，330930点且张量有限。
完整comparison.json、comparison_train/test.png、final_audit.json在runs/neural_material_calibrated_sdf_pilot。
协议与后续解析材质对照见[sdf_geometry](sdf_geometry.md)。


## 固定几何的神经/解析材质对照（2026-09-23）

Pixiu同35k模型、冻结几何和相机，两组reset材质后各5k，neural/ggx全562 fit PSNR25.085764/24.879871，
LPIPS .131759/.136268；16train亮点MAE .169378/.177121，召回 .594805/.564750。
解析GGX未胜出，不宣称预训练已被替代；没有用official test选这两个候选。
比较图、完整指标和固定权重审计在runs/analytic_material_pilot。
后续共享光强尺度试验由反照率饱和和亮度范围诊断推动，协议见[sdf_geometry](sdf_geometry.md)。


## 共享光强尺度对照（2026-09-23）

fixed/calibrated同初始化各5k、固定几何/相机，完整71test PSNR21.271131/21.232289，
LPIPS .152717/.152907；亮点MAE .364083/.360717、召回 .176502/.184075。
全局gain学到1.360451。高光部分指标微幅改善，整体未改善，未设为默认。
完整train/test对比、图像和审计位于runs/neural_material_light_scale_pilot。


## SDF直接图像监督可行性（2026-09-23）

主Gaussian完全固定，128px同16train视图、两组各3k。Gaussian参考/固定距离场/更新距离场：
PSNR23.912845/23.033681/22.525679，LPIPS .113197/.176761/.159646，alpha L1 .026104/.028959/.028304。
更新场有部分感知/掩码改善但PSNR更差，两个SDF分支均缺乏细节，不能称为主GS高光改善。
全部权重、相机、场更新计数和边界审计通过，证据runs/sdf_volume_feasibility。


## SDF空间细节对照完成（2026-09-23）

`sdf_volume_detail_pilot`两组各3k及同16训练视角评价完成，128px、seed0、全部562训练帧。
从同一`runs/sdf_volume_feasibility/fixed_field`模型扩展（实际路径见下文）；两组均增加外观网格并更新距离场，
只有geometry_grid增加SDF网格，所有主GS/传输/相机固定，无重复头预热。

| 同16训练视图 | appearance_grid | geometry_grid | 不变的Gaussian参考 |
|---|---:|---:|---:|
| PSNR |23.122093|23.502200|23.912845|
| SSIM |.871221|.876693|.882637|
| LPIPS |.129547|.126037|.113197|
| alpha L1 |.029109|.026025|.026104|

几何网格相对外观网格对照提高0.3801dB，alpha误差下降10.6%；支持进一步检验几何表示分辨率。
两分支仍明显平滑，细小高光未恢复，整体图像指标仍低于主GS参考。这里没有几何GT，也没有held-out验证；
不能将此解释为真实几何精度或主GS高光改善，本轮不修改默认方法，不加强SDF→GS监督。

最终逐位审计：GS/主传输/相机均与源相同；两组field_steps8000/head_steps6000，场均更新，所有张量有限。
两组外观网格投影非零，geometry_grid几何网格投影非零；六面32²采样的边界SDF均正，最小.340119/.345413。
16个评价帧及保存的校正相机一致。协议仅output/sdf_detail不同，首次记录loss严格相同。

两训练视图0/35的采样加倍检查：64粗+64细改为128粗+128细。
外观组前景平均绝对RGB变化.000366/.000653，几何组.000730/.000822（量化后的观察空间）。
对应PSNR变化为−.038/−.012、−.101/−.038dB；局部最大像素差分别达.161/.239。
平均变化小但局部尚不稳定，只有两视图，不能宣称采样已完全收敛，也不能排除薄结构漏采样。

所有训练/评价/采样进程结束，GPU0/1已释放。结果与审计：
`runs/sdf_volume_detail_pilot/{comparison.json,comparison.png,final_audit.json,protocol_audit.json}`；
`sampling_appearance_grid.json`、`sampling_geometry_grid.json`及对应源码归档保留。
本次没有再开启训练组；用户要求收尾后在新主任务接续，当前工具缺少create_thread/handoff_thread。


## 2026-09-23：SDF亮点采样是负结果

`sdf_volume_peak_sampling_pilot`同源固定距离场/GS，各3k、512px/512rays，
同16训练视图：uniform/peaks PSNR23.284552/21.152530、LPIPS.157497/.165776；
5390个GT峰MAE.343188/.114115，对比度/GT.016977/.136336，召回.009091/.154174。
候选亮度更接近但亮斑过宽/错位，所有16帧PSNR均下降，不采纳为恢复高光方案。
主GS冻结，同16帧参考PSNR26.078235，未有主模型/test/真实几何改善。
完整输入诊断、邻环误差、参数审计及图像见[实验记录](sdf_peak_sampling.md)。


## 2026-09-23：峰邻域监督部分缓解宽斑，提示编码未胜出

`sdf_peak_shape_pilot`两候选各3k及同16train/512px评价完成，控制复用上轮peaks。
context/hint_encoding PSNR21.882991/20.986815，LPIPS.164928/.165802；
峰MAE.149558/.112894，对比度/GT.153857/.129656，precision .243472/.176086、recall .226160/.133952。
context相对peaks PSNR+0.730461dB、4帧邻环过亮误差−33.80%，但峰MAE增加31.06%、小高光仍偏宽。
它仍低于原uniform PSNR1.402dB、低于冻结主GS4.195dB；hint编码未改善位置/对比/整体质量。
两组不推进official test，不设为默认。保留context作为更充分训练/射线覆盖检验的候选；
不能把拟合、alpha/sharpness变化当真实几何或主GS改善。
全部冻结模块逐位相同、状态有限、hint新增参数实际更新；源码/配置/图像/审计全部归档。
完整结果见[形状对照](sdf_peak_shape.md)。


## 2026-09-23：四倍射线预算仅有有限局部收益

`sdf_peak_budget_pilot`完成两组同source/512px/3k、2048ray，各6,144,000射线。
peaks/context PSNR21.164989/21.882586、LPIPS.164840/.164196，
峰MAE.110257/.145756、contrast/GT.155515/.178574、P/R .207569/.194249与.255305/.286642。
相对各自512ray控制，PSNR仅+0.012459/−0.000406dB；context召回.226160→.286642，
但图像仍偏宽/错位，4帧环域过亮只下降3.03%；peaks的环域过亮增加8.37%。
训练各约445秒，是控制约1.87倍墙时、4倍射线预算，不能称同成本收益。
固定模块逐位不变、状态有限、全部评价帧和GT峰池相同；无test/主GS/真实几何改善结论。
[完整结果及体积分机制诊断](sdf_peak_budget.md)。


## 2026-09-23：更薄SDF积分层未恢复细小高光

`sdf_volume_thickness_pilot`两组固定距离场及β、同context2048ray/128粗+128细/3k已完成。
β183.884/735.536对应同16fit PSNR21.822377/21.497218、LPIPS.159842/.149252；
峰MAE.149246/.155914、contrast/GT.174169/.158370、precision.245128/.213452。
薄层轮廓更锐、LPIPS较低，但小高光没有改善。4帧1–4px桶（168组件/307px）
recall .188925→.169381、contrast .134288→.114227、any-hit组件27→24；
这4帧小/中/大桶命中像素净变化−6/−5/+31，不能把总recall微升解释为小核恢复。
两组β在31日志点恒定，主GS/transport/camera/SDF逐位相同，其他head参数确实更新且有限。
各约719秒/9.866GiB，6,144,000ray×256区间；不推进此两组test。
[完整记录及分桶图像](sdf_volume_thickness.md)。
