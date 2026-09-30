# 同预算局部成对RGB差分监督

2026-09-23。协议在实现前固定，现有训练与测试入口已扩展，短测、两组30k、全部评价、终端审计及固定图块目视检查均已完成。
完整目标仍为准确小高光及整体重光照质量。
[可移动中心实验](gs_residual_movable_centers.md)失败后，独立复核选择此项，暂不继续增加角度结构。
[已做监督核查](../research/residual_objectives.md)明确旧的峰RGB+11px neutral contrast已试，本项不是首次提出局部对比思想。

## 要检验的问题

现有峰/全局环独立采样不能保证两端属于同一处高光。窄核虽然提高少量命中，却增加宽斑/误亮/色块。
新实验比较相同局部配对样本下的RGB L1与RGB L1+成对RGB差分，检验相对形状监督的实际作用。
它不是保证锐化的数学机制；两端误差符号相反时，差分L1相当于加重两端误差，需要实际验证。
不同时加入neutral-min约束、SSIM、inverse-PDF、其它正则或新的方向结构。

## 结构与source

两组均从原geometry_grid source新建geometry + multiply + wide的零输出残差头，所有初始state完全相同，
包括相同wide尺度buffer。不续学过的wide模型；不改中心结构、clamp、光强/距离因子或主模型/相机。
唯一实验变量：control的pair权重0，treatment为.25；两组都使用相同新配对采样。
历史wide及multiply只作参照，不能把它们当同轮匹配控制。

## 每步2048 draw预算和样本关系

1. 512个GT峰锚点，每个对应1个局部环像素，共1024个端点。
2. 另加512个当前规则的foreground、512个rays.valid样本；重复抽样按次数进入损失。
3. renderer仍只查询unique covered pixels，两个端点必须都包含在residual_indices中。

沿用neutral_peak_mask。锚点在rays.valid内，候选环为该锚点中心11×11方窗，排除所有GT峰、
要求GTalpha>.9且rays.valid；从有非空环的锚点中均匀抽峰，再从其自身环均匀抽1点。
不按模型误差、预测亮度、组件是否已命中或任何test结果选点。
为保留现有分布，锚点按合格峰像素采样，不引入额外的小组件均匀采样策略。

若整帧无eligible anchor，不制造伪pair，2048全部按现有foreground/valid各半规则回填；
foreground为空时全用valid；pair项为有计算图连接的零。记录峰总数/eligible数、组件桶覆盖及退化帧次数。
这种采样会改变历史全局环分布，所以必须有本轮共同sampler的RGB-only控制。

采样后用本次冻结GS alpha检查两端均>0；不支持的pair不进入差分项、不重采样，普通RGB项仍保留。
差分项始终按原512个pair槽位归一化，不能因有效pair很少而放大权重；记录排除率。
若未来允许不同ray预算，则要求能整分为4，pair槽数为N/4，两个基础池各N/4；本正式实验仍固定2048。

## 固定损失

在原有未量化observation RGB上计算，使用重复draw的权重：

L_rgb = sum_{2048 draws,3 channels}|P−T| / (2048*3)

L_pair = sum_{supported pairs,3 channels}|(P_peak−P_ring)−(T_peak−T_ring)| / (512*3)

L = L_rgb + lambda*L_pair，lambda取0或.25。

选择逐通道RGB差分而非仅minRGB：当前失败包括色块，RGB差分同时约束局部颜色变化，
也不把梯度只送入当前最小通道。.25=512/2048使每端点每通道的差分梯度系数与其普通RGB系数一致；
这不保证参数梯度范数相等，参数共享及重复样本仍会改变实际范数。
两组都记录未加权pair项、加权项、RGB项和总loss；初始RGB/pair原值相同，总loss因lambda不同应不同。
不要沿用“首步总loss相同”的旧审计断言。

## 实现与必要检查

复用train.py、现有采样辅助模块及测试入口，不新建训练器，不从部分修正全图计算11px pool损失。
建议--residual-paired-context与--residual-pair-weight，默认关闭/0保持旧数值路径；正权重要求配对mode，
配对mode要求residual及各.25峰/环配额。本stage mode/weight写config，续训需显式匹配；它们不影响推理公式。
GT局部候选池可按现有掩码缓存方式预计算，不同时实现冻结receiver缓存或其它渲染优化。

CPU构造例检查2048等比例预算、局部关系/边界/排除全GT峰、重复权重、退化帧回填、支持掩码、
两端梯度及lambda0控制；真实3+3检查初值完全一致、实际head更新、源状态冻结、保存重载与损失合同。
帧采样仍Python seed0，ray generator仍独立seed+3；两组样本选择不得依赖其不同的预测。
日志记录RGB/未加权pair/加权pair、总pair和支持数、抽样计数。
可在预定step1/100/3000/12000/30000记录两项对末RGB权重的梯度RMS/余弦；只作解释，不动态调权。

## 固定预算与评价门槛

canonical新name建议gs_residual_paired_loss_pilot，variants rgb/paired（仅权重0/.25不同），
各30k/full562/512px/2048 draws/seed0，LR在3k衰减至.00028后保持，save3k/12k/30k。
最多2空闲GPU并行，ssd-gs不升级。三个节点fit16、终端full562，先eval_test:false。

treatment相对匹配control仍须满足原七项数字条件及固定crop：

- tiny abs(contrast/GT−1)下降≥.03、recall增加≥.03、RGB MAE下降；
- 全峰precision下降≤.01，固定4fit11px环域neutral过亮增加≤5%；
- PSNR下降≤.1dB，LPIPS增加≤.002；固定GT图块支持可靠窄峰改善。

本轮开跑前额外固定treatment对原GS的整体质量条件：完整562 PSNR≥25.259189694−.1，
LPIPS≤.129588453+.002，SSIM≥.891718217−.002。精确基值读取gs_residual_budget_reference/fit_full/metrics.json，
不以文档舍入数替代。此前几乎全部帧SSIM/LPIPS退化，只赢弱控制不足以再次使用开发期test。
此新绝对门槛仅约束未来本轮，不追溯改写已完成SG实验的门槛或结果。

全部预定条件通过才两组固定30k一起完整71test；不测单一胜者、不降门槛、不改预算。
原GS和历史wide/multiply的指标及代价都必须报告。562是源已见训练拟合；71是开发期观察过的未训练帧，
非新盲测、纯固定相机换光或真实几何质量证明。完整目标尚未完成。

## 已完成实现检查

新增residual_sampling.py，扩展train.py与既有test_methods.py/test_method_integration.py；不改head/renderer/evaluate。
7项CPU检查通过：局部候选/边界/预算、像素均匀抽样与组件覆盖、空环资格、退化回填、重复权重与两端梯度、
支持掩码及固定分母、lambda0控制。首次重复RGB测试例恰好让重复均值等于唯一均值，断言失败；
只修正测试数据后通过，生产代码未因此修改，首失败日志保留为cpu_tests_initial_failure.log。

两组native512/2048真实3+3、严格续训模式/权重、冻结状态、保存重载和CLI评价均首次通过。
25个初始tensor字节完全一致，配置仅output和pair weight不同；两stage抽样/支持/资格/组件覆盖计数完全一致。
首RGB .07610297203063965和原pair .18261705338954926均相同；总loss分别.07610297203063965/.12175723910331726，
符合权重0/.25。主GS/transport/camera/radius精确固定，head更新。短测不是质量结果。
证据runs/gs_residual_paired_loss_smoke/{cpu_tests.log,verification.json,source.tar,rgb,paired}。


## 历史正式启动记录（现已完成）

`bash launch_validation.sh configs/validation.json`，新name gs_residual_paired_loss_pilot；GPU0/1先检查空闲。
2026-09-23 05:04:20 UTC开始，rgb/paired实际训练进程228851/228852、scheduler228845。
实际训练日志和GPU活动已核对；无test任务。预算与门槛按上文固定，最终结果如下。

终端CPU分析脚本/tmp/port_residual_paired_loss_audit.py经独立schema与root完整复核、编译后，首次执行通过。
startup_audit.json/audit_preparation.json保留启动与准备证据，final_audit.json/log记录完整结果。

## 训练采样总体核查（非质量结果）

12k保存状态已包含全部562帧的固定GT资格，双方完全一致，汇总于sampling_population.json。
原浮点训练mask共212982峰像素，全部在valid域且都有非空局部环，没有退化帧。
1–4px桶为21252组件/40444px；5–16px为8474组件/72324px；>16px为3022组件/100214px，
全部组件都有合格锚点，且到12k均至少被抽中一次。这不证明各组件被充分优化或准确拟合。
12k每组6144000对中6142037对两端有GS支持，过滤1963对（约.03195%）；没有重新抽样或改变分母。
训练mask使用原浮点图，不能与量化评价的212877峰像素、21237个最小组件混用，差异不是帧遗漏。

## 完成结果：局部指标收益伴随整体退化，不采用

2026-09-23 05:40:45 UTC全部canonical任务完成。没有追加test，没有改变默认模型。
完整562帧均为训练拟合；以下指标不是未见灯光泛化或真实几何质量证据。

| 方案 | PSNR | SSIM | LPIPS | 全峰precision |
|---|---:|---:|---:|---:|
| 原GS | 25.259190 | 0.891718 | 0.129588 | 0.631668 |
| 历史multiply | 25.242683 | 0.876765 | 0.137921 | 0.566809 |
| 历史wide | 25.365816 | 0.878177 | 0.137976 | 0.575195 |
| 匹配RGB | 25.173248 | 0.876975 | 0.138497 | 0.536719 |
| paired | 24.732770 | 0.871918 | 0.141648 | 0.469261 |

1–4px桶固定21237组件/40429px，先池化raw统计再派生：

| 方案 | RGB MAE | 对比/GT | recall(2px) |
|---|---:|---:|---:|
| 原GS | 0.202598 | 0.294762 | 0.358357 |
| 历史multiply | 0.140797 | 0.410018 | 0.503277 |
| 历史wide | 0.144804 | 0.406718 | 0.498231 |
| 匹配RGB | 0.139224 | 0.414460 | 0.517450 |
| paired | 0.138402 | 0.456402 | 0.571768 |

paired相对匹配RGB：小峰对比提升.041941、recall+5.4317pp、MAE下降.000823；
固定4fit11px环neutral过亮.065297891→.067466088（+3.3205%），这四项通过预定门槛。
但全峰precision下降6.7459pp、PSNR下降.440478dB、LPIPS增加.003151，三项失败。
对原GS的PSNR/SSIM/LPIPS三项整体质量门槛也全部失败，未下调任何门槛。
RGB相对原GS的SSIM562/562、LPIPS554/562帧退化；paired分别562/562、558/562帧退化。
历史multiply/wide只作参照，其全局环采样不同，不能作为本轮匹配控制。

固定8个GT图块目视检查已完成且失败：frame0B的孤立亮点仍未解析，并增加黑块；
frame35A出现大量错误亮粒及合并亮块；frame105A浅色区域的孤立GT亮点仍大多缺失。
若干响应更亮，但不构成可靠的形状和定位恢复。manual_review.json及三个gate副本均已同步completed/false。
不追加71test，不将本轮峰对比/召回改善描述为总体重光照成功。

## 实现、来源与成本审计

最终CPU审计首次exit0：25个初始tensor字节相同，config仅output和权重不同；
全部六个checkpoint的GS/transport/camera/radius逐位冻结，head确实更新且所有状态有限。
源码与正式归档核对一致，抽帧顺序/LR/完整预算及两组资格、支持、累计采样和组件覆盖一致。
没有保存每步全部pixel IDs，因此不把计数/种子/相同采样器核对夸大为逐步原始索引日志审计。
fit16/full562的RGB有3项、paired有5项激活统计末位差，最大分别7.45e−9/1.49e−8；
均在预先声明rtol1e−6/atol1e−8范围内，GT/alpha/count和图像/组件指标一致。

每组61.44M draws、15.36M pairs，两端有支持15355503对、过滤4497对（约.02928%），没有退化回填。
源/新头全562均见，stage每帧34–80次更新。训练耗时2049.78/2060.71秒，峰分配显存均7.80154GiB。
相同硬件同期paired约比RGB慢.53%；历史wide时间只是另次执行参照，不能作为受控加速测量。
GPU0/1已释放；全部日志、checkpoint与评价保留，不重启完成的canonical名字。

产物位于runs/gs_residual_paired_loss_pilot：comparison.json/png/crops、training_gate.json、manual_review.json、
final_audit.json/log、small_peak_audit.json、paired_loss_curve.png、paired_sampling_loss.png、analysis_source.tar；
startup_audit.json、sampling_population.json、audit_preparation.json保存阶段证据。

## 用户指引后的研究主线调整

用户指出连续不改善可能是方案本身有问题，允许从底层换方案。结合多轮可靠负结果，
停止把当前冻结GS上的自由RGB残差头带宽、采样或损失微调作为下一主线；不再默认当前PORT/材质分解必须保留。
正在核查实际SSD-GS/GS3完整checkpoint、来源与高光表现，再决定基础表示与光传输替代。
这是下一研究方向，尚未选定或实现新架构；不能把本轮负结果宣称为所有类似表示不可能。
完整目标仍active。
