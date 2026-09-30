# 可移动外观中心的宽／窄球面高斯对照

2026-09-23。实现、短测、两组30k及完整训练范围评价、最终审计和人工图块检查全部完成。窄核未通过5/7数值门槛，不扩test、不改默认；完整研究目标仍active。
[局部提示审计](gs_residual_angular_cues.md)完成后独立复核选择此项；完整目标仍为准确小高光及重光照质量。

## 动机与主比较

固定geometry normal下，遗漏小组件87.21%的局部提示极值离GT超过2px，仅0.55%满足窄核筛选。
这不证明几何错误，也不证明网络容量上限，但不支持直接再做固定法线窄核30k。
下一项让两个组的外观角度中心都可学习，主比较为相同中心自由度下的宽／窄响应。
入射×出射PE乘积留作后续候选，它没有直接检验此次发现的局部中心错位。

这不是旧normal-field实验重跑：旧场修改受限的主着色法线并通过冻结BRDF；本项只生成残差头的额外角度特征，
不传给几何、shader、阴影或PORT。不是GS³复现，也不称恢复物理法线或物理BRDF。

## 固定最小结构

从原geometry_grid源新建geometry-normal的multiply残差头，不续已学残差。
保留当前94维输入、空间网格、原网格投影及multiply交互，不改变旧normal/cos/log hints。
令k是同一次网格查询的24维值，PE(x)是原27维位置编码：

- D：Linear(51,32) → ReLU → Linear(32,3)，含bias；最后Linear的权重和bias为零，共1763参数。
- d=D([PE(x),k])，m=normalize(n_geometry.detach()+d)。m只控制下面的新核。
- K_j=exp((clamp(m·h,−1,1)−1)/tau_j²)，h来自实际wi/wo。tau用弧度。
- Q：Linear(8,128,bias=False)，普通随机初始化，1024参数；首层ReLU前增加Q(K)。
- 原末RGB继续零初始化，初始最终图像保持原GS；本项新增2787参数，总2200784。
- 两组都训练D/Q以及原残差；wide尺度8°–128°、narrow尺度2°–32°，各8个对数均匀尺度。
- 两组所有可学习初始张量及公共buffer逐位一致；仅角度尺度buffer明确不同。保存fresh初值进行核对。

该球面高斯不用训练期acos，避免可学习中心在dot=±1处的奇异导数；在小角度处cos(theta)−1≈−theta²/2，
与诊断用的角度高斯有相同局部曲率。它在大角度处是不同函数，尤其128°尺度。
tau应称局部曲率对应的角度尺度，不称严格角度标准差。固定中心的所有尺度仍共享极值位置。

normalize沿用项目约定；不临时加入中心限幅、正则、Fresnel、各向异性或学习带宽。
第一步因末RGB零，中心梯度为零是预期；D第一层还需等待零末层开始更新。
窄核远离中心处可能下溢、缺少梯度，是需要记录的失败模式，不事后改尺度补救。

## 接口与检查要求

建议--residual-angular-bank none/wide/narrow，默认none保持原state键及行为；本实验启用时要求
radiance-residual、residual-interaction=multiply、residual-normal=geometry。mode存config并严格续训/评价载入。
只需扩展现有head/train/evaluate/renderer和已有测试入口，不新建训练器或替换共享依赖。
新增模块在旧模块之后初始化，复用一次grid查询，不改变主GS/transport/camera/radius状态。

必要检查：初始图像精确保持；两组初始可学习张量/公共buffer相同且尺度差正确；中心/Q的后续梯度、
真实中心移动、输入停止梯度、光强齐次、保存重载与模式检查；角度dot=±1处梯度有限。
完成真实3+3短测后再正式启动。普通CPU测试须CUDA_VISIBLE_DEVICES为空；GPU启动前核查空闲。

记录每核激活、D/Q梯度尺度、detach后的中心转角，以及||n+d||最小值/低范数比例与有限性。
分块或跨帧最小值必须取min，不能沿用普通均值的查询像素加权；分位数若记录须由原始分布计算，
不能平均各chunk分位数。没有正则或其它机制据这些诊断动态干预训练。

## 预算、评价与预定继续门槛

新run建议gs_residual_movable_center_pilot；canonical configs/validation.json设variants wide/narrow，
各30000步、完整562fit、512px、2048 rays、25%峰+25%11px邻环、seed0、geometry normal，
LRdecay3000后保持.00028，save3k/12k/30k。最多2张启动前检查空闲GPU并行，ssd-gs不升级。
固定三节点fit16和终端full562，eval_test:false；不以中途结果改预算或挑checkpoint。

narrow相对wide完整562的1–4px桶须同时满足：

1. abs(contrast_ratio−1)下降≥.03，recall增加≥.03，RGB MAE下降。
2. 全峰precision下降≤.01；固定4fit的11px环域neutral过亮增加≤5%。
3. 整体PSNR下降≤.1dB，LPIPS增加≤.002。
4. 固定GT crop有可靠窄峰改善，不能只新增宽亮斑。

全部通过才两组固定终端模型一起完整71test；否则不扩test，不降低门槛。
原GS及历史multiply都需报告；历史模型不是同轮中心自由度消融，不能把对它的差异全部归因于中心学习。
即使胜wide也不自动采用；与原GS相比的SSIM/LPIPS、误亮和泛化代价必须如实报告。
562帧已经参与源模型及先验训练，结果仅为拟合；71test也是开发期观察过的未训练帧，非新盲测或纯固定相机换光。

主要失败风险：中心记忆已见视角、窄核缺梯度或由宽端独占、错误接收位置使单一位置中心无法同时解释多帧。
结果不能单独证明真实几何/灯位错误。灯位继续使用独立世界坐标；[元数据核查](../research/light_camera_contract.md)
不支持相机刚性灯架假设，因此不自动把灯随保存相机offset搬动。


## 实现与正式启动前检查

现有head/renderer/train/evaluate已接入--residual-angular-bank，默认none保持旧行为。
尺度在CPU构造后转设备，严格载入时核对保存buffer与配置，避免CPU/GPU的logspace末位差异。
最小值按非空query/frame取min，其它均值和比例按query像素加权；空query最小值为0哨兵，不参与非空汇总。
运行中记录中心两个Linear、Q的参数/梯度RMS，各核Q列梯度，以及激活/旋转/raw中心范数诊断。

既有9项残差/交互CPU检查通过，新4项MovableAngularCenterTests首跑通过（日志cpu_tests.log）。
GPU0/1的wide/narrow真实512px、2048射线、3+3步、保存重载及CLI评价均首跑exit0。
D/Q实际更新、后续梯度非零，首步零RGB导致D/Q梯度为零；原GS/传输/相机/radius全部精确保持。
保存再载入实际渲染逐位相同；缺bank、错误bank及续训切换bank均拒绝。
两组24个公共初始state张量逐位相同，唯一不同为8维角度尺度buffer，宽度相差4倍；
配置仅output和residual_angular_bank不同。独立manifest审查确认正式命令同样只有这两项区别。
短测产物runs/gs_residual_movable_center_smoke/{wide,narrow}/report.json、verification.json、initial_gradient_checks.json。
没有失败重试；源及命令分别归档。最终两条测试断言强化后用已保存首步日志和错误日志复核通过，未重复GPU检查。


## 完成结果：窄核未可靠恢复小峰

以下均为同一562帧训练拟合，最小桶21237组件/40429px。历史multiply没有新增中心，
仅作参照，本轮主控制是两组都有可学习中心的wide/narrow。

| 完整562fit | 原GS | 历史multiply | wide | narrow |
|---|---:|---:|---:|---:|
| PSNR |25.259190|25.242683|25.365816|25.183454|
| SSIM |.891718|.876765|.878177|.876021|
| LPIPS |.129588|.137921|.137976|.139161|
| 小桶MAE |.202598|.140797|.144804|.137793|
| 小桶对比/GT |.294762|.410018|.406718|.416703|
| 小桶recall |.358357|.503277|.498231|.508323|
| 全峰precision |.631668|.566809|.575195|.546558|
| 全峰recall |.536869|.733409|.742875|.747474|

narrow相对wide：小桶对比改善.009985、recall增加.010092，均低于.03要求；
全峰P下降.028638，固定4fit11px邻环neutral过亮.064498→.072086（+11.7645%），
PSNR下降.182362dB，以上五项未通过。小桶MAE下降.007011、LPIPS增加.001185，在对应两项允许范围内。
不能将MAE下降或更多命中当作可靠窄峰恢复：预测峰总数375834→399257，precision同步下降。
小桶命中20143→20551px、any-hit9905→10151，只有有限增加。

wide的完整fit PSNR虽比原GS高.106626dB，但SSIM561/562更差、LPIPS551/562更差；
narrow的SSIM全部562更差、LPIPS549/562更差。两个模型都不能作为整体质量赢家。

| 同16fit | wide3k | narrow3k | wide12k | narrow12k | wide30k | narrow30k |
|---|---:|---:|---:|---:|---:|---:|
| PSNR |25.734663|25.777032|25.724910|25.787009|26.057339|25.922768|
| LPIPS |.126384|.126388|.127726|.126515|.126459|.127897|
| 小桶对比/GT |.333739|.335328|.401264|.387358|.435237|.457458|
| 小桶recall |.414239|.432848|.487055|.490291|.533172|.546926|

固定8个GT图块人工检查：frame0B两颗独立GT点仍未恢复，两组在宽错位亮斑周围新增暗色/色块；
35A窄核比宽核出现更密白色碎斑和合并亮纹，宽核也有错误紫色块；35B保留局部亮尖，
但历史multiply已做到；70A条状高光仍部分合并；105A浅色区域多个GT亮点仍缺失。
因此人工门槛也不通过。manual_review.json与training_gate/comparison/final_audit的三份gate已同步。
未下调门槛、未追加71test、未采用默认；不能由本轮训练结果推断未见灯光泛化。

## 中心与激活：实际学到了变化，但没有转化为可靠峰形

完整562全GS覆盖查询共38945262像素，两组相同。中心平均转角wide14.572866°、narrow27.844391°。
D的两层、Q以及其它头参数实际更新，六个节点都有限；并非中心一直冻结或没有梯度。
这些是外观特征中心，不能当真实几何/法线恢复。m还依赖实际投影receiver的冻结混合法线。

wide最小||n+d||=.04616679，低于.1的仅3个完整查询像素（frames465/399/199各1），
合池比例7.703e−8；narrow最小.24569882，低于.1比例0。训练稀疏日志未见<.1，
因此不能只用训练抽样日志替代全图诊断，也没有证据这3点解释整体失败。

narrow各核在完整562查询的精确零值比例为60.6621%、27.0345%、3.3203%、.03037%、0、0、0、0；
wide全部为0。2度核的稀疏训练日志无权平均零值比例51.8445%，不能与全图合池60.6621%混用。
所有8个Q列在首步之后的300个记录点都有正且有限的梯度；不是证明每个训练步骤都非零。
窄核稀疏和下溢是实际参数化特征，不能单凭此统计把失败归为未激活，更不能事后改带宽补救。

## 最终审计与输出

复用canonical命令bash launch_validation.sh configs/validation.json，source基版本47028ea0ff77f427aa6a016cb64e039432935a5e；
实际脏源码以run/source.tar为准。仅GPU0/1各30k、61.44M draws；耗时1972.27/1985.37秒，峰allocated约7.632GiB。
两个节点3k/12k时长wide197.73/788.95秒，narrow197.62/791.27秒。
帧抽样和累计计数一致；3k每帧0–17、12k10–35、30k34–80，完整来源仍是562训练帧。

最终CPU审计首跑exit0无失败。24个公共初值tensor逐字节相同，唯一不同的尺度buffer与保存配置匹配、全程不变；
六个checkpoint的主GS/transport/camera/radius与原source完全相同，渲染配置/PORT/阴影状态一致。
逐帧和合池指标重算通过，min正确只取非空查询最小值；所有GT、alpha与组件计数精确一致。
独立fit16/full562仅wide1项/narrow4项激活统计末位差，最大1.49e−8/9.31e−10，在预定紧容差内记录；
其它数值包括图像metric和组件统计都相同。源、初始化和测试日志均保留，没有训练或审计失败重试。
所有训练/评价/审计已结束，GPU0/1释放，无后续任务运行。

产物：runs/gs_residual_movable_center_pilot；模型/fit_003000/fit_012000/fit/fit_full在{wide,narrow}/Real_NRHints/Pixiu。
根目录comparison.json/png、comparison_crops.png、training_gate.json、manual_review.json、final_audit.json/log、
small_peak_audit.json、movable_center_curve.png、interaction_scales.png、angular_kernel_scales.png、appearance_center_scales.png、startup_audit.json和analysis_source.tar。
CPU分析脚本/tmp/port_residual_movable_center_audit.py及其三个历史helper已归档。


## 结案记录与后续状态

本轮benchmark于2026-09-23 04:38:01 UTC（13:38:01 JST）完成；最终数值审计及04:43:56 UTC的人工检查均已完成。
结案只读核查见verification.json：status及10个训练/评价步骤完成，旧进程退出，GPU计算进程为空；
training_gate.json与comparison.json/final_audit.json中保存的gate、独立manual_review.json一致。
归档source.tar与analysis_source.tar共同的六份核心源码逐字节一致；不再要求当前工作区train.py与旧run相同，
因为下一项已经开始原地实现。source.tar/analysis_source.tar和原始日志未覆盖，docs.tar保存本次结案的五份专属/汇总文档。

下一项[同预算局部成对RGB差分监督](gs_residual_paired_loss.md)正在实现，尚未启动新训练或得到新结果。
新轮额外原GS整体质量条件属于未来预定协议，不追溯修改本轮门槛。完整研究目标仍未完成。
