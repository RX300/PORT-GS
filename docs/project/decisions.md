## 2026-09-30：训练代码按职责拆分，默认行为不变

用户要求在代码审阅后重构 PORT-GS（选择“重构+修复，默认行为不变，新行为只能显式开启”）。
`train.py` 拆为固定顺序的编排循环与 `training/` 包（选项、源 checkpoint 契约、初始化、调度、相机/灯位、SDF/体积/法线场、辐射残差）；
manifest 评价阶段改为共享构造函数，并记录源码脏树状态、diff 哈希及 third_party 修订（HEAD 单独不能代表脏树代码）。
修复：rotation 相机源上的残差/体积独享阶段因 SparseAdam 拒绝 lr=0 而崩溃；续训丢失累计规范平移；`--val-limit 0` 除零；渲染器运行期 assert。
显式选项 `--opacity-reset-every`（默认 3000 等同原行为）用于检验保留导入 GGGS 不透明度。端口划分分块重算，传输峰值显存约减 6.7 GiB。
备选“删除 SDF/体积/残差/法线场等辅助分支”未采用：09-30 清理明确保留这些分支。端口渐入属于研究改动，未实现。
验证：10 场景首步逐位一致、冻结阶段整体逐位一致、其余差异在旧代码自身非确定性范围内、manifest 输出一致、93 项单元测试通过。
[训练模块](../architecture/modules/training.md) · [验证记录](../experiments/training_refactor_20260930.md)。

## 2026-09-30：按用户选择收敛为三个方法

用户明确保留 directional_port_v1、surface_attention、neural_material；其余六个方法/几何入口删除。
移除专用训练、评价、初始化、诊断、manifest 分派及测试；保留共享数学、神经材质、相机校正与已有 GGGS 初始化读取。
历史模型、数据、结果、规范实验配置不变，退役源码与 Wrapping 依赖先归档。
本次不重新排名三个保留方法，也不启动正式实验；Attention 的最新相机协议效果仍未验证。
[范围与验证](../experiments/method_retirement_20260930.md)。

## 2026-09-30：采用R2b为新的默认实验配置；不采用逐帧灯位

R2b（directional_port_v1、GGGS初始化、`--optimize-cameras --camera-mode rotation --camera-start 2000 --camera-lr 1e-3 --camera-lr-final 1e-5 --camera-gauge translation`）
在三种评价口径、两场景上均优于此前PORT与GS3/SSD-GS。R2c灯位校正差异≤0.06dB，属于噪声范围，不增加自由度。
ports消融显示其在标定正确后有效（Pixiu约0.7dB），保留。固定标定test仍以原始标定为主指标，不拟合test位姿。

## 2026-09-30：标定修正后改用directional_port_v1为主基底；加入平移规范

证据（第一轮，旋转相机校正、30k）：对齐诊断下default优于neural（Cat 29.12 vs 27.90，Pixiu 32.19 vs 31.86）；
固定标定Pixiu default 24.699/.8697/.1111，三项均超过GS3与SSD-GS。无规范约束时共享俯仰校正使坐标系竖直漂移
（Cat test偏移+10.6/+14.7px），固定标定Cat分数崩溃。决策：后续以default为基底，所有相机自标定都加`--camera-gauge translation`；
逐帧灯位校正在default基底上检验；原neural的gauge_light排队任务在启动前取消（输出目录保护，已留CANCELLED说明）。
[实验报告](../experiments/calib_camrot.md) · [研究结论](../research/calibration_consistent_relighting_20260930.md)。

## 2026-09-30：以标定为首要瓶颈；模型选择改用对齐诊断，主指标不变

证据：审计显示真实场景固定标定test误差由相机偏移主导，PORT各变体的固定PSNR差异主要来自偏移大小。
决策：(1) 重光照训练加入训练相机自标定，采用固定相机中心的旋转校正，避免单一锚点相机把自身误差带入整个重建；
(2) 主指标保持原始标定official test，不拟合test位姿；同时报告逐视角平移对齐后的PSNR/SSIM/LPIPS作为区分标定与外观质量的诊断；
(3) 不再用固定标定test的小幅差异挑选材质表示。备选“对test位姿做拟合”（SSD-GS代码实际行为）不作为主协议。
[审计](../experiments/calibration_audit_20260930.md)。

## 2026-09-27：两轮联合优化保留为实验，不推广

默认与DNA从同一GGGS先验几何独立训练各30k，均开启几何更新。DNA有更高PSNR，但LPIPS低于新默认的质量水平，细节依然缺失；相对历史DNA也没有一致胜出。
默认3DGS深度与同一GGGS连续深度回放均显示粗糙化/轮廓代理下降。后者按源坐标归一化，过滤只烘焙一次，初始物理参数和深度/alpha回放检查通过。
这不能量化真实3D误差，却说明本次接法没有保持初始几何表现。下一步可检验保留GGGS几何约束、限制几何更新幅度的方案；本轮没有额外启动该实验。
旧默认、旧DNA和源GGGS权重均不覆盖，保留四个最终联合模型及完整证据。[报告](../experiments/gggs_dna_joint.md)。

## 2026-09-27：默认完成后顺序运行DNA，保持GGGS三维几何

用户新增第二个实验：默认方法结束后，独立接DNA并允许几何继续优化。
使用同一原始GGGS权重，不继承默认方法拟合后的几何；保留3D尺度。
DNA沿用GGX和归一化32分量传输，补协方差短轴法线输入；旧2D圆盘路径不变。
原2DGS特有的畸变/表面自一致损失显式禁用，不用占位3D损失冒充；光度与分布损失保留。
关键测试及默认渲染器前后等价检查通过，实际短测及正式训练排在默认实验完成之后。
[协议与边界](../experiments/gggs_dna_joint.md)。

## 2026-09-27：默认方案采用GGGS初始化并联合更新几何

用户要求接默认relighting，随后明确允许继续优化几何；正式训练前改为联合更新位置/尺度/旋转/透明度和材质，保留默认增密裁剪。
导入GGGS滤波协方差及透明度补偿，默认gsplat渲染和expected-depth/deferred着色保持原样。
不将其描述为GGGS原生渲染或纯材质控制变量实验，不复用前轮local_transport材质权重。
原始几何和历史终端模型不覆盖。[协议](../experiments/gggs_default_joint.md)。

## 2026-09-27：局部光传输完成，但不替换默认

两场景各30k材质后，PSNR较DNA约+.38/+.40dB；Cat的LPIPS略好，Pixiu变差，两者感知指标均未优于Default。固定图像复核仍见毛发模糊、细小高光缺失和边缘拖影。保留代码与两个最终模型用于研究，不宣称解决几何或完整物理材质辨识。所有几何buffer与输入完全一致；本轮仅实现冻结几何的第一阶段。[完整结果](../experiments/local_transport.md)。

## 2026-09-27：当前几何直接进入材质阶段

用户明确要求先采用当前GGGS法线/深度几何，再接此前局部逐贡献神经光传输并跑Cat/Pixiu。
复用上一轮完成的几何，先冻结其位置/尺度/透明度与Mip参数，只训练新神经材质；
本轮不启用更大方案的联合几何更新或人口重分配，避免把实现范围夸大为完整重设计。
用作者投影与遮挡列表逐贡献求值，以三点条件高斯积分表示有限厚度；不转回旧DNA deferred材质。
材质使用全官方train，原几何internal-val成为材质fit，最终仅official test作未拟合图像评价。
不把已知几何缺陷作为继续实验的阻断条件，但仍如实报告重光照效果与物理解释边界。

## 2026-09-27：GGGS先验结果不推广；修正评价坐标数值尺度

法线+相对深度联合先验抑制Pixiu局部尖刺，但RGB分数几乎不变、LPIPS与边界准确性下降，
Cat存在新折痕。两场景完整30k与fit/validation已完成，不据平滑度宣称几何恢复，不进入relighting。
终端发现少量连续深度像素在世界坐标求解时离群，改为原训练坐标求解后输出世界深度；
没有删像素、截断深度或更改权重，新旧GGGS均按同一修正协议完整重评，原证据保留。
[结果与证据](../experiments/gggs_normal_depth.md)。

## 2026-09-27：按用户要求对GGGS添加法线与深度先验

这次明确授权同时使用法线和深度预训练监督，覆盖此前深度禁用约束；SDF仍禁用。
保持GGGS core原初始化、相机、增密和图像损失，新增StableNormal/DA3先验，独立fresh30000。
历史DA3相机条件绝对深度存在跨视图形状偏差，因此采用去全局log尺度的相对深度形状约束。
先验在1000至2999步渐进启用，避免等几何增密结束后才介入。记录这是normal+depth组合实验，
不将结果单独归因于其中一个教师；没有切换默认relighting方法。
[协议与来源](../experiments/gggs_normal_depth.md)。

## 2026-09-27：Wrapping/StableNormal 对照完成后继续拒绝几何验收

无先验Wrapping未改善Pixiu边界后，按用户条件授权加入fit-only StableNormal法线先验；无SDF或深度教师。
两场景两条件完整30k及fit/validation完成。教师符合度改善，但真实深度灰模仍有错误形状，
边界F1和IoU未改善，因此不将法线平滑作为成功，也不进入relighting。
保留两条件各自30k最终模型及固定比较证据，不增加中间模型；它们是同一Wrapping实现的对照选项。
下一候选为NCC因果消融和边界约束覆盖检查，未启动；不要把这一建议当成已验证修复。
[证据和数值](../experiments/gaussian_wrapping.md)。

# 2026-09-23：采用fit-only表面种子做fresh受控初始化实验

随机包围盒初始化绝大多数中心不在一致轮廓内，取消截断单独无高光恢复，因此改变初始化分配/法线/scale的bundle。
用现有CPU依赖做全506fit轮廓95%投票、2px容差、64→192网格signed EDT零面采样；保留不完美标定、非保守轮廓外壳/凹陷缺失边界。
不把它当几何GT或持续约束；只复制新鲜模型初始means/quats/scales，opacity/material/light保持共同初态。
可行性4CPU+100step profile+初始4fit覆盖检查已通过，进入预先声明的各10000 fresh更新，双方均intersection/cutoff0；
不继承profile权重、不利用validation构造、不据单帧RGB选预算，不默认推广。正式完整质量结果尚无。

# 2026-09-23：停止cutoff-only恢复预算，转向前景条件初始化

5000步公平对照仅得到有限全图指标收益，完整fit和validation小峰召回均0，人工检查失败。
取消阈值解决了直接梯度通路，未解决真实细节拟合；不以短程点数重激活或PSNR小幅增益替代研究目标。
下一优先改变初始表面样本分配及维持机制。全506fit-mask诊断表明随机包围盒初始化只有2.683%中心符合95%轮廓，
但也有大量mask-consistent中心变透明，后期存活中心一致性更好；两种机制都需考虑，不把点淘汰一律判错。
候选需只用fit来源，明确视觉外壳/标定误差边界，先完成可行性与受控预算设计；本结案未实施新初始化、未启动新GPU实验。

# 2026-09-23：隔离透明度截断造成的梯度失活

100k交点反射对照失败，原始100000点仅511/844个opacity>=1/255，且fit也无可靠小高光。
选择最早保留的intersection25k作为共同起点，单独比较alpha cutoff1/255与0；不再叠加旧RGB残差。
这是对低透明度点失去直接渲染梯度的可检验机制研究，不预设所有低透明度点都应被保留或失活就是唯一根因。
默认旧路径不变，截断为0仍有rho<=9和near-plane支持，精确0片元排除。100步profile及来源/采样/保存检查通过，
初始状态逐字节等于原25k，只有cutoff/output不同；按预先声明的资源可行性条件进入固定5000步配对训练。
采样/损失/相机/材质/原数据划分相同，Adam/RNG等同重置，不能叫精确续训。所有fit与validation质量需完整评估后决定。
GT固定图块及8项数值/人工门槛沿用旧协议；阈值合格点增多不直接等于可见有效容量、真实几何或高光恢复。

## 资源与控制检查通过，固定新基础100k对照并启动 — 2026-09-23

保留原提议100k预算：native512/100k点/4×64²patch的100步资源测试只需13.217/9.875GiB，首步后.295/.242s每步，
两张空闲49GB卡可执行；外推耗时数小时，不能将短测或单fit帧分数当质量选型。08:50:31UTC按canonical启动两组。
与全局RNG耦合的首次3步检查失败已修复为专用frame RNG，并完成两组3+3及共享light-scale接口检查；全部旧失败保留。
正式两组都开放一个正值场景标量，避免把任意median irradiance归一化当绝对辐射标定，迫使有界diffuse用specular补亮度。
这不是重新做旧冻结神经材质的光强微调：新几何/局部BRDF与合成基础都重建；共享标量在两组完全同配置、fit-only优化。
主检验是unoccluded局部反射合成顺序；不把失败泛化为所有表面方法失败，也不把胜出直接当完整重光照完成。
固定100k、同初态/采样/优化器和观察模型，25k/50k只在训练后做fit诊断，终端full506fit+full56val；无official71。
预定全56val数值门槛和GT固定8crop人工检查全部必需；4固定全帧环域不缩成显示crop。旧full562模型不能是此val的干净控制。
[已固定协议](../experiments/intersection_reflectance.md)。完整研究目标未完成，不设默认或宣称已恢复真实细小高光。

以下为历史决策，顶部优先。

## 改换合成与着色基础，不再微调冻结RGB残差 — 2026-09-23

完整基础审计显示GS3在Pixiu整图得分较好仍缺小峰，在Cat亦无一致优势；不直接复制ASG作为答案。
选择检验非线性反射与属性聚合的顺序：真实surfel交点先着色再合成，对照同交点先聚合物理材质再着色。
全新fit-only初始化并联合优化几何/材质，不继承已训练562帧的模型、相机或先验。
使用解析两瓣GGX不是新BRDF；候选支持、排序、参数数目、预算和损失保持两组相同。收益若存在仍须拆清假设边界。
现有native片元枚举按中心深度提前停止、bbox对有限透视盘不保守，因此只复用tile候选加速，重新实现交点与排序。
固定点数避免伪造native absgrad/densification统计；阴影语义和真实资源消耗必须在长训练前确定。
材质/基础算子检查通过不等于质量改善。100k预算目前只是预案，组合与真实可行性检查后才冻结canonical配置。
[协议](../experiments/intersection_reflectance.md) · [已完成基础审计](../experiments/foundation_comparison.md)。

以下为历史决策，顶部优先。

## 停止残差微调，先核查已有基础表示的真实高光表现 — 2026-09-23

用户提出底层方案可能有问题，允许重新换方案。当前多轮残差研究没有同时恢复可靠细小高光与整体质量，
不再把增加带宽/容量或换辅助loss作为下一主线。已发现完整GS3/SSD100k模型实际存在，不能按旧文档重复训练。
先对已保存基础模型在共同GT上重算完整小峰/误亮及整图质量；预算/监督/标定不同，明确是观察性比较。
旧neural模型与当前着色语义不同，必须按自身归档导出；原4PNG及全指标精确一致已验证。
SSD训练用test做标定，即使raw评价也只能做污染参照；GS3输入pointcloud完整来源继续核查。
根据实际小峰定位与Cat/Pixiu差异再选择基础重建路线，不把ASG名称或单场景整图分数当作解决方案。
[固定审计协议](../experiments/foundation_comparison.md)。

## 成对差分未通过质量门槛，停止残差头微调主线并复核基础方案 — 2026-09-23

匹配RGB/paired两组各30k、三个fit16节点、终端完整562fit于05:40:45 UTC完成，CPU完整审计及固定8图块人工检查完成。
只有pair权重0/.25不同；25个初始张量相同，原GS/transport/camera/radius固定，采样/资格/支持/覆盖与LR核查通过。
paired的小峰对比、recall、MAE及环域过亮共4/7数值条件通过；全峰precision、PSNR、LPIPS三项失败。
完整562的PSNR/SSIM/LPIPS为24.732770/.871918/.141648，匹配RGB为25.173248/.876975/.138497，
原GS为25.259190/.891718/.129588；相对原GS的三项预定质量门槛全部失败。
固定图块仍有假亮点、宽/错位响应、色块和暗斑，人工门槛false；不降低门槛、不追加71test、不设默认。
保留小峰训练拟合的有限正向变化，但它不足以支持准确恢复或未见条件质量提升。

每组15360000对仅4497缺少双端支持，全部raw-mask eligible组件已被采到；这排除了“普遍没抽到峰”这一简单解释，
不证明每个组件优化充分。训练mask212982峰像素与量化评价212877峰像素不同，不能混用计数。
RGB/paired耗时2049.78/2060.71s、峰值显存均7.80154GiB；旧任务退出、GPU0/1释放，没有新训练。
[本轮结果](../experiments/gs_residual_paired_loss.md)及[结案核查](../../runs/gs_residual_paired_loss_pilot/verification.json)保留完整证据。

用户指出多次失败可能表明核心方案有问题，要求考虑从基础替换。当前采纳的方向是停止把残差头带宽、采样或损失的
进一步微调作为下一主线，重新审视基本表示、光传输与监督目标，并据已有证据选择替代方案。
这是研究方向调整，不是已证明某个单一模块为唯一原因；基础复核正在进行，替代模型尚未选定或实现。
完整研究目标继续active，历史模型、实验和负结果保留；没有新训练或test授权由本结案自动产生。

以下为历史决策，保留当时的阶段措辞；顶部方向优先。

## 历史决策：可移动中心窄核未过门槛；下一项只检验同预算局部差分监督 — 2026-09-23

wide/narrow同中心自由度、同初值和同预算30k的完整562fit已完成，最终数值审计与固定图块检查已完成。
窄核小峰对比/recall的增益仅.009985/1.009pp，伴随全峰precision下降.028638、邻环误亮+11.7645%、PSNR−.182362dB；
5/7数值门槛及人工窄峰门槛未通过。保留MAE有限改善，不将增加命中或真实中心移动包装成准确高光恢复。
不降门槛、不追加71test、不设默认；wide/narrow相对原GS的SSIM/LPIPS几乎全部帧退化，二者均非整体质量赢家。
历史multiply作为参照保留，不当作本轮新增中心的匹配消融。训练拟合结果不推断未见条件泛化，外观中心不代表物理法线。
[已完成结果与审计](../experiments/gs_residual_movable_centers.md)。

下一项[同预算局部成对RGB差分监督](../experiments/gs_residual_paired_loss.md)已经固定协议并开始实现，尚未启动。
从原source新建相同multiply+wide零头，两组共享局部配对采样，唯一变量为pair权重0/.25，避免将采样变化当成损失收益。
旧峰RGB+neutral contrast已试，不声称首次引入局部对比思想，也不同时增加角度结构或正则。
新轮除原匹配控制门槛外，预先增加相对原GS的整体PSNR/SSIM/LPIPS条件；仅约束未来新轮，不追溯重写本轮门槛。
先完成CPU和真实3+3必要检查，完整研究目标继续active。

以下为历史决策，保留当时的阶段措辞。

## 先拒绝固定中心窄核投入，再检验可移动外观中心 — 2026-09-23

完整562train局部cue审计及独立复核完成：遗漏组件87.21%的geometry提示极值偏离>2px，仅0.55%通过窄核筛选。
缩带宽无法移动同中心极值，因此不做固定法线窄核30k。此结果不证明真实几何错误，也不否定所有方向表示。
选择下一项两组均有可学习外观中心的宽／窄SG控制；保留原主法线/shader，检验中心自由度下的带宽作用。
采用无训练期acos的球面高斯，公开与此次theta²核在大角度处不同，避免隐瞒数值实现变化。
同参数/初值、固定30k预算和原继续门槛；历史multiply只作参照，不能将收益全部归于中心学习。
灯位保留独立世界坐标合同，不用未经支持的刚性灯架假设移动灯。
[已完成诊断](../experiments/gs_residual_angular_cues.md) · [未实施的固定协议](../experiments/gs_residual_movable_centers.md)。

## 保留乘性交互的有限拟合证据，不扩大到test — 2026-09-23

同初值/同预算/同source的add/multiply30k完整评价已完成。乘性组在完整562fit的PSNR、LPIPS、
小峰MAE/对比/recall及全峰precision均优于add，不能把实用继续门槛未通过说成没有作用。
但小峰对比/召回增益.026480/2.508pp未到预定.03/3pp；固定图块仍有缺失/错位/宽斑，
相对原GS的SSIM/LPIPS几乎全部帧退化。不下调门槛、不追加test、不改默认；测试迁移本轮未被检验。
初始尺度差已披露，终端乘性支路RMS与加性接近、梯度实际非零，不能归为简单持续失活。
完整来源/预算/冻结状态/指标审计通过，负面边界与局部收益一并保留。
下一项先审计当前遗漏小峰的局部角度提示定位与可分性，再决定是否值得固定法线的窄角度基。
不重复旧的平均n=h角度检查，不立即启动另一轮30k。
[结果与证据](../experiments/gs_residual_interaction.md)。

## 全数据预算未转化为测试峰形，下一项检验显式交互 — 2026-09-23

完整562帧的连续3k/12k/30k只缓慢提高训练小峰，终端test峰局部对比仍近零，整体SSIM/LPIPS所有71帧退化。
不因recall或峰MAE微升宣称恢复、不延长同一方案；主冻结/来源与指标实现审计通过，所以保留为可信负结果。
当前30k完整test仅2.386%的可覆盖小峰像素有任一通道clip，不能以全图14.5%截断解释绝大多数缺失。
选择等参数add/multiply空间—方向交互，初始状态完全相同、末RGB零、A/B/W常规随机，明确add可被原首层吸收，
因此是重参数化控制而非相同函数类。先按预定完整train门槛筛选，再成对评价终端test。
门槛、初始化/尺度/梯度检查与适用边界见[预算结果末节](../experiments/gs_residual_budget.md)。
尚未实施下一模型；不把“交互不足”当已证实原因。

## 少帧能拟合窄峰，先验证完整数据优化预算 — 2026-09-23

单帧3k与四帧12k在固定原GS/相机下均恢复了直接拟合图的细小亮核，环域误差同步下降。
四帧实际每帧约3000更新，源仍全部562fit，不能改写成少帧从头或held-out结果。
其余已见视图严重退化，所以不将局部拟合成功当重光照完成，也不据单/四帧差异唯一归因为容量/视角冲突。
覆盖与分区检查排除了大部分峰无支撑或被clamp的简单解释，但前景黑点是真实残留问题。
选择下一项保持结构和合成公式，以新单链30k检查完整562帧上的优化预算；
LR decay3000，固定3k/12k/30k节点，只有终端完整test，不以test选择checkpoint。
[完整证据与协议](../experiments/gs_residual_subset.md)。
空间—方向显式交互作为后备假设，有原始论文/代码核查；固定照明NVS结论不能直接转用于换光。
[研究备忘](../research/high_frequency_relighting.md)。

## 保留主GS并学习残差：实现通过但未恢复高光 — 2026-09-23

在SDF体分支多项受控负结果之后，选择固定较强主GS并学习零初始化光照条件有符号修正，
避免放弃已拟合纹理；只比较新头输入的几何/既有着色法线，冻结其余全部状态。
完成2×3k及16fit/71test，准确小峰仍未恢复，SSIM/LPIPS退化；不采用为默认。
源码/检查证明训练只更新新头，但参数冻结不是画质保持保证。
下一步先做单帧与少帧拟合诊断，检验局部表达/优化是否能产生窄峰，而不是直接增加编码。
单帧3k与四帧12k大致匹配每帧曝光；必须明确总预算差异及无法唯一归因的限制。
[固定实验和后续具体协议](../experiments/gs_radiance_residual.md)。

## 高光恢复：先验窄峰、材质路径与着色法线

用户要求恢复GT中的高光。采用程序化角度扫描和固定训练帧分项渲染定位，
发现旧先验对窄峰严重低估，且旧gate会衰减高光；只修这两项的2000步短检没有恢复亮点。
加入有界材质着色法线与固定GT亮点代理监督后，训练帧亮点误差和对比度改善，
但仍存在宽度/整体画质代价。该组合不是各组成项独立因果贡献的完整消融。
最终选择固定原几何、Cat/Pixiu各追加8000步验证，使用同一冻结先验，末端完整official test。
保留两轮pilot和原结果，报告30k+8k预算；不将追加训练结果包装成同预算从头训练。
[完整诊断与协议](../experiments/highlight_recovery.md)。

完整test否定了“亮度更高即高光恢复”的判断：GT亮点局部对比度没有改善。
随后三项训练帧对照降低法线偏差却没有提高亮点重建，因此不进一步推广这些训练开关。
保留全部负结果并移除无收益的试验代码；当前候选不是质量赢家，默认不变。

## 共享材质先验替换局部材质网络 — 2026-09-22

用户指定Yu等SIGGRAPH2026的共享神经材质decoder，排除扩散生成模型。
实现neural_material作为独立注册方法，预训练一次、Cat/Pixiu共用并冻结权重。
每Gaussian前6个feature通道作为材质logit，base作为独立漫反射logit；
decoder不接收位置或光强。选择现有2DGS提供表面法线，继承方向端口处理非局部项。
不把任意椭球的旋转轴直接当作真实表面法线，也不声称端口是物理路径追踪。
公开细节不足处及程序化采样/方向嵌入/反照率近似均在方法文档列明。
仅Cat/Pixiu，固定30k/seed0全train、末端checkpoint及完整官方test，
比较已有默认/attention及历史surfel结果；不以test选择结构或提前停止。
正式实验已完成：两场景PSNR/SSIM均提高，平均PSNR较默认高0.921686dB，
但LPIPS均退化，均值增加0.015101。保留该方法作为研究候选，默认方法不变。
没有追加其他场景、延长预算或依据test改模型；[完整证据](../experiments/results.md)。

材质预训练首轮恒定0.001学习率在50k末端反弹，独立验证BRDF相对L2=0.229、
透射MAE=0.219，不作为正式实验先验。保留该诊断后，仅将学习率改为余弦衰减至1e-5，
同seed/预算/模型/数据重新训练；验证改善至0.062/0.028，固定此先验后启动场景训练。
这一选择只使用程序化材质验证，没有读取Cat/Pixiu test来选择先验。

## 按用户要求移除两个候选及其历史指标 — 2026-09-22

删除surface_kernel与surface_diffusion的方法实现、注册入口、专用测试和文档中的实验指标。
保留其余6种方法、已有有效结果与表面先验；默认仍为directional_port_v1。
SSD-GS参考指标继续单独保存在research_summary.json，不包含已删除候选的pilot记录。
canonical配置恢复为已完成的surface_attention cosine/30k设置，使用未启动的
surface_attention_validation名称；此次清理不启动训练或评价。

## 以受控消融选择简洁的非局部表示 — 2026-09-22

在用户要求排查2DGS退化并简化间接光后，先测试两个可分离的变化：
法线监督0.05→0.005，以及将方向端口换成单次源到像素cross-attention。
三个条件均为30k/seed0、相同train内灯光划分、相同深度监督，不再使用固定RGB预热和颜色重置。
同位姿的教师法线不一致是减弱约束的依据；尚不能认定它是全部退化原因。
三个场景完整validation用于选择，再冻结设置开展原六场景全train→official test。
保留原默认方法和已有输出；不以训练loss或少量图像选择模型，不在当前队列中途改条件。

用户进一步要求以优雅方案为目标、避免为创新堆模块。候选只增加一个attention算子，
保留光强线性并明确空间压缩近似；参数量是网络模块计数，不能冒充全部模型存储收益。
是否保留该算子由质量/速度证据决定。attention和学习光传输本身已有充分先例，
不预先声称首创、物理直接/间接解耦或顶会竞争力。
[诊断、协议与相关工作](../architecture/modules/attention.md)。

## 用户授权两个候选各跑六场景30k — 2026-09-17

方案一paired_port与方案二local_frame分别从头训练，使用现有六场景完整train/test协议。
两张空闲GPU并行：GPU1专用于方案一队列，GPU3专用于方案二队列，每张卡一个worker。
保持原30k基线日程和数据设置，只切换方法；方案二frame_width=32。
配置冻结为767575f，各实验独立保存源码快照和精确命令，保留原基线输出。
[协议](../experiments/research_methods_validation_20260917.md)。

## 多方法架构与两个研究候选 — 2026-09-17

原工作区已保存为Git提交2204cdc，接口重构单独提交cbc3545。
在feature/selectable-transport-methods分支完成methods注册表与公共基类，
移除根目录重复transport实现，保留原模型参数键及已保存checkpoint读取合同。
研究A1/B1分别作为paired_port/local_frame接入，默认方法仍为directional_port_v1。
新方法是否提升图像质量待正式实验；本次只进行算子与最小真实数据集成验证。

[接口](../architecture/modules/methods.md) · [实现](../architecture/modules/research_methods.md) ·
[验证记录](../experiments/method_refactor_20260917.md)。

## User cancellation and30k restoration — 2026-09-16

Stop the60k experiment and delete only its current output directory. Restore
30000 steps in the train parser and validation config. Preserve completed30k
results and other projects. Do not launch a replacement run. Other architecture,
phase timings and final-only saving settings remain unchanged.

## User-authorized two-GPU execution — 2026-09-16

Use at most two available GPUs. The user's final allocation isGPU1/3 because
GPU2 became occupied. Preserve Cat's active training process and optimizer;
expand only the scheduler, with Pixiu and subsequent jobs on the two workers.
No training protocol or model change accompanies this allocation update.

## User-requested 60k budget — 2026-09-16

Repeat the same six-scene directional rank512 experiment from scratch with
60000 steps, preserving the30k results. Keep5000-step shadow/port start,
25000-step refinement stop, and no intermediate checkpoints. Existing normalized
learning-rate schedules stretch to the longer budget; do not restart optimizer
state from30k or alter other settings.

## User-requested training schedule — 2026-09-15

Replace the in-progress directional rank512 experiment: delete its outputs,
restart from scratch, disable periodic validation/intermediate model saves,
set shadow_start=port_start=5000 and refine_stop=25000. Keep final checkpoint
and full test evaluation. This supersedes the previous rank-only comparison.

## User-requested port count — 2026-09-15

Increase directional_port_v1 from64 to512 spatial ports and rerun the same six
scenes from fresh initialization, holding all other configured settings fixed.
Keep the64-port experiment as the primary comparator. No architecture changes
beyond port count or extra experiments are included.

## Current architecture — 2026-09-15

The active model is directional_port_v1: 64 spatial ports, four direction
channels and a shared material direction MLP. Direct and nonlocal radiance are
combined at pixel receivers. See [directional architecture](../architecture/modules/directional_transport.md).
The earlier architecture below is historical.

# Material decisions — completed 2026-09-13

## Resume research under the new user instruction

The September 12 request restores PORT-GS development and supersedes the earlier
retirement arrangement. At most three rounds may review, improve and retrain the
method. The target is Cat above 20 dB under original test calibration, with better
quality where possible. The historical 22.000071 dB score remains a previous
method result. SSD-GS contributes only metric references in this cycle.

## Accept round two and freeze after two structural rounds

The second-round fixed 30k checkpoint gives 22.316405 dB / .778143 SSIM /
.228932 standard LPIPS on the same 52 validation frames. Relative to round one,
PSNR decreases .224227 dB while detail energy increases .268290 → .339026
(+26.4%) and LPIPS improves .247135 → .228932. Mean visible support decreases
17.338 → 12.646px. Silhouette IoU decreases .957555 → .954375; edge accuracy
remains a limitation. Difficult frames 233 and 158 improve .249891/.268827 dB.
Relative to historical matched 30k validation, PSNR remains +.114553 dB.

Independent paired review finds LPIPS improvement on all 52 held-out frames,
SSIM improvement on 35/52 and PSNR improvement on 22/52. Alpha L1 increases
.020662 → .023228 and mean silhouette distance increases 3.840 → 4.217px.
The decision prioritizes perceptual/detail gains while recording these PSNR and
outline tradeoffs. Structural iteration ends after two rounds. The two
first-round startup corrections were engineering restarts within round one.
GPU operator/receiver audits and training are complete. Archive
`cat_r2_source.tar` freezes the selected code, and its resolved training settings
are the basis for all subsequent runs.

Fresh 30k full fits completed serially on GPU 0: Cat 522/66 train/test,
GS³ Translucent 2000/400 at 512px white background, and SSS-GS Bunny 500/500 at
256px black background with unit intensity 1. Final test used original metadata,
and the source/configuration freeze preceded testing. All three results are
complete and GPU 0 is released. External runs used frozen settings without new
tuning, ablations, other-method training or duplicate data downloads.

Cat reaches 21.501174 dB, meeting the >20 dB target. Compared with historical
full Cat, PSNR decreases .498897 dB and LPIPS improves .021971; LPIPS improves on
65/66 frames, while alpha L1 worsens on every frame. This retains a measured
perception/pixel/outline tradeoff. Translucent reaches 28.304924 dB with qualified
GS³ paper references. Bunny reaches 37.600658 dB, but its weakest views are
12.143550/12.436795 dB and the SSS-GS aggregate is not a Bunny-specific comparator.
These final test outcomes did not trigger additional method selection.

Matched-scope source counts are 2009 → 1531 physical lines across eight production
modules (-23.79%) and 4133 → 2242 across top-level Python including tests (-45.75%).
All earlier source/checkpoint archives remain intact. See
[final results](../experiments/results.md) and
[cleanup accounting](../../runs/research_20260912/cleanup_files.json).
[Round-two evidence](../../runs/research_20260912/cat_r2_s0/analysis_summary.json).

The following candidate decisions record how this selection was reached.

## Advance to pixel receivers after first-round validation

The first-round fixed 30k checkpoint reaches 22.540632 dB on 52 Cat validation
frames, +.338780 dB over historical 22.201852. It uses 399,358 Gaussians; detail
energy remains .268290 of GT versus roughly .271 previously. Mean visible support
falls to 17.338px, so smaller support alone has not restored detail. Fit32 remains
blurred as well. These measurements motivate a second round rather than candidate
acceptance. The 20k monitor peak of 22.682633 dB is unquantized and belongs to an
earlier checkpoint; reporting uses the fixed-last 30k quantized result.

The second-round renderer replaces constant per-Gaussian shaded RGB with
pixel-receiver shading. Rasterization blends base/features/visibility and expected
camera-Z; pixel-center back-projection reconstructs world receivers. The material
network gains 51 spatial channels from eight-band world-position encoding. Every
Gaussian remains in the source integral, while pixel receivers query its same
partition and feature-conditioned exchange fraction.

The renderer sets `channel_chunk=attributes.shape[-1]+1`, so the default 36
attributes and depth share one rasterization pass. The dependency's default
32-channel chunks overwrite `means2d.absgrad` across passes, losing part of the
geometry-growth signal. This explicit pass width supplies the complete attribute
and depth contribution to absolute projected gradients.

Source-node conservation and reversibility retain their precise discrete scope.
The continuous receiver extension and expected-depth mixing introduce separate
approximations. GPU audits and fresh 30k training were scheduled on the same 470/52 split,
seed 0 and fixed-last selection; their completion and acceptance are recorded above.
There are at most three rounds in total. First-round weights use the archived
`cat_r1_source.tar`; second-round changes are not a first-round improvement claim.
[First-round evidence](../../runs/research_20260912/cat_r1_s0/analysis_summary.json).

## Replace additive RGB transport with conservative irradiance exchange

One material-conditioned exchange fraction controls both source pooling and
receiver redistribution. The receiver's shared angular response acts on the
resulting irradiance. This directly replaces the weakly coupled additive branch
and the historical representation options. The exchange preserves a discrete
quadrature-weighted irradiance sum and detailed balance at fixed geometry.
These guarantees stop before local material response and image formation.
The candidate's empirical benefit and publication novelty remain open.

## Split broad image supports directly

Historical Cat diagnosis found camera-contribution-weighted support near 22px
and detail energy around 27% of GT. The new training configuration adds split
candidates when projected radius exceeds 3% of the image's long edge (15.36px at
512px), alongside the gradient-based criterion. Duplication excludes split
candidates. `refine_scale2d_stop_iter=refine_stop` keeps this rule active during
refinement and `grow_scale2d=0.03` defines the threshold.

The dependency also couples this switch to screen-radius pruning, whose parent
radii become stale after splitting. Explicit `prune_scale2d=inf` keeps pruning
based on opacity/world size while avoiding that invalid child decision. A first
startup was interrupted before validation; its logs remain identified as
`interrupted_screen_pruning`, and the first structural round restarts with this
correction. Geometry budget control is established engineering rather than a
novelty claim. Exchange and geometry changes are evaluated together, so a result
cannot isolate their individual effects.

## Factor exchange normalization for efficient GPU execution

The expanded `N x R x RGB` tensor and dimension-0 softmax were an implementation
bottleneck as the first Cat run approached 400k points and about 0.5 seconds per
training step. The replacement stores contiguous `R x N` source weights and
normalizes along the final dimension, then handles RGB exchange fractions through
two `R x 3` matrix products. Cancellation of the common source normalization
preserves the exact exchange formula and `O(NR)` complexity.

On 167,692 real Cat Gaussians with rank 32 and GPU 0, CUDA-event median operator
forward/backward time decreased from 126.353 to 0.982 ms (128.7x); forward time
decreased from 75.648 to 0.246 ms. These timings cover the isolated operator.
Complete training also includes material response, shadows, rasterization,
refinement and optimization, whose throughput is measured separately.

Float64 maximum relative differences are 3.40e-15 for outputs and at most
1.76e-13 for gradients. Float32 output error against the float64 reference is
1.52e-4 for the archived expanded layout and 2.13e-6 for the factored layout;
checked gradient accuracy also improves. The
[factorization audit](../../runs/research_20260912/operator_factorization_audit.json)
and [operator audit](../../runs/research_20260912/operator_audit.json) preserve
inputs, measurements and numerical checks.

That first-round implementation pause and restart were recorded under
`interrupted_tensor_expansion`. The restarted `cat_r1_s0` has now completed;
its quality evidence is recorded separately. These layout timings remain a
first-round operator measurement as the second-round pixel path is introduced.

## Preserve history through source and experiment provenance

The original source is archived in `runs/research_20260912/source_before.tar`.
Previous logs, checkpoints, data and research documents remain intact. Historical
checkpoints use their archived code; the active implementation uses one selected
representation. Current checkpoints omit optimizer state because initialization
starts fresh optimization rather than resuming an interrupted run.

## Keep selection in training-derived validation

Use Cat's 470/52 train-light split, 512px, seed 0 and 30k steps on GPU 0. Run at
most three structural rounds and stop early on accepted improvement. Freeze the
accepted configuration before fresh full fitting of 522 official train frames
and one evaluation of the 66 official test frames. Training-camera corrections,
when enabled, belong to fitted frames; held-out evaluation uses original camera
and light metadata. Fit evaluation restores its learned camera offsets.

## Reuse complete external data first

GS³ Translucent is the first additional dataset candidate. Existing GS³ and
SSS-GS small data are complete with respect to their local JSON references.
Reuse shared datasets; record actual execution separately. Environment-light
ReCap data requires a different input path. Published metrics become matched
comparisons only after their data and observation protocols are aligned.

The following decisions describe the completed September 11 work. They preserve
historical reasoning; current scheduling and method choices are defined above.

# Historical material decisions — 2026-09-11

## Retire PORT under the revised research objective

The user's September 11 instruction superseded the earlier decision to stop after any
measured improvement. Two repair rounds leave Cat at 22.000 dB and Pixiu at
20.972 dB, well below the local SSD-GS references. PORT development ends here.
This is a research investment decision based on measured quality and images;
different training budgets leave the theoretical capacity of PORT unresolved.
All historical decisions below remain a record of the completed repair cycle.
See [retirement](retirement.md) for evidence and the test-image correspondence audit.

## Own the refinement schedule

Independent review found a deterministic opacity-reset bug in the installed
gsplat dependency. PORT's replacement invokes existing geometry/statistics
operations with the correct reset condition. Reusing the shared environment
preserves the SSD-GS stack. The point budget limits growth while pruning and
opacity resets remain scheduled through the configured refinement phase.

## Repair verified behavior before adding a new appearance model

Existing records already found little benefit from radiance moments and a shared
camera response, and losses from the camera/object learning-rate change. Current
work first repairs verified optimization behavior. Shadow-model limitations are
recorded as hypotheses until rerun evidence establishes their practical effect.

## Preserve experiment membership and provenance

Initialized weights retain their checkpoint's fit/validation membership;
`--fit-all` explicitly consumes all train frames. Exact launch arguments, GPU,
seed, metrics and outputs are recorded within PORT-GS. User data, previous logs,
checkpoints and research documents stay intact. Existing controls are historical
references, and this cycle launches only the repaired method.

## 2026-09-15: User selects the original anchor512 model

Restore the model/render/evaluation implementation from `9e9596a` at the user's
request. Retain useful loss logging/plotting and the JSON launch interface,
while removing HashGrid and residual parameters from active code/configuration.
Preserve all prior results and model snapshots. This is a code rollback, not
authorization to start another benchmark. See [restore record](restore_anchor512_20260915.md).

## 2026-09-15: Residual decoder after queried HashGrid features

At the user's request, replace the plain direct RGB decoder with a width128
stem and two two-layer residual blocks followed by an RGB head. Keep inputs,
HashGrid settings, radiometric scaling, losses and six-scene 30k/seed0 protocol.
This includes one more hidden affine layer than the previous plain decoder, so
it is a residual-architecture comparison rather than a parameter-matched skip
ablation. Save the block count in JSON/checkpoints and update the evaluator.
No source pooling or 512-channel mixture is reintroduced. Preserve all prior
scored runs for comparison; verify residual gradients, rendering, saved weights
and loss curves before launch. See [protocol](../experiments/residual_hashgrid_validation_20260915.md).

## 2026-09-14: Remove pooling and directly decode HashGrid queries

The user clarified that the intended HashGrid replacement should directly
decode queried features, without the 512-channel illumination mixture. Remove
the spatial partition head, quadrature/source integral, exchange fractions and
port activation schedule from the canonical implementation and all callers.
Use a 175-input, four-layer width-128 SiLU RGB decoder with light/view/material
and visibility conditioning. Keep point-light intensity/inverse-square scaling
explicit; use visibility as an input so shadowed queries can have learned
nonzero radiance. This relinquishes the historical discrete conservation claim.

Repeat the same six scenes, 30k and seed0; HashGrid is active from step1. Retain
the NVIDIA encoding config, dataset protocol, geometry/shadows, and image losses.
Record total and weighted loss terms every100 steps and at the first/final steps,
and generate loss.png after training. Checks now cover direct query independence,
input/native gradients, intensity scaling, rendering, serialization and plotting.
See [protocol](../experiments/direct_hashgrid_validation_20260914.md).

## 2026-09-14: NVIDIA HashGrid replaces learned spatial anchors

The user's next experiment replaces continuous anchor centers/widths with
NVIDIA tiny-cuda-nn HashGrid, not a hand-written grid. Save the prior source as
Git commit `9e9596a` and exclude outputs/dependencies from Git. Retain 512
exchange channels, the angular response, and 30k/seed0/six-scene protocol to
compare directly with the completed anchor512 run. Grid resolutions 16–2048
use capped per-level tables rather than dense 2048^3 weights. This improves
spatial representation capacity but does not remove low-rank exchange pooling.

The environment's installed tinycudann binary requires unavailable GLIBC_2.33.
Build the same recorded NVIDIA source revision locally as an encoding-only
wheel and use project-local PYTHONPATH, preserving the shared PyTorch/CUDA stack.
Experiment JSON and HashGrid JSON are tracked, and each run freezes copies.
See [protocol](../experiments/hashgrid_validation_20260914.md).

## Simplify source/query integration

Forward Gaussians and deferred pixels now share one transport evaluation. Source
weights have one implementation; identical source/target bases are reused within
the call. Observation rendering is also shared by training and evaluation.
GPU bounds eliminate unnecessary whole-image transfers. Temporary formatting
tools reside under `/tmp` and do not alter the training environment.


## Select round 2 from actual Cat images and error data

Round 1 reduces Cat's point count but leaves PSNR slightly lower. Inspection of
84 images shows blurred fit texture and broad validation illumination/structure
errors, especially in partial-shadow regions. Direction-dependent deterioration
and low correlation with alpha changes prioritize the coupling of photometric
error and shadow geometry as the next hypothesis. See the image-diagnosis report
for limitations. The candidate's unchanged forward values, convergent opacity
finite differences and finite geometry gradients justify one same-budget rerun;
quality remains to be established from its fixed-last validation and images.


## Accept measured round-2 improvement and freeze

The final 30k Cat result improves PSNR and LPIPS with about half the historical
Gaussian count. Matched-frame images show lower coarse RGB error and slightly
higher detail energy, while several difficult frames and outline errors remain.
This is sufficient for the user's early-stop option after improvement; it is
not a claim of complete real-scene accuracy. A fresh full fit uses the same
frozen 30k budget and is evaluated once on official test. A third structural
change is outside the accepted scope of this cycle.

## Directional evaluation outcome — 2026-09-15

The user-requested directional architecture is implemented and retained. Six
fixed-budget experiments completed without tuning on test. Results do not show
an overall improvement over legacy rank512: mean PSNR−0.185288dB, with the main
regression on bunny_small and improvement on AnisoMetal. Keep these results as
evidence; no unrequested architecture revision or new ablation was launched.
See [experiment report](../experiments/directional_port_validation_20260915.md).
# 2026-09-21：新增PORT的2DGS表面监督变体

用户要求在默认方向端口基础上引入2DGS；澄清后使用StableNormal预测法线，
随后明确加入DA3深度监督。注册directional_surfel，未改变默认方法。
保留原512×4端口和直接光MLP，只配套修改圆盘几何、光栅化、面积权重、阴影偏移、
增密及表面损失。采用冻结教师的train-only离线预测，推理不依赖教师。
DA3-Large逐帧相对深度使用去log尺度损失，不当作绝对深度；StableNormal法线按
作者2DGS实现转换到OpenCV坐标。使用本地隔离venv共享基线Torch/CUDA。
当前中心深度分层阴影仍有倾斜/大圆盘误差，未实现精确次级射线。
[完整实现与限制](../architecture/modules/surface.md)。
## 保留原法线强度对照，并修复确定的数值错误 — 2026-09-22

首轮弱法线条件在bunny validation下降1.856dB，dragon下降0.100dB，
尽管轮廓误差改善；不接受“一概减弱法线就能改善重光照”的假设。
增加attention+原法线权重0.05的三场景对照，算法仍保持一个attention算子。

dragon attention失败的直接原因是几何梯度出现NaN。单圆盘最小复现确认，
gsplat 1.5.3多通道padding使用torch.empty，废弃通道的非法值仍参与反向传播。
在PORT的渲染适配层显式补零，不修改共享依赖，不用丢弃非法梯度来掩盖问题。
已在真实失败批次验证前向输出完全一致、修复后梯度有限，并通过最小GPU集成检查。
保留失败产物，在新的followup队列重跑该条件，再完成补充对照和六场景最终测试。
## 2026-09-22：以注意力饱和诊断增加Q/K归一化对照

dragon弱法线attention在数值修复后完成，但validation退化3.483dB。抽查2个fit与2个
validation帧，attention退化到单源格（有效源数1，最大权重均值1），部分帧该源几乎未受光。
据此在原attention实现内加入dot/cosine评分选项，cosine限制Q/K尺度，无新增网络参数。
保留dot对照的原行为；新实验使用原法线权重0.05，与正在完成的dot组匹配。
仍使用固定30k/seed0和同一train灯光留出划分；先GPU1跑dragon，再完成其他两场景，
选型后从头跑六场景official test。归一化是标准稳定化方法，不单列为论文创新。
## 2026-09-22：固定attention六场景评估配置

三场景五条件validation已全部完成。cosine/normal_weight=0.05的PSNR均值31.816408，
为attention候选中最高；方向端口原权重为32.457148，仍是本次消融质量更强的对照。
cosine在同30k预算下三场景日志训练时间合计约为对照的54.6%，显存明显降低。
因此以该设置完成用户要求的新方法六场景从头全train/official test，评估质量与效率折中；
该决定不等于质量超越或论文创新验收。固定时间2026-09-22 01:04:59 UTC。
正式输出surface_attention_validation_20260922，GPU0/1/2、seed0、30k、warmup0、
surface_start1000、shadow/port5000、refine_stop25000，复用既有StableNormal/DA3逐图先验。
后续不以该run的官方test结果修改这次冻结参数；保留原默认directional_port_v1。
## 2026-09-22：正式测试完成，保留默认并记录效率折中

surface_attention/cosine六场景全部30k与1937张test完成，0失败；PSNR28.622846，
相对首轮2DGS提高0.150060dB，仍低于原默认3DGS 0.467129dB。
SSIM0.918468高于两个对照，LPIPS0.090192略差。训练时间合计相对首轮2DGS约1.92倍速度，
最大已分配显存19.467→9.990GiB。保留surface_attention作为更轻的研究候选，
不替换directional_port_v1默认，也不声称已解决直接/间接光解耦或具备投稿结论。
用户要求的调查、调整及三GPU完整重训/测试已完成；数据、源码、失败日志与对照均保留。
[全部指标和图像](../experiments/results.md)。

## 2026-09-23：用训练辅助SDF验证几何假设

按用户要求加入SDF↔2DGS。优先用连续场的零面/梯度约束已存在的深度与法线，不同时改BRDF。
避免零场退化：窄带有符号距离、方向、Eikonal、视线前方自由空间；双向分离梯度并预热。
相同初始化、预算与外部图像先验的control/SDF对照，最多两张空闲GPU。
这不是GSDF全流程复现，没有独立SDF体渲染或几何GT；效果必须通过完整评价确认。

## 2026-09-23：先验质量与SDF耦合

首轮SDF自一致性提高但高光未恢复；小步续训仍未改善整体拟合。
不继续在这两个因素上反复扫权重。相机条件多视图DA3在8真实训练帧上优于单图先验与
StableNormal的一致性，故固定视图数8、共享4个锚点，替换全部train深度作等预算试验。
训练仍使用原相对深度损失，不把尺度对齐后的DA3深度当成真值绝对深度。
旧normal先验复用，所有原先验/检查点保留；只用训练集结果判断是否值得完整测试。

## 2026-09-23：检验SDF直接图像监督

前两种几何改动没有恢复高光。仅互监督的SDF没有独立图像信号，可能只是平滑GS预测。
下一组在相同已拟合field上比较辅助模式与SDF梯度法线直接PBR；保持材质、外部先验与预算不变。
这需要field作为推理模型的一部分，已显式接入训练/评价/诊断与checkpoint恢复，不隐式替换旧结果。

## 2026-09-23：交点深度短程对照后改为从头验证

解析测试证明中心深度不满足倾斜平面几何；修正梯度、真实短训已验证，但同38k模型续训2k并未恢复高光。
不据0.015dB的fit PSNR增量升级默认，也不以几何解析正确推断真实图像改善。
下一步两组从相同随机初始化开始、统一修正深度，独立检验SDF互监督；延迟到3000步拟合field，避免早期随机表面。
20k到400k增密和30k完整预算检验此前固定点数短续训未覆盖的情形。保持最多两张空闲GPU，不请求常规决策确认。


## 2026-09-23：隔离冻结BRDF近似，准备直接GGX对照

当前SDF/先验/相机诊断表明几何和输入对齐存在问题，但没有证明它们是全部高光误差的来源。
材质码离线拟合中最窄GGX宽度仍存在偏差，故增加`--material-model ggx`作为同一管线的受控材质分支。
保持六维容量、法线修正、点光、间接传输与观察变换；不引入固定灯光的新视角RGB残差。
它是独立F0、两种alpha和混合系数的解析模型，不声称覆盖所有真实材料。
先做真实旧模型转换/保存重载检查，再同一35k场景初始化、冻结几何和相机、两组重置材质码、各5k。
两组保留已拟合法线、漫反射底色和PORT权重，以训练诊断评价材质收敛；尚未启动此场景对照。
SDF当前5k对照继续原协议，后处理仅在对应训练和fit评价完成后执行。


## 2026-09-23：保留SDF空间细节候选，暂不加大对GS反馈

同外观容量的受控3k对照显示SDF网格改善16训练视图RGB/轮廓指标，但仍模糊且未超过固定GS图像指标。
保留实现与模型；下一步先检验泛化、局部几何及采样/分辨率，避免将可拟合颜色等同于几何真值。
用户要求新主任务继续，本组收尾后不在当前任务开启下一训练组。可用工具缺少创建/移交接口，接续文件已写全。


### 接续评估约束：不能事后从全train模型创建“未见”验证集

已核对sdf_volume_detail_pilot两组fit_all=true、fit562/validation0，且共同源经过全train拟合。
后续若检验train内留出泛化，必须从未拟合留出帧的初始化开始，并覆盖GS/相机/SDF/头的完整训练来源。
直接续训当前checkpoint只可作为拟合诊断。不得仅改变evaluate的帧选择而声称消除了训练泄漏。


## 2026-09-23：512px训练亮点射线配额对照

原生亮点缩128px后局部对比只保留40.28%，且原512射线预算在512px下每步期望只命中2.10个proxy亮点。
下一次小试选用两组共同512px与固定距离场，只改变GT亮点采样配额0→.25；
保留512总射线、64粗+64细、同一geometry_grid起点、3k步/seed0。
该选择优先检验监督不足，不继续向可能补偿外观误差的SDF局部法线增加自由度。
ray RGB/mask/geometry项均被显式重权，sharpness仍学习；不是无偏方差降低或完全固定体渲染权重。
评价仅同16训练视图，统一512px，报告整体与亮点位置/对比，不调用test。
完整源模型经过562帧全train训练，不能事后创建未见划分；后续必须审查先验的多视图context来源。


## 2026-09-23：拒绝峰亮度改善即高光恢复，检验形状约束

完成的25%峰采样把峰MAE降低66.75%，但16/16视图PSNR均下降；4帧峰邻环正误差升5.53倍。
因此不提升该候选为默认、不直接投放official test。选择两项独立小试：保持128峰quota，
额外128邻环射线以增强局部负监督；或保持采样不变，只给现有log提示加无π四频编码。
两者同source/512px/512rays/3k，与已完成peaks比较，不重复训练控制；新代码关闭功能后的
输出/默认初始化/采样和随机数状态已与旧归档CPU逐位审计，完整新路径真实512px短测通过。
编码只借鉴NRHints使用高光提示PE，当前逐样本log提示不是其每ray GGX提示；不把研究动机当收益证据。


### 上述形状实验的决策结果

两组均已完成。context确实部分减轻局部过亮（−33.80%）、提高位置召回，但整体仍低于uniform和主GS，
保留为研究候选，不晋级test/默认；hint仅微降亮点MAE、位置/对比/整体退化，当前不继续扩大该结构。
下步优先检验收敛/射线覆盖，避免继续堆叠结构：3k定向quota384000次、212982 GT峰，
平均1.803次/峰且有重复。覆盖统计不是收敛证明，增加预算也必须如实标注额外计算成本，
并继续固定源模型和同一训练评价集合。下一预算实验尚未启动。


## 2026-09-23：只增加射线预算检验监督覆盖

在上轮context部分缓解邻域过亮、hint编码失败后，保持模型/数据/损失/3k优化步数/学习率日程不变，
将每步射线512→2048；peaks/context各自与其已完成512ray控制比较。
两组总射线预算均6,144,000，明确是4倍射线成本，不将相同步数称同成本。
本轮检验覆盖与梯度方差，不据覆盖计数断言已收敛，也不把更多rays称独立epoch。
2048ray真实512px短测通过，再在两张检查空闲GPU启动；最终保持同16训练视图及4帧局部诊断。


### 四倍射线对照后的选择：检验积分层厚度

两组已完成。context的峰召回/对比有小幅增益、整体PSNR不变，peaks宽斑误差甚至增加。
不继续仅堆每步射线，不把3k负结果当旧网络容量上限或已充分收敛；保留context2048作后续协议。
CPU固定射线诊断发现当前积分法线RMS离散中位8.62°，将sharpness乘4可降到2.98°，
采样加倍不消除原混合宽度；直接改已训练头的sharpness反而增大RGB误差，不能直接当修复。
下一计划为两组同geometry_grid起点/context2048ray/3k，冻结距离场及sharpness，
仅比较源sharpness183.8839569091797与四倍735.5358276367188。两组均须新训，
此前会学习sharpness的模型不能替代固定值控制。尚未实现该固定选项或启动下一轮。
采用共同128粗+128细：64→128曾有一条RGB变化.238/法线42°，128→256该点降为.000307/0.28°；
全512射线仍有RGB最大.0393/法线9.59°尾部，不能宣布采样收敛。记录约翻倍积分区间成本。


## 2026-09-23：将厚度变为受控固定参数

已实现可选固定sharpness，不改变默认方法或旧模型键；由本次CLI明确指定、加载权重后设置并冻结。
两个固定值对照均从geometry_grid原source出发，固定全部主模型/场/β，仅学辐射头和外观网格。
两组共同context2048ray、128粗+128细、3k/seed0；唯一差异是β183.8839569091797/735.5358276367188。
旧adaptive模型不是固定值控制，两个新固定组均需训练。
CPU真实小批量检查通过；完整GPU批量短测后再启动，不以初始诊断RGB变化代替重训结果。


### 厚度结果与研究转向

固定四倍β使LPIPS改善，但PSNR、峰MAE、对比度和precision退化，1–4px小峰召回也下降。
这是当前固定场/3k协议的负结果，不证明所有表面渲染或长训练无效；不继续只调整体分支参数。
主GS仍明显更好。下一优先实现零初始化、有真实wi/wo/light_position条件并按I/d²缩放的
有符号响应残差，冻结原主模型，保留其全部纹理/直接光/PORT/alpha；
两组仅比较新头所用的冻结geometry/material法线，避免给法线新增自由度。
该残差不是只读view方向的固定光照NVS项；必须验证光强正齐次，且不能据此声称物理BRDF。
不输入teacher RGB，不额外加gate或正则；隐藏层随机、RGB末层零，不迁移SDF体积分的隐表示。
原GS是零残差控制，无须重训。固定3k终端协议后评价完整71个未训练official-test帧，
用原始test相机，不做逐帧适配或选中间checkpoint；称开发期评价而非独立盲测。
实现/测试/保存契约与风险见[方案](../research/gs_radiance_residual.md)，尚未实现或启动。


## 2026-09-24 JST: bounded user-requested closeout

The user requests completion of the current method/improvement and full Cat/
Pixiu test results, followed by redundant-file cleanup, method/result inventory,
and pause. This supersedes earlier internal no-automatic-official-test notes
for the later final phase only. Current pilot budgets remain unchanged.
Do not keep adding candidate studies indefinitely or mark highlight recovery
achieved merely because the closing sequence has finished.

Mask-surface initialization is retained as the bootstrap for this new foundation:
matched10000-step development-validation PSNR14.367→21.250 andLPIPS.302→.182,
but tiny recall4/3770 and manual gates fail. This is not a default-method promotion.
One final improvement tests a half-uniform tiny-component patch proposal with
inverse-probability compensation, preserving the existing expected uniform-origin
L1/SSIM/mask objective. Unlike older GT quotas it does not deliberately change
objective weights. Four mathematical/sampling tests and actual100-step profile
passed, followed by predeclared equal5000-step stages from mask10000.
Equal steps/patch counts are not equal visible fragments, GPU time or variance.
Source stage RNG repeats the original uniform stream; record this limitation.

After development selection, final fresh30000 fulltrain per scene uses no prior
checkpoint. Surface seeds and any proposal derive only from522Cat/562Pixiu
training frames. Official66/71 tests use original calibration and no adaptation;
previous research observed these scenes, hence development testing, not blind.
The original default remains the primary practical baseline; originalneural
results provide context. Fixed GT crop frames are inherited, not selected after
new predictions. No test-based retuning before cleanup/pause.


## 2026-09-24 JST: close current study and honor requested pause

Both final full-train30000 runs,137official test frames, all training-fit frames
and terminal audits/manual review completed. Cat PSNR21.119731, Pixiu21.672545;
tiny recall0/575 and6/5324. No reliable tiny-highlight recovery, no default promotion.
Record this implementation/budget as a negative result, without claiming all
physical surface methods impossible. No posttest tuning or further trial.
User-authorized redundant-file cleanup deleted117files/~2.072GiB; final models,
all tests,59training inputs, failed evidence and source/config remain. Preserve
old documentation in pre_cleanup_docs.tar.gz, provide concise current entry pages,
and pause the goal as explicitly requested after these closing conditions.


## 2026-09-24 — Photo-supervised distribution material

Adopt normalized discrete-surface/vMF mixtures and independent RGB energy from
8DNA’s factorization, with analytic direct GGX. Path supervision unavailable; use
image-marginal KL plus photometry and explicitly do not claim original 8DNA
training. Fresh occupancy-only surface seeds and 2DGS self-consistency; no SDF,
learned geometry priors, old checkpoints or dependency upgrades. Two GPUs,
30000 steps per scene, full official test after fixed training.


## 2026-09-24 — PORT-DNA-2DGS outcome

Complete the requested prototype study after fixed fresh30000 Cat/Pixiu and all137
official tests. PSNR/SSIM improve over the default, LPIPS worsens, fixed fur/texture/
highlight crops remain blurred. Do not promote the method or tune after test.
The normalized neural branch is used (16fit direct-only intervention), but this
is not proof of physical path recovery or an isolated KL benefit. Keep all models,
source/config, seed provenance and report-generation failure records.


## 2026-09-24 — User-requested final-only run retention

Keep only five registered methods with extant official final artifacts (18 final models).
Remove pilot/ablation/smoke/initial weights and generated training caches; final-only request
supersedes earlier failed-run retention. Consolidate complete terminal exports into the
owning final test folders before deleting audit containers. Preserve original source/config
and historical docs, label deleted paths and retraining cache requirements explicitly.
Shared datasets and other projects remain untouched. See runs_cleanup_final_only_20260924.


## 2026-09-26 — Redesign recommendation, not an accepted quality improvement

Complete-train evaluation confirms micro-detail underfitting in both scenes. More than half
of the stored surfels cannot contribute through the native alpha cutoff, while population
refinement is disabled. Recommend effective-population management and local per-contribution
transport, with separate Cat thickness / Pixiu surface checks and train-only calibration
evidence. Do not repeat only rank/loss/step increases or treat photo KL as path likelihood.
No new training was launched in this diagnostic pass; the design itself remains unvalidated.
Correct the NRHints new-capture/S-log generalization: Cat/Pixiu belong to the reused DNL
scene family, whose published approximate response is gamma2.2. See 20260926 diagnosis/design.


## 2026-09-26 — Native reconstruction before new relighting

User requested geometry first. Use unmodified author GaussianModel/CUDA, SH3 and
author densification/normal schedule; adapt dataset initialization and exact K.
Keep moving-light limitation explicit. Train-only occupancy positions replace
unverified supplied point clouds; no SDF/pretrained geometry. Review independent
appearance-free geometry views and self-consistency, not RGB metrics alone.
Local extension builds avoid incompatible installed simple_knn GLIBC and preserve
the shared stack. New relighting follows only if geometry is usable.


## 2026-09-26 geometry-first experiment decision

Use the actual author 2DGS model/CUDA to test the user's geometry-first prerequisite, with explicit full-K/mask-initialization adaptations. The native control formed opaque black sheets; explicit alpha supervision fixes gross support, but geometry remains over-smoothed/terraced. Camera-extent coordinate normalization is an isolated third profile, verified projection-invariant and exported back to world coordinates; it does not materially solve visual artifacts. Both scenes fail the geometry gate. Do not label their good silhouette scores or smooth self-consistency as accurate geometry, and do not transfer them as a frozen geometry teacher to the approved new material. Next evidence should come from stable-light observations or validated fit-only classical geometry/calibration constraints. Preserve one final geometry model per scene and archive development-control evidence without its intermediate weights, following the user's cleanup instruction. [Evidence](../experiments/native_2dgs_geometry.md).


## 2026-09-27 topology repair and conditional relighting decision

User authorized repair/retraining, followed by relighting only if geometry is good. Implemented project-local radius lineage correction, then corrected our first repair's cross-interval stale statistics: use interval maxima for the decision, reset afterward. Archived first repair fails the new regression; final four native tests pass. Four fresh30k profiles per scene quantify prune-only, generic depth/regularizer recipe, object-scale splitting, and final interval timing. Functional behavior and population accounting are correct, but both scenes still fail manual geometry acceptance. No relighting training was started. Retain final native pair and complete control evidence, remove superseded weights under final-only policy. No inference that fixing a software defect proves geometry quality, and no claim that illumination/calibration is the sole remaining cause. [Evidence](../experiments/native_2dgs_topology_repair.md).

## 2026-09-27：转向通用几何定义，保留无教师约束

不采用按毛发/半透明类别专门设计的分支。优先 GGGS 连续深度核心，再按证据评估 Wrapping；
CoMVS 的独立深度监督须先证明匹配可靠。当前简单匹配不足，不把不确定深度全局强加给高斯。
首轮 GGGS 使用相同数据、分割、预算，作者模型/增密/学习率与 native 不同，因此不声称严格单变量消融。
先验收灰模、轮廓和多视图几何，再引入光照条件外观及最终 relighting。
用户授权淘汰冗余失败结果；源码依赖保留，权重/图片删除，关键实验依据压缩存档。

GGGS核心的完整30k重跑只带来有限改善，两场景均不通过几何验收，relighting条件不满足。
连续深度替换的负结果不用于否定完整GGGS/Wrapping；停止以SH-only后端轮换作为主线，优先成像模型/几何证据适配。

## 2026-09-27：用户指定方法退役

删除surface_reflectance和directional_surfel，不保留不可达训练/渲染分支；同时移除独占的surface_fragments与surface_sampling。
保留DNA依赖的neural_material材质函数、GGX、2DGS光栅器和轮廓初始化，不改变DNA结构、权重或训练配置。

## 2026-09-27：继续退役两个端口变体

用户指定删除paired_port和local_frame；移除实现、注册、专用单元测试及共享测试中的专属分支。
没有保留方法依赖这两个模块；保留directional/base共享实现及历史研究记录。当前共7个独立方法入口。

## 2026-09-27：用户授权Gaussian Wrapping及条件StableNormal监督

先试Wrappingsurface的完整关键训练组件，不再只换深度内核。保留与此前相同数据/分割/预算。
若无先验结果仍有明显尖刺或形状错误，再以相同fresh预算加入StableNormal法线正则，前向模型与其余损失保持相同。
新的明确授权允许法线教师，覆盖此前禁止预训练几何监督的范围；SDF和深度教师仍禁用。

## User-requested GGGS + neural material trial

Reuse the original GGGS normal/depth geometry independently; continue joint geometry updates as requested. Keep the existing frozen neural BRDF implementation, recreate its retired procedural prior using canonical50k config, and share one decoder between Cat/Pixiu. Use covariance normals through the already-tested3D adapter. Use positive global light-scale fitting as DNA, no image-specific exposure; disclose that this differs from default joint. Keep the30k/default activation schedule and full-data evaluation fixed before observing final metrics. No architectural rewrite or inferred geometry improvement.

## GGGS + neural material outcome

Both30k completed. Small gains over DNA (.078/.386dB), nearly unchanged LPIPS; worse LPIPS than default. Cat/Pixiu surface roughness proxy15.74→47.77°/13.31→68.31° despite improved Pixiu silhouette proxies. Frozen decoder check passed, no implementation failure found. Keep terminal models for comparison, do not promote. Evidence supports constrained geometry updates as a next hypothesis; no new trial authorized by inference or launched here. [Results](../experiments/gggs_neural_material_joint.md).

## 2026-09-28: user-requested1M cap ablation

Change only max-points to1000000 in canonical validation config. Fresh originalGGGS initialization with identical decoder and seed, not a continuation after density refinement ended. Both Cat and Pixiu requested by inherited scene scope; record attained counts because a cap does not force growth. Preserve400k finals. No new architecture or regularization changes. [Protocol](../experiments/gggs_neural_material_1m.md).

## 1M cap result reviewed

Completed and reviewed both full datasets. Cat nearly doubles final points but slightly reduces PSNR/SSIM, increases time59% and memory89%; LPIPS change tiny. Pixiu cap inactive under both settings. Geometry roughness remains; do not promote1M or infer benefit from small nondeterministic differences. No further training launched. [Report](../experiments/gggs_neural_material_1m.md).
