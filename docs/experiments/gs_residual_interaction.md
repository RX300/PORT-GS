# 空间—方向交互受控实验

状态：两组30k、三个节点fit16、终端full562、CPU审计和人工图块检查全部完成。乘性组有有限训练拟合收益，但未通过预定继续门槛；未评价test，不采用为质量赢家。完整研究目标仍active。

## 预先固定的假设与实现

少帧残差直接拟合可恢复训练小峰，完整562帧30k仍低拟合且测试退化。当前模型截断仅涉及
2.386%的covered测试小峰像素，不能解释绝大部分缺峰。下一项检验空间与方向交互的参数化，
不是已经证明交互不足或收敛。研究范围与来源见[研究备忘](../research/high_frequency_relighting.md)。

现有24维网格输出k与原94维输入扣除位置PE后的67维a，新增无bias A:24→16、B:67→16、W:16→128。
两组均保留原输入、首Linear及网格投影，在第一层ReLU前分别加W(Ak+Ba)或W((Ak)⊙(Ba))。
复用一次网格查询，各增加3504参数，总2197997。add可吸收到原首层及网格投影，属于加性重参数化控制；
同参数量不等于同函数类或有效容量。默认none继续原接口与checkpoint。

两组相同随机种子、完全相同原头/网格/A/B/W初始张量，保存residual_initial.pt实证核对。
A/B/W常规随机初始化、末RGB为零保证原图。网格初始化±1e−4导致add/multiply信号尺度不同；
记录k、Ak、Ba、交互输出的绝对均值/均方以及参数/梯度RMS，不临时归一化或改变损失。
保留真实目标lightpos/wi/wo和I/(scale·distance²)，固定源GS/传输/相机；不解释为物理BRDF或真实几何改善。

## 固定协议

- source：runs/sdf_volume_detail_pilot/geometry_grid/Real_NRHints/Pixiu/last.pt；新零头，不续旧残差。
- canonical configs/validation.json，name gs_residual_interaction_pilot，variants add/multiply。
- 各30000步、512px、2048 rays、.25峰+.25邻环、完整562train、seed0、geometry normal。
- LR .001按3000步衰减后保持.00028；保存3k/12k/30k。两张已检查空闲GPU，ssd-gs环境不升级。
- 固定三个节点fit16和终端fit_full562。eval_test:false；不根据中间结果改单组预算或选终点。
- source整条来源已拟合全部562，因此所有fit只能解释为训练拟合。official71未参与权重/相机/先验拟合，
  但已在开发期间多次观察，且相机和灯光同时变化，不能称新盲测或纯固定视角换光。

## 训练继续门槛（在本轮结果之前固定）

multiply相对add完整562帧的GT1–4px组件桶同时满足：

1. abs(contrast_ratio−1)下降至少.03，recall增加至少.03，RGB MAE下降。
2. 全峰precision下降不超过.01。
3. 固定4fit帧[0,35,70,105]的11px邻环neutral过亮增加不超过5%。
4. 整体PSNR下降不超过.1dB，LPIPS增加不超过.002。
5. 人工查看相同GT定义crop，确认窄峰改善而非更宽的错误亮斑。

这是实用研究筛选阈值，不是统计显著性检验。全部通过才将两组固定30k模型一起完整71test；
未通过则记录负结果、不扩test。即使胜add也需对比原GS和历史方法，不能通过弱控制自动宣称质量改善。

## 验证与结果

CPU9项（旧5+新4）通过，含共享初始化、零图保持、add吸收恒等式、新因子梯度/光强齐次/输入隔离、
单次网格查询及4097点分块统计。CPU测试的fork_rng已显式devices=[]，另用CUDA_VISIBLE_DEVICES空运行，避免初始化CUDA上下文。
真实add_retry/multiply_retry各3+3短测通过：原图/alpha精确保持，源GS/transport/camera/radius固定，
A/B/W均实际更新且梯度非零，续训模式不匹配拒绝、缺权重/缺mode拒绝、保存重载逐位相同，native512 CLI评价成功。
两组首次短测训练/续训已通过，后因测试第二个head留在CPU而图像重载检查失败；仅修正测试.cuda()，
原add/multiply失败目录与日志保留，独立retry完成。生产路径未因此改变。
证据runs/gs_residual_interaction_smoke/，CPU日志cpu_tests.log、真实retry/report.json。
独立CPU代码/manifest审查通过，两组训练命令仅mode和output不同；正式结果见下节。

诊断解释：multiply的A乘c、B除c会保持乘积不变，所以单个因子范数不能独立表示分支贡献或容量。
激活RMS用于描述尺度，梯度RMS用于检查实际学习信号；两者都不是高光质量指标。
RGB末层初始为零，因此首步A/B/W梯度为零符合链式法则；后续须检查实际非零梯度及权重变化。
正式启动已核对19个初始状态张量逐位一致，原配置差异仅output/residual_interaction；
首步交互输出RMS为add .3201527、multiply 5.46069e−6，来源startup_audit.json。


## 完成结果：有限拟合收益，继续门槛未通过

以下均为训练拟合；源模型已训练全部562帧。小桶为完整21237个GT组件/40429像素，
2px命中是代理指标，不能代表精确形状。历史none只作参照，本轮匹配控制为add。

| 完整562fit | 原GS | 历史none30k | add30k | multiply30k |
|---|---:|---:|---:|---:|
| PSNR |25.259190|25.155463|25.061057|25.242683|
| SSIM |.891718|.878691|.875183|.876765|
| LPIPS |.129588|.138400|.140676|.137921|
| 1–4px MAE |.202598|.151896|.145893|.140797|
| 1–4px 对比/GT |.294762|.355182|.383538|.410018|
| 1–4px recall |.358357|.436098|.478196|.503277|
| 全峰precision |.631668|.569858|.538856|.566809|
| 全峰recall |.536869|.681807|.712961|.733409|

multiply相对add：小桶对比距离改善.026480（要求≥.03）、recall增加.025081（要求≥.03），
两项未达标；小桶MAE下降.005096、全峰precision提高.027953、PSNR提高.181626dB、LPIPS降低.002755。
固定4fit的11px环域neutral过亮.072574→.069218（−4.624%），其余五项数值门槛通过。
阈值保持事先固定，不因接近门槛而降低，也不改用逐帧不加权平均替代预定合池统计。
这是实用筛选未通过，不是证明乘性交互无效或旧模型容量已达上限。

multiply比add的562帧PSNR有349帧改善/213退化，LPIPS404改善/158退化。
但相对原GS，multiply的SSIM仅1帧改善/561退化、LPIPS15改善/547退化；
整体PSNR仍低.016507dB，LPIPS高.008333。固定4fit环域过亮也仍高于原GS约9.37%。
不能因为赢add而自动称主模型质量提升。

| 同16fit节点 | add3k | mul3k | add12k | mul12k | add30k | mul30k |
|---|---:|---:|---:|---:|---:|---:|
| PSNR |25.856418|25.761495|25.819568|25.733068|25.756139|25.912111|
| LPIPS |.123531|.125083|.126445|.126127|.129070|.126623|
| 小桶对比/GT |.318810|.340734|.343294|.392981|.418500|.438225|
| 小桶recall |.384304|.425566|.402913|.478964|.526699|.525890|

人工查看相同GT定义的8个64×64图块：frame35B亮尖更局部；35A相对add减少一些错误白色碎斑，
但仍有多余宽亮纹。frame0B的两个独立GT小点仍未恢复，预测保留偏下的大亮斑；
70A条状峰部分合并，105A浅色上部的多个亮点仍大多缺失。记录为局部收益存在、可靠窄峰改善未确认，
人工门槛不通过；不是声称所有像素都毫无改善。manual_review.json已与三个gate副本同步。
最终不扩展71test、不采用新默认；本轮没有未见灯光或真实几何改善证据。

## 运行与最终审计

命令：`bash launch_validation.sh configs/validation.json`；git基版本47028ea0ff77f427aa6a016cb64e039432935a5e，
实际脏源码以run/source.tar为准。配置、全部CLI命令、环境与GPU分配保存在validation.json/manifest.json。
仅使用检查空闲的GPU0/1，各30000步/61.44M pixel draws；耗时1857.73/1862.30秒，峰allocated7.632GiB。
两组逐步帧序列和抽样累计计数一致；3k每帧0–17（3帧本stage未抽），12k10–35、30k34–80，
终端均覆盖全部562。所有source帧先前已经训练，不能将stage未抽帧称未见。

CPU最终审计首次执行exit0，无审计失败。19个初始state张量逐位一致，首loss相同；
三个节点主GS/传输/相机/radius逐位保持、源渲染配置和PORT/阴影启用状态保持，全部tensor有限。
A/B/W实际更新、日志梯度有限并非持续为零。终端full562交互输出RMS add .370040、mul .373701，
初始尺度差没有导致乘积分支持续失活；这不排除优化动力学差异，也不是容量上界证明。
全部指标按逐帧raw sums重算通过；GT、alpha和计数精确一致。
独立fit16/full562仅add frame105的delta_abs_mean差5.59e−9，已在预设rtol1e−6/atol1e−8内记录，
其余重叠值相同。训练/渲染/评价/测试源文件与启动source.tar逐字节一致。
所有训练/评价/审计进程结束，GPU0/1释放，无下一训练正在运行。

输出：`runs/gs_residual_interaction_pilot/`；模型和评价位于`{add,multiply}/Real_NRHints/Pixiu/`。
根目录包含comparison.json/png、comparison_crops.png、interaction_curve.png、interaction_scales.png、
small_peak_audit.json、training_gate.json、manual_review.json、final_audit.json/log、audit_attempts.json、
startup_audit.json、source_consistency.json、analysis_source.tar。完整CPU分析脚本在/tmp/port_residual_interaction_audit.py并已归档。

同期只读核查[BiGS入射—出射表示](../research/bidirectional_angular_representations.md)和
[GS³角度高斯](../research/angular_lobe_representation.md)；没有安装其环境、修改SSD-GS或改变本轮协议。
这些候选尚未实现或验证，不能由论文效果推断本项目缺失小峰已解决。

## 下一诊断决策

先做当前遗漏小峰的局部角度提示定位/可分性审计，暂不增加训练分支。
以当前multiply30k的完整562fit及正确相机，计算固定窄/宽half-vector核在每个1–4px组件及11px邻环的信号，
区分已经命中/部分/完全未命中，检查极值是否在正确位置。相同中心的核缩窄不会移动极值，
此诊断能检验候选窄核是否有定位正确的输入信号，而非重复旧的平均n=h角误差。
单GPU、只前向、不用test、不优化GT方向；完整预定统计、分母及实用筛选见[接续](../project/research_handoff.md)。
尚未实现或启动，不把拟议诊断当已有证据。
