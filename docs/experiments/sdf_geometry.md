# SDF几何试验 — 2026-09-23

目的：检查共享连续SDF是否能改善2DGS几何，并帮助恢复未准确呈现的高光。

## 固定协议

复用ssd-gs/CUDA12.1，GPU0/1最多两张。根git revision 47028ea，具体源码以source.tar为准。
入口：`bash launch_validation.sh`，配置：configs/validation.json，目录：runs/neural_material_sdf_geometry。
Cat/Pixiu均从neural_material_highlight_recovery的38k checkpoint开始，追加3000步，seed0，
512px黑底，全部522/562训练帧，评估完整66/71个official test帧，原相机。

两组control/sdf的材质、高光权重0.05、初始化、学习率、图像与几何监督相同。
几何全部解冻，不增删点(refine-stop=0)。StableNormal=0.005、DA3=0.05、
深度法线一致性=0.01、distortion=0.01，从第1步启用。
SDF组仅额外启用双向场，预热500步、渐增500步，随后完整权重2000步。
共享decoder冻结，保留6维材质码及原有3维有界着色法线，不再重置材质。

比较PSNR/SSIM/VGG LPIPS、GT中性局部亮点的RGB MAE与对比度、轮廓IoU、
渲染法线/深度法线夹角、归一化distortion。亮点mask是图像代理而非真实specular分量；
几何一致性指标无几何GT，不能单凭一致性下降就认定几何更准确。
预算/权重在本轮test评价前固定，不用test选检查点。

## 进度

CPU定向5项检查通过；真实Pixiu梯度+3步训练+3步继续训练通过。
正式对照已全部完成，完整结果见后文。

## 下一轮：小步位置更新的训练集对照

首轮两场景test PSNR：control 21.517294/21.019045，SDF 21.517097/21.062508。
相对control只出现Pixiu约0.043dB的微小改善，不能称为恢复高光；两组均低于38k初始化。
发现位置lr重启为原末端约100倍，下一轮先隔离这个优化因素。
canonical配置改为neural_material_sdf_small_step_pilot：仅Pixiu，原38k初始化，
control/SDF各2000步，将position-scale从camera换为object，position-decay从0.01换为0.1。
初始位置lr从0.00236665降低到0.000224326，末端0.0000224326接近原30k末端。
其余材质、先验、SDF参数及点数不变，使用fit评价和固定训练帧诊断，不追加test调参。
首轮配置仍完整保存在runs/neural_material_sdf_geometry/validation.json。

## 首轮完整结果

Cat/Pixiu共137个official test帧，两个等预算分支均完成；每个分支全部训练3000步。
17项CPU检查与真实Pixiu短训/重载通过，最终checkpoint有限、decoder与38k初始化完全相同。
SDF确实更新了Gaussian位置/朝向/尺度/opacity；final_audit.json逐项记录。

|场景|分支|PSNR|SSIM|LPIPS|法线/深度夹角°|亮点RGB MAE|亮点对比度/GT|
|---|---|---:|---:|---:|---:|---:|---:|
|Cat|initial|22.3332|0.7763|0.2572|17.053|0.1301|0.0418|
|Cat|control|21.5173|0.7676|0.2741|15.742|0.1193|0.0557|
|Cat|sdf|21.5171|0.7679|0.2742|14.351|0.1187|0.0681|
|Pixiu|initial|21.2051|0.8450|0.1604|13.840|0.3693|-0.0034|
|Pixiu|control|21.0190|0.8444|0.1656|16.477|0.3521|-0.0013|
|Pixiu|sdf|21.0625|0.8447|0.1651|15.194|0.3523|-0.0051|

结论：SDF改善相对control的几何自一致性，Pixiu PSNR仅提高0.04346dB；
两场景仍比38k初始化差。Pixiu亮点对比度仍接近零，局部图像可见高光位置/宽度未恢复。
Cat亮点代理只有1101个像素且包含部分纹理；Pixiu26147个像素。
不把本轮视为高光恢复成功，也不替换默认方法。

产物：runs/neural_material_sdf_geometry/comparison.json、comparison.md、comparison.png、
final_audit.json；每个分支/场景的test/与geometry_diagnostics/，以及initial_diagnostics/。
诊断增加的SDF误差代码在diagnostics_source.tar（叠加source.tar），命令在diagnostics_manifest.json。

## 小步位置更新结果：未恢复高光

Pixiu两分支各2000步、全fit评价及固定16个训练帧诊断完成；未追加official test。

|分支|16帧PSNR|亮点RGB MAE|亮点对比度/GT|法线深度夹角°|
|---|---:|---:|---:|---:|
|initial|24.7608|0.197135|0.298859|13.5557|
|control|24.4302|0.196523|0.274583|14.5430|
|sdf|24.4111|0.192705|0.279793|13.8291|

相同5390个训练亮点代理像素。SDF略减小亮度误差，但对比度仍低于38k初始化，
整体拟合未提升。小步位置更新不足以解释或解决高光问题，不再沿此参数继续扫。
下一步检查相机条件多视图DA3先验，为连续场加入图像产生的额外几何信息。

## 多视图深度先验检查与对照

真实Pixiu前8帧、多视图上下文仅来自train。固定4个跨批相机锚点，每批最多8视图，
传入已知OpenCV相机外参/K；没有新增依赖或预训练权重。
深度导出法线与StableNormal的平均夹角：单图34.464°，多视图24.949°，8帧均下降。
输出深度尺度与训练相机至物体中心距离相近。参照法线与物体中心都不是GT，
这一检查只确认有继续实验价值，不能证明真实几何精度提升。
检查产物runs/sdf_multiview_depth_check/quality_check.json与normal_comparison.png。

生成全部562训练帧的多视图深度于runs/sdf_multiview_depth_priors/Real_NRHints/Pixiu/；
normal.pt链接已有StableNormal先验，保持法线监督相同。
命令：`CUDA_VISIBLE_DEVICES=0 third_party/da3_env/bin/python prepare_surface_priors.py --kind depth --scene /workspace/datasets/SSD-GS/data/Real_NRHints/Pixiu --output runs/sdf_multiview_depth_priors/Real_NRHints/Pixiu --resolution 512 --processing-resolution 512 --depth-views 8 --fit-all`。

下一实验neural_material_sdf_multiview_pilot使用相同38k初始化/小步位置lr/SDF/2000步，
仅替换深度先验，直接复用上一轮sdf组作为单图先验对照，不重复训练。
训练和诊断仍仅使用fit，固定同16帧；不查看新official test。

补充先验审计：前8训练帧前景深度(95%−5%)/中位数均值，单图=0.189819、多视图=0.064462。
单图相对深度起伏更大，但没有几何GT，尚不能单凭这个量判定新深度正确。

## 多视图先验续训结果：未改善高光

全562帧fit PSNR：单图23.573149、多视图23.573587，差异只有0.00044dB。
以下为同16个训练帧、相同5390个亮点代理像素：

|分支|PSNR|亮点RGB MAE|亮点对比度/GT|法线/深度夹角°|
|---|---:|---:|---:|---:|
|initial|24.7608|0.197135|0.298859|13.5557|
|sdf_monocular|24.4111|0.192705|0.279793|13.8291|
|sdf_multiview|24.4381|0.193683|0.281047|13.8126|

新先验的几何参考一致性提高，没有转化为显著渲染或高光收益。
不据此扩展official test、不继续扫深度权重。全部输出与源码保留，默认方法不变。
最终checkpoint 2000步/562 fit帧/无validation，字段有限，decoder保持冻结原值。
下一研究候选为SDF连续法线直接参与PBR，接受图像误差监督；尚未实现。

## 连续法线直接着色：固定对照协议

runs/neural_material_sdf_shading_pilot：两组均从多视图SDF试验的最终checkpoint开始，
同样Pixiu全部562 train、seed0、512、追加2000步、object位置lr/decay=.1，不增删点。
auxiliary继续原几何互监督；shading额外使SDF基础法线进入同一神经材质PBR。
两组均保留field拟合、Eikonal、双向几何损失、相同先验和highlight_weight=.05，decoder冻结。
评价全fit及相同16帧训练诊断，暂不新增official test。
仅当图像/高光整体改善时才扩展验证，不以SDF loss下降代替效果证据。

## SDF法线直接着色结果：负结果

auxiliary/shading两组各2000步，全562帧fit及固定16帧诊断完成。
全fit PSNR为23.584880/23.469237，直接着色更差。配置审计证明两组只差sdf_shading开关。

|分支|16帧PSNR|亮点RGB MAE|亮点对比度/GT|GS法线/深度夹角°|
|---|---:|---:|---:|---:|
|initial|24.4381|0.193683|0.281047|13.8126|
|auxiliary|24.4248|0.187286|0.293798|13.8215|
|shading|24.2824|0.202236|0.232265|13.2826|

前4个诊断帧共有1210个亮点代理像素：实际着色法线与半向量夹角从auxiliary 19.334°
增至shading 26.674°；基础法线则从23.733°增至32.761°。连续性增加没有恢复正确高光方向。
直接着色分支保留为已验证但未采用的实验开关，默认仍辅助模式，不扩展该候选的test评价。

最终field步数4000、decoder完全不变、所有权重有限，见final_audit.json。
训练帧固定对比图comparison.png、完整比较comparison.json及protocol_audit.json均位于本run。

## 单Gaussian与合成表面的差异审计

在auxiliary模型399399个Gaussian中，opacity>0.5的93295个点有33.36%离SDF零面超过0.02场景半径；
按面积×opacity归一化的交换测度，有40.91%位于该阈值之外。中位距离0.01591场景半径。
这不是几何真值误差，包含场近似误差、遮挡点和混合层，不能据此直接删除Gaussian。
但它证实当前只约束合成深度/法线，没有把单个Gaussian约束到零面。
下一候选：对近似可见、高opacity的Gaussian施加SDF零面软约束，保持现有GS法线着色与点数。
用等预算辅助模式作对照，避免直接删除点造成轮廓缺损；代码尚未实现。

## 单Gaussian约束：固定协议

neural_material_sdf_primitive_pilot从上一轮auxiliary最终模型（总42k、SDF4000步）开始。
control/primitive各2000步，仅primitive权重0.05不同，其余材质、先验、位置lr、点数、SDF项相同。
筛选近似可见opacity>0.5中心，Z深度窗口0.05半径；每步1024点，无剪枝、无SDF直接着色。
仅全fit与相同16帧train诊断；同时核验单Gaussian离零面的分布及渲染指标，避免只优化代理指标。

本轮诊断额外记录GT亮点的half-vector specular反事实探针：仅在GT选择的像素替换高光项法线，
其余颜色分量不变。这不能用于真实预测，也不是理想法线的真值，只用于区分法线与材质限制。
诊断源码更新保存在本run的diagnostics_source.tar，叠加训练source.tar；训练/eval代码未变。

## 单Gaussian软约束结果

全fit PSNR control=23.597247、primitive=23.611050，提升0.01380dB；LPIPS基本相同。
固定16帧亮点对比度/GT .303435→.306375，亮度MAE .184482→.185347，没有实质性高光恢复。

固定使用初始化opacity>0.5的相同93295点核验，避免通过更改opacity改变统计集合：
|分支|到初始field平均距离|初始field外0.02比例|到自身field平均距离|自身field外0.02比例|
|---|---:|---:|---:|---:|
|initial|0.019500|0.333619|0.019500|0.333619|
|control|0.019639|0.338303|0.019279|0.312728|
|primitive|0.018193|0.277560|0.021309|0.389196|

单点确实更接近初始field，但field联合更新后零面也变化；这可能使几何追随移动目标。
当前近似可见点到自身field的距离mean由control .013554增至primitive .015418，不能声称表面贴合整体改善。
所有点数399399不变、decoder不变、权重有限，field更新至6000步；原始日志和审计保留。

反事实探针（前4帧1210像素）：control实际对比度.281864，半向量specular探针.670520；
MAE .163723→.113686。probe使用GT选择像素，不能当作预测或证明可泛化，但显示法线方向仍有改善空间。

下一项固定field的几何精修从同一个初始auxiliary模型出发，保持primitive_weight=.05及其余设置，
仅--freeze-sdf：不更新field、不计算GS→SDF拟合/Eikonal，仍保留SDF→GS的几何损失。
输出neural_material_sdf_fixed_field_pilot，与已完成primitive组比较，预算仍2000步，只fit评价。

## 固定SDF精修结果：小幅变化，尚未恢复高光

全562帧fit PSNR=23.606072，普通续训23.597247、联合primitive=23.611050；整体差异很小。
固定16帧：PSNR=24.453213，亮点MAE=0.183768，
亮点对比度/GT=0.315260（普通续训.303435、联合primitive .306375）。
可见候选点自身field平均距离=0.012477，超过0.02半径的比例=0.112070，
普通续训对应.013554/.134720。表面贴合代理有改善，但无几何GT，不能认定真实形状精度提升。

以同一初始高opacity集合统计，到初始field距离mean=0.018131，
>0.02半径比例=0.271215，初始分别.019500/.333619。
field全部张量逐位不变，sdf_steps=4000；Gaussian点数399399不变、decoder不变、权重有限。
全部训练/诊断完成，未新增official test评价。没有恢复准确高光，不提升为默认方法。

## 连续法线细节场：固定协议

neural_material_normal_field_pilot：从fixed_field最终模型（总44k、SDF4000步）出发，
control/normal_field各追加2000步，保留freeze-sdf和primitive_weight=.05、相同多视图深度、
相同材质与原法线角度范围；仅normal_field组增加零初始化位置残差及1e-4残差正则。
GPU0/1最多两张，全部562 train、seed0、512px；评价全fit和固定16帧，暂不新增test。
真实短测证明新分支初始化不改变图像，场通过RGB学到非零参数，保存/恢复完整。

## 连续法线细节场结果：负结果

control/normal_field全562帧fit PSNR23.628176/23.635938。相同16帧PSNR24.495669/24.501015，
高光RGB MAE .177761/.180121、对比度/GT .328722/.325596。整体变化很小，高光没有改善，
不扩展official test、不提升为默认。训练346.64/352.27秒，峰值分配13.44/13.62 GiB。
协议仅normal_field与输出不同；final_audit.json确认SDF逐张量不变、计数4000、decoder不变、权重有限。

## 训练高光提示与跨视图几何审计

runs/specular_cue_coherence保留562 train帧211070个高光代理像素，未使用test。
raw GS点按.005场景半径体素汇聚、每帧一票、至少3视图，留一视图半向量预测误差22.856°；
SDF零面投影后24.197°。红色主体诊断子集仍20.733°/22.184°，不能将亮点半向量当真值法线。
排除被预测视图、仅选择其他视图方向一致且相机多样的体素，伪法线依然差于当前模型。
更细体素没有消除问题；DA3绝对深度在同一像素重建的点比GS更不一致，不作为绝对几何监督。
4096同源点、64训练视图的轮廓探针：GS平均落在膨胀2px轮廓外比例9.04%，DA3为34.26%。
这些是相机/掩码/表面之间的一致性诊断，不是几何真值误差。
原始银行normal_cues.pt；细节见geometry_cue_check.json、confident_cue_check.json。

## 2DGS交点深度：实现与固定协议

审计gsplat 1.5.3的rendering.py和RasterizeToPixels2DGSFwd.cu发现：
RGB+ED实际混合的是Gaussian中心相机Z；光栅核计算射线与圆盘的局部交点，却未将交点Z用于深度。
在world z=3、camera yaw=.4 rad的单圆盘测试中，返回深度恒为3.06318；
正确交点深度范围2.57146–3.87188，原反投影离平面mean .312713、max .700570。
项目内新增可选surface-depth=intersection：复用原贡献列表、相同高斯核/alpha与中心排序，
通过射线变换求交点Z后期望合成。2D屏幕低通支撑继续用中心Z；不声称它也是平面交点。
这是可微PyTorch参考路径，未修改共享依赖；原RGB属性、alpha、深阴影与中心Z distortion代理不变。
几何GT解析测试max误差4.77e-7；位置/旋转解析梯度、双层位置/opacity梯度、空视图通过。
真实SDF短训/重载/CLI评价见runs/sdf_intersection_depth_smoke；不以短测评价质量。
4个已训练视角交点深度平均改变1.1%–1.4%场景半径，法线一致性直接替换变差：模型需适应新几何语义。
诊断证据runs/specular_cue_coherence/depth_intersection_probe.json。

neural_material_sdf_intersection_pilot：同38k highlight_recovery初始化，center/intersection各2k，
seed0、512px、562 train、相同多视图相对深度/法线先验、位置scale=object、decay=.1。
两组均从新SDF开始，500步warmup+500步渐增，SDF互监督权重.05/.01；不冻结旧场、不开单点约束/细节法线。
只surface-depth不同。GPU0/1最多两张，评价全fit及相同16帧train；尚未产生质量结论。

本轮在原有高光误差/对比度外，补充亮点定位诊断：预测与GT都用相同neutral_peak_mask、
相同GT前景区域；分别累计预测点与GT点在对方2px（Chebyshev）邻域内的比例作为precision/recall。
规则在本轮训练结果完成前固定，防止把错误位置的亮点误认为恢复。仍只是图像代理，不是真实BRDF标签。
诊断源码新增于diagnostics_source.tar；训练源码source.tar保持不变。

## 交点深度2k结果：未恢复高光

center/intersection两组训练与562帧fit、固定16帧诊断全部完成，均未用test。
全fit PSNR23.569698/23.584735，SSIM .872597/.872557，LPIPS .145551/.145900。
16帧PSNR24.411436/24.449635；高光MAE .194929/.194901、对比度/GT .277504/.276885。
高光定位recall@2px .466234/.455473、precision@2px .570334/.590730；GS法线/深度角13.825°/14.977°。
提升很小且指标混合，不能宣称高光或几何质量改善。comparison.png显示细节与光斑仍模糊。
训练360/426秒，约18%时间开销；field2000、399399个点、decoder不变、参数有限。
配置只surface_depth/output不同。此实验检验从长期中心深度模型适应新深度，不代表从头训练的结论。

## 从头训练的SDF互监督：固定协议

neural_material_sdf_fresh_validation：Pixiu全562 train，seed0，512px，两组各30k，从相同随机20k点开始。
两组统一intersection深度、改进材质先验、highlight_weight=.05、normal_weight=.005、相对depth_weight=.05，
position_scale=camera、decay=.01、max_points400000、refine_stop25000，阴影/端口5000启用，表面先验1000启用。
仅SDF开关不同：gaussian组无field，sdf组从第3000步拟合field，500有效步预热+500步渐增反馈，权重.05/.01。
不使用init-checkpoint、primitive约束、SDF直接着色或法线细节场，避免旧深度几何与旧field的历史影响。
增密统计明确使用原CUDA属性路径的absgrad，不包含独立交点深度路径的逐像素绝对梯度；两组协议相同。
保存10k、20k、30k用于确认是否仍在改善；队列评价各检查点全fit，正式test留待训练候选确定。
新`--sdf-start`默认1保持现有协议，显式用于从头训练的几何预热。SDF shading模式的RGB梯度仍按原方式更新field。

首轮fresh smoke发现原子index_add导致重载图像5个像素有7.45e-9差异，严格重载检查失败，原日志保留。
已用按ray固定顺序的segment_reduce替代原子求和；不放宽重载相等断言。
短程已完成结果配套原source.tar；完整从头实验使用新的源码快照，不能混淆两个版本的数值顺序。

从头对照prefix_audit.json：首步帧/点数/loss完全一致；3000步SDF启动前已有CUDA训练数值差异，
2900步点数91728/90399。固定seed和相同配置不意味着整个增密轨迹逐位相同；小幅最终差异仍需重复seed验证。

## 从头10k的独立轮廓一致性检查（非最终结果）

hull_audit_010000.json/png使用64个均匀训练视图、512px GT掩码、2px膨胀，排除相机后和画面外点。
各分支从opacity>.5中心按opacity×切向面积独立抽4096次（有放回、seed0）；没有使用GPU或test。
无SDF/有SDF的平均轮廓外视图比例 .028564/.031221；至少10%视图外的点比例 .074219/.087646。
这是不同分支的分布采样，不是点对点匹配；高opacity中心也不等价于真实表面点。
有SDF分支用10次Newton迭代投影到零面，要求|f|<1e-4半径且位移<.1半径，保留3763个样本。
投影后平均轮廓外比例 .073856，至少10%视图外的点比例 .276641；JSON同时保存同一接受子集的投影前统计。
此中间结果表明SDF零面尚未更符合独立轮廓约束，不据此提前停止或改变30k实验。
这些指标检查给定相机与GT掩码的一致性，不能替代几何GT精度或最终高光/重光照评价。

### 10k SDF整体轮廓投影（CPU）

为避免只看抽样点，进一步在归一化[-1,1]^3内计算256^3 SDF网格，对16个均匀训练相机做128px投影，
每条与包围盒相交的射线均匀采256点，以三线性SDF<0判断占据；GT alpha按area缩到128并用.5阈值。
平均IoU .867599、precision .912057、recall .946566。前4视图用128^3网格复核，IoU变化均<.0016。
结果不是原生2DGS轮廓，不能直接与此前512px GS IoU横比；它表明独立SDF形状仍有可见轮廓误差。
证据sdf_silhouette_010000.json/png；不使用GPU或test，不修改正在运行的训练。

### 10k完整分辨率的相同16帧诊断

额外检查GPU1有约28GiB空闲后，在同一GPU短暂运行诊断，进程allocator上限20%，实际峰值2.734GiB。
没有使用第三张GPU，没有停止/改动训练；GPU1训练墙钟时间包含这次共享开销，不据它作严格速度比较。
复用diagnose_image_errors.py，512px、相同16个train帧、geometry_frames=2、highlight_frames=4。
无SDF/有SDF：PSNR22.421174/22.297388，高光MAE .285786/.287173、对比度 .055256/.049164，
召回@2px .124490/.108163，精确率 .403509/.405948；GS silhouette IoU .887441/.889280，
法线/深度夹角21.1661°/18.9541°。SDF内部几何一致性改善，亮点质量尚未改善。
这是10k中间结果，与原定30k最终/完整fit评价分开；comparison_010000.json/png与两组diagnostics_010000保存。
整体SDF体素投影的归一化包围盒六面均为正，最小f=.356315，没有负体积被盒边裁断。

## 独立SDF轮廓CPU小试：改善很小

入口diagnose_image_errors.py新增--sdf-mask-fit-steps，仅CPU、仅更新field，不改变Gaussian/材质/辐射。
从sdf_fresh的10k SDF出发，36个训练视图（fit[::16]）拟合、18个不同训练视图（fit[8::32]）检查；
检查视图仅排除于本次轮廓拟合，并非原始场景训练的held-out。没有official test参与。
128px，每步64射线（半数轮廓带、半数均匀有效射线），每射线128采样，logit=-min(SDF)/.005；
BCE权重1或0、Eikonal .1、Adam1e-4、1000步、seed0。CPU相同初始化，source/config/history/field/图像完整保存。
输出runs/sdf_mask_cpu_pilot/{control,mask}。初始检查IoU .865792；Eikonal-only后.596272，mask后.867382。
不能用失去表面位置约束而退化的control，宣称相对初始大幅提升；新约束尚未证明能改善场景高光。
新增解析球轮廓与SDF梯度回归通过，当前21项CPU检查全部通过。该probe未接入主训练。

## 轮廓与输入相机的独立一致性审计

不使用学习的几何，仅从相同36个训练掩码构造256^3视觉外壳，再投影到18个检查视图。
128px原掩码严格交集的检查IoU .825665、召回 .837196；512px掩码膨胀2px后 .848189/.873543。
把射线采样从256加至512，仅改变IoU约.0005；所有保留体素至少被2视图观察，未知体素未造成截断。
扩大至8px容差，召回升至.939695而精度降至.903329，IoU只有.854095；允许2个视图违反也未改善IoU。
在参与构造的视图中，512px+2px外壳IoU .848747、召回 .860613，说明问题并非仅检查视角插值。
这不是几何GT误差；标定、掩码误差和离散化均可能影响。证据visual_hull_masks*.json/png。

进一步做Pixiu训练SIFT审计：40个均匀anchor、各两个约3°相邻姿态候选、前景SIFT、双向ratio .7，
图像F用1px RANSAC；只在其inliers上比较原始K/pose导出的F。14对有至少20个inliers，
成对中位Sampson误差的中位数为1.70498px，图像估计F为.15929px；半像素坐标修正不消除差异。
限定两张图前景最低35%区域、局部R<1.5G以偏向平台纹理，18对至少8点，标定误差中位数1.77582px。
其中140/474有55个平台匹配，误差10.243px；对应纹理的连线图已人工查看。
这是标定/图像不一致的证据，不能据少量、可能共面的匹配估计完整3D误差或声称全部高光问题均由它造成。
原6°审计另存pixiu_camera_audit.json；更近视角为pixiu_camera_near_audit.json，平台子集为pixiu_camera_board_audit.json。
数据匹配在pixiu_camera_correspondences.npz，所有图像与指标在当前fresh run。

20k额外共享GPU诊断因主训练显存峰值而OOM，主训练继续正常；失败日志diagnostics_020000.log及
部分产物diagnostics_020000_failed_shared_gpu保留。后续诊断等待相应训练进程退出，不再与训练共享GPU。

后续判断：完成当前预定30k，并比较末端的固定16帧与全部fit。最终固定30k两组还应各做完整official test，
用于检验正则是否改善泛化；不在10k/20k上反复测试或按test选检查点。下一候选优先训练相机修正，
其效果需同时看独立匹配残差、fit高光和原始测试相机下的评价，不继续盲加SDF权重。

## 从头30k完整结果：SDF未恢复高光

两组30k、全部562 fit、固定16帧train诊断、固定末端的全部71 official test及高光诊断完成。
无SDF/有SDF的全fit PSNR23.443276/23.365305；test PSNR21.248259/21.315856、
SSIM .843846/.844464、LPIPS .161047/.160689。小幅test增益不足以宣称明显改善。
测试亮点MAE .362038/.361996基本相同，对比度/GT为-.015385/-.011767，召回@2px .160898/.151375。
高光位置/对比度仍明显不对，不能认定高光恢复成功。测试法线/深度夹角16.9343°/14.3808°，内部一致性改善。
原30k神经材质test PSNR21.478632、SSIM .851217、LPIPS .161733；当前也未全面超过原30k。
完整comparison.json、comparison_train.png、comparison_test.png及各场景final_audit.json保留。
点数330930/332996；两组decoder逐张量不变、参数有限；SDF更新26989步。默认方法不变。

## 下一阶段：训练相机修正受控对照

neural_material_camera_refine_pilot：从fresh/gaussian最终30k出发，两组各5000步，seed0、512px、全部562 train。
初始化选择依据是该模型更好的fit结果及隔离相机因素，不依据test选择检查点。
两组共同做几何/材质/传输小步精修：position_scale=object、decay=.1、无增密、阴影/端口第1步启用，
保留相同先验、修正交点深度与冻结decoder；没有SDF辅助项。仅camera组启用optimize-cameras，camera_start=1、lr=.0003。
第一个训练相机固定，其他相机用既有有界SE(3)模块；不新建相机表示或改数据集元数据。
报告校正训练相机下的fit须明确标注；最终test仍用原始相机。独立平台SIFT对应点未进入训练loss，用于事后几何检查。
真实相机训练、anchor不动、alpha到pose梯度、保存/重载/CLI corrected-fit全部通过，证据runs/intersection_camera_smoke。

## 训练相机修正5k结果：整体画质改善，高光仍未恢复

control/camera均完成同30k初始化后的5k精修、562 fit、固定16帧train诊断、全部71 test与高光诊断。
全fit PSNR23.609348/25.005606；camera的fit使用保存的修正训练相机，不能当作相同原相机下的比较。
原始测试相机下PSNR21.059702/21.322745，SSIM .842716/.843663，LPIPS .161373/.155105。
测试亮点MAE .360458/.357679、对比度/GT -.013584/-.010549、召回 .161204/.155773：准确高光未恢复。
独立匹配误差由1.704978降至.789299px；平台子集1.775818降至.682851px，支持真实对齐改善。
两组点数330930，decoder不变、参数有限；test的corrected_fit_frames=0，camera首帧raw为零，核验记录完整。
输出runs/neural_material_camera_refine_pilot，comparison.json/train.png/test.png、final_audit.json齐全。

相机偏移简单迁移诊断：在每对匹配两端都排除其自身偏移，用pose-space最近4个训练相机、逆平方距离权重，
预测两端的camera-local raw校正。所有匹配/平台误差分别1.712198/1.860533px，未优于原始1.704978/1.775818。
这是训练内部诊断，不是严格独立场景测试；该预测方案被否决，没有用于测试相机。
采用保存的校正训练相机重做相同视觉外壳检查，512px+2px容差IoU .881694（原.848189）。
这些检查视图曾参与场景/相机拟合，因此只证明训练内部轮廓一致性改善。源代码与结果均保留。

## 固定校正相机后的SDF互监督：预定协议

neural_material_calibrated_sdf_pilot：从camera_refine/camera最终35k出发，两组各5k、无增密，
相同图像、先验、材质、位姿、位置步长和intersection深度，仅sdf开关不同。
两组optimize-cameras=true用于恢复已学习校正，但camera_lr=0，从首步应用并保持所有相机参数不变。
SDF组重新拟合field，500有效步预热+500步渐增反馈，权重.05/.01；不加primitive、SDF着色或细节法线场。
固定已改善的标定可避免把后续相机变化混入SDF效果。真实3+3步、重载、逐张量相机不变检查通过，
证据runs/calibrated_sdf_smoke。评价先看全fit与固定16帧，再仅对最终5k模型做原始测试相机评价。


### 固定相机SDF对照实际启动确认

`neural_material_calibrated_sdf_pilot`两训练子进程194525/194526已确认在GPU0/1运行，
history超过3000步；protocol_audit.json仅output和sdf不同。其他GPU未使用。
后处理exec会话43113/54342等待各自完整benchmark job完成，再执行16帧train、71帧test和高光诊断。
下一材质分支目前仅CPU验证，不占用第三张GPU；运行中训练采用已有source.tar，未修改其协议。


### 固定校正相机的SDF对照最终结果（完整）

输出`runs/neural_material_calibrated_sdf_pilot`。两组各5k完成，562 fit、固定16 train诊断、
完整71 official test及71帧高光诊断完成；测试使用原始相机，没有测试位姿优化。

| 指标 | control | sdf |
|---|---:|---:|
| 全fit PSNR |25.275051|25.259190|
| test PSNR |21.308830|21.331654|
| test SSIM |.842869|.843040|
| test LPIPS |.152874|.152835|
| test高光RGB MAE |.359883|.358917|
| test高光对比度/GT |−.006987|−.005892|
| test高光召回@2px |.178108|.179523|
| test法线/深度夹角 |17.734166°|16.568914°|
| 16 train高光MAE |.156602|.158159|
| 16 train高光对比度/GT |.450253|.451671|
| 16 train召回@2px |.600000|.603525|

final_audit确认两组相机offset逐位等于35k初始检查点、冻结decoder不变、330930点、所有张量有限；
SDF有效更新5000步。comparison_train/test.png人工查看仍缺少正确细小亮点。
**结论：即使改善训练相机对齐，当前辅助SDF互监督仍只带来小幅内部一致性改善，不能认定高光恢复。**
它不排除更强独立SDF表面观测的作用，但当前不继续调大互监督权重；默认方法保持原样。

### 下一对照：固定几何和相机的解析材质

canonical `configs/validation.json`配置`analytic_material_pilot`，neural/ggx各5k、Pixiu全部562 train。
同一camera_refine_pilot/camera的35k模型初始化；此前已选择该检查点，不根据本轮test选模型。
冻结means/scales/quats/opacities及相机，refine-stop=0；两组均reset-material，保留原着色法线、底色和PORT。
neural重置为改良prior初始码；ggx重置F0=.04、alpha=.1/.3、mix=.9，无预训练权重。
优化材质、底色、有界着色法线特征和PORT；两组初始RGB不会完全一致，这是表达替换而非零残差增加。
统一原始点光、intersection接收点、直接高光保留、highlight=.05、512px、seed0、相同学习率和预算。
关闭固定几何下的常量先验损失，未增加SDF；先报告全部fit及固定16帧train诊断，尚未安排official test。

`runs/analytic_material_smoke`真实512px 3+3步证明神经→解析转换、续训、有限值、逐位渲染重载和CLI评价；
几何与保存的相机张量逐位未变。23项CPU检查通过。改动保存在下一run/source.tar，旧实验使用旧快照。


### 训练平台匹配的统一内参检查：未通过留序列验证

CPU使用既有SIFT bank中18对、548个静态平台代理对应点（每对至少8个，来自同一捕获序列），
在原始外参下拟合一个共享K：fx/fy乘子限.8–1.2，cx/cy偏移限±64px，soft-L1 Sampson残差。
只读train，没有读取test图像或test元数据；不改变数据文件或训练协议。
全对拟合使每对误差中位数1.77585→1.61001px，但拟合结果不能作为独立验证。
排除整个捕获序列后，在未用于内参拟合的序列上检查：

| 排除/检查序列 | 检查对数 | 原误差px | 修正误差px |
|---|---:|---:|---:|
| Seq1 |10|2.47076|1.51917|
| Seq2 |4|1.27180|2.73045|
| Seq3 |4|1.77585|3.58824|

第三折cx触及边界，焦距/主点在不同折不稳定。有限平面纹理匹配也限制可辨识性。
因此没有证据支持将这一统一内参修正用于模型或测试相机；未采用，也不继续在这18对上调边界搜参。
输出runs/neural_material_camera_refine_pilot/intrinsics_probe/metrics.json与source.tar。
这不证明真实内参完全准确，只是否决了此具体共享修正估计。


### 解析材质对照最终训练结果：未胜出

`runs/analytic_material_pilot`两组5k、全部562 fit和固定16train诊断完成。
冻结几何及校正相机逐位不变、330930点；神经decoder不变，GGX无decoder权重，全部张量有限。

| 指标 | neural | ggx |
|---|---:|---:|
| 全fit PSNR |25.085764|24.879871|
| 全fit SSIM |.889809|.886569|
| 全fit LPIPS |.131759|.136268|
| 16train亮点MAE |.169378|.177121|
| 16train亮点对比度/GT |.414580|.398267|
| 亮点precision@2px |.652495|.631619|
| 亮点recall@2px |.594805|.564750|

可视化comparison_train.png已查看，解析材质没有改善亮点；没有扩大到official test，也没有替换默认。
GT亮点处GGX两叶片alpha均值约.11–.21，并未自动拟合到离线可表达的.005窄峰。
因此“能表达窄高光”不等于场景优化能得到正确高光；本结果不是GGX全部优化方案的上界。
comparison.json、final_audit.json和analysis_source.tar已保存，旧后处理会话83415/47567已退出。

### 新诊断：物理漫反射幅度与光强归一化

前一固定相机control40k中，opacity>.5的84803个点，红通道base.sigmoid>.95占88.60%，中位数.99965；
这是未按投影面积加权的点统计。其16train渲染非局部分量占比约60–90%，不能直接视作物理间接能量真值。
本轮neural/ggx在可见前景内域的红通道base>.95占94.41%/91.23%，表明饱和也发生在渲染属性上。
相同几何/相机下，65.20%的内域像素其GT红通道线性化亮度超过当前归一化光强下单位Lambertian上界
（cos=1、无遮挡）；这个上界只针对直接漫反射，specular/间接光可以超过它，不能据此断言曝光错误。
下一步进一步看低部灰色静态平台代理区域，隔离大面积红色半透明主体的影响，再决定是否检验场景光强尺度。

2026-09-26更正：下述NRHints3.4的Sony A7II/S-log描述针对其新采集流程，不应直接归到复用DNL的Cat/Pixiu；DNL采用gamma2.2近似线性化。本地PNG精确处理链仍未校准。历史引用：
https://arxiv.org/html/2308.13404 。不据此给PNG盲目套用S-log逆变换，也不修改测试图像。


### 像素亮度范围补充与光强尺度试验

`analytic_material_pilot/capacity_diagnostics`在相同16train帧分析212855个底座代理像素：
低部35%前景、局部R<1.5G、排除neutral peaks。其RGB通道超过无遮挡/cos=1的单位Lambertian上界比例
为56.93%/32.87%/10.98%。这仍是启发式区域和当前gamma下的直接漫反射上界，间接能量未排除。
与红色base大面积饱和共同构成检查场景光强尺度的依据，不直接证明材质、照明或相机响应的唯一根因。

下一run `neural_material_light_scale_pilot`：同analytic_material_pilot/neural最终模型初始化，
Pixiu全562train，512px，seed0，fixed/calibrated各追加5k。选择neural依据本轮fit和高光训练诊断更好，
没有对这两个材质候选运行official test。两组仍固定几何/校正相机，无SDF/增密，无材质重置，
继续训练材质、底色、有界法线特征和PORT。仅calibrated开放一个正值全场景light_scale。
全fit与固定16train诊断后比较整体、高光位置/强度、反照率饱和和非局部份额；不将尺度改变本身当成功。
`runs/light_scale_smoke`真实检查通过，默认不变，模型无需新增checkpoint权重键。


### 材质码支持范围检查（CPU，仅训练模型）

`analytic_material_pilot/latent_support`使用同一已冻结prior的encoder编码12288个程序化材质，
前8192作参考，后4096作同分布检查；另从neural最终模型opacity>.5的点均匀抽4096个（seed0）。
神经decoder逐位等于prior。拟合码每维超出参考1–99%分位范围的比例约1.4–7.2%，
其各维1%分位约.24–.39、99%分位约.72–.84，**不支持“六维码普遍卡在0/1边界”的解释**。

联合六维的参考最近邻L2中位数：程序化留出.03886，场景码.12859；场景59.74%超过程序化留出的99%距离.10459。
这说明场景优化后的码与有限程序化编码样本有分布偏移，但不是冻结decoder表达上界、也不证明远码无效。
因此保留作为后续分析证据，不据此新增latent惩罚或开放decoder训练，当前光强尺度对照保持单变量。
统计与程序保存在metrics.json/source.tar；没有使用test RGB。


### 共享光强尺度5k：训练评估完成，已固定模型进入test

`neural_material_light_scale_pilot` fixed/calibrated全部562 fit和16train诊断完成。
全fit PSNR25.210354/25.228696、LPIPS .128963/.129224；改善很小，LPIPS未改善。
16train亮点MAE .159175/.153562，对比度/GT .457533/.462542，precision .650609/.647591、recall .636178/.641929。
标量gain=1.360451（light_scale=.14099163→.10363597）。可见红色base>.95比例91.26%→90.75%，
非局部辐射份额的逐帧平均78.50%→72.19%；仍高度依赖非局部项。
底座上界超出红通道比例56.93%→39.84%，此指标随gain机械变化，不能单独作为物理分解变好的证据。

final_audit确认两组几何/相机/decoder逐位不变、330930点、所有张量有限，fixed光强尺度不变。
已固定最终检查点，决定完整评价两组71 official test及高光；不优化测试相机/曝光或选择中途checkpoint。
训练后处理旧会话53933/55468已结束。test会话10632(fixed)/22371(calibrated)，GPU0/1。


### 共享光强尺度最终test：未设为默认

完整71test固定检查点、原始测试相机，fixed/calibrated PSNR21.271131/21.232289，
SSIM .842118/.841802、LPIPS .152717/.152907。亮点MAE .364083/.360717，
对比度/GT −.008714/−.008638、precision .173746/.172852、recall .176502/.184075。
亮点强度/召回有微小改善，整体指标没有改善，仍未恢复准确细小高光，因此保留为可选实验参数而非默认。
两组的71帧计数、原始相机、固定权重和有限值均已审计；comparison_test.png与完整comparison.json已保存。
旧test会话10632/22371已完成，目前没有训练或GPU诊断在运行。

### 转向直接图像监督的SDF分支

已有多组互监督只改善自一致性，故下一结构实验优先给SDF引入独立RGB/alpha体渲染观测。
当前仅完成`sdf_ray_weights`原语及解析平面深度/梯度、前后表面遮挡、空射线和极端值检查；
全部25项CPU检查通过，没有宣称已经完成NeuS或启动新训练。实现范围见[体渲染模块](../architecture/modules/sdf_volume.md)。


### 独立SDF图像监督：实现和真实检查完成

sdf_volume.py已有位置/法线/入射/出射/光位置条件的辐射头、粗细射线采样、CDF合成和双向渲染几何监督。
28项CPU检查通过。sdf_volume_smoke联合3+3步、sdf_volume_only_smoke固定GS模式和
sdf_volume_sampling_smoke更新采样后的3+3步/重载/CLI评价通过，GS/相机固定模式逐位不变。
失败目录sdf_volume_smoke_failed_plot、sdf_volume_only_smoke_failed_guard保留；不是训练成功证据。

预检查sdf_volume_domain_audit覆盖562 train、128px、1952284个前景像素，零前景射线落在采样区域外；
源SDF立方体边界min=.387124，全部正值。采样选择64粗+64细，避免均匀PDF细点与粗边界重合。

### 已固定的SDF体渲染可行性协议

canonical配置sdf_volume_feasibility：Pixiu全部562train、128px、seed0，两组各3000步、512射线/步。
同calibrated_sdf_pilot/sdf的40k模型和5000更新的SDF初始化；选择它是因为已有场和校正相机，不按test选优。
全部GS参数（含材质与PORT）和保存的相机均固定，阴影/PORT状态继承源模型，主模型本轮不会改变。
两组同一辐射头初始化，头+sharpness均训练；fixed_field固定距离场参数，image_field在500步头预热后更新场。
头Adam .001按原网络日程下降；可训练场lr=.0001，RGB权重1、alpha .1、Eikonal .1、到GS教师depth .05/normal .01。
射线几何项在预热后500步渐增；这里仅GS→SDF方向有优化作用，完整联合模式的反向梯度已另行验证。
预计fixed_field sdf_steps维持5000，image_field最终7500，两者volume_steps均3000。
评价相同16个train视图的SDF分支（fit_sdf），不是独立held-out，也不是主GS改善指标；不安排official test。
协议、代码与配置在run/source.tar/validation.json/manifest.json记录；最多GPU0/1。


### SDF体渲染可行性3k结果：辅助分支可训练，细节仍不足

两组3k及相同16fit视图评价完成。全部GS/主传输/相机逐位不变；fixed_field场逐位不变，
image_field从5000计数到7500，双方head计数3000，所有张量有限。源/新场边界仍全正，min分别.387124/.348366。

| 128px、同16train视图 | Gaussian参考 | 固定距离场 | 图像更新距离场 |
|---|---:|---:|---:|
| PSNR |23.912845|23.033681|22.525679|
| SSIM |.882637|.850849|.848512|
| LPIPS |.113197|.176761|.159646|
| alpha L1 |.026104|.028959|.028304|

图像更新场改善部分感知和掩码指标，但PSNR更差，仍未超过固定的GS参考；可视化明显缺少小结构与纹理。
没有据此启动更强GS反馈或宣称主GS质量改善。结果、图、权重审计保存在comparison.json/png、final_audit.json，
程序保存在analysis_source.tar。全部训练与Gaussian参考评价已结束。

下一可检验假设是空间表示分辨率不足，尚不能排除训练时长或优化问题。
计划从当前模型零残差增加多分辨率空间细节，保持初始函数相同，再做受控对照。
运行时探测发现已安装tinycudann二进制要求GLIBC_2.33，而主机为2.31，不能导入；未修改环境或安装库。
已独立实现纯PyTorch三线性多分辨率网格和零投影残差，30项CPU检查通过，尚未接入SDF/辐射头。
具体后续接口与待验证项见[模块](../architecture/modules/sdf_volume.md)。


### 空间细节接入检查与受控对照

SDF标量残差和辐射头第一层位置残差均已接入，分别由--sdf-detail和--sdf-volume-detail控制。
保留旧网络形状，新增投影为0。31项CPU及sdf_detail_smoke真实完整射线批量3+3步通过，
初始颜色/alpha/深度/法线、重载严格零容差一致，主GS/相机未变，峰值allocated约5.49GiB。

canonical run：sdf_volume_detail_pilot，source为sdf_volume_feasibility/fixed_field最终checkpoint，
依据更好的辅助RGB fit和严格保持的初始距离场选择，不使用test。Pixiu全部562train、128px，各追加3k。
两组都开放场更新、都新增相同的外观网格；只有geometry_grid增加SDF标量网格，appearance_grid仍用原Fourier场。
所有主GS/相机/主传输固定，ray/学习率/损失沿用可行性协议。头已累计3000步、预热不重复；
场从5000到8000、头从3000到6000。输入checkpoint相同，新残差起点为0。
评价同16fit SDF视图，不安排official test。后续须比较mask/RGB/场边界，并做采样分辨率复核，
不能仅因更大外观容量改善颜色就认定几何更好。


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
