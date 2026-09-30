# 基础表示的同GT高光审计

2026-09-23。本轮已完成全部137帧、9个模型/场景流的619张预测共同GT审计及固定图块人工检查。
GS³在Pixiu改善整体分数，但没有可靠恢复细小高光；Cat也未通过预定条件。不直接复制或推广现成ASG。
用户提出从底层换方案后，冻结GS自由RGB残差微调主线已停止；下一项
[逐ray表面交点反射原型](intersection_reflectance.md)正在实现，尚无新质量训练结果，完整研究目标仍active。

## 数据、模型与来源

Pixiu全部71个official test、Cat全部66个official test，native512、黑背景、原始测试标定。
这些图像早已在开发中观察过，不能称新盲测；相机和灯同时变化，不是固定视角纯换光。

| 模型 | 来源与边界 |
|---|---|
| GS³100k | GS3/runs/full_benchmark_20260913；实际checkpoint/PLY/全量PNG已完成，旧文档及pending准备清单失效。只用train优化模型及训练相机/灯，测试无opt_pose |
| SSD100k | SSD-GS既有checkpoint及real_fixed_calibration_20260913原始标定导出；训练曾用test RGB拟合test相机/灯。test分支不更新Gaussian optimizer，但完整流程有测试曝光；仅污染参考，排除主决策 |
| PORT default30k | directional_port512_validation_20260915，完整train，原始test标定 |
| PORT neural30k | neural_material_real_validation_20260922，完整train，从新2DGS几何及冻结材质先验学习 |
| PORT冻结来源，仅Pixiu | sdf_volume_detail_pilot/geometry_grid，残差各实验共同来源；全部562参与完整源链训练，71无权重/标定训练但开发已观察 |

Pixiu主控制为port_source，Cat为port_neural；另报告port_default。预算、几何、训练监督和实现不同，
因此是基础方案的观察性比较，不能隔离某一个ASG、法线或光传输模块的因果效果。
SSD历史opt-pose 31.17dB不进入表格；重置测试标定不能消除其训练期测试曝光。
SSD原训练完整源码归档缺失，现有证据为未修改训练代码、cfg、保存标定和归档scene/evaluator合同。
新增来源核查见`runs/foundation_comparison_audit/source_provenance.json`：GS³两场景保存的`input.ply`位置逐位匹配
`RandomState(0)`生成的100000个随机点，后续随机颜色匹配，法线全零；不是缓存点云、SfM或单目先验初始化。
保存的训练相机名字、顺序和原点对应全部Cat522/Pixiu562训练帧，优化相机数量一致，没有test相机优化文件。
GS³本次实现的模型和相机/灯优化均只用train；这不证明公开数据标定最初如何取得，也不把开发已观察test改称盲测。
SSD的测试曝光限定为test相机/灯标定分支，不能误写为test直接更新Gaussian optimizer；它仍在所有主决策副本中为`primary_eligible:false`。

## 忠实导出，而非静默改变旧模型

SSD/GS³已有全量独立GT/预测PNG。PORT旧评价每个目录只有前4张pair，因此需要补齐已评价基础模型的图片，
不训练、不修改checkpoint，也不为本轮失败的paired/RGB头追加test。
旧neural_material的features6:9原是传输特征，当前代码将其解释为着色法线偏移且已改变specular门控。
因此旧PORTdefault和neural必须各用自己的source.tar，在独立/tmp目录恢复；唯一改动是
`if output and idx < 4:`→`if output:`，保存全部pair。环境仍为ssd-gs，不安装或升级。
当前冻结source没有残差，其现用着色公式与旧reference一致；evaluate.py新增默认关闭的--save-all。
五个PORT导出流均已与各自旧reference全部metrics/views及前4PNG字节精确一致：Pixiu三组各71帧、Cat两组各66帧。
旧neural的features6:9原语义由各自归档源码保留，未按当前着色法线偏移含义重新解释。
导出来源/单行差异记录在runs/foundation_comparison_audit/export_sources.json，失败日志保留，不覆盖旧结果。

## 统一评价与预定选择

canonical配置继续使用configs/validation.json，本次mode为foundation_comparison；入口为：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_image_errors.py \
  --foundation-comparison configs/validation.json \
  --output runs/foundation_comparison_audit/analysis --foundation-device cuda:0 --lpips
```

此配置用于诊断入口，不调用训练launcher；上一完成训练配置保存在paired_loss_pilot/validation.json。
图像、掩码、PSNR/SSIM和组件均在CPU比较；已检查空闲GPU仅用于VGG LPIPS，最多2GPU约束保持。
GT由同一SceneDataset/target_image生成并量化为uint8；所有预测只读取PNG，不作曝光、颜色或像素对齐。
核对原metadata帧序/名字、全部71/66图片数量、512尺寸和GT-left布局。SSD缺逐帧metric映射，只依赖归档loader的
索引顺序并明确限制，不能进入干净决策。保存GT的逐像素字节差也完整报告；>1byte差触发对齐复核，不能静默消除。

复用现有neutral_peak_mask、组件分桶、raw合池、2px precision/recall及11px排峰GTalpha>.9环域。
记录全峰false-positive像素、1–4/5–16/>16px MAE/对比/召回、全部图像质量、全帧环误差。
固定图块帧在查看预测前确定：Pixiu[0,17,35,53]，Cat[0,16,33,49]；每帧取按8连通label顺序前两个
1–4px GT组件，64×64原生crop。若少于2个，记录缺额，不按预测效果替换。所有方法相同crop。

基础候选相对各场景主控制的实用筛选：小峰对比距1误差下降≥.03、召回+≥.03、MAE下降；
全峰precision下降≤.01、固定四帧11px环过亮增加≤5%、PSNR下降≤.1dB、LPIPS增加≤.002、SSIM下降≤.002，
再目视检查定位与窄峰。这是预定研究筛选，不是显著性检验，不自动采用模型，也不更改已完成paired门槛。

如果干净GS³确实恢复PORT缺失的细小峰，下一实现主线转向联合学习几何和显式表面反射的基础；
若仅整图分数更好但同样丢峰，则不能把ASG复制当作答案，应重新检验表面反射求值与空间合成机制。
Cat用于材料/场景边界检查，Pixiu收益不能直接推广成所有场景成功。完整目标保持active。

## 已完成共同GT结果

下表来自`analysis/summary.json`的统一CPU输入协议；SSD只列作有测试标定曝光的背景参考。
小桶为1–4px GT组件；recall/precision使用2px容差，不能仅据命中推断准确窄峰形状。

| 场景/模型 | PSNR | SSIM | LPIPS | 小桶RGB MAE | 小桶对比/GT | 小桶recall | 全峰precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pixiu GS³ |23.723631|.864052|.117342|.265749|.000257|.150075|.385122|
| Pixiu SSD参考 |23.403902|.862117|.119268|.269461|−.003937|.126597|.342482|
| Pixiu PORT default |20.578249|.845557|.156690|.353612|−.004904|.008640|.227990|
| Pixiu PORT neural |21.478632|.851218|.161733|.365245|−.003446|.015778|.308672|
| Pixiu PORT source（控制） |21.331655|.843041|.152835|.318331|−.006346|.126221|.178755|
| Cat GS³ |19.123925|.716881|.231927|.155471|.010593|.036522|.026099|
| Cat SSD参考 |18.003657|.703663|.248188|.196733|.010436|.033043|.026827|
| Cat PORT default |21.536184|.766470|.230405|.185985|.031460|.000000|.000000|
| Cat PORT neural（控制） |22.479174|.780935|.255563|.174941|.055782|.013913|.144068|

Pixiu共有2880个小桶组件/5324px；GS³相对source的小桶对比距1改善约.006603、recall增加2.3854pp，
均低于.03/3pp要求，故仅6/8数值条件通过。其整体PSNR/SSIM/LPIPS确有改善，全峰假阳性像素25521→11307，
但小峰对比仍接近零。Cat仅368个小桶组件/575px；GS³只有MAE和LPIPS两项通过，2/8条件通过，PSNR/SSIM下降。
这也说明Cat的少量亮点代理包含毛发和纹理，不能视为物理高光真值。

固定图块按GT选取，Pixiu8个、Cat6个；Cat帧33/49各只有1个合格组件，保留缺额，没有按预测补选。
root人工检查为completed/false：缺失、错位、宽化的亮点和细节仍在，未确认可靠窄峰恢复。
全部场景gate与summary已同步；所有候选均未推广。失败的paired/RGB残差头没有新增test。

## 完整性核查与数值边界

三个实现检查通过；独立CPU核查覆盖全部619预测的数量、尺寸、映射、canonical GT字节、固定crop标签/边界，
以及逐帧GT桶计数、raw组件合池、全峰precision/recall和全帧/固定四帧环域池化，结构与池化检查通过。
Pixiu source的全部71组件行及合池与新export、旧`gs_residual_budget_reference/test`精确一致。
GS³/SSD保存GT与canonical差异最大1byte，二者差异记录相同；所有PORT保存GT逐字节相同。
原始4份分析源码在核查时与`analysis/source.tar`逐字节匹配，manifest与planned/canonical相同。

严格跨后端标量核查没有全通过：以既定rtol1e−6/atol1e−8比较source的71帧，13帧SSIM、57帧LPIPS超差，
最大差分别9.059906e−6与1.683831e−6；PSNR最大差3.814697e−6仍在容差内，三项整组均值也在容差内。
初失败日志/源码保留为`analysis_verification_initial_failure.log`和`analysis_verification_initial_failure_source.tar`，
最终报告明确`structural_integrity_checks_passed:true`、`scalar_reference_tolerance_checks_passed:false`，没有扩大容差。

单帧18的`numerics_probe.json`使用同一保存PNG，无重渲染：CPU/GPU的byte/255转换最大相差5.960464e−8，
126个byte值的转换不同。SSIM CPU float32为.7855463027954102、由原bytes计算的CPU float64为.7855349302711744，
GPU自身除255为.7855372428894043，恰等于旧export；GPU使用CPU除255输入则为.7855347990989685。
同一GPU LPIPS的GPU除法/CPU除法输入分别得到.20233534276485443/.20233550667762756，精确复现旧export/共同审计值。
此探针说明输入转换和SSIM浮点后端会改变末位，不宣称逐帧全部原因已逐一定位。
本轮所有模型的共同CPU byte/255、CPU float32 SSIM及统一VGG输入协议保持；差异量级未接近预定门槛的关键结论边界。
不重写旧export指标、不把严格标量失败隐藏为“全部检查通过”。

证据在`runs/foundation_comparison_audit/`：`analysis/summary.json`、各场景`metrics.json`/`crops.png`、
`export_verification.json`、`source_provenance.json`、`analysis_verification.json`、`numerics_probe.json`、
`manual_review.json`、`completion.json`与`docs.tar`。原source、analysis及首失败证据保留。

## 下一基础原型已选，质量结果尚待验证

下一项是[逐ray表面交点反射对照](intersection_reflectance.md)：在ray–surfel交点计算非负GGX着色后合成，
对比先聚合属性再求BRDF；从新初始化联合学习几何/材质，不使用PORT、冻结decoder或自由RGB残差。
截至本审计结案，`SurfaceReflectanceTransport`文件和6项CPU检查已完成但尚未注册，渲染后端正在实现和有限GPU0检查中。
计划新建506fit/56validation划分，不复用已见完整562帧的source或先验；尚未启动正式质量训练。
本轮审计完成不等于完整研究目标完成，也不证明新方案必然更好；不据此宣称当前所有GPU空闲。
canonical仍是已完成的`foundation_comparison`诊断manifest，不要把它交给训练launcher或同名重跑。

## 历史准备状态（现已完成）

已实现foundation_comparison.py并扩展现有diagnose_image_errors.py；三个针对布局/映射/环池化的CPU检查通过。
当时全量PORT预测正在忠实导出，正式统一高光审计尚未启动，尚无共同GT质量结论；现以上文完成状态为准。
