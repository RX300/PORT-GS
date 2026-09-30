# 固定主GS上的光照条件残差：下一研究方案

状态：2026-09-23，设计、真实短测及正式对照全部完成，准确窄峰仍未恢复。
[结果与下一诊断](../experiments/gs_radiance_residual.md)；不采用为默认。
本页保留结果出现前固定的方案与评价协议。

## 决策依据

SDF辐射头在位置细节网格、原生512px、亮点/邻域采样、4倍ray预算、固定薄层对照后，
仍有明显宽斑和错位，训练整体质量显著低于固定主GS。薄层改善LPIPS却使1–4px小峰的
对比度和召回下降，不能继续把“更锐”当恢复高光。
固定主GS同16fit PSNR26.078、峰recall60.35%，其纹理和细节更好。
下一项优先保留该基线，只学习其图像误差的有符号修正；不是把整幅GS颜色换成弱辅助分支。

## 方法与范围

在renderer得到原transport foreground之后、alpha合成之前加入：

L_new = max(0, L_GS + I/(light_scale*distance_squared) * R(x,n,wi,wo,light_position_normalized))

x=(receiver_world−center)/radius。R的输入同时有实际灯位置、入射方向和出射方向；
不输入frame ID或GT，不添加teacher RGB输入、额外gate或新正则。
这是单点光源条件的有效响应残差，不是固定照明NVS残差，不宣称内在BRDF或能量守恒。
其对同一灯位置的正标量光强缩放保持正齐次；非负截断不保证任意多灯叠加线性。

主GS所有属性、原transport/decoder/PORT、相机、光强尺度完全固定。
原直接光和PORT贡献均保留，原alpha保持；只优化新残差头。
使用现有94维位置/方向/normal/cosine/log-hint特征结构、3×128 ReLU及空间detail网格作为起点，
隐藏层随机初始化，RGB最后Linear的weight和bias全零，输出有符号logits而非Softplus。
不要把整个网络置零；不迁移已适配SDF体积分的隐层，这属于未来独立初始化消融。
新残差初始为零，必须验证原GS线性/观察图和alpha保持；一开始末层先学习属正常过程。

计划两组只改变残差头使用的冻结法线：
- geometry：原receivers['normals']，世界坐标GS几何混合法线。
- material：原NeuralMaterialTransport.material_normal(receivers)，包含既有有界着色法线修正。
原GS分支始终保持它本来的着色法线，只有新头输入不同；两个新头权重初始化/预算相同。
选择两组是为隔离旧BRDF拟合法线作为先验的影响，不再给法线增加自由度。

## 最小接入约束

当前renderer的receivers是局部dict，诊断通过hook抓取；正式实现应增加显式可选接口，
复用已有反投影/alpha除法，不复制另一套receiver计算，不把hook用作训练接口。
首先限定当前neural_material checkpoint和只训练残差模式。
新模块作为独立state/config保存；原GS/transport参数键保持，缺少已启用残差权重时不能静默忽略。
评价根据保存配置明确启用，报告branch名称；未启用时所有7个原方法及默认行为保持。
同一学习头的训练/评价必须走同一着色函数。训练仅在抽样像素查询残差，
评价在全部GS覆盖像素分块查询；重复抽样像素的损失权重必须保留，不用GT补足未覆盖区域。
世界坐标接收点、原foreground和原光强尺度停止梯度，仍沿用现有PNG前景gamma/alpha观察模型。
记录截断通道比例、残差幅度及原foreground幅度，以检测大范围抵消或破坏正确纹理。

## 验证与固定实验

复用现有train/evaluate/launch_validation.sh与canonical configs/validation.json，新run目录。
共同源优先当前geometry_grid checkpoint（主GS已逐位证明等于40k参考），
两组native512、2048像素/步、3k/seed0、相同LR日程；25%峰+25%11px非峰邻环+原混合。
不修改默认方法；冻结原GS是零残差控制，无须重复训练控制。
虽然像素数/步相同，单点头远少于体分支查询次数，不宣称匹配FLOPs或训练成本。

最小核心检查：零输出原图/alpha保持；零光与正标量光强倍增；梯度只进残差；
所有主状态/相机逐位不变；真实2048像素3+3训练、续训、严格重载与CLI评价。
先做同16fit与4帧分桶/邻环图像诊断，并固定终端checkpoint，记录两组均有的失败/退化。
该主渲染候选将评价完整71帧official-test、512px、原始test相机，与原GS同协议比较，
不做测试曝光/相机拟合、不据测试挑中间checkpoint。已有来源审计确认这些帧未进入模型拟合；
测试集已在以往研究中观察过，因此称开发期未训练帧评价，不称独立盲测。
若测试显示泛化退化，完整记录；训练改进不能替代目标中的重光照质量改善。

## 风险与未选择的替代方案

残差可能记忆训练灯位或补偿错误几何；光照条件输入本身不证明泛化。
固定alpha无法补回GS完全未覆盖的几何，截断会影响暗区梯度。
完整GS表面辐射头也可行，但同时改变颜色表示并放弃原主模型的纹理优势；
GS表面与SDF体分支比较还同时改变位置/法线/alpha，不能全部归因于去掉体混合。
仅把SDF体分支改为其平均表面点查询可作算子诊断，但本次优先直接改善更强的主GS结果。
