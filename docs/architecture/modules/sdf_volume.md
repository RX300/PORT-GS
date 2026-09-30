# 直接图像监督的SDF体渲染分支

## 实现与边界

`sdf_volume.py`在既有SurfaceSDF上实现光照条件体渲染，并接入train.py/evaluate.py。
这是借鉴GSDF双渲染分支和NeuS式CDF的实验，不是完整论文复现：
https://city-super.github.io/GSDF/ 、https://github.com/Totoro97/NeuS/blob/main/models/renderer.py 。
SDF仍是4频率、3×64的原场；没有同时更换hash编码、安装nerfacc或改变共享环境。

`sdf.sdf_ray_weights`查询有序区间边界的SDF，只有logistic CDF下降产生opacity。
log域生存概率和透射率避免直接相除接近零的CDF。与原NeuS的中点梯度/cos退火不同，
本实现查询实际区间端点；这一差异明确保留，不把它称为逐行复现。

## 射线和外观

相机射线经过以scene radius归一化的[-1,1]^3区域。每条射线先分64段，再按CDF权重追加64个位置；
PDF含总质量.01的均匀分量。粗细数量相同，使空PDF的新位置落在粗区间中点，避免半数细采样重复粗边界。
采样位置和相机射线不反传，最终端点SDF、合成权重和场法线均可接受图像梯度。
积分距离乘radius、再乘射线camera-Z方向余弦，得到与GS一致的camera-Z深度。

SDFRadiance为94输入、3×128 ReLU、RGB Softplus的轻量场景内网络，连同log-sharpness共45572个参数。
输入包含位置/入射方向/出射方向的4频编码、场法线、四个角度余弦、三个log高光提示及归一化光位置。
高光提示停止梯度，法线和其余几何路径保留梯度。输出乘已知点光的强度及距离平方衰减，保持对光强线性。
它没有预训练，也不是内在BRDF估计器；光照条件网络近似有效辐射响应，没有独立SDF阴影射线。
不能把固定光照的新视角网络作为移动光源模型，或把辅助头的RGB拟合当作主GS高光改善。

训练射线一半来自GT前景，一半来自区域有效射线，允许重复抽样；没有将该平衡采样重加权为全图均匀损失。
RGB遵循现有观察模型：PNG先对前景应用gamma再alpha合成，HDR为整幅图gamma。
alpha直接监督来自原始GT。默认主渲染仍是GS，只有evaluate.py --sdf-volume选择独立体渲染。

## 训练与梯度契约

--sdf-volume-weight默认为0；大于0需--sdf，并不与SDF直接着色、单点primitive约束或RGB几何预热混用。
--sdf-volume-rays默认512，--sdf-volume-samples默认64，--sdf-volume-warmup默认500。
辐射头Adam从.001按主网络相同日程下降；--sdf-lr默认.001，可显式指定微调已拟合场的学习率。

普通联合模式：头预热期间RGB看到停止梯度的SDF几何，场继续原GS点拟合；预热后改用独立RGB/alpha、
Eikonal和逐射线的双向depth/normal一致性，不再同时叠加旧点拟合/单点反馈损失。
有效互监督要求GT内域、GS alpha>.8及SDF alpha>.8。SDF→GS停止场目标，GS→SDF停止Gaussian目标；
权重随预热后的同长度窗口渐增。RGB、mask分别乘volume_weight和.1*volume_weight，Eikonal权重.1。
辅助RGB不会更新主光强尺度或相机。普通GS图像损失仍按原流程更新GS和材质。

--sdf-volume-only用于可行性验证：全部Gaussian、材质、主传输、相机、光强保持固定，
自动恢复已有训练相机校正及源checkpoint的阴影/PORT激活状态，不投影、增密或裁剪Gaussian。
只记录并优化SDF分支损失；GS→SDF保留，SDF→GS项删除。头预热期间场也固定，之后才允许场接受图像监督。
--freeze-sdf可以继续固定距离场参数；两组的辐射头和CDF sharpness仍都会学习。
这一区分很重要：固定距离场不等于固定全部体渲染权重。

## 检查点与评价

保存sdf_volume与sdf_volume_steps，另保存已有sdf/sdf_steps。分别统计辐射头和场的实际更新，
续训不重复已完成的头预热。场、头、sharpness都随checkpoint恢复，不需要外部网络文件。
evaluate.py --sdf-volume使用相同保存参数；fit恢复训练相机校正，test使用原始测试相机。
默认evaluate.py仍渲染GS。批量配置eval_branch=sdf_volume、eval_limit=N，输出fit_sdf等独立目录。

## 已验证与当前实验

28项CPU检查通过，覆盖解析平面深度/导数、背面/双层遮挡、空射线/极端值、薄表面补采样、
光强线性、RGB→field梯度、预热隔离、双向监督目标隔离及批量评价参数。
真实512px联合3+3步通过：runs/sdf_volume_smoke；固定GS模式通过sdf_volume_only_smoke。
更新采样后再做真实3+3步与CLI评价：runs/sdf_volume_sampling_smoke，GS/相机逐位不变、场与头有限、渲染逐位重载。
首轮失败为loss绘图假定项集合恒定，已改为绘制全部阶段项、未启用项贡献记0；另一次固定模式触发旧双向保护，
已仅对明确volume-only模式放行。失败日志保留，不混同成功检查。

sdf_volume_domain_audit检查128px全部562训练视图：1952284个前景像素全部有区域有效射线，
所有相机在区域外，源场边界采样min SDF=.387124，无负值。尚须检查学习后的边界和几何质量。

当前小规模试验见docs/experiments/sdf_geometry.md：同一已拟合场，两组都固定GS；
比较固定距离场与图像监督更新场。这里只验证SDF分支是否稳定学到图像和表面，不能据此声称GS已经改善。


## 空间细节残差设计（下节已实现并完成对照）

3k可行性结果已完成，更新场改善LPIPS/alpha但损失PSNR，两个SDF分支均明显模糊，未超过主GS参考。
设计时只把“空间分辨率不足”作为待检验假设，不宣称已经定位唯一原因。
借鉴多分辨率网格编码：https://nvlabs.github.io/instant-ngp/ 。

已新增materials/spatial_detail.py：12层、每层2通道、格点分辨率16至256；小层用密集索引，
大层使用上限2^17的哈希表，纯PyTorch三线性插值。零初始化输出投影确保新增残差的初始输出为0。
SDF残差可乘在边界及外部值/一阶导数均为0的窗函数，保护已有场的区域边界；辐射头不需要该窗。
30项CPU检查通过，含仿射插值、二阶导数、法线损失到网格参数的梯度及边界条件。

没有使用tcnn后端：实际导入失败，已安装二进制依赖主机不存在的GLIBC_2.33（主机2.31）。
纯PyTorch实现是明确选择，没有自动后端切换，也未修改共享环境。探测记录在sdf_volume_feasibility/hash_runtime_probe.json。

设计接口（现已实现）：将零残差接入SurfaceSDF和辐射头第一层，保存明确配置和新增权重；
从旧checkpoint显式保留原网络、只初始化新残差，验证初始渲染保持一致、续训/重载和显存。
建议区分外观细节与SDF细节开关，以相同外观容量比较几何空间分辨率的作用；仍先固定主GS验证，
不能把辅助头的局部颜色拟合误认作真实几何改善。

GPU512点、完整默认网格的零输出及法线损失到网格参数高阶梯度也已验证，
记录pytorch_grid_probe.json；这不证明完整射线训练的显存/速度，集成后仍需真实小试。


## 空间细节接入与保存（2026-09-23）

--sdf-detail向SurfaceSDF增加带边界窗的标量残差；--sdf-volume-detail向辐射头第一个Linear输出增加128维位置残差，
位于ReLU之前。两个开关独立，原网络键、形状和计算顺序保留；不开启时行为不变。
从无细节checkpoint显式扩展时，保留所有旧权重，仅初始化新网格和零投影。已有细节模型必须按保存配置严格加载，
训练若漏掉对应开关会报错，避免丢弃学到的细节。评价自动恢复两个开关。

31项CPU检查通过，新增完整场值/法线/辐射响应的零容差初始保持检查。
sdf_detail_smoke真实128px、512射线×(64粗+64细)的3+3步已通过：扩展前后RGBA/深度/法线零容差一致，
续训与重载、主GS/相机固定、场与头有限值、两分支CLI评价通过；峰值allocated memory约5.49 GiB。
这验证数值/状态契约，不证明质量改善。

两组3k空间细节对照已结束，详细指标与采样限制见[实验记录](../../experiments/sdf_geometry.md)。
几何网格改善辅助train指标，但主GS不变，小高光仍未准确恢复。


## 训练亮点射线配额（2026-09-23）

`--sdf-volume-peak-fraction`默认0，保持原半前景/半domain-valid的采样顺序和随机数路径。
设为.25时，在512总射线预算内先抽128条训练GT neutral-peak射线，再抽192条前景和192条domain-valid射线；
均有放回。亮点定义复用evaluate.neutral_peak_mask，GT掩码只用于训练抽样，不进入推理。
若当前帧无有效亮点，全部预算仍使用原混合。这个策略显式重加权RGB、mask以及射线几何loss，
不是原目标的无偏重要性采样估计。history的sdf_volume_peak_rays_total包含重复抽样，
计数仅累计本次训练调用，不是唯一像素数，也不是checkpoint全训练生涯计数。

`evaluate.py --highlights`现在可对GS或SDF分支使用同一量化观察域亮点指标；
`--resolution 512`支持在统一分辨率评价，报告另保留training_resolution。
精度/召回使用2px邻域，MAE/局部对比按GT亮点像素池汇总；这些是图像代理，不是真实镜面分量。
canonical配置用eval_resolution与eval_highlights接入批量评价。
本轮固定距离场的受控实验见[亮点采样](../../experiments/sdf_peak_sampling.md)。


### 峰周围监督与提示编码的独立候选

`--sdf-volume-peak-context-fraction`默认0。候选.25与peak-fraction=.25组合时，512总射线分为
128 peak、128 context、128前景、128domain。context是11×11 GT峰膨胀区域去除峰且alpha>.9，
实际候选池再与domain相交；空池的未用预算回到原前景/domain分配。这里的11px指kernel宽，半径5px。
该策略保留原峰配额，增强对附近非峰像素的监督，是显式目标重权。

`--sdf-volume-hint-encoding`默认关闭。现有停止梯度的3维log提示按[1,2,4,8]做sin/cos（不乘π），
24维经无bias Linear到128维，在原首层Linear及位置残差之后、ReLU之前相加。新增3072参数，
权重初始全零，保持完整初始渲染。旧网络键/形状不变；训练必须显式开启以扩展旧模型，
若已有编码checkpoint缺少开关则报错，evaluate按保存配置严格恢复。
这是本项目逐采样点log提示的编码实验，不是NRHints使用每射线原始GGX提示的复现。

两者独立与已经完成的25% peak控制比较，协议见[高光形状](../../experiments/sdf_peak_shape.md)。
默认主GS渲染和默认方法均不改动；GT采样掩码不进入推理。


## 可选固定CDF sharpness

`--sdf-volume-fixed-sharpness <正有限数>`在加载辐射头权重后设置其现有log_sharpness参数为指定值的log，
并在创建Adam之前冻结该参数。默认省略时保持原可学习行为；状态键和评价公式不变。
checkpoint保存实际log值及本次配置，evaluate直接使用保存值，无新的渲染分支。
`--init-checkpoint`仍只加载权重并重建optimizer；冻结由本次CLI指定，不隐式继承旧冻结设置。

这改变CDF积分层厚度，而不改变SDF零面。固定场、相机和sharpness后，alpha/depth/normal相关项
可成为没有可训练参数路径的常量；仍记录原loss，辐射头由RGB监督学习。
更改β会故意改变初始渲染，只有新残差的零初始化比较可在同β下声称保持，不能声称等于原source。
完整厚度协议及采样限制见[厚度对照](../../experiments/sdf_volume_thickness.md)。

既有test_method_integration.py新增--sdf-volume-samples和--sdf-volume-fixed-sharpness，
可覆盖2048ray×(128粗+128细)真实批量，检查β恒定、头更新及保存/续训/重载。
