# SDF 与 2DGS 双向几何监督

2026-09-23，训练辅助模块 `sdf.py`，通过 `train.py --sdf` 启用，适用于可训练的2DGS几何。
不新增光传输方法或依赖。默认辅助模式不改变推理；可选直接着色模式改变基础法线来源，
需要推理加载field，详见末节。两种模式均保留共享材质decoder。

## 连续表面

固定使用Gaussian center/radius归一化世界坐标与距离。3层64宽Softplus MLP，
输入坐标及4频率位置编码，以半径0.6的球面距离加可学习残差初始化。
这是本项目独立实现的连续几何正则，不是完整GSDF/NeuS双渲染器复现。
参考[GSDF作者项目](https://city-super.github.io/GSDF/)的双向几何约束思想。

## 两个梯度方向

每步从当前训练帧的alpha>0.8、GT前景且3×3邻域有效的位置采样1024点。
用相机Z深度和K/viewmat恢复世界坐标；圆盘合成法线朝向相机后作为外法线。

- 2DGS→SDF：detach坐标和法线，拟合f(x)=0；在法线两侧0.005–0.02场景半径
  的窄带拟合正/负距离；沿视线向相机前移0.05–0.25半径约束为空间外侧。
  法线方向损失权重0.05，窄带与空间均匀采样的Eikonal损失权重0.1，前方自由空间权重0.1。
- SDF→2DGS：通过functional_call使用detach后的field参数，保持坐标梯度。
  零面距离损失权重0.05，场梯度法线与圆盘法线的方向损失权重0.01。
  场梯度法线作为固定目标，不对field求二阶几何梯度。
  深度/法线渲染反传至Gaussian位置、旋转以及影响合成的尺度/opacity。

前500个有效SDF更新只拟合场，随后500步线性增加反向几何约束。
SDF使用Adam lr=0.001。图像、mask、已有StableNormal/DA3与深度法线一致性继续约束几何。
监督来自多视图预测及已有图像先验，没有新几何真值；相互一致不等于几何准确。
遮挡边界已排除，但透明表面、混合深度、薄结构仍可能违反局部有符号表面假设。
此版本没有SDF体渲染、独立RGB分支、SDF增删Gaussian或网格监督。

## 接口与检查点

`--sdf --sdf-warmup-steps 500 --sdf-samples 1024 --sdf-weight .05 --sdf-normal-weight .01`。
要求geometry=2dgs且不冻结几何。没有高可信前景像素的帧不更新SDF，也不计入预热。
checkpoint保存sdf state与sdf_steps，--init-checkpoint继续启用--sdf会恢复两者；
与现有训练初始化规则相同，optimizer重新创建。默认辅助模式渲染/评价只需要2DGS与transport；直接着色模式还需field。

测试覆盖双向梯度隔离、独立球面带符号拟合、真实Pixiu的means/quats梯度、
CLI短训、field保存与重载、继续训练计数。真实测试产物：runs/sdf_geometry_smoke。

诊断入口diagnose_image_errors.py新增geometry-frames，可输出法线/深度图；所有2DGS记录
深度法线夹角与distortion。包含SDF的checkpoint还记录零面距离、梯度法线一致性及Eikonal误差。

## SDF法线直接参与PBR（2026-09-23）

新增`--sdf-shading`，需neural_material、--sdf与已完成预热的SDF checkpoint，禁用RGB几何预热。
已有辅助分支仍不改变推理。直接着色模式在Gaussian期望深度位置查询grad f并归一化，
朝向相机作为材质的基础法线，再沿用已有有界着色法线修正和冻结BRDF decoder。
训练时create_graph=True，RGB通过法线的导数直接更新SDF，也能更新接收点位置；
辅助GS↔SDF监督仍读取原Gaussian法线，避免field用自己作伪标签。
推理在no_grad外层内局部enable_grad计算同一个一阶场梯度，不保留训练图。

检查点仍保存sdf/sdf_steps，config新增sdf_shading；evaluate.load_surface_field负责恢复，
训练、验证、CLI评价、图像诊断均显式将同一field传入render_observation/renderer。
新模式的推理必须有field，不能拿只读Gaussians/transport的旧归档代码复现。
该方案不是仅在训练增加loss，而是改变基础着色法线的来源，需要独立对照验证。

18项CPU检查通过；真实Pixiu的RGB→field二阶链式梯度有限且非零、decoder无梯度、
CLI短训/续训(2000→2003→2006)、重载逐像素一致、SDF法线确实改变图像、CLI评价通过。
证据runs/sdf_shading_smoke；这些验证不代表高光质量提升。

## 单Gaussian零面软约束

`--sdf-primitive-weight .05`要求--sdf。投影Gaussian中心到当前相机，在GT前景、alpha>0.8的
3×3内域中，选opacity>0.5且中心相机Z与双线性采样渲染深度差<0.05场景半径的点。
这是近似可见性筛选，不是精确光线追踪；阈值内薄层仍可能同时被选中。
每步用独立随机生成器抽取sdf-samples个中心，优化平均|f((mu-center)/radius)|。
field参数在这项损失中detach，只有means收到梯度；筛选不对opacity/深度图反传。
沿用SDF反向监督的预热/渐增；history记录sdf_primitive及候选点数。保留点数与原GS法线着色。
解析平面检查遮挡/背景/低opacity排除及梯度隔离，真实Pixiu3+3步训练/恢复通过。
证据runs/sdf_primitive_smoke。启用该项不会自动采用SDF直接着色。

诊断对当前近似可见高opacity中心记录平均SDF距离和超过0.02半径的比例，
与全场景中心分布及渲染指标一起评估该约束的实际作用。

## 固定field的分阶段几何精修

`--freeze-sdf`要求--sdf及已完成预热的field checkpoint。冻结field参数并停止field optimizer、
GS→SDF拟合/方向/Eikonal/空域损失；Gaussian继续接受原RGB及SDF→GS损失和可选primitive损失。
field更新计数sdf_steps保持不变。用已完成joint primitive组作对照，检验目标零面移动的影响。
真实3+3步短训/重载逐张量验证field完全不变，并确认primitive loss实际激活。

### 从头训练的启动时机

`--sdf-start`（默认1）指定首次拟合field、采样GS表面并启用互监督流程的训练迭代。
此前SDF辅助分支不采样、不更新、不增加sdf_steps；现有有效更新计数的warmup/ramp从第一次成功采样开始。
拟合过的SDF直接着色模式仍允许RGB更新field；该开关只延迟几何互监督，不能用来冻结着色field。
从头对照设为3000，让GS先形成初步表面；默认值保持已完成续训协议。

### 独立轮廓可行性诊断（未接入训练）

silhouette_rays在归一化[-1,1]^3内构造固定相机射线；silhouette_logits用沿射线的最小SDF值给出soft occupancy。
它只提供轮廓约束，不是RGB或完整NeuS渲染器。相机/射线固定，梯度只进入field。
`diagnose_image_errors.py <SDF checkpoint> --output <fresh path> --sdf-mask-fit-steps 1000 --sdf-mask-weight 1`
运行CPU拟合；weight=0是Eikonal-only对照。保存field.pt独立产物，不能将其当作完整场景last.pt使用。
球体解析轮廓与扩张梯度检查通过。Pixiu初始轮廓只获很小改善，尚不作为主训练开关。


独立图像监督的体渲染分支已接入主训练、保存与独立评价，并完成可行性对照；
尚未改善主GS高光。接口、结果与空间细节扩展的开发状态见[sdf_volume](sdf_volume.md)。
