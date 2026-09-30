# 连续着色法线细节场

`materials/normal_field.py`的NormalResidualField以归一化世界接收点位置为唯一输入，
8频率Fourier编码、3层128宽ReLU、3维输出，共40067个可学习参数。
不输入视角、灯光或图像；每个场景直接用现有训练图优化，无额外预训练数据。

输出作为Gaussian法线特征features[6:9]的logit残差，相加后仍执行原来的0.5*tanh、
切向投影与单位化。因此总法线角度范围不扩大，原6维材质码、冻结BRDF和PORT传输不变。
末层权重/偏置全零，初始渲染与无细节场逐像素相同；随机初始化不改变其他模块的RNG序列。

训练通过--normal-field启用，仅支持neural_material、无RGB几何预热。
独立Adam，学习率复用transport日程(.001→.00028，本次2k)，增加1e-4*mean(residual²)正则。
RGB梯度同时到达细节场和接收点位置；SDF正则仍约束宏观几何。该场不进入非局部特征输入。

checkpoint顶层normal_field保存模型和坐标归一化buffer，config.normal_field记录开关。
evaluate.load_normal_field严格恢复；render_observation/evaluate_samples/renderer显式传normal_field。
renderer把位置查询结果放入receivers.normal_residual；NeuralMaterialTransport在法线变换中使用。
训练、验证、CLI评价、图像诊断共用此路径。关闭开关时原模型参数与推理不变。

20项CPU检查通过；真实Pixiu零初始化逐像素相同、RGB→field梯度有限非零、decoder冻结、
3+3步训练/继续、细节场学到非零权重、field重载像素相同、CLI评价通过。
固定SDF逐张量不变同时验证，证据runs/normal_field_smoke。
这是对细节表达假设的实验实现，尚未证明高光或泛化改善。
