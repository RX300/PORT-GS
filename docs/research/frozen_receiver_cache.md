# 冻结接收点缓存边界核查

2026-09-23，只读当前代码；未实现、未运行，不改正在进行的成对RGB实验。
这是潜在等价加速，不是高光质量改进或已测得的速度收益。

## 可复用边界

固定源的transport生成未合成foreground RGB之后、renderer.apply_radiance_residual之前。
每帧保存排序后的covered flat IDs、原base RGB、实际world XYZ、残差所选冻结normal、GS alpha，
以及已校正camera eye、原世界light position/intensity、图像shape和完整covered count。
source center/radius及transport.light_scale保持。material模式可保存最终material_normal，
无需保留其余receiver features；base必须始终来自原shader，不能随残差normal选择而改变。

训练采样只需要covered∩rays.valid，但全covered缓存更便于统一完整查询。源原GS、transport、
camera、光强尺度、shadow/PORT激活在残差stage均固定；必须先按完整fit映射校正camera再构建。
当前只有decoded GT/mask和局部pair候选缓存，没有receiver/base cache。

## 必须保持的计算合同

- head查询排序后的unique covered draw IDs，保留当前4096分块；RGB和pair损失仍按原draw重复计权。
- 原base+delta先clamp_min(0)，再线性alpha/background合成，然后调用原observation_image。
  PNG路径还会用alpha.clamp_min(1e-8)恢复foreground、gamma、再合成；不可代数简化，
  否则极低alpha/非零背景及浮点消去行为可能改变。HDR对已合成线性图做gamma。
- 训练观察RGB不量化、不clamp到[0,1]；无覆盖draw保留背景，无query保留连接head的零梯度。
- 配对支持仍是两端GS alpha>0，分母仍3×nominal pairs，不按支持数量重新归一化。
- requested计数含重复；queried计数及激活统计针对唯一覆盖点，均值按query数加权，min取min。
- 不缓存会随训练变化的grid features、interaction、center或SG输出。采样候选顺序、帧顺序及独立CUDA RNG消耗保持。

现有renderer helper依赖完整图像及receiver字典。要获得实际加速，需抽取一个共享的查询/合成接口，
不能为缓存分支复制另一套公式，也不能将紧凑tensor直接假装成现有完整输入。
cache是运行期派生数据，不替代checkpoint/source；当前checkpoint/head合同应保持。

## 内存数量级

按此前完整562帧共38,945,262个covered pixels：FP32 baseRGB+XYZ+normal+alpha为10 floats，
约1.451GiB；另加int64 covered IDs后约1.741GiB。仅baseRGB约.435GiB。
未计模型/optimizer/候选池；现有562×512²的RGB+GTalpha本身约2.195GiB。
若缓存固定94维网络输入另需约13.64GiB，且只查2048draw时重算便宜，无优先理由。
GPU或host驻留尚未选择，没有速度测量，也不在本轮实验中途添加。

若未来实施，先在同一源/样本/初值下检查实际predictions、RGB/pair loss、head gradients、
alpha/覆盖/统计和短训练轨迹，再考虑增加预算；不能把实现差异与预算变化合成一个因果结论。
来源：[训练](../../train.py)、[渲染](../../renderer.py)、[观察变换](../../evaluate.py)、
[配对采样](../../residual_sampling.py)。
