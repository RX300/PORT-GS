# Cat/Pixiu质量瓶颈复核（2026-09-26）

本轮完成只读分析、保留终端模型的完整训练集补评和设计建议，没有改模型参数、生产渲染代码、正式配置或数据，没有启动新训练。新增评价放在原最终run的`fit_full/`，没有重新创建清理掉的中间run。历史负实验只引用仍存文档，不能宣称其已删除模型再次验证通过。

## 新的完整训练集证据

来源：`runs/distribution_material_final/Real_NRHints/{Cat,Pixiu}/{fit_full,test}/metrics.json`。
使用该run原源码一致的evaluate/renderer/data/distribution_material，512px、固定30000步模型，无相机或曝光再拟合。

| 场景 | 全train / test帧数 | train PSNR | test PSNR | train / test LPIPS | train tiny命中 | test tiny命中 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Cat | 522 / 66 | 22.531665 | 22.141679 | .265215 / .267714 | 6/3999 = .1500% | 0/575 |
| Pixiu | 562 / 71 | 25.126019 | 21.502959 | .148296 / .167022 | 205/40429 = .5071% | 22/5324 = .4132% |

Cat当前版本首先是训练拟合失败；不能只归咎于未见视角。Pixiu存在3.62306dB的总体train/test差距，同时训练微小高光也几乎未学到。全train与test的条件分布不同，这些均值不是严格匹配条件的泛化误差分解。
先前16fit的21.10/26.14dB不是全train均值；以本次522/562帧补评为准。
tiny是量化GT中1–4px中性亮峰的代理，不等于物理镜面反射标签。Cat的毛发纹理还必须单独评估。

## 已证实的实现限制

### 1. 固定40000并不等于40000个有效表面点

读取最终checkpoint：Cat21975点、Pixiu22647点的sigmoid opacity低于1/255，分别54.94%/56.62%。
可通过这一必要条件的上限只有18025/17353点；实际可见数还会更少。
本机gsplat1.5.3的`cuda/include/Common.h:54`定义ALPHA_THRESHOLD=1/255；
`RasterizeToPixels2DGSFwd.cu:361–365`使用alpha=min(.999,opacity*exp(-sigma))，sigma非负；低于阈值直接跳过，反向也跳过。
因此上述低opacity点不能贡献相机图像；非局部源路径仍可使用这些点，但其source mass仅占.4408%/.3967%，不能据此宣称全部梯度或全部计算均消失。
`configs/validation.json`的points=max-points=40000、refine-stop=0，使点死亡后没有增密/重分配机制。点仍保存在权重里，并不代表有可用相机渲染容量。

这是结构/配置事实，不是“只要补点就一定提高质量”的因果证明。旧交点方案取消cutoff单独恢复曾失败，说明需把初始化、有效点分配、局部属性容量与几何梯度一起设计。

### 2. 当前神经间接项是受限的全局混合

实现`methods/distribution_material.py:85–122`把全部源点先压成32个RGB信号，然后由每个接收点混合。对固定出射查询和任意输入照明，该分支每颜色的离散传输矩阵秩最多32；直接项不受这个上限约束。
源空间分布的熵等效点数中位数：Cat约1058、Pixiu约1337；这说明典型分量汇集很多源点，不是精确测得的散射传播范围。
vMF的kappa上限32.05；单个角度叶片的最小半高半宽约11.94°。全部source×component的未加权叶片中位半高半宽Cat16.91°、Pixiu18.18°，不是可见度/传输贡献加权值。
Cat有116/1280000个极宽叶片在单位球内根本不降到半峰，统计已单列，不能对它们用acos生成NaN。
这些角宽是**入射方向响应**，不能换算成图像高光宽度的严格下界；锐利直接GGX、阴影变化与空间/出射条件仍能产生高频。

因此该分支适合广域平滑传输，但不能作为承担所有高频外观的充分表示。Pixiu55.47%的16fit线性能量来自间接项，Cat22.30%；这不等于物理多次散射比例，也不能仅靠比例判定分支“作弊”。
当前opacity合格点上的核心GGX alpha中位数Cat.327、Pixiu.213，alpha<.01的比例仅3.90%/5.00%。这是全体合格点、非可见度加权统计，不可据此判定物体真实粗糙度；它与输出偏平滑相容。

### 3. 神经材质仍在一个混合后的虚拟表面上着色

`renderer.py:467–485`先平均base/features/visibility和中心深度，再反投影为单一接收点、归一化混合法线，最后计算非线性材质。
一般有 f(sum w m_j, normalize(sum w n_j)) != sum w f(m_j,n_j)。
当毛发、轮廓、多个表层贡献混在一起时，该近似可能移动高光或使其变宽；相机改变时参与混合的点也改变。
但这只是机制分析：已有逐ray–surfel交点GGX对照同样失败，所以“改为先着色再合成”不是已经验证的单独解药。

### 4. 8DNA训练信号没有被保留下来

8DNA学习已知可渲染资产的路径吞吐量加权条件密度，并回归独立能量；其路径采样可直接提供哪个入射端与哪个出射端相连的样本。
本项目只有图像，没有真实路径。现有image_distribution_loss是每张观察图的通道空间KL，约束亮度在哪些像素出现，不能识别8D隐含散射路径。
概率归一化是有效数学约束，但本适配不继承原文的路径监督或低方差训练保证。形状、局部材质、非局部传输和场景光强仍可互相补偿。
[8DNA原文§3](https://arxiv.org/html/2604.25129v1)。

## 数据与相机：有风险，但不能无证据当成总解释

本地512px图像的fx为Cat3160.438、Pixiu3095.451；小角度一阶近似下，.01°旋转对应.552/.540px横向变化（光轴附近、忽略畸变）。这只是灵敏度，不是测得的位姿误差。
历史Cat宽视角训练匹配报告metadata-F中位残差约3.687px；匹配也受重复毛发、可见性和光照影响，不能直接当真实相机误差。旧训练相机优化提高了部分纹理，却降低原始验证相机下指标；不能重试无约束逐帧pose就宣布解决。
本次中心视角/灯角最近邻的联合角距中位数Cat13.82°、Pixiu12.83°；与test PSNR的Pearson仅−.067/−.184。这种粗代理不足以解释主要误差，更不能断言“测试都是外推”。

NRHints作者版明确讨论了标定误差在3D表示中导致模糊，官方README推荐真实场景训练相机优化；这是训练侧证据，不授权test RGB位姿拟合。
[作者版比较与标定讨论](https://arxiv.org/html/2308.13404v1) · [官方README](https://github.com/iamNCJ/NRHints#training)。

### 修正此前的捕获来源归纳

NRHints区分新采集的Cat on Decor/Cup and Fabric/Pikachu与复用DNL数据。其README把Cat与CatSmall分列；DNL作者项目页直接展示当前Cat和Pixiu类型，原文§4列出Cat/Pixiu由双DSLR采集，并采用gamma2.2近似线性化、假设没有其他主要光源。
因此，之前把NRHints新采集流程的Sony A7II+iPhone/S-log描述泛化到本地Cat/Pixiu是不充分的；不能把S-log失配或强环境光当作已确认病因。
这仍不是本地压缩包从原视频到PNG的逐文件辐射校准证明。当前gamma2.2有相关源文支持，保持不变；精确曝光、白平衡、闪光发射形状误差仍未定量辨识。
[DNL作者项目](https://yuedong.shading.me/project/defnlight/defnlight.htm) · [DNL原文§4](https://gao-duan.github.io/publications/neuralrelighting/DeferredNeuralLighting_low_resolution.pdf)。

## 已有负结果限制下一步选择

以下是保留文档中的历史结论，相关中间权重已按要求删除：

- 少量条件可拟合：Pixiu四帧残差拟合的tiny命中295/307、对比/GT约.947；但其他已见帧受到破坏。说明局部图像存在拟合能力，不证明正确材质或泛化。[记录](gs_residual_subset.md)
- 更多预算/乘性交互/可移动角度中心：部分训练指标改善，整体质量和验证未通过，不能再次包装为全新答案。[预算](gs_residual_budget.md) · [交互](gs_residual_interaction.md)
- 只强调峰损失：paired使完整训练tiny召回51.75%→57.18%，但PSNR−.44dB、precision下降、黑块/亮粒增加；因此不能仅追recall。[记录](gs_residual_paired_loss.md)
- 表面初始化明显帮助低频形状与颜色（开发验证PSNR14.37→21.25），但tiny仍4/3770。[记录](mask_surface_initialization.md)
- 4倍tiny采样曝光没有转化为验证质量改善。[记录](surface_patch_proposal.md)
- 精确交点局部GGX100k与聚合对照均缺细峰；取消cutoff也非充分修复。[记录](intersection_reflectance.md)
- 本地GS³对Pixiu的干净历史test为23.72/.1173，明显好于PORT；Cat却只有19.12。不存在已有证据支持的通用替代赢家。SSD部分旧高分涉及test相机/灯优化，不能作为干净目标。[来源](foundation_comparison.md)

## 优先级与结论

1. **高置信事实、第一工程检查**：恢复有效几何容量管理，查透明点死亡与错误空间覆盖；同时用小问题测试几何与局部材质能否共同拟合。
2. **强机制依据、尚需因果验证**：逐贡献着色、跨视图稳定的空间/方向条件，避免平均后的表面代理和全局低秩分支包办高频。
3. **需独立证据**：标定残差、Cat多层毛发表示、Pixiu几何法线与镜面条件；分别设计小对照，不把它们一次全改。
4. **低优先级**：再换归一化、增加rank、继续加峰权重、仅增加训练步数、未经数据来源核实就改gamma/灯位。

建议转向[局部光传输与可更新几何方案](../research/local_transport_redesign_20260926.md)。本轮诊断没有证明唯一病因或保证该方案获胜。
完整新数值、阈值/角宽定义、命令、环境和来源在[quality_diagnosis_20260926.json](quality_diagnosis_20260926.json)。
