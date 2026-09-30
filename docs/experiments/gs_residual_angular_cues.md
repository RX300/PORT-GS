# 当前遗漏小峰的局部角度提示定位审计

状态：实现、CPU6项、native16真实检查、完整562帧前向、独立CPU核查与人工图块检查全部完成。固定法线窄角度核筛选未通过，不启动该训练扫描。
这是交互对照后决定的只前向诊断，不是新的训练或测试集评价；完整研究目标仍active。

## 问题与固定协议

multiply30k相对add有有限训练拟合收益，但未通过继续门槛。两个模型的源GS/相机全部固定。
新核以现有法线为中心，缩小带宽不会移动极值；先测它们在当前剩余失败位置是否有定位正确且
有区分度的信号，而非重复旧平均n=h角误差。方案依据见[接续](../project/research_handoff.md)。

- checkpoint：runs/gs_residual_interaction_pilot/multiply/Real_NRHints/Pixiu/last.pt。
- 固定native512、全部562train，完整保存相机映射；一张检查空闲GPU，一次前向遍历，无优化器、不读test。
- reference：同模型fit_full/metrics.json，逐帧复核PSNR/SSIM/raw_MSE/alpha、残差统计和全峰/组件指标。
  GT/alpha/count精确，其它独立渲染浮点rtol1e−6/atol1e−8，保留实际差异；不重复LPIPS。
- 复用真实transport receivers，并核对残差头逐块点/法线与receiver顺序；重建原观察图要求逐位相同。
- 同GT代理的8连通1–4px组件：21237组件/40429px。按现有2px预测峰命中分missed/partial/all_hit。
  已有评价any-hit10060，因此完全未命中分母应为11177组件；无支持者不从此分母消失。
- 每组件11px膨胀邻环排除全部GT峰、要求GT alpha>.9及有效GS覆盖/角度；低GS alpha≤.9单列，不额外剔除。
- geometry normal为主，material normal只作描述；角度来自实际wi/wo的half vector。
- 两组各8个单位峰值核exp(-theta²/(2tau²))，tau对数均匀2°–32°和8°–128°，无1/tau振幅缩放。
  数值在CPU float64计算；它们是原始特征提示，不是预测辐射，不乘光强/距离因子，也不优化方向。

## 统计、筛选与边界

保存每核core/ring均值和差值、每组最佳差值/带宽；按GT选择最佳核仅为离线乐观上界，不输入模型。
定位为组件与有效ring中最小角度到GT组件的Chebyshev像素距离。精确并列极值记录最近/最远距离，
平坦域或极值跨core/ring不称可靠定位。带宽不同不改变极值位置。
完整core有效且ring非空才有资格；无支持者保留原始记录并计筛选失败。
统计包括组件及GT组件面积加权均值/分位数，明确eligible统计与全部missed分母的区别。
逐组件ring均值的加权汇总不是全图ring并集的像素均值，不能混称。

在新诊断结果之前固定的实用筛选：geometry完全未命中组件中，满足以下全部条件者至少60%，
并获固定crop支持，才值得后续短训练验证：可靠极值在2px内、最佳窄核core−ring≥.1，
且比最佳宽核增加≥.05。超过半数missed组件非正窄核对比或极值偏离>2px时，不直接做固定法线窄核30k。
这是参数化投入的实用筛选，不是模型容量/可表达性定理；小信号也可能被后续可学习层放大。
空间系数及其它原输入仍可能修正失败，不能由本诊断否定一切角度表示或推断真实法线。

复用此前固定8个64×64 GT crops（0/35/70/105帧各2处），展示GT/当前预测/两种法线角度及固定核热图，
不按本轮结果挑图。保存4帧NPZ原始角度/alpha/图像，未绘全图不等于未计入完整统计。

## 入口与检查

入口仍为diagnose_image_errors.py，新增--angular-cues、--reference-metrics、--cue-crops、--resolution。
此模式--limit 0遍历完整源fit；正数复用均匀选帧，仅作短测。拒绝test及与其它诊断模式混用。
纯CPU算子位于angular_cues.py；6项AngularCueTests覆盖单位峰/弧度、居中/偏移/平坦/并列极值、
8连通与ring排除、2px分组、缺支持/低alpha、加权合并及空组，首跑全部通过。
当前检查日志runs/gs_residual_angular_cue_smoke/cpu_tests.log。

完整结果与执行记录如下。


## 完整结果

所有562帧来自既有multiply30k训练范围；没有新的权重优化或test评价。完整GT高光32721组件/212877px，
其中最小桶21237组件/40429px，命中20347px、any-hit10060组件，逐项等于既有评价。
完全未命中11177组件/19169px；部分命中625/1838px，全部命中9435/19422px。
小桶仅4px无GS覆盖，分布于3个完全未命中组件；628px低GS alpha≤.9（遗漏组334px），没有从分母中移除。

| 完全未命中组件（分母11177） | geometry主分析 | material描述性参考 |
|---|---:|---:|
| 有完整core且有效ring |11174|11174|
| 提示极值可靠定位≤2px |1426（12.76%）|1908（17.07%）|
| 全部窄核条件合格 |62（0.55%）|76（0.68%）|
| 最佳窄核core−ring均值 |.065535|.111639|
| 最佳宽核core−ring均值 |.064030|.110018|
| 窄−宽差值均值 |.001505|.001621|
| 窄−宽差值中位数 |.000091|.000099|

上表均值/中位数是eligible组件统计，不是全局ring并集或GT像素加权值。
geometry主分析的9748/11177（87.21%）提示极值偏离>2px；4386个最佳窄核对比非正，
两者并集9870（88.31%）。仅62个同时达到定位、.1对比和.05相对增益，远低于预设60%。
无支持3个计不合格。极值距离中位数为5px，即当前11px局部域边界；不能把它当成无限域真实错位距离。
窄/宽核同中心，所以其极值位置本就一致。材料法线仅是描述性参考，没有重新选择主分析或门槛。

已命中组也存在角度提示错位：全部命中的9435组件中，geometry可靠定位1731（18.35%），
material3098（32.84%）。这说明原始提示的局部定位问题并不只发生在当前遗漏组；
不能据相关性宣称它是每个残差预测失败的唯一原因。当前头仍可通过其它输入/空间系数产生亮核。
选中2度核也不等于真实粗糙度：当所有对比均非正时，接近零的窄核也可能取得最大值。

固定8个crop检查与筛选不通过一致：frame0B的两颗GT小点没有对应2度提示，少量提示集中在偏下宽响应附近；
多数2度图在GT高光处接近黑色，32度形成宽结构，128度接近均匀。frame35B的模型已恢复局部亮尖，
但2度提示基本缺失，进一步表明不能把原始核等同于模型能力。manual_review.json记录原图范围与结论。
因此不启动固定法线窄核30k；不能据本诊断证明真实几何错误、法线GT误差、表达能力上限，
也不能否定可移动中心、多叶片、空间修正或更一般的双向表示。

## 运行、复核与产物

复用ssd-gs，无依赖变更，仅使用启动前确认空闲的GPU0。
真实16帧首跑通过：674组件/1236px；完整562诊断107.41秒（含作图），前向部分104.38秒。
新CPU代码angular_cues.py、新入口均已归档；原GS/传输/残差头/相机state与radius逐位保持。
562帧均核对实际receiver/头输入顺序和内部观察图重建逐位一致；逐帧参考指标通过。
22帧有23项独立渲染浮点末位差异，主要是残差统计，另含3个raw_MSE、1个SSIM；最大1.79e−7，
均满足rtol1e−6/atol1e−8。全部GT/alpha/整数计数精确，完整组件池化与参考相同。
首4个fullfit已有PNG逐像素核对通过；没有将仅有保存图像的字节检查夸大成562张独立参考PNG检查。

独立CPU核查：raw components.jsonl重算全部汇总字段精确；固定4帧168组件/307px的两种法线、
所有16个核、局部ring、定位和筛选由独立NumPy实现重算，所有原始值一致。
native16的674组件重叠记录一致，4帧NPZ的GT/预测/alpha/NaN覆盖mask相同。
第一次独立核查过严要求角度NPZ逐位相同，frame70坐标(x=250,y=433)的一个像素有
geometry5.96e−8/material1.19e−7弧度差；修改为原定紧浮点容差并记录差异后retry通过。
失败日志、首版脚本及重试保留；不是模型或科学指标失败，没有重跑GPU或改生产代码。
39份运行源码与当前版本逐字节相同。所有进程已结束，GPU0释放。

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
/workspace/ubuntu2004_cuda12_1/utils/conda-envs/ssd-gs/bin/python -u diagnose_image_errors.py \
runs/gs_residual_interaction_pilot/multiply/Real_NRHints/Pixiu/last.pt \
--angular-cues --reference-metrics runs/gs_residual_interaction_pilot/multiply/Real_NRHints/Pixiu/fit_full/metrics.json \
--cue-crops runs/gs_residual_interaction_pilot/comparison.json --resolution 512 --split train --limit 0 \
--output runs/gs_residual_angular_cue_audit
```

该输出已经完成，不要同名重跑。产物：runs/gs_residual_angular_cue_audit/{config.json,metrics.json,components.jsonl,
cue_crops.png,frame_000/035/070/105.npz,manual_review.json,verification.json,verification_source.tar,attempts.json}。
日志runs/gs_residual_angular_cue_audit.log；失败日志verification_failure_01.log与failure_source_01.tar保留。
独立核查脚本/tmp/port_angular_cue_audit_verify.py已归档。CPU/真实短测位于gs_residual_angular_cue_smoke。
