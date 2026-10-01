# 重构代码：默认方法与 neural_material 六场景重训

## 范围与来源

开始于2026-10-01（日本时间）。代码HEAD为26de5bd（refactor/port-gs-structure），开始检查时工作区干净。用户授权当前PORT-GS任务，最多使用3张GPU。

方法：directional_port_v1、neural_material。每种覆盖Real_NRHints Cat/Pixiu、Synthetic_GS3 AnisoMetal/Translucent、Synthetic_SSS-GS bunny_small/dragon_small，共12组。

开始时runs/为空，没有旧checkpoint、GGGS几何或材质decoder可复用。因而采用原生fresh初始化；这不是旧GGGS初始化的R2b复现。只比较当前两个方法，不恢复退役的几何训练入口。

## 训练和评价

- 共享ssd-gs环境，PyTorch2.4.1、CUDA12.1、gsplat1.5.3；不升级依赖。
- 30,000步、seed0、完整官方train；默认20,000初始高斯/400,000上限。默认方法原生3DGS，neural_material原生2DGS。
- Real/GS3为512px，SSS为256px；GS3白背景，其余黑背景；gamma2.2；SSS单位灯强1。
- 真实数据沿用文档推荐的训练相机rotation校正与translation gauge，训练相机学习率.001→.00001，从2000步启用；合成数据不启用相机优化。不拟合测试相机或灯位。
- 完整official test：66、71、400、400、500、500帧。输出PSNR、SSIM、标准LPIPS和完整预测/GT；无test-time calibration、无shift-align排名，不按测试指标选checkpoint。
- neural_material重建一次共享50k程序化材质decoder，场景训练时冻结；StableNormal/DA3仅生成官方train先验，使用现有独立推理环境和教师权重。保留原生normal/depth/surface/distortion损失，不做无先验简化。

## 入口与记录

规范配置：configs/validation.json、configs/material.json。使用pretrain_material.py、make_validation_manifest.py、launch_validation.sh、run_benchmark.py和evaluate.py。

正式目录：runs/refactored_six_scene_20261001/<方法>/<数据类别>/<场景>/。
材质依赖：runs/refactored_six_scene_20261001_dependencies/material_decoder/。
表面先验：正式目录下dependencies/surface_priors/<数据类别>/<场景>/。
预检：runs/refactored_six_scene_20261001_preflight/。日志：logs/refactored_six_scene_20261001/及manifest声明的场景日志。

生成manifest时冻结配置和source.tar，并记录Git及第三方来源。为同一队列混合原生3DGS/2DGS，只对2DGS任务安排surface_preprocessing；未改训练器、模型或渲染公式。

GPU0/1/2各一个串行worker，GPU3不使用。启动核对实际子进程、训练进度和GPU；随后按小时检查。固定训练→评估阶段自动顺序推进，不反复人工轮询。

## 状态

已完成12/12组，0失败，共3,874个测试评估帧。完整指标和核验记录见[最终结果](../../runs/refactored_six_scene_20261001/RESULTS.md)。默认方法在六场景的PSNR、SSIM、LPIPS均更优。

## 启动验证（10:25 JST）

共享50k decoder及默认/Neural集成预检均完成。三卡队列通过canonical launch_validation.sh启动，tmux socket为port-validation-refactored_six_scene_20261001。默认Cat/Pixiu/AnisoMetal实际PID100288/100289/100290，step1900/2100/500；GPU利用率67%/40%/83%，GPU3空闲。3运行、9待执行，无失败。正式指标尚未产出。
