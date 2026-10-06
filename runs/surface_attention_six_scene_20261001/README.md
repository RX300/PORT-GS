# Surface Attention 六场景实验（2026-10-01）

对齐 refactored_six_scene_20261001：surface_attention 原生2DGS，从头训练30,000步，seed0；六场景 Cat、Pixiu、AnisoMetal、Translucent、bunny_small、dragon_small。Real/GS3 512px，SSS 256px；完整官方train与test（66/71/400/400/500/500）。真实训练启用rotation相机校正和translation gauge，测试保持原始标定，不拟合测试图。方法其余参数使用当前默认值。

直接复用上一轮相同训练图的 StableNormal/DA3 先验，路径保存在validation.json与manifest.json；无须重新生成，也不使用测试先验。不需要神经材质decoder。GPU0/1/2，GPU3保留空闲。

入口：configs/validation.json → launch_validation.sh → run_benchmark.py。正式目录 runs/surface_attention_six_scene_20261001/surface_attention/<family>/<scene>/；各场景保存last.pt、test/metrics.json及完整配对图。manifest、配置及source.tar保存命令与来源。上一轮结果不覆盖。

启动前执行现有test_method_integration.py的surface_attention训练/重载/评估检查，输出单独位于runs/surface_attention_six_scene_20261001_preflight，不计入正式结果。启动确认真实训练进程、日志和GPU后，每小时只检查一次。

状态：19:15:48 JST全部完成，19:35按小时核验通过。6/6组，0失败，1,937个test帧。最终指标见runs/surface_attention_six_scene_20261001/RESULTS.md与results.csv。

预检完成：surface_attention CLI训练、最终权重重载、CLI评价及2DGS梯度检查通过。首次预检因测试入口32px与完整512px先验不匹配被拒绝；失败日志保留，改用上一轮32px预检先验后通过（_preflight_retry）。正式配置无需修改。
