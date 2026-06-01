# 可视化脚本说明

本文件夹用于存放视觉伺服模拟结果的可视化脚本。目前主要脚本为：

visualize_servo_results.py

该脚本基于 servo_sim.py all 生成的全序列结果文件，对 AOT 原始掩码结果与 AOT+GC 梯度校正结果进行指标对比和可视化展示。

## 运行方式

在项目根目录下运行：

python visualization/visualize_servo_results.py

## 输入文件

脚本默认读取以下文件：

servo_eval/all_sequences/per_sequence.csv
servo_eval/all_sequences/summary.json

其中，per_sequence.csv 记录每个 DAVIS-2017 val 序列在不同方法下的视觉伺服控制指标；summary.json 为全序列汇总结果。

## 输出目录

脚本生成的图表和统计结果会保存到：

servo_eval/visualizations/

主要输出包括：

dashboard_summary.png
mean_improvement_bar.png
win_rate_bar.png
mean_metric_comparison.png
per_sequence_compare_tracking_rmse_px.png
per_sequence_compare_tracking_p99_px.png
per_sequence_compare_control_energy.png
per_sequence_compare_centroid_jerk_rms.png
improvement_by_sequence_tracking_rmse_px.png
improvement_by_sequence_tracking_p99_px.png
improvement_by_sequence_control_energy.png
improvement_by_sequence_centroid_jerk_rms.png
boxplot_tracking_rmse_px.png
boxplot_tracking_p99_px.png
boxplot_control_energy.png
boxplot_centroid_jerk_rms.png
visualization_summary.json
visualization_summary.md

## 指标说明

当前脚本主要可视化以下视觉伺服控制指标：

tracking_rmse_px

表示平均追踪误差，单位为像素。数值越低，说明虚拟相机对目标质心的跟踪越稳定。

tracking_p99_px

表示 99 分位追踪误差，反映极端情况下的较大追踪偏差。数值越低，说明最坏情况下的控制误差越小。

centroid_jerk_rms

表示目标质心轨迹的 RMS jerk，用于衡量轨迹抖动程度。数值越低，说明轨迹变化更平滑。

control_energy

表示控制能耗或控制输入强度。数值越低，说明控制过程所需调整更少，控制行为更平稳。

n_missing_frames

表示目标缺失帧数量。

missing_rate

表示目标缺失帧比例。

在当前脚本中，上述指标均按照“数值越低越好”进行处理。因此，AOT+GC 相比 AOT(ori) 的 improvement 为正值时，表示梯度校正后的结果更优。

## 可视化内容

脚本主要生成以下几类图：

1. 逐序列指标对比图
对每个 DAVIS 序列分别比较 AOT(ori) 与 AOT+GC 的 tracking error、control energy 和 centroid jerk 等指标。

2. 箱线图
展示两种方法在所有序列上的指标分布差异。

3. 改善率图
计算 AOT+GC 相对 AOT(ori) 的逐序列改善比例，用于观察梯度校正在不同序列上的效果差异。

4. 胜率图
统计 AOT+GC 在多少比例的序列上优于 AOT(ori)。

5. 综合面板图
将平均指标、平均改善率、胜率和 tracking RMSE 逐序列改善情况整合在一张图中，便于报告展示。

## 当前结果解释

根据当前运行结果，AOT+GC 相比 AOT(ori) 在平均 tracking RMSE、tracking P99、control energy 和 centroid jerk 上均有轻微改善，但整体平均提升幅度较小。

从胜率图可以看到，AOT+GC 在多数序列上能够降低 tracking RMSE 和 control energy，说明梯度校正对下游视觉伺服控制有一定正向影响。但从均值来看，这种提升并不显著，因此更适合通过逐序列改善图和胜率图进行展示，而不是只依赖平均指标图。

逐序列改善图也显示，梯度校正的效果具有明显的序列依赖性：部分序列改善较明显，部分序列变化很小，也有少数序列略有下降。

## 推荐用于报告或 PPT 的图

建议优先使用以下图：

dashboard_summary.png
mean_improvement_bar.png
win_rate_bar.png
improvement_by_sequence_tracking_rmse_px.png
improvement_by_sequence_tracking_p99_px.png
per_sequence_compare_tracking_rmse_px.png

其中，win_rate_bar.png 和逐序列 improvement 图最能体现 AOT+GC 在多数序列上的轻微优势。mean_improvement_bar.png 可以用于说明整体平均提升幅度较小。

不建议重点使用 boxplot_missing_rate.png 和 boxplot_n_missing_frames.png，因为大多数序列的目标缺失帧为 0，这两张图的信息量较低。

## 备注

当前脚本主要完成 README 中要求的“逐序列指标对比、AOT(ori) vs AOT+GC 胜率/箱线图”等指标级可视化。

轨迹叠加动画和将质心轨迹回投到 JPEG 帧上的可视化，需要额外的逐帧轨迹日志或 DAVIS 原始 JPEG 图像。由于当前仓库中不包含 DAVIS 原始 JPEG 帧，因此这一部分后续可在数据补全后继续扩展。
