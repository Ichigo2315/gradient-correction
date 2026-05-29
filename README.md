# ECE 228(SP26) Final Project
## Refining Video Object Segmentation with Test-time Gradient Correction

AOT (Associating Objects with Transformers) [[1]](#参考文献) 是一种用于视频目标分割(Video Object Segmentation, VOS)的模型。原论文中，用于模型输出质量的指标有两个，F分数和J分数：**J分数 (Jaccard Index)** 衡量区域相似度，计算预测掩码与真实掩码的交并比 (IoU)；**F分数 (Boundary F-measure)** 衡量边界准确度，基于预测掩码边界与真实掩码边界的精确率 (Precision) 和召回率 (Recall) 计算调和平均数。

本项目通过引入推理时的**梯度校正（Gradient Correction / GC）** [[2]](#参考文献) 来优化输出掩码质量，并搭建**视觉伺服模拟器**来验证这种优化对物理控制带来的实际提升。

---

### 核心任务

#### 1. 应用梯度校正（已完成）

核心任务：复现Yuxi Li等人提出的梯度校正方法，输出高质量的 Mask

1. 复现 AOT Baseline：运行 AOT 在 DAVIS2017数据集上的训练->测试流程，获取基础的预测掩码。
2. 引入梯度校正 (Gradient Correction)：
   - 将视频第一帧的 Ground-Truth (真实掩码) 作为锚点。
   - 在推理阶段，利用循环一致性损失计算梯度。
   - 通过梯度下降对当前帧的预测掩码进行校正。
3. 输出结果：得到优化后预测掩码。

#### 2. 视觉伺服模拟器

核心任务：搭建模拟器，验证掩码质量对控制轨迹和指标的影响**

1. 搭建模拟器：构建一个基于图像的视觉伺服闭环仿真环境。
2. 轨迹计算：
   - 接收模块一生成的 **Pred Mask**（预测掩码）和 **Ground-Truth Mask**（真实掩码）。
   - 提取目标的质心，并使用卡尔曼滤波器（Kalman Filter）进行平滑处理。
   - 将处理后的信号输入 PID 控制器，模拟虚拟云台相机追踪该目标的运动轨迹。
3. 计算物理控制指标：
   - 对比“原生 AOT 掩码”与“梯度校正后掩码”在控制端的表现。
   - 核心指标：**追踪误差 (Tracking Error)**、**控制能耗/平滑度 (Control Energy/Jerk)** 以及是否发生目标丢失 (Lock-loss)。

---

### 数据集及项目结构

我们使用 `DAVIS-2017` 数据集的验证集 (val) 完成全部实验。

**DAVIS-2017 (Densely Annotated VIdeo Segmentation)** 是半监督视频目标分割领域最常用的基准数据集之一。其特点为：仅提供每个序列**第一帧**的真实掩码 (Ground-Truth) 作为输入，模型需在后续所有帧中分割并跟踪这些目标。本项目使用其480p版本。


数据集目录（`aot-benchmark/datasets/DAVIS/`，原始 JPEG 帧因体积较大不纳入仓库，可从 [DAVIS 官网](https://davischallenge.org/davis2017/code.html) 下载）：

```
DAVIS/
├── JPEGImages/480p/<seq>/*.jpg     # 视频帧（输入，未入库，需自行下载）
├── Annotations/480p/<seq>/*.png    # 真实掩码 GT（入库保留）
└── ImageSets/2017/val.txt          # val 子集 30 个序列名
```

整体项目结构：

```
gradient-correction/
├── aot-benchmark/        
│   ├── networks/managers/evaluator.py    # 推理 + 梯度校正
│   ├── tools/eval.py                     
│   ├── configs/                          # 配置（含梯度校正超参数）
│   ├── pretrain_models/                  # 预训练权重
│   └── results/davis2017/                # 预测掩码 (Pred Mask)，以 zip 提供（~2000 PNG/run，需解压到同名目录）
│       ├── ..._noGCfull_....zip      # AOT(original)
│       └── ..._legacy20k1full_....zip# AOT+GC (K=1, α=20)
├── davis2017-evaluation/          # 官方 J&F 评测工具包
├── servo_sim.py                   # 视觉伺服仿真（code stub，待实现：single / all 子命令）
├── servo_eval/                    # 伺服仿真结果（图 + 指标）
│   ├── car-roundabout/            # 单序列示例输出
│   └── all_sequences/            # 全序列聚合（CSV / JSON / 图）
└── run_official_eval.py           # 评测入口脚本
```

---

### 开始编写视觉伺服模拟器

模块二（视觉伺服模拟器）以 `servo_sim.py` 的 **code stub** 形式提供：文件中已写好全部函数/类的签名、数据契约与行为说明（docstring），但函数体均为 `raise NotImplementedError`，需要组员按注释填充实现。下面是从零开始的完整流程。

#### 1. 克隆仓库

```bash
git clone <本仓库地址> gradient-correction
cd gradient-correction
```

#### 2. 配置环境

模块二（`servo_sim.py`）只依赖 NumPy / Matplotlib / Pillow，**不需要 PyTorch / CUDA**（梯度校正掩码已随仓库提供）：

```bash
conda create -n servo python=3.9 -y
conda activate servo
pip install numpy matplotlib pillow
```

> 若还要重跑 AOT 推理或官方 J&F 评测，则需另装 PyTorch(CUDA) 等依赖，并下载 DAVIS 原始 JPEG 帧，详见“数据集及项目结构”。

#### 3. 解压掩码（GT + 预测）

仓库中掩码以 zip 形式存放（PNG 数量过多），首次使用前需就地解压：

```powershell
# DAVIS GT 真值标注 -> aot-benchmark/datasets/DAVIS/Annotations/480p/<seq>/*.png
Expand-Archive aot-benchmark/datasets/DAVIS/Annotations.zip -DestinationPath aot-benchmark/datasets/DAVIS/

# 预测掩码 -> aot-benchmark/results/davis2017/<run>/Annotations/480p/<seq>/*.png
Expand-Archive aot-benchmark/results/davis2017/davis2017_val_noGCfull_AOTT_PRE_ckpt_unknown.zip     -DestinationPath aot-benchmark/results/davis2017/
Expand-Archive aot-benchmark/results/davis2017/davis2017_val_legacy20k1full_AOTT_PRE_ckpt_unknown.zip -DestinationPath aot-benchmark/results/davis2017/
```

（Linux/macOS 用 `unzip <zip> -d <目标目录>` 即可。）

#### 4. 实现 `servo_sim.py`

按 stub 中的 docstring 逐个补全实现，建议顺序：

1. **掩码 I/O**：`load_mask_sequence` / `extract_centroid` / `object_diag`
2. **滤波与控制**：`KalmanCV2D`（常速卡尔曼）、`PID2D`（带饱和的离散 PID）
3. **闭环仿真**：`simulate`（逐帧 感知 → 卡尔曼 → PID → 虚拟云台，输出 `RunLog`）
4. **指标**：`compute_metrics`（RMSE / P99 / jerk / 控制能耗 / lock-loss 等）
5. **驱动**：`run_single`（单序列 + 出图）、`run_all`（全序列聚合）、`build_parser` / `main`

#### 5. 生成结果

实现完成后运行两个子命令：

```bash
# 单序列对比（默认 GT vs AOT+GC，car-roundabout），产出 4 张图 + metrics.json
python servo_sim.py single

# 全序列聚合（AOT(ori) vs AOT+GC），产出 per_sequence.csv / summary.json / summary_bar.png
python servo_sim.py all
```

输出统一写入 `servo_eval/`（`car-roundabout/` 与 `all_sequences/`）。

#### 6. 编写可视化脚本（开放，组员自行发挥）

最后一步**不提供 stub**，请组员基于 `servo_eval/` 中的产物自行设计可视化/报告脚本，例如：

- 输入：`servo_eval/all_sequences/per_sequence.csv`、`summary.json`、各序列 `metrics.json`
- 可做：逐序列指标对比、AOT(ori) vs AOT+GC 胜率/箱线图、轨迹叠加动画、把质心轨迹回投到 JPEG 帧上的可视化等
- 目标：直观呈现“梯度校正后掩码对下游物理控制的影响”，服务于最终报告

---


### 参考文献

- [1] Zongxin Yang, Yunchao Wei, and Yi Yang. "Associating Objects with Transformers for Video Object Segmentation." *Advances in Neural Information Processing Systems* (NeurIPS), 2021. [[PDF]](AOT.pdf)
- [2] Yuxi Li, Ning Xu, Jinlong Peng, John See, and Weiyao Lin. "Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation." *Advances in Neural Information Processing Systems* (NeurIPS), 2020. [[PDF]](cyclic%20mechanism.pdf)
