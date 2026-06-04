# ECE 228 (SP26) Final Project
## Refining Video Object Segmentation with Test-time Gradient Correction

[English](README.md) | **中文**

AOT (Associating Objects with Transformers) [[1]](#参考文献) 是一种用于视频目标分割(Video Object Segmentation, VOS)的模型。原论文中，用于模型输出质量的指标有两个，F分数和J分数：**J分数 (Jaccard Index)** 衡量区域相似度，计算预测掩码与真实掩码的交并比 (IoU)；**F分数 (Boundary F-measure)** 衡量边界准确度，基于预测掩码边界与真实掩码边界的精确率 (Precision) 和召回率 (Recall) 计算调和平均数。

本项目通过引入推理时的**梯度校正（Gradient Correction / GC）** [[2]](#参考文献) 来优化输出掩码质量，并搭建**视觉伺服模拟器**来验证这种优化对物理控制带来的实际提升。

---

### 核心贡献

#### 1. 应用梯度校正

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


数据集目录`aot-benchmark/datasets/DAVIS/`，原始 JPEG 帧因体积较大不纳入仓库，可从 [DAVIS 官网](https://davischallenge.org/davis2017/code.html) 下载：

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
├── visualization/                 # 绘图脚本（指标图表、流程图）
└── run_official_eval.py           # 评测入口脚本
```

---

### 复现实验

本节给出端到端的完整流程：**环境配置 → 生成掩码 → 运行伺服模拟器 → 产出结果图表。**

#### 1. 克隆仓库

```bash
git clone https://github.com/Ichigo2315/gradient-correction.git
cd gradient-correction
```

#### 2. 配置环境

按复现深度分两种范围：

- **完整复现**（重跑 AOT 推理 + 官方 J&F 评测）——需要带 CUDA 的 PyTorch：

```bash
conda create -n ECE228 python=3.9 -y
conda activate ECE228
# 先安装与你的 GPU/驱动匹配的 CUDA 版 PyTorch，然后：
pip install numpy scipy matplotlib pillow pandas opencv-python tqdm scikit-image
```

> 推理需要 DAVIS 原始 JPEG 帧，请自行下载（见“数据集及项目结构”）并放到 `aot-benchmark/datasets/DAVIS/JPEGImages/480p/`。

- **仅伺服复现**（直接用仓库自带掩码）——无需 PyTorch / CUDA：

```bash
conda create -n servo python=3.9 -y
conda activate servo
pip install numpy scipy matplotlib pillow pandas
```

#### 3. 生成掩码

需要在 `aot-benchmark/` 下准备三套掩码：DAVIS 真值 (GT)，以及两次预测（AOT 无 GC、AOT+GC）。

**方式 A — 使用仓库自带掩码（快）。** 掩码以 zip 存放（PNG 数量过多），首次就地解压：

```powershell
# DAVIS Ground Truth -> aot-benchmark/datasets/DAVIS/Annotations/480p/<seq>/*.png
Expand-Archive aot-benchmark/datasets/DAVIS/Annotations.zip -DestinationPath aot-benchmark/datasets/DAVIS/

# 预测掩码 -> aot-benchmark/results/davis2017/<run>/Annotations/480p/<seq>/*.png
Expand-Archive aot-benchmark/results/davis2017/davis2017_val_noGCfull_AOTT_PRE_ckpt_unknown.zip     -DestinationPath aot-benchmark/results/davis2017/
Expand-Archive aot-benchmark/results/davis2017/davis2017_val_legacy20k1full_AOTT_PRE_ckpt_unknown.zip -DestinationPath aot-benchmark/results/davis2017/
```

（Linux/macOS 用 `unzip <zip> -d <目标目录>` 即可。）

**方式 B — 从零重跑生成掩码**（需完整环境 + JPEG 帧）。分别跑两次推理，再评测 J&F：

```bash
cd aot-benchmark

# AOT(ori) baseline —— 关闭梯度校正
python tools/eval.py --exp_name noGCfull --stage pre --model aott \
  --dataset davis2017 --split val \
  --ckpt_path pretrain_models/AOTT_PRE_YTB_DAV.pth --no_gc

# AOT+GC —— legacy 梯度校正，每帧校正 (K=1)，内迭代 20 步
python tools/eval.py --exp_name legacy20k1full --stage pre --model aott \
  --dataset davis2017 --split val \
  --ckpt_path pretrain_models/AOTT_PRE_YTB_DAV.pth \
  --gc_legacy --gc_interval 1 --gc_iter 20
cd ..

# 两次结果的官方 DAVIS-2017 val J&F（区域 J + 边界 F）
python run_official_eval.py --runs noGC legacy20_k1
```

DAVIS-2017 val 参考值（×100）：AOT(ori) **J&F 79.29 / J 76.59 / F 81.99**；AOT+GC **J&F 79.62 / J 76.63 / F 82.60**（+0.33 J&F，增益主要在边界 F）。推理速度（单卡）：AOT(ori) ≈ 57.6 FPS，AOT+GC (K=1, N=20) ≈ 1.5 FPS。

#### 4. 运行视觉伺服模拟器

`servo_sim.py` 提供两个子命令，输出统一写入 `servo_eval/`：

```bash
# 单序列对比（GT vs AOT+GC，car-roundabout）：4 张图 + metrics.json
python servo_sim.py single

# 全序列聚合（AOT(ori) vs AOT+GC）：per_sequence.csv / summary.json / summary_bar.png
python servo_sim.py all
```

> `servo_sim.py` 以 **code stub** 形式提供：全部函数/类的签名、数据契约与行为说明（docstring）已写好，但函数体均为 `raise NotImplementedError`。运行上述命令前，请按 docstring 补全实现（建议顺序：掩码 I/O → 卡尔曼/PID → 闭环 `simulate` → `compute_metrics` → `run_single` / `run_all`）。

#### 5. 产出结果图表

在 `servo_eval/all_sequences/per_sequence.csv` 就绪后，生成指标对比图：

```bash
python visualization/visualize_servo_results.py
```

该脚本读取 `servo_eval/all_sequences/`，把图表（逐序列对比、箱线图、逐序列提升、平均提升、胜率、dashboard，以及 `improvement_percent.csv` / `win_rate.csv` / `visualization_summary.json`）写入 `servo_eval/visualizations/`。

---

### 参考文献

- [1] Zongxin Yang, Yunchao Wei, and Yi Yang. "Associating Objects with Transformers for Video Object Segmentation." *Advances in Neural Information Processing Systems* (NeurIPS), 2021.
- [2] Yuxi Li, Ning Xu, Jinlong Peng, John See, and Weiyao Lin. "Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation." *Advances in Neural Information Processing Systems* (NeurIPS), 2020.
