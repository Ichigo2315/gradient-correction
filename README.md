# ECE 228 (SP26) Final Project
## Refining Video Object Segmentation with Test-time Gradient Correction

**English** | [中文](README.zh.md)

AOT (Associating Objects with Transformers) [[1]](#references) is a semi-supervised Video Object Segmentation (VOS) model that segments and tracks objects in a video given only the mask of the first frame. Output quality is measured with two metrics: the **J score (Jaccard Index)** measures region similarity as the Intersection-over-Union (IoU) between the predicted and ground-truth masks; the **F score (Boundary F-measure)** measures contour accuracy as the harmonic mean of boundary precision and recall.

This project improves the output mask quality with test-time **Gradient Correction (GC)** [[2]](#references), and builds a **visual servoing simulator** to verify the practical benefit of this refinement for downstream physical control.

---

### Key Contributions

#### 1. Test-time Gradient Correction

Core task: reproduce the gradient-correction method of Yuxi Li et al. to produce higher-quality masks.

1. Reproduce the AOT baseline: run AOT's train→test pipeline on DAVIS-2017 to obtain the base predicted masks.
2. Introduce Gradient Correction:
   - Use the first-frame Ground-Truth mask as the anchor.
   - At inference time, compute gradients from a cycle-consistency loss.
   - Refine the current-frame predicted mask via gradient descent.
3. Output: the refined predicted masks.

#### 2. Visual Servoing Simulator

Core task: build a simulator to verify how mask quality affects control trajectories and metrics.

1. Build a closed-loop, image-based visual servoing simulation.
2. Trajectory computation:
   - Take the **Pred Mask** (predicted) and **Ground-Truth Mask** from Module 1.
   - Extract the object centroid and smooth it with a Kalman filter.
   - Feed the processed signal into a PID controller that drives a virtual pan/tilt camera to track the object.
3. Physical control metrics:
   - Compare "native AOT mask" vs. "gradient-corrected mask" at the control side.
   - Key metrics: **Tracking Error**, **Control Energy / Jerk (smoothness)**, and **Lock-loss**.

---

### Dataset & Project Structure

All experiments use the validation set (val) of the `DAVIS-2017` dataset.

**DAVIS-2017 (Densely Annotated VIdeo Segmentation)** is one of the most widely used benchmarks for semi-supervised VOS. It provides the Ground-Truth mask of only the **first frame** of each sequence as input; the model must segment and track those objects across all subsequent frames. We use the 480p version.

The dataset lives in `aot-benchmark/datasets/DAVIS/`. The raw JPEG frames are large and are not committed; download them from the [DAVIS website](https://davischallenge.org/davis2017/code.html):

```
DAVIS/
├── JPEGImages/480p/<seq>/*.jpg     # video frames (input, not committed, download yourself)
├── Annotations/480p/<seq>/*.png    # Ground-Truth masks (committed)
└── ImageSets/2017/val.txt          # the 30 sequence names of the val subset
```

Overall project structure:

```
gradient-correction/
├── aot-benchmark/        
│   ├── networks/managers/evaluator.py    # inference + gradient correction
│   ├── tools/eval.py                     
│   ├── configs/                          # configs (incl. GC hyper-parameters)
│   ├── pretrain_models/                  # pretrained weights
│   └── results/davis2017/                # predicted masks (Pred Mask), shipped as zip (~2000 PNG/run, unzip in place)
│       ├── ..._noGCfull_....zip      # AOT(original)
│       └── ..._legacy20k1full_....zip# AOT+GC (K=1, α=20)
├── davis2017-evaluation/          # official J&F evaluation toolkit
├── servo_sim.py                   # visual servoing simulator (code stub: single / all subcommands)
├── servo_eval/                    # servo simulation outputs (figures + metrics)
│   ├── car-roundabout/            # single-sequence example output
│   └── all_sequences/             # all-sequence aggregation (CSV / JSON / figures)
├── visualization/                 # plotting scripts (metric charts, flowchart)
└── run_official_eval.py           # evaluation entry script
```

---

### Reproducing the Experiments

#### 1. Clone the repository

```bash
git clone https://github.com/Ichigo2315/gradient-correction.git
cd gradient-correction
```

#### 2. Set up the environment

There are two scopes depending on how far you want to reproduce:

- **Full reproduction** (re-run AOT inference + official J&F evaluation) — needs PyTorch with CUDA:

```bash
conda create -n ECE228 python=3.9 -y
conda activate ECE228
# install a CUDA build of PyTorch matching your GPU/driver, then:
pip install numpy scipy matplotlib pillow pandas opencv-python tqdm scikit-image
```

> The raw DAVIS JPEG frames are required for inference; download them (see "Dataset & Project Structure") and place them under `aot-benchmark/datasets/DAVIS/JPEGImages/480p/`.

- **Servo-only reproduction** (masks already shipped in the repo) — no PyTorch/CUDA needed:

```bash
conda create -n servo python=3.9 -y
conda activate servo
pip install numpy scipy matplotlib pillow pandas
```

#### 3. Generate the masks

You need three sets of masks under `aot-benchmark/`: the DAVIS Ground-Truth and the two prediction runs (AOT without GC, and AOT+GC).

**Option A — use the masks shipped in the repo (fast).** They are stored as zip (too many PNGs to commit raw); unzip them in place:

```powershell
# DAVIS Ground Truth -> aot-benchmark/datasets/DAVIS/Annotations/480p/<seq>/*.png
Expand-Archive aot-benchmark/datasets/DAVIS/Annotations.zip -DestinationPath aot-benchmark/datasets/DAVIS/

# Predicted masks -> aot-benchmark/results/davis2017/<run>/Annotations/480p/<seq>/*.png
Expand-Archive aot-benchmark/results/davis2017/davis2017_val_noGCfull_AOTT_PRE_ckpt_unknown.zip      -DestinationPath aot-benchmark/results/davis2017/
Expand-Archive aot-benchmark/results/davis2017/davis2017_val_legacy20k1full_AOTT_PRE_ckpt_unknown.zip -DestinationPath aot-benchmark/results/davis2017/
```

(On Linux/macOS use `unzip <zip> -d <target-dir>`.)

**Option B — regenerate the masks from scratch** (requires the full environment + JPEG frames). Run inference twice, once for each method, then evaluate J&F:

```bash
cd aot-benchmark

# AOT(ori) baseline — gradient correction disabled
python tools/eval.py --exp_name noGCfull --stage pre --model aott \
  --dataset davis2017 --split val \
  --ckpt_path pretrain_models/AOTT_PRE_YTB_DAV.pth --no_gc

# AOT+GC — gradient correction, every frame (K=1), 20 inner steps
python tools/eval.py --exp_name legacy20k1full --stage pre --model aott \
  --dataset davis2017 --split val \
  --ckpt_path pretrain_models/AOTT_PRE_YTB_DAV.pth \
  --gc_legacy --gc_interval 1 --gc_iter 20
cd ..

# Official DAVIS-2017 val J&F (region J + boundary F) for both runs
python run_official_eval.py --runs noGC legacy20_k1
```

Reference numbers on DAVIS-2017 val (×100): AOT(ori) **J&F 79.29 / J 76.59 / F 81.99**; AOT+GC **J&F 79.62 / J 76.63 / F 82.60** (+0.33 J&F, mostly on boundary F). Inference speed (single GPU): AOT(ori) ≈ 57.6 FPS, AOT+GC (K=1, N=20) ≈ 1.5 FPS.

#### 4. Run the visual servoing simulator

`servo_sim.py` exposes two subcommands. Outputs are written under `servo_eval/`.

```bash
# Single-sequence comparison (GT vs AOT+GC, car-roundabout): 4 figures + metrics.json
python servo_sim.py single

# All-sequence aggregation (AOT(ori) vs AOT+GC): per_sequence.csv / summary.json / summary_bar.png
python servo_sim.py all
```

> `servo_sim.py` ships as a **code stub**: every function/class signature, data contract and behaviour is documented in the docstrings, but the bodies raise `NotImplementedError`. Implement it following the docstrings (suggested order: mask I/O → Kalman/PID → closed-loop `simulate` → `compute_metrics` → `run_single` / `run_all`) before running the commands above.

#### 5. Produce the result charts

With `servo_eval/all_sequences/per_sequence.csv` in place, generate the metric comparison figures:

```bash
python visualization/visualize_servo_results.py
```

This reads `servo_eval/all_sequences/` and writes the charts (per-sequence comparison, box plots, per-sequence improvement, mean improvement, win rate, dashboard, plus `improvement_percent.csv` / `win_rate.csv` / `visualization_summary.json`) to `servo_eval/visualizations/`.

---

### References

- [1] Zongxin Yang, Yunchao Wei, and Yi Yang. "Associating Objects with Transformers for Video Object Segmentation." *Advances in Neural Information Processing Systems* (NeurIPS), 2021.
- [2] Yuxi Li, Ning Xu, Jinlong Peng, John See, and Weiyao Lin. "Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation." *Advances in Neural Information Processing Systems* (NeurIPS), 2020.
