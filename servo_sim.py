"""
Visual-servoing evaluation harness for VOS masks.

This file was completed from an implementation stub that specifies the public
API (function and class signatures), the data contracts, and the expected
behaviour via docstrings/comments.

-------------------------------------------------------------------------------
What the finished script must do
-------------------------------------------------------------------------------
Pipeline (per mask sequence):
    DAVIS-style indexed PNG masks  ->  per-object centroid (observation z_t)
                                  ->  constant-velocity Kalman filter (state x_t)
                                  ->  PID controller driving a virtual pan-tilt
                                       camera toward the image center
                                  ->  metrics + figures

Two sub-commands:

  single  Run the loop on one sequence for an arbitrary set of mask sources and
          save per-run figures + metrics.json. Defaults compare GT vs AOT+GC on
          car-roundabout:

            python servo_sim.py single

          Add methods like this:

            python servo_sim.py single \
              --mask-dir aot-benchmark/datasets/DAVIS/Annotations/480p/car-roundabout --label GT \
              --mask-dir aot-benchmark/results/davis2017/davis2017_val_noGCfull_AOTT_PRE_ckpt_unknown/Annotations/480p/car-roundabout --label "AOT(ori)" \
              --mask-dir aot-benchmark/results/davis2017/davis2017_val_legacy20k1full_AOTT_PRE_ckpt_unknown/Annotations/480p/car-roundabout --label "AOT+GC(legacy)"

  all     Aggregate four downstream metrics over the whole DAVIS-2017 val set for
          the two fixed methods (AOT(ori) / AOT+GC) and save
          per_sequence.csv, summary.json, summary_bar.png:

            python servo_sim.py all

Notes:
    * GT masks are taken as the silver standard (used as the normalization
      reference; in `single` mode metrics are computed against the GT-driven
      controller trajectory).
    * Pixel-space tracking error is a perception-for-control surrogate, not a
      calibrated physical quantity (see proposal).
"""

from __future__ import annotations

import argparse
import csv
import json 
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def load_mask_sequence(mask_dir: Path) -> list[np.ndarray]:
    """Load all frames in `mask_dir` (sorted by filename) as uint8 ndarrays.

    DAVIS masks are indexed PNGs: pixel value = object id (0 = background).

    Returns one HxW uint8 array per frame, ordered by filename.
    Raise FileNotFoundError if the directory contains no .png files.
    """
    mask_dir = Path(mask_dir)
    png_files = sorted(mask_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG masks found in {mask_dir}")
    return [np.asarray(Image.open(path), dtype=np.uint8) for path in png_files]


def extract_centroid(mask: np.ndarray, obj_id: int = 1) -> np.ndarray | None:
    """Return (cx, cy) of the `obj_id` pixels as float64, or None if absent.

    cx/cy are the mean column/row index of all pixels equal to `obj_id`.
    """
    rows, cols = np.nonzero(mask == obj_id)
    if rows.size == 0:
        return None
    return np.array([cols.mean(), rows.mean()], dtype=np.float64)


def object_diag(mask: np.ndarray, obj_id: int = 1) -> float:
    """Bounding-box diagonal of the object in pixels (used for normalization).

    Return NaN if the object is absent in this frame.
    """
    rows, cols = np.nonzero(mask == obj_id)
    if rows.size == 0:
        return float("nan")
    width = cols.max() - cols.min() + 1
    height = rows.max() - rows.min() + 1
    return float(np.hypot(width, height))


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity, 2-D position observations)
# ---------------------------------------------------------------------------

class KalmanCV2D:
    """Constant-velocity Kalman filter.

    state       x = [cx, cy, vx, vy]^T
    observation z = [cx, cy]^T

    Suggested init: x = [init_pos, 0, 0], P = 100*I(4),
    F = constant-velocity transition with step `dt`, H = position selector,
    Q = q*I(4) (process noise), R = r*I(2) (observation noise).
    """

    def __init__(self, dt: float, q: float, r: float, init_pos: np.ndarray):
        """Store dt and build x, P, F, H, Q, R per the class docstring."""
        self.dt = float(dt)
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0],
                          dtype=np.float64)
        self.P = 100.0 * np.eye(4, dtype=np.float64)
        self.F = np.array([[1.0, 0.0, self.dt, 0.0],
                           [0.0, 1.0, 0.0, self.dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.Q = float(q) * np.eye(4, dtype=np.float64)
        self.R = float(r) * np.eye(2, dtype=np.float64)

    def predict(self) -> None:
        """Time update: x <- F x ; P <- F P F^T + Q."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Measurement update with observation z = [cx, cy] (standard KF eqs)."""
        innovation = np.asarray(z, dtype=np.float64) - self.H @ self.x
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        gain = self.P @ self.H.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ innovation
        self.P = (np.eye(4, dtype=np.float64) - gain @ self.H) @ self.P

    def step(self, z: np.ndarray | None) -> np.ndarray:
        """Run predict(), then update() iff z is not None; return a copy of x."""
        self.predict()
        if z is not None:
            self.update(z)
        return self.x.copy()


# ---------------------------------------------------------------------------
# Discrete PID
# ---------------------------------------------------------------------------

class PID2D:
    """Independent 2-D PID controller with symmetric output saturation u_max."""

    def __init__(self, kp: float, ki: float, kd: float, dt: float,
                 u_max: float = 50.0):
        """Store gains/dt/u_max; zero-initialise the integral and prev-error."""
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.u_max = float(u_max)
        self.integral = np.zeros(2, dtype=np.float64)
        self.prev_error = np.zeros(2, dtype=np.float64)

    def __call__(self, e: np.ndarray) -> np.ndarray:
        """Return clipped u = kp*e + ki*integral(e) + kd*derivative(e).

        Update internal integral and previous-error state each call.
        """
        e = np.asarray(e, dtype=np.float64)
        self.integral += e * self.dt
        derivative = (e - self.prev_error) / self.dt
        self.prev_error = e.copy()
        u = self.kp * e + self.ki * self.integral + self.kd * derivative
        return np.clip(u, -self.u_max, self.u_max)


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------

@dataclass
class RunLog:
    """Per-run trace produced by `simulate` and consumed by metrics/plots."""
    label: str
    times: np.ndarray
    z: np.ndarray            # (T, 2)  raw centroid observations (pixel)
    x_hat: np.ndarray        # (T, 4)  Kalman state estimate
    e: np.ndarray            # (T, 2)  pre-control error in IMAGE frame
    u: np.ndarray            # (T, 2)  control command
    cam: np.ndarray          # (T, 2)  virtual camera offset
    obj_diag: np.ndarray     # (T,)    bbox diagonal of GT object (px)
    missing: np.ndarray      # (T,)    True if no observation in that frame
    metrics: dict = field(default_factory=dict)


def simulate(masks: list[np.ndarray], *, label: str, dt: float, target: np.ndarray,
             kalman_r: float, kalman_q: float, pid_gains: tuple[float, float, float],
             obj_id: int, gt_masks: list[np.ndarray] | None = None) -> RunLog:
    """Run the perception-control loop on a single mask sequence.

    Convention (everything stays in *absolute* image-frame pixel coordinates):
        z_t        : observed centroid of the object in the image
        x_hat_t    : Kalman-smoothed estimate of z_t        (state in abs frame)
        cam_view   : where the virtual camera is pointing   (in abs frame)
        e_t        : x_hat_t - cam_view                     (camera-lead error)
        u_t        : PID(e_t) drives cam_view toward x_hat_t

    Per-frame loop:
        1. zt = extract_centroid(mask, obj_id)  (mark `missing` if None)
        2. x_hat_t = KalmanCV2D.step(zt)
        3. e_t   = x_hat_t[:2] - cam_view
        4. u_t   = PID2D(e_t)
        5. cam_view += u_t * dt
        6. obj_diag_t = object_diag(gt_masks[t]) if gt_masks given (else NaN)

    Init the camera locked on the first observed centroid; raise RuntimeError if
    object `obj_id` never appears. `target` is only for plotting (image center),
    not used in the control law. `gt_masks` only feeds the normalization scale.

    Higher mask noise -> jittery x_hat_t -> jittery u_t (control energy / jerk
    rise) and larger steady-state lag (RMSE rises).

    Return a fully-populated RunLog (metrics left empty for compute_metrics).
    """
    if not masks:
        raise ValueError("Cannot simulate an empty mask sequence")
    if gt_masks is not None and len(gt_masks) != len(masks):
        raise ValueError("Mask sequence and GT sequence must have equal lengths")

    observations = [extract_centroid(mask, obj_id) for mask in masks]
    init_pos = next((z for z in observations if z is not None), None)
    if init_pos is None:
        raise RuntimeError(f"Object id {obj_id} never appears in {label}")

    _ = target
    n_frames = len(masks)
    kalman = KalmanCV2D(dt, kalman_q, kalman_r, init_pos)
    pid = PID2D(*pid_gains, dt)
    cam_view = init_pos.copy()
    z_log = np.full((n_frames, 2), np.nan, dtype=np.float64)
    x_hat_log = np.empty((n_frames, 4), dtype=np.float64)
    error_log = np.empty((n_frames, 2), dtype=np.float64)
    control_log = np.empty((n_frames, 2), dtype=np.float64)
    camera_log = np.empty((n_frames, 2), dtype=np.float64)
    diag_log = np.full(n_frames, np.nan, dtype=np.float64)
    missing = np.zeros(n_frames, dtype=bool)

    for frame_idx, observation in enumerate(observations):
        missing[frame_idx] = observation is None
        if observation is not None:
            z_log[frame_idx] = observation
        state = kalman.step(observation)
        error = state[:2] - cam_view
        control = pid(error)
        cam_view = cam_view + control * dt
        x_hat_log[frame_idx] = state
        error_log[frame_idx] = error
        control_log[frame_idx] = control
        camera_log[frame_idx] = cam_view
        if gt_masks is not None:
            diag_log[frame_idx] = object_diag(gt_masks[frame_idx], obj_id)

    return RunLog(label=label,
                  times=np.arange(n_frames, dtype=np.float64) * dt,
                  z=z_log,
                  x_hat=x_hat_log,
                  e=error_log,
                  u=control_log,
                  cam=camera_log,
                  obj_diag=diag_log,
                  missing=missing)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# The four headline downstream metrics (lower = better) and their plot labels.
METRIC_KEYS = ["tracking_rmse_px", "tracking_p99_px",
               "centroid_jerk_rms", "control_energy"]
PRETTY = {"tracking_rmse_px": "RMSE [px]", "tracking_p99_px": "P99 err [px]",
          "centroid_jerk_rms": "jerk RMS", "control_energy": "ctrl energy"}


def compute_metrics(log: RunLog, lock_thresh_frac: float = 0.5) -> dict:
    """Compute physical / engineering metrics for one run; store on log.metrics.

    Let err_pix = ||e_t|| per frame. The returned dict must contain:
        tracking_rmse_px    : RMS of err_pix
        tracking_p99_px     : 99th percentile of err_pix
        tracking_rmse_norm  : RMS of err_pix / obj_diag   (over valid frames)
        tracking_p99_norm   : 99th percentile of the normalised error
        control_energy      : sum_t ||u_t||^2 * dt
        centroid_jerk_rms   : RMS magnitude of the 3rd time-derivative of x_hat
                              position (via np.gradient); NaN if T < 4
        lock_loss_frames    : #frames with err_pix > lock_thresh_frac*median(diag)
                              (-1 if no valid diag)
        n_missing_frames    : number of frames with no observation
        n_total_frames      : total number of frames

    Also assign the dict to `log.metrics` and return it.
    """
    err_pix = np.linalg.norm(log.e, axis=1)
    valid_diag = np.isfinite(log.obj_diag) & (log.obj_diag > 0)
    if np.any(valid_diag):
        norm_error = err_pix[valid_diag] / log.obj_diag[valid_diag]
        tracking_rmse_norm = float(np.sqrt(np.mean(norm_error ** 2)))
        tracking_p99_norm = float(np.percentile(norm_error, 99))
        lock_threshold = lock_thresh_frac * float(np.median(log.obj_diag[valid_diag]))
        lock_loss_frames = int(np.count_nonzero(err_pix > lock_threshold))
    else:
        tracking_rmse_norm = float("nan")
        tracking_p99_norm = float("nan")
        lock_loss_frames = -1

    if len(log.times) >= 4:
        dt = float(log.times[1] - log.times[0])
        velocity = np.gradient(log.x_hat[:, :2], dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        jerk = np.gradient(acceleration, dt, axis=0)
        centroid_jerk_rms = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))
    else:
        dt = 0.0 if len(log.times) < 2 else float(log.times[1] - log.times[0])
        centroid_jerk_rms = float("nan")

    metrics = {
        "tracking_rmse_px": float(np.sqrt(np.mean(err_pix ** 2))),
        "tracking_p99_px": float(np.percentile(err_pix, 99)),
        "tracking_rmse_norm": tracking_rmse_norm,
        "tracking_p99_norm": tracking_p99_norm,
        "control_energy": float(np.sum(log.u ** 2) * dt),
        "centroid_jerk_rms": centroid_jerk_rms,
        "lock_loss_frames": lock_loss_frames,
        "n_missing_frames": int(np.count_nonzero(log.missing)),
        "n_total_frames": int(len(log.times)),
    }
    log.metrics = metrics
    return metrics


# ---------------------------------------------------------------------------
# Plotting (single-sequence mode)
# ---------------------------------------------------------------------------

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def plot_runs(logs: list[RunLog], out_dir: Path, target: np.ndarray,
              hw: tuple[int, int]) -> None:
    """Save four comparison figures (dpi=150) into `out_dir` (created if needed):

        tracking_error.png : ||e_t|| vs time for each run
        trajectories.png   : absolute-frame z_t (raw), x_hat (Kalman) and
                             cam_view (PID) paths; image-rectangle + center star;
                             square=start, triangle=end; y-axis inverted
        control.png        : ||u_t|| vs time for each run
        metrics_bar.png    : grouped bars of the four METRIC_KEYS per run

    hw = (H, W) is the frame size; `target` is the image center marker.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for idx, log in enumerate(logs):
        ax.plot(log.times, np.linalg.norm(log.e, axis=1),
                color=PALETTE[idx % len(PALETTE)], label=log.label)
    ax.set(xlabel="time [s]", ylabel="tracking error [px]",
           title="Visual-servo tracking error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_error.png", dpi=150)
    plt.close(fig)

    height, width = hw
    fig, ax = plt.subplots()
    for idx, log in enumerate(logs):
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(log.z[:, 0], log.z[:, 1], ":", color=color, alpha=0.45,
                label=f"{log.label}: raw")
        ax.plot(log.x_hat[:, 0], log.x_hat[:, 1], "-", color=color,
                label=f"{log.label}: Kalman")
        ax.plot(log.cam[:, 0], log.cam[:, 1], "--", color=color,
                label=f"{log.label}: camera")
        ax.plot(log.x_hat[0, 0], log.x_hat[0, 1], "s", color=color)
        ax.plot(log.x_hat[-1, 0], log.x_hat[-1, 1], "^", color=color)
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], "k-", lw=1)
    ax.plot(target[0], target[1], "k*", markersize=12, label="image center")
    ax.set(xlabel="x [px]", ylabel="y [px]", title="Centroid and camera trajectories")
    ax.set_aspect("equal", adjustable="box")
    points = np.concatenate([
        path[np.all(np.isfinite(path), axis=1)]
        for log in logs
        for path in (log.z, log.x_hat[:, :2], log.cam)
    ])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_margin = max(10.0, 0.1 * (x_max - x_min))
    y_margin = max(10.0, 0.1 * (y_max - y_min))
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for idx, log in enumerate(logs):
        ax.plot(log.times, np.linalg.norm(log.u, axis=1),
                color=PALETTE[idx % len(PALETTE)], label=log.label)
    ax.set(xlabel="time [s]", ylabel="control magnitude",
           title="PID control command")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "control.png", dpi=150)
    plt.close(fig)

    x_positions = np.arange(len(METRIC_KEYS))
    bar_width = 0.8 / len(logs)
    fig, ax = plt.subplots()
    for idx, log in enumerate(logs):
        offset = (idx - (len(logs) - 1) / 2) * bar_width
        values = [log.metrics[key] for key in METRIC_KEYS]
        ax.bar(x_positions + offset, values, bar_width, label=log.label,
               color=PALETTE[idx % len(PALETTE)])
    ax.set_xticks(x_positions, [PRETTY[key] for key in METRIC_KEYS])
    ax.set(title="Downstream servo metrics", ylabel="metric value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# `single` sub-command
# ---------------------------------------------------------------------------

DEFAULT_GT  = "aot-benchmark/datasets/DAVIS/Annotations/480p/car-roundabout"
DEFAULT_PRD = ("aot-benchmark/results/davis2017/"
               "davis2017_val_legacy20k1full_AOTT_PRE_ckpt_unknown/"
               "Annotations/480p/car-roundabout")


def run_single(args: argparse.Namespace) -> None:
    """Driver for `single`: compare arbitrary mask sources on one sequence.

    Steps:
        * If --mask-dir not given, default to [DEFAULT_GT, DEFAULT_PRD] with
          labels ["GT (DAVIS)", "AOT+GC"]; require one --label per --mask-dir.
        * Load the first mask-dir as GT reference; derive (H, W), target=center,
          dt = 1/fps.
        * For each (mask-dir, label): simulate() then compute_metrics(). Use
          kalman_r_gt when the label starts with "gt" (case-insensitive), else
          kalman_r_pred. Print metrics per run.
        * plot_runs(...) into args.out_dir and dump {label: metrics} to
          out_dir/metrics.json.

    Expected args: mask_dir, label, obj_id, fps, kalman_q, pid, out_dir,
    kalman_r_gt, kalman_r_pred.
    """
    mask_dirs = args.mask_dir or [DEFAULT_GT, DEFAULT_PRD]
    labels = args.label or ["GT (DAVIS)", "AOT+GC"]
    if len(mask_dirs) != len(labels):
        raise ValueError("Provide exactly one --label for each --mask-dir")

    mask_dirs = [ROOT / path for path in mask_dirs]
    gt_masks = load_mask_sequence(mask_dirs[0])
    height, width = gt_masks[0].shape
    target = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    dt = 1.0 / args.fps
    logs = []

    for mask_dir, label in zip(mask_dirs, labels):
        masks = load_mask_sequence(mask_dir)
        kalman_r = args.kalman_r_gt if label.lower().startswith("gt") else args.kalman_r_pred
        log = simulate(masks, label=label, dt=dt, target=target,
                       kalman_r=kalman_r, kalman_q=args.kalman_q,
                       pid_gains=tuple(args.pid), obj_id=args.obj_id,
                       gt_masks=gt_masks)
        metrics = compute_metrics(log)
        logs.append(log)
        print(f"\n{label}")
        print(json.dumps(metrics, indent=2))

    out_dir = ROOT / args.out_dir
    plot_runs(logs, out_dir, target, (height, width))
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({log.label: log.metrics for log in logs}, file, indent=2)


# ---------------------------------------------------------------------------
# `all` sub-command  (full DAVIS-2017 val aggregation)
# ---------------------------------------------------------------------------

# GT path template and the two fixed methods to compare (path templates take a
# `{seq}` placeholder). AOT(ori) = no-GC baseline; AOT+GC = legacy K=1, alpha=20.
GT_TPL = ROOT / "aot-benchmark/datasets/DAVIS/Annotations/480p/{seq}"
METHODS = {
    "AOT(ori)": ROOT / "aot-benchmark/results/davis2017/"
    "davis2017_val_noGCfull_AOTT_PRE_ckpt_unknown/Annotations/480p/{seq}",
    "AOT+GC": ROOT / "aot-benchmark/results/davis2017/"
    "davis2017_val_legacy20k1full_AOTT_PRE_ckpt_unknown/Annotations/480p/{seq}",
}
# lab-coat's ids 1 and 2 are tiny regions that disappear almost immediately in
# the predictions. Track one of its three persistent person masks instead.
OBJECT_ID_OVERRIDES = {"lab-coat": 3}


def _path(tpl: Path, seq: str) -> Path:
    """Substitute `{seq}` into a path template and return a Path."""
    return Path(str(tpl).format(seq=seq))


def run_all(args: argparse.Namespace) -> None:
    """Driver for `all`: aggregate the four metrics over DAVIS-2017 val for
    AOT(ori) vs AOT+GC.

    Steps:
        * Read sequence names from args.val_list (one per line); dt = 1/fps.
        * For each sequence: load GT masks (skip seq if missing). For each method
          in METHODS: load masks, simulate() with kalman_r=args.kalman_r and the
          GT masks as normalization, compute_metrics(); collect a row
          {sequence, method, <METRIC_KEYS>}. Record skips for missing dirs/errors.
        * Write per_sequence.csv (metrics + missing-frame columns) and a compact
          missing_frames.csv with one row per sequence.
        * Compute, in summary.json:
            - means          : per-method mean of each metric over all sequences
            - win_vs_AOT_ori : per-metric count of seqs where AOT+GC < / == / >
                               AOT(ori)  (lower = better)
            - skipped, n_total_seqs
        * Save summary_bar.png: grouped bars of the four metrics (one bar per
          method: AOT(ori) vs AOT+GC).
        * Print a mean table to the console.

    Expected args: val_list, out_dir, obj_id, fps, kalman_q, pid, kalman_r.
    """
    val_list = Path(args.val_list)
    sequences = [line.strip() for line in val_list.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / args.fps
    rows = []
    skipped = []

    for seq in sequences:
        try:
            gt_masks = load_mask_sequence(_path(GT_TPL, seq))
            height, width = gt_masks[0].shape
            target = np.array([width / 2.0, height / 2.0], dtype=np.float64)
            obj_id = OBJECT_ID_OVERRIDES.get(seq, args.obj_id)
            sequence_rows = []
            for label, template in METHODS.items():
                masks = load_mask_sequence(_path(template, seq))
                log = simulate(masks, label=label, dt=dt, target=target,
                               kalman_r=args.kalman_r, kalman_q=args.kalman_q,
                               pid_gains=tuple(args.pid), obj_id=obj_id,
                               gt_masks=gt_masks)
                metrics = compute_metrics(log)
                sequence_rows.append({
                    "sequence": seq,
                    "selected_obj_id": obj_id,
                    "method": label,
                    **{key: metrics[key] for key in METRIC_KEYS},
                    "n_missing_frames": metrics["n_missing_frames"],
                    "n_total_frames": metrics["n_total_frames"],
                    "missing_rate": metrics["n_missing_frames"] / metrics["n_total_frames"],
                })
            rows.extend(sequence_rows)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            skipped.append({"sequence": seq, "reason": str(error)})
            print(f"Skipping {seq}: {error}")

    fieldnames = ["sequence", "selected_obj_id", "method", *METRIC_KEYS,
                  "n_missing_frames", "n_total_frames", "missing_rate"]
    with (out_dir / "per_sequence.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    means = {}
    for label in METHODS:
        method_rows = [row for row in rows if row["method"] == label]
        means[label] = {
            key: float(np.mean([row[key] for row in method_rows]))
            if method_rows else float("nan")
            for key in METRIC_KEYS
        }

    paired = {}
    for row in rows:
        paired.setdefault(row["sequence"], {})[row["method"]] = row

    missing_fieldnames = ["sequence", "selected_obj_id",
                          "AOT(ori)_missing_frames", "AOT+GC_missing_frames",
                          "n_total_frames"]
    with (out_dir / "missing_frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=missing_fieldnames)
        writer.writeheader()
        for seq, seq_rows in paired.items():
            writer.writerow({
                "sequence": seq,
                "selected_obj_id": seq_rows["AOT(ori)"]["selected_obj_id"],
                "AOT(ori)_missing_frames": seq_rows["AOT(ori)"]["n_missing_frames"],
                "AOT+GC_missing_frames": seq_rows["AOT+GC"]["n_missing_frames"],
                "n_total_frames": seq_rows["AOT(ori)"]["n_total_frames"],
            })

    win_counts = {}
    for key in METRIC_KEYS:
        counts = {"better": 0, "equal": 0, "worse": 0}
        for seq_rows in paired.values():
            if set(METHODS).issubset(seq_rows):
                gc_value = seq_rows["AOT+GC"][key]
                baseline_value = seq_rows["AOT(ori)"][key]
                if np.isclose(gc_value, baseline_value):
                    counts["equal"] += 1
                elif gc_value < baseline_value:
                    counts["better"] += 1
                else:
                    counts["worse"] += 1
        win_counts[key] = counts

    summary = {
        "means": means,
        "win_vs_AOT_ori": win_counts,
        "missing_frames_by_sequence": {
            seq: {
                row["method"]: {
                    "selected_obj_id": row["selected_obj_id"],
                    "n_missing_frames": row["n_missing_frames"],
                    "n_total_frames": row["n_total_frames"],
                    "missing_rate": row["missing_rate"],
                }
                for row in seq_rows.values()
            }
            for seq, seq_rows in paired.items()
        },
        "skipped": skipped,
        "n_total_seqs": len(sequences),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    x_positions = np.arange(len(METRIC_KEYS))
    bar_width = 0.35
    fig, ax = plt.subplots()
    for idx, label in enumerate(METHODS):
        offset = (idx - 0.5) * bar_width
        ax.bar(x_positions + offset, [means[label][key] for key in METRIC_KEYS],
               bar_width, label=label, color=PALETTE[idx])
    ax.set_xticks(x_positions, [PRETTY[key] for key in METRIC_KEYS])
    ax.set(title="DAVIS-2017 val mean servo metrics", ylabel="mean metric value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close(fig)

    print("\nDAVIS-2017 val mean servo metrics")
    print("method\t" + "\t".join(METRIC_KEYS))
    for label in METHODS:
        values = "\t".join(f"{means[label][key]:.6g}" for key in METRIC_KEYS)
        print(f"{label}\t{values}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with `single` and `all` sub-commands.

    Shared options (both sub-commands):
        --obj-id (int, default 1)         object id to track (DAVIS uses 1, 2...)
        --fps (float, default 24.0)
        --kalman-q (float, default 1.0)   process-noise variance
        --pid KP KI KD (default 0.45 0 0.05)

    `single` adds:
        --mask-dir (repeatable), --label (repeatable, one per --mask-dir)
        --out-dir (default "servo_eval/car-roundabout")
        --kalman-r-gt   (float, default 1.0)    obs-noise R for the GT run
        --kalman-r-pred (float, default 25.0)   obs-noise R for predicted runs
      -> func = run_single

    `all` adds:
        --val-list (default ROOT/aot-benchmark/.../ImageSets/2017/val.txt)
        --out-dir  (default ROOT/servo_eval/all_sequences)
        --kalman-r (float, default 25.0)        obs-noise R for all methods
      -> func = run_all

    The selected sub-command must set `func` so that main() can dispatch.
    """
    parser = argparse.ArgumentParser(description="Visual-servoing evaluation for VOS masks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--obj-id", type=int, default=1)
        subparser.add_argument("--fps", type=float, default=24.0)
        subparser.add_argument("--kalman-q", type=float, default=1.0)
        subparser.add_argument("--pid", type=float, nargs=3, metavar=("KP", "KI", "KD"),
                               default=(0.45, 0.0, 0.05))

    single = subparsers.add_parser("single", help="run one sequence")
    add_shared_options(single)
    single.add_argument("--mask-dir", action="append")
    single.add_argument("--label", action="append")
    single.add_argument("--out-dir", default="servo_eval/car-roundabout")
    single.add_argument("--kalman-r-gt", type=float, default=1.0)
    single.add_argument("--kalman-r-pred", type=float, default=25.0)
    single.set_defaults(func=run_single)

    all_sequences = subparsers.add_parser("all", help="run DAVIS-2017 val")
    add_shared_options(all_sequences)
    all_sequences.add_argument(
        "--val-list",
        default=ROOT / "aot-benchmark/datasets/DAVIS/ImageSets/2017/val.txt")
    all_sequences.add_argument("--out-dir", default=ROOT / "servo_eval/all_sequences")
    all_sequences.add_argument("--kalman-r", type=float, default=25.0)
    all_sequences.set_defaults(func=run_all)
    return parser


def main() -> None:
    """Parse args and dispatch to the selected sub-command's `func`."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
