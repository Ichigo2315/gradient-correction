from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("servo_eval/all_sequences")
OUT_DIR = Path("servo_eval/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "per_sequence.csv"

METHOD_ORI = "AOT(ori)"
METHOD_GC = "AOT+GC"

LOWER_IS_BETTER_METRICS = [
    "tracking_rmse_px",
    "tracking_p99_px",
    "centroid_jerk_rms",
    "control_energy",
    "n_missing_frames",
    "missing_rate",
]

MAIN_METRICS = [
    "tracking_rmse_px",
    "tracking_p99_px",
    "centroid_jerk_rms",
    "control_energy",
]


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CSV_PATH}. Please run `python servo_sim.py all` first."
        )

    df = pd.read_csv(CSV_PATH)

    if "sequence" not in df.columns or "method" not in df.columns:
        raise ValueError("per_sequence.csv must contain `sequence` and `method` columns.")

    metrics = [m for m in LOWER_IS_BETTER_METRICS if m in df.columns]
    main_metrics = [m for m in MAIN_METRICS if m in df.columns]

    if not metrics:
        raise ValueError("No expected servo metric columns were found.")

    return df, metrics, main_metrics


def compute_improvement(df, metrics):
    wide = df.pivot(index="sequence", columns="method", values=metrics)

    methods = set(df["method"].unique())
    if METHOD_ORI not in methods or METHOD_GC not in methods:
        raise ValueError(
            f"Expected methods `{METHOD_ORI}` and `{METHOD_GC}`, but found {sorted(methods)}."
        )

    improvement = pd.DataFrame(index=wide.index)

    for metric in metrics:
        ori = wide[(metric, METHOD_ORI)]
        gc = wide[(metric, METHOD_GC)]

        with np.errstate(divide="ignore", invalid="ignore"):
            improvement[metric] = (ori - gc) / ori * 100

        improvement[metric] = improvement[metric].replace([np.inf, -np.inf], np.nan)

    improvement.to_csv(OUT_DIR / "improvement_percent.csv")
    return improvement


def plot_per_sequence_comparison(df, metrics):
    for metric in metrics:
        plot_df = df.pivot(index="sequence", columns="method", values=metric)
        keep = [m for m in [METHOD_ORI, METHOD_GC] if m in plot_df.columns]
        plot_df = plot_df[keep]

        ax = plot_df.plot(kind="bar", figsize=(16, 6), width=0.8)
        ax.set_title(f"Per-sequence Comparison: {metric}")
        ax.set_xlabel("Sequence")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=75)
        ax.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"per_sequence_compare_{metric}.png", dpi=300)
        plt.close()


def plot_boxplots(df, metrics):
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        df.boxplot(column=metric, by="method")
        plt.title(f"Distribution Comparison: {metric}")
        plt.suptitle("")
        plt.xlabel("Method")
        plt.ylabel(metric)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"boxplot_{metric}.png", dpi=300)
        plt.close()


def plot_improvement_by_sequence(improvement, metrics):
    for metric in metrics:
        series = improvement[metric].dropna().sort_values(ascending=False)

        plt.figure(figsize=(16, 6))
        series.plot(kind="bar")
        plt.axhline(0, linewidth=1)
        plt.title(f"AOT+GC Improvement over AOT(ori): {metric}")
        plt.xlabel("Sequence")
        plt.ylabel("Improvement (%)")
        plt.xticks(rotation=75, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"improvement_by_sequence_{metric}.png", dpi=300)
        plt.close()


def plot_mean_improvement(improvement, metrics):
    mean_improvement = improvement[metrics].mean(skipna=True).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    mean_improvement.plot(kind="bar")
    plt.axhline(0, linewidth=1)
    plt.title("Average Improvement of AOT+GC over AOT(ori)")
    plt.ylabel("Mean improvement (%)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_improvement_bar.png", dpi=300)
    plt.close()

    return mean_improvement


def plot_win_rate(improvement, metrics):
    win_rate = {}

    for metric in metrics:
        valid = improvement[metric].dropna()
        win_rate[metric] = np.nan if len(valid) == 0 else (valid > 0).mean() * 100

    win_df = pd.DataFrame(
        {
            "metric": list(win_rate.keys()),
            "win_rate_percent": list(win_rate.values()),
        }
    )
    win_df.to_csv(OUT_DIR / "win_rate.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(win_df["metric"], win_df["win_rate_percent"])
    plt.title("Win Rate of AOT+GC over AOT(ori)")
    plt.ylabel("Win rate (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "win_rate_bar.png", dpi=300)
    plt.close()

    return win_rate


def plot_mean_metric_comparison(df, metrics):
    mean_df = df.groupby("method")[metrics].mean().T
    keep = [m for m in [METHOD_ORI, METHOD_GC] if m in mean_df.columns]
    mean_df = mean_df[keep]
    mean_df.to_csv(OUT_DIR / "mean_metrics_by_method.csv")

    ax = mean_df.plot(kind="bar", figsize=(11, 5), width=0.75)
    ax.set_title("Mean Servo Metrics by Method")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean value")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_metric_comparison.png", dpi=300)
    plt.close()


def plot_dashboard(df, improvement, metrics):
    selected = [
        m
        for m in ["tracking_rmse_px", "tracking_p99_px", "control_energy", "centroid_jerk_rms"]
        if m in metrics
    ]

    if not selected:
        return

    mean_metrics = df.groupby("method")[selected].mean().T
    mean_improvement = improvement[selected].mean(skipna=True)
    win_rate = improvement[selected].apply(lambda s: (s.dropna() > 0).mean() * 100)

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    mean_metrics.plot(kind="bar", ax=ax1)
    ax1.set_title("Mean Metrics by Method")
    ax1.set_ylabel("Mean value")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(2, 2, 2)
    mean_improvement.plot(kind="bar", ax=ax2)
    ax2.axhline(0, linewidth=1)
    ax2.set_title("Mean Improvement of AOT+GC")
    ax2.set_ylabel("Improvement (%)")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(2, 2, 3)
    win_rate.plot(kind="bar", ax=ax3)
    ax3.set_ylim(0, 100)
    ax3.set_title("AOT+GC Win Rate")
    ax3.set_ylabel("Win rate (%)")
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(axis="y", alpha=0.25)

    ax4 = fig.add_subplot(2, 2, 4)
    if "tracking_rmse_px" in selected:
        rmse_imp = improvement["tracking_rmse_px"].dropna().sort_values(ascending=False)
        rmse_imp.plot(kind="bar", ax=ax4)
        ax4.axhline(0, linewidth=1)
        ax4.set_title("Tracking RMSE Improvement by Sequence")
        ax4.set_ylabel("Improvement (%)")
        ax4.tick_params(axis="x", rotation=75)
        ax4.grid(axis="y", alpha=0.25)
    else:
        ax4.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "dashboard_summary.png", dpi=300)
    plt.close()


def write_summary(df, metrics, mean_improvement, win_rate):
    summary = {
        "input_csv": str(CSV_PATH),
        "num_sequences": int(df["sequence"].nunique()),
        "num_rows": int(len(df)),
        "methods": sorted(df["method"].unique().tolist()),
        "metrics_used": metrics,
        "mean_improvement_percent": {
            k: None if pd.isna(v) else float(v)
            for k, v in mean_improvement.to_dict().items()
        },
        "win_rate_percent": {
            k: None if pd.isna(v) else float(v)
            for k, v in win_rate.items()
        },
        "note": (
            "For all selected metrics, lower values are better. "
            "Positive improvement means AOT+GC performs better than AOT(ori)."
        ),
    }

    with open(OUT_DIR / "visualization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Visualization Summary",
        "",
        f"Number of sequences: {summary['num_sequences']}",
        f"Number of rows: {summary['num_rows']}",
        f"Methods: {', '.join(summary['methods'])}",
        "",
        "For all selected metrics, lower values are better. Positive improvement means AOT+GC performs better than AOT(ori).",
        "",
        "## Mean Improvement (%)",
    ]

    for k, v in summary["mean_improvement_percent"].items():
        lines.append(f"- {k}: {v:.2f}" if v is not None else f"- {k}: NA")

    lines.append("")
    lines.append("## Win Rate (%)")

    for k, v in summary["win_rate_percent"].items():
        lines.append(f"- {k}: {v:.2f}" if v is not None else f"- {k}: NA")

    with open(OUT_DIR / "visualization_summary.md", "w") as f:
        f.write("\n".join(lines))


def main():
    df, metrics, main_metrics = load_data()

    improvement = compute_improvement(df, metrics)

    plot_per_sequence_comparison(df, main_metrics)
    plot_boxplots(df, metrics)
    plot_improvement_by_sequence(improvement, main_metrics)

    mean_improvement = plot_mean_improvement(improvement, metrics)
    win_rate = plot_win_rate(improvement, metrics)

    plot_mean_metric_comparison(df, main_metrics)
    plot_dashboard(df, improvement, metrics)
    write_summary(df, metrics, mean_improvement, win_rate)

    print("Metric visualization finished.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
