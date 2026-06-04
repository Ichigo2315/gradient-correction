from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


BASE_DIR = Path("servo_eval/all_sequences")
OUT_DIR = Path("servo_eval/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "per_sequence.csv"

METHOD_ORI = "AOT(ori)"
METHOD_GC = "AOT+GC"

# --------------------------------------------------------------------------- #
# Academic paper style (serif / Times New Roman, muted palette, black edges)
# --------------------------------------------------------------------------- #
SERIF_FONTS = ["Times New Roman", "Times", "Nimbus Roman",
               "Liberation Serif", "STIXGeneral", "DejaVu Serif"]

# Muted, print-friendly palette inspired by SPEC-style charts.
COLOR_ORI = "#a6a6a6"   # neutral gray
COLOR_GC = "#1f6f8b"    # deep teal-blue
COLOR_POS = "#1f6f8b"   # improvement > 0
COLOR_NEG = "#c0504d"   # improvement < 0
COLOR_SINGLE = "#2e7ebb"

METHOD_COLORS = {METHOD_ORI: COLOR_ORI, METHOD_GC: COLOR_GC}

EDGE_COLOR = "black"
EDGE_WIDTH = 0.7
BAR_WIDTH = 0.8


def set_style():
    """Configure a clean serif (Times-like) academic style globally."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_FONTS,
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#2b2b2b",
        "axes.axisbelow": True,          # grid behind bars
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_axes(ax, *, grid_axis="y"):
    """Apply the shared grid + legend styling to an Axes."""
    ax.grid(axis=grid_axis, color="#b8b8b8", linestyle="--",
            linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_edgecolor("black")


def colors_for(columns):
    """Pick palette colors matching a list of method column names."""
    return [METHOD_COLORS.get(c, COLOR_SINGLE) for c in columns]

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

        ax = plot_df.plot(kind="bar", figsize=(16, 6), width=BAR_WIDTH,
                          color=colors_for(keep), edgecolor=EDGE_COLOR,
                          linewidth=EDGE_WIDTH)
        ax.set_title(f"Per-sequence Comparison: {metric}", fontsize=19)
        ax.set_xlabel("Sequence", fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis="x", rotation=75)
        ax.legend(title="Method")
        style_axes(ax)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"per_sequence_compare_{metric}.png")
        plt.close()


def plot_boxplots(df, metrics):
    methods_present = [m for m in [METHOD_ORI, METHOD_GC]
                       if m in df["method"].unique()]
    for metric in metrics:
        groups = [df.loc[df["method"] == m, metric].dropna().values
                  for m in methods_present]

        fig, ax = plt.subplots(figsize=(7, 5))
        bp = ax.boxplot(groups, tick_labels=methods_present, widths=0.5,
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.4),
                        whiskerprops=dict(color="#2b2b2b", linewidth=1.0),
                        capprops=dict(color="#2b2b2b", linewidth=1.0),
                        flierprops=dict(marker="o", markersize=3.5,
                                        markerfacecolor="#888888",
                                        markeredgecolor="#555555", alpha=0.6))
        for patch, m in zip(bp["boxes"], methods_present):
            patch.set_facecolor(METHOD_COLORS.get(m, COLOR_SINGLE))
            patch.set_edgecolor(EDGE_COLOR)
            patch.set_linewidth(EDGE_WIDTH)
            patch.set_alpha(0.9)

        ax.set_title(f"Distribution Comparison: {metric}")
        ax.set_xlabel("Method")
        ax.set_ylabel(metric)
        style_axes(ax)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"boxplot_{metric}.png")
        plt.close()


def plot_improvement_by_sequence(improvement, metrics):
    for metric in metrics:
        series = improvement[metric].dropna().sort_values(ascending=False)
        bar_colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in series]

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(series.index, series.values, width=BAR_WIDTH,
               color=bar_colors, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
        ax.axhline(0, linewidth=1.0, color="#2b2b2b")
        ax.set_title(f"AOT+GC Improvement over AOT(ori): {metric}",
                     fontsize=19)
        ax.set_xlabel("Sequence", fontsize=16)
        ax.set_ylabel("Improvement (%)", fontsize=16)
        ax.tick_params(axis="x", rotation=75)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        legend_handles = [
            mpl.patches.Patch(facecolor=COLOR_POS, edgecolor=EDGE_COLOR,
                              label="AOT+GC better"),
            mpl.patches.Patch(facecolor=COLOR_NEG, edgecolor=EDGE_COLOR,
                              label="AOT(ori) better"),
        ]
        ax.legend(handles=legend_handles)
        style_axes(ax)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"improvement_by_sequence_{metric}.png")
        plt.close()


def plot_mean_improvement(improvement, metrics):
    mean_improvement = improvement[metrics].mean(skipna=True).sort_values(ascending=False)
    bar_colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in mean_improvement]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(mean_improvement.index, mean_improvement.values, width=0.62,
           color=bar_colors, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.axhline(0, linewidth=1.0, color="#2b2b2b")
    ax.set_title("Average Improvement of AOT+GC over AOT(ori)")
    ax.set_ylabel("Mean improvement (%)")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_improvement_bar.png")
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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(win_df["metric"], win_df["win_rate_percent"], width=0.62,
           color=COLOR_SINGLE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.axhline(50, linewidth=1.0, color="#c0504d", linestyle="--",
               alpha=0.8)
    ax.set_title("Win Rate of AOT+GC over AOT(ori)")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "win_rate_bar.png")
    plt.close()

    return win_rate


def plot_mean_metric_comparison(df, metrics):
    mean_df = df.groupby("method")[metrics].mean().T
    keep = [m for m in [METHOD_ORI, METHOD_GC] if m in mean_df.columns]
    mean_df = mean_df[keep]
    mean_df.to_csv(OUT_DIR / "mean_metrics_by_method.csv")

    ax = mean_df.plot(kind="bar", figsize=(11, 5), width=0.75,
                      color=colors_for(keep), edgecolor=EDGE_COLOR,
                      linewidth=EDGE_WIDTH)
    ax.set_title("Mean Servo Metrics by Method")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean value")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.legend(title="Method")
    style_axes(ax)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_metric_comparison.png")
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

    keep = [m for m in [METHOD_ORI, METHOD_GC] if m in mean_metrics.columns]
    mean_metrics = mean_metrics[keep]

    ax1 = fig.add_subplot(2, 2, 1)
    mean_metrics.plot(kind="bar", ax=ax1, color=colors_for(keep),
                      edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax1.set_title("Mean Metrics by Method")
    ax1.set_ylabel("Mean value")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(title="Method")
    style_axes(ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    mi_colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in mean_improvement]
    ax2.bar(mean_improvement.index, mean_improvement.values, width=0.62,
            color=mi_colors, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax2.axhline(0, linewidth=1.0, color="#2b2b2b")
    ax2.set_title("Mean Improvement of AOT+GC")
    ax2.set_ylabel("Improvement (%)")
    ax2.tick_params(axis="x", rotation=30)
    style_axes(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(win_rate.index, win_rate.values, width=0.62,
            color=COLOR_SINGLE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax3.axhline(50, linewidth=1.0, color="#c0504d", linestyle="--", alpha=0.8)
    ax3.set_ylim(0, 100)
    ax3.set_title("AOT+GC Win Rate")
    ax3.set_ylabel("Win rate (%)")
    ax3.tick_params(axis="x", rotation=30)
    style_axes(ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    if "tracking_rmse_px" in selected:
        rmse_imp = improvement["tracking_rmse_px"].dropna().sort_values(ascending=False)
        r_colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in rmse_imp]
        ax4.bar(rmse_imp.index, rmse_imp.values, width=BAR_WIDTH,
                color=r_colors, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
        ax4.axhline(0, linewidth=1.0, color="#2b2b2b")
        ax4.set_title("Tracking RMSE Improvement by Sequence")
        ax4.set_ylabel("Improvement (%)")
        ax4.tick_params(axis="x", rotation=75)
        style_axes(ax4)
    else:
        ax4.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "dashboard_summary.png")
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
    set_style()
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
